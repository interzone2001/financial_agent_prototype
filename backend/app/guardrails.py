"""Guardrails (spec §4.1): pure functions called by graph nodes, traced as parsers.

Advice regexes match *advisory phrasing* (subject + modal + trade verb, ratings, price
targets), not bare buy/sell/hold, so filing text like "shares held by" / "sell-through"
passes. The Haiku check on chat is the backstop (cuttable: pass llm=None).
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from app.llm import LLM
from app.models import ChatAnswer, FilingsSummary
from app.tracing import traceable

SECTIONS = ("key_developments", "risk_factors", "financial_highlights", "material_events")
_TRADE = r"(?:buy|sell|hold|invest|purchase|short|accumulate|trim)"

INJECTION_PATTERNS = [
    (r"\b(?:ignore|disregard|forget|override)\s+(?:(?:all|any|the|your|of)\s+)*"
     r"(?:previous|prior|above|earlier|preceding|system)\s+(?:instructions?|prompts?|messages?|rules)"),
    r"\bsystem\s+prompt\b",
    (r"\b(?:reveal|print|show|repeat)\s+(?:me\s+)?(?:your|the)\s+(?:hidden\s+|initial\s+)?"
     r"(?:instructions|prompt)\b"),
    r"\byou\s+are\s+now\b",
    r"\bpretend\s+(?:to\s+be|you\s+are)\b",
    r"\b(?:developer|god|jailbreak)\s+mode\b|\bjailbreak\b",
    r"^\s*(?:system|assistant|developer)\s*:",
    r"</?\s*(?:system|instructions?|im_start|im_end)\s*>",
    r"\bnew\s+instructions\s*:",
]
ADVICE_PATTERNS = [
    (rf"\b(?:you|investors?|clients?|advisors?|one)\s+(?:should|must|ought\s+to|might\s+want\s+to)"
     rf"\s+(?:consider\s+)?{_TRADE}\b"),
    rf"\bshould\s+(?:i|we|you)\s+{_TRADE}\b",
    (r"\b(?:i|we)(?:'d|\s+would)?\s+(?:recommend|suggest|advise)\s+"
     r"(?:buying|selling|holding|investing|purchasing|shorting)\b"),
    r"\b(?:buy|sell|hold)\s+(?:rating|recommendation)\b",
    r"\bstrong\s+(?:buy|sell)\b",
    r"\bprice\s+target\b",
    r"\b(?:go|going|stay|be)\s+(?:overweight|underweight)\b",
    (r"\b(?:is|looks|seems)\s+(?:like\s+)?(?:an?\s+)?(?:good|great|attractive|compelling|bad|poor)"
     r"\s+(?:buy|investment|entry\s+point)\b"),
    r"\b(?:good|right|great)\s+time\s+to\s+(?:buy|sell)\b",
    r"\b(?:buy|sell)\s+(?:it\s+)?now\b",
]
_INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in INJECTION_PATTERNS),
                           re.IGNORECASE | re.MULTILINE)
_ADVICE_RE = re.compile("|".join(f"(?:{p})" for p in ADVICE_PATTERNS), re.IGNORECASE)

INJECTION_REFUSAL = ("I can only answer questions about this security's market data and SEC "
                     "filings, and I can't change how I operate.")
ADVICE_REFUSAL = ("I can't provide investment advice or recommendations. I can explain what the "
                  "SEC filings and market data say, for example risk factors or recent results.")
OFF_TOPIC_REFUSAL = ("I can only help with factual questions about {ticker}'s market data and SEC "
                     "filings, and I can't give investment advice.")
ADVICE_CHECK_SYSTEM = (
    "You are a compliance checker for a wealth-management research tool. Decide whether the text "
    "gives investment advice: an explicit or implied recommendation to buy, sell, hold or size a "
    "position, a price target, or a judgement that the security is a good/bad investment. "
    "Factual reporting of filings or market data is NOT advice.")


class AdviceCheck(BaseModel):
    is_advice: bool
    reason: str


@traceable(run_type="parser", name="guardrails.detect_injection")
def detect_injection(text: str) -> bool:
    return bool(_INJECTION_RE.search(text))


@traceable(run_type="parser", name="guardrails.contains_advice")
def contains_advice(text: str) -> bool:
    return bool(_ADVICE_RE.search(text))


@traceable(run_type="parser", name="guardrails.check_summary")
def check_summary(summary: FilingsSummary) -> tuple[FilingsSummary, list[str]]:
    """Drop claims citing ids outside citation_index, or reading as advice."""
    warnings: list[str] = []
    kept: dict[str, list] = {}
    for field in SECTIONS:
        kept[field] = []
        for claim in getattr(summary, field):
            if not claim.citations or any(c not in summary.citation_index for c in claim.citations):
                warnings.append(f"Dropped an ungrounded claim in {field} (unknown sources).")
            elif contains_advice(claim.text):
                warnings.append(f"Dropped a claim in {field} that read as investment advice.")
            else:
                kept[field].append(claim)
    return summary.model_copy(update=kept), warnings


@traceable(run_type="parser", name="guardrails.llm_advice_check")
def llm_advice_check(text: str, llm: LLM) -> bool:
    msgs = [{"role": "user", "content": f"<text>{text}</text>"}]
    return llm.parse("fast", ADVICE_CHECK_SYSTEM, msgs, AdviceCheck).is_advice


@traceable(run_type="parser", name="guardrails.check_chat_answer")
def check_chat_answer(answer: ChatAnswer, llm: LLM | None) -> ChatAnswer:
    """Advice (regex, then Haiku if llm) -> refusal; filings answers need >=1 chunk citation."""
    if answer.route == "off_topic":
        return answer
    if contains_advice(answer.text) or (llm is not None and llm_advice_check(answer.text, llm)):
        return ChatAnswer(text=ADVICE_REFUSAL, route=answer.route, warnings=[
            *answer.warnings, "Answer withheld: it read as investment advice."])
    warnings = list(answer.warnings)
    if answer.route in ("filings", "both") and not any(c.chunk_id for c in answer.citations):
        warnings.append("This answer is not backed by a cited filing passage; "
                        "verify against the original filings.")
    return answer.model_copy(update={"warnings": warnings})
