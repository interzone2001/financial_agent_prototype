import type { ChatAnswer, CreateSessionResponse, ErrorResponse, Report } from "./types";

export class ApiError extends Error {
  status: number; // 0 = network failure
  error_code: string;
  constructor(status: number, error_code: string, message: string) {
    super(message);
    this.name = "ApiError"; this.status = status; this.error_code = error_code;
  }
}

export const THREAD_KEY = "fap.thread_id";
const useFixtures = () => import.meta.env.VITE_USE_FIXTURES === "1"; // read per call so tests can stubEnv
const fixtureFiles = import.meta.glob<{ default: unknown }>("./fixtures/*.json");

async function fixture<T>(name: "session" | "report" | "chat"): Promise<T> {
  const load = fixtureFiles[`./fixtures/${name}.json`];
  if (!load) throw new ApiError(0, "fixture_missing", `Missing fixture ${name}.json`);
  return (await load()).default as T;
}

async function toApiError(res: Response): Promise<ApiError> {
  let body: unknown = null;
  try { body = await res.json(); } catch { /* non-JSON body, e.g. proxy error page */ }
  if (body && typeof body === "object") {
    const b = body as Partial<ErrorResponse> & { detail?: unknown };
    if (typeof b.error_code === "string" && typeof b.message === "string") return new ApiError(res.status, b.error_code, b.message);
    if (typeof b.detail === "string") return new ApiError(res.status, "invalid_input", b.detail);
    if (Array.isArray(b.detail) && typeof b.detail[0]?.msg === "string") return new ApiError(res.status, "invalid_input", b.detail[0].msg);
  }
  return new ApiError(res.status, "http_error", `Request failed (HTTP ${res.status})`);
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body) });
  } catch {
    throw new ApiError(0, "network_error", "Can't reach the server. Check your connection and retry.");
  }
  if (!res.ok) throw await toApiError(res);
  return (await res.json()) as T;
}

export const api = {
  createSession: (): Promise<CreateSessionResponse> => (useFixtures() ? fixture("session") : post("/api/sessions")),
  getReport: async (thread_id: string, ticker: string): Promise<Report> => {
    if (!useFixtures()) return post("/api/report", { thread_id, ticker });
    if (ticker !== "AAPL") throw new ApiError(404, "ticker_not_found", `Ticker not found: ${ticker}`);
    return fixture("report");
  },
  chat: (thread_id: string, message: string): Promise<ChatAnswer> =>
    useFixtures() ? fixture("chat") : post("/api/chat", { thread_id, message }),
};

export async function ensureSession(): Promise<string> {
  try { const stored = sessionStorage.getItem(THREAD_KEY); if (stored) return stored; }
  catch { /* storage blocked: fall through to a fresh session */ }
  const { thread_id } = await api.createSession();
  try { sessionStorage.setItem(THREAD_KEY, thread_id); } catch { /* keep in memory only */ }
  return thread_id;
}

export function errorMessage(e: unknown): string {
  if (!(e instanceof ApiError)) return "Something went wrong. Please retry.";
  if (e.status === 404) return "Ticker not found. Check the symbol and try again.";
  if (e.status === 409) return "Generate a report first";
  return e.message;
}

export const isRetryable = (e: unknown) => e instanceof ApiError && (e.status === 0 || e.status >= 500);
