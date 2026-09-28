export function Warnings({ warnings }: { warnings: string[] }) {
  if (warnings.length === 0) return null;
  return (
    <div className="warnings" role="alert">
      <strong>Heads up:</strong>
      <ul>{warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>
    </div>
  );
}

export function Disclaimer({ text }: { text: string }) {
  return <p className="disclaimer">{text}</p>;
}
