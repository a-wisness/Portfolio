// Renders the synthesized answer, styling inline [n] citation markers, plus a
// per-query performance footer (stage latency, token usage, estimated cost).

function renderWithCitations(text) {
  // Split on bracketed numbers like [1] or [12], keeping the delimiters.
  const parts = text.split(/(\[\d+\])/g);
  return parts.map((part, i) => {
    if (/^\[\d+\]$/.test(part)) {
      return (
        <sup key={i} className="citation">
          {part}
        </sup>
      );
    }
    return <span key={i}>{part}</span>;
  });
}

function fmtMs(ms) {
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${Math.round(ms)}ms`;
}

function MetricsBar({ metrics }) {
  return (
    <div className="metrics" title="Per-query performance">
      <span className="metrics__item">embed {fmtMs(metrics.embed_ms)}</span>
      <span className="metrics__sep">·</span>
      <span className="metrics__item">retrieve {fmtMs(metrics.retrieve_ms)}</span>
      <span className="metrics__sep">·</span>
      <span className="metrics__item">Claude {fmtMs(metrics.synthesize_ms)}</span>
      <span className="metrics__sep">·</span>
      <span className="metrics__item metrics__item--total">
        total {fmtMs(metrics.total_ms)}
      </span>
      <span className="metrics__sep">·</span>
      <span className="metrics__item">
        {metrics.input_tokens.toLocaleString()} in /{" "}
        {metrics.output_tokens.toLocaleString()} out
      </span>
      <span className="metrics__sep">·</span>
      <span className="metrics__item">
        ~${metrics.estimated_cost_usd.toFixed(4)}
      </span>
    </div>
  );
}

export default function AnswerCard({ answer, metrics }) {
  if (!answer) return null;
  return (
    <div className="answer">
      <div className="answer__label">Answer</div>
      <div className="answer__body">{renderWithCitations(answer)}</div>
      {metrics && <MetricsBar metrics={metrics} />}
    </div>
  );
}
