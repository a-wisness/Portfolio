// Shows the retrieved source passages backing the answer. Each is collapsible.

export default function ResultList({ sources }) {
  if (!sources || sources.length === 0) return null;
  return (
    <div className="sources">
      <div className="sources__label">Sources</div>
      {sources.map((s) => (
        <details key={s.index} className="source" open={s.index === 1}>
          <summary>
            <span className="source__num">[{s.index}]</span>
            <span className="source__file">{s.filename}</span>
            <span className="source__score">
              similarity {s.score.toFixed(2)}
            </span>
          </summary>
          <p className="source__text">{s.text}</p>
        </details>
      ))}
    </div>
  );
}
