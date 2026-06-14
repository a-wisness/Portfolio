export default function LikedList({ liked, onRemove, onRecommend, busy }) {
  return (
    <div className="panel">
      <h2>2 · Your picks</h2>
      {liked.length === 0 ? (
        <p className="muted">Add a few movies to get recommendations.</p>
      ) : (
        <>
          <ul className="chip-list">
            {liked.map((m) => (
              <li key={m.movie_id} className="chip">
                {m.title}
                <button
                  className="chip-remove"
                  aria-label={`Remove ${m.title}`}
                  onClick={() => onRemove(m.movie_id)}
                >
                  ×
                </button>
              </li>
            ))}
          </ul>
          <button className="primary-btn" onClick={onRecommend} disabled={busy}>
            {busy ? "Recommending…" : `Recommend from ${liked.length} movie(s)`}
          </button>
        </>
      )}
    </div>
  );
}
