// Displays a ranked list of recommendations. Each card shows the model score,
// genres, and a button to pivot to "similar movies" for that title.

export default function RecommendationGrid({ title, strategy, items, onSimilar }) {
  if (!items) return null;
  return (
    <div className="results-block">
      <div className="results-head">
        <h2>{title}</h2>
        {strategy && <span className="strategy-tag">{strategy}</span>}
      </div>
      <div className="grid">
        {items.map((m, i) => (
          <div key={m.movie_id} className="movie-card">
            <div className="rank">#{i + 1}</div>
            <div className="movie-title">{m.title}</div>
            <div className="movie-genres">{m.genres.join(" · ")}</div>
            <div className="movie-footer">
              <span className="score-badge" title="Model score">
                {m.score.toFixed(3)}
              </span>
              <button className="similar-btn" onClick={() => onSimilar(m)}>
                Similar
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
