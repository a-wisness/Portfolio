import { useEffect, useState } from "react";
import { searchMovies } from "../api.js";

export default function MovieSearch({ onAdd, likedIds, onError }) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);

  useEffect(() => {
    let active = true;
    const handle = setTimeout(async () => {
      try {
        const movies = await searchMovies(query, 25);
        if (active) setResults(movies);
      } catch (err) {
        onError(err.message);
      }
    }, 200); // debounce
    return () => {
      active = false;
      clearTimeout(handle);
    };
  }, [query, onError]);

  return (
    <div className="panel">
      <h2>1 · Pick movies you like</h2>
      <input
        className="text-input"
        type="text"
        placeholder="Search the catalog…"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      <ul className="result-list">
        {results.map((m) => {
          const added = likedIds.has(m.movie_id);
          return (
            <li key={m.movie_id} className="result-row">
              <div className="result-meta">
                <span className="result-title">{m.title}</span>
                <span className="result-genres">{m.genres.join(" · ")}</span>
              </div>
              <button
                className="add-btn"
                disabled={added}
                onClick={() => onAdd(m)}
              >
                {added ? "Added" : "+ Add"}
              </button>
            </li>
          );
        })}
        {results.length === 0 && <li className="muted">No matches.</li>}
      </ul>
    </div>
  );
}
