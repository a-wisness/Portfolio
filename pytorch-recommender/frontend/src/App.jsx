import { useCallback, useEffect, useState } from "react";
import { getHealth, recommend, similarMovies } from "./api.js";
import MovieSearch from "./components/MovieSearch.jsx";
import LikedList from "./components/LikedList.jsx";
import RecommendationGrid from "./components/RecommendationGrid.jsx";

export default function App() {
  const [liked, setLiked] = useState([]);
  const [recBlock, setRecBlock] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [health, setHealth] = useState(null);

  useEffect(() => {
    getHealth().then(setHealth).catch((e) => setError(e.message));
  }, []);

  const likedIds = new Set(liked.map((m) => m.movie_id));

  const addLike = (movie) =>
    setLiked((prev) =>
      prev.some((m) => m.movie_id === movie.movie_id) ? prev : [...prev, movie]
    );
  const removeLike = (id) =>
    setLiked((prev) => prev.filter((m) => m.movie_id !== id));

  async function handleRecommend() {
    setBusy(true);
    setError("");
    try {
      const res = await recommend([...likedIds], 12);
      setRecBlock({
        title: "Recommended for you",
        strategy: res.strategy,
        items: res.recommendations,
      });
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  const handleSimilar = useCallback(async (movie) => {
    setBusy(true);
    setError("");
    try {
      const res = await similarMovies(movie.movie_id, 12);
      setRecBlock({
        title: `Similar to “${movie.title}”`,
        strategy: res.strategy,
        items: res.recommendations,
      });
      window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }, []);

  const modelMissing = health && !health.model_loaded;

  return (
    <div className="app">
      <header className="header">
        <h1>
          Cine<span className="accent">Match</span>
        </h1>
        <p className="tagline">
          Movie recommendations from a Neural Collaborative Filtering model,
          trained in PyTorch on MovieLens.
        </p>
        {health && health.model_loaded && health.metrics && (
          <p className="metrics-line">
            {health.num_users.toLocaleString()} users ·{" "}
            {health.num_items.toLocaleString()} movies · HR@10{" "}
            {health.metrics["hr@10"]?.toFixed(3)} · NDCG@10{" "}
            {health.metrics["ndcg@10"]?.toFixed(3)}
          </p>
        )}
      </header>

      {modelMissing && (
        <div className="banner">
          No trained model is loaded. Run{" "}
          <code>docker compose run --rm backend python -m app.train</code> (or{" "}
          <code>python -m app.train</code> locally), then refresh.
        </div>
      )}

      {error && <div className="error">{error}</div>}

      <main className="layout">
        <aside className="sidebar">
          <MovieSearch onAdd={addLike} likedIds={likedIds} onError={setError} />
          <LikedList
            liked={liked}
            onRemove={removeLike}
            onRecommend={handleRecommend}
            busy={busy}
          />
        </aside>

        <section className="main-col">
          {recBlock ? (
            <RecommendationGrid
              title={recBlock.title}
              strategy={recBlock.strategy}
              items={recBlock.items}
              onSimilar={handleSimilar}
            />
          ) : (
            <div className="empty-state">
              <p>
                Search for movies you enjoy, add a few, then hit{" "}
                <strong>Recommend</strong>. Or click <strong>Similar</strong> on
                any result to explore the learned embedding space.
              </p>
            </div>
          )}
        </section>
      </main>

      <footer className="footer">
        PyTorch NeuMF (GMF + MLP) · implicit feedback · FastAPI · React
      </footer>
    </div>
  );
}
