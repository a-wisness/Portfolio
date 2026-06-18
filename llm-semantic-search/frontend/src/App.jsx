import { useEffect, useState } from "react";
import { getStats, reset, search } from "./api.js";
import UploadPanel from "./components/UploadPanel.jsx";
import SearchBar from "./components/SearchBar.jsx";
import AnswerCard from "./components/AnswerCard.jsx";
import ResultList from "./components/ResultList.jsx";

export default function App() {
  const [stats, setStats] = useState({ documents: [], total_chunks: 0 });
  const [result, setResult] = useState(null);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState("");

  async function refreshStats() {
    try {
      setStats(await getStats());
    } catch (err) {
      setError(err.message);
    }
  }

  useEffect(() => {
    refreshStats();
  }, []);

  async function handleSearch(query) {
    setSearching(true);
    setError("");
    setResult(null);
    try {
      setResult(await search(query));
    } catch (err) {
      setError(err.message);
    } finally {
      setSearching(false);
    }
  }

  async function handleReset() {
    if (!confirm("Clear all indexed documents?")) return;
    try {
      await reset();
      setResult(null);
      await refreshStats();
    } catch (err) {
      setError(err.message);
    }
  }

  const hasDocs = stats.total_chunks > 0;

  return (
    <div className="app">
      <header className="header">
        <h1>
          Semantic Search <span className="accent">Studio</span>
        </h1>
        <p className="tagline">
          Search your documents by meaning. Answers grounded in your sources,
          synthesized by Claude.
        </p>
      </header>

      <main className="layout">
        <aside className="sidebar">
          <UploadPanel onIngested={refreshStats} onError={setError} />

          <div className="panel">
            <h2>Indexed</h2>
            {hasDocs ? (
              <>
                <ul className="doclist">
                  {stats.documents.map((d) => (
                    <li key={d.filename}>
                      <span className="doclist__name">{d.filename}</span>
                      <span className="doclist__count">{d.chunks}</span>
                    </li>
                  ))}
                </ul>
                <p className="muted">{stats.total_chunks} chunks total</p>
                <button className="link-btn" onClick={handleReset}>
                  Clear index
                </button>
              </>
            ) : (
              <p className="muted">Nothing indexed yet.</p>
            )}
          </div>
        </aside>

        <section className="main-col">
          <h2>2 · Ask a question</h2>
          <SearchBar
            onSearch={handleSearch}
            busy={searching}
            disabled={!hasDocs}
          />

          {error && <div className="error">{error}</div>}

          {searching && (
            <div className="loading">Retrieving passages and synthesizing…</div>
          )}

          {result && !searching && (
            <div className="results">
              <AnswerCard answer={result.answer} metrics={result.metrics} />
              <ResultList sources={result.sources} />
            </div>
          )}
        </section>
      </main>

      <footer className="footer">
        Local embeddings (sentence-transformers) · Vector store (ChromaDB) ·
        Answers by Claude (claude-opus-4-8)
      </footer>
    </div>
  );
}
