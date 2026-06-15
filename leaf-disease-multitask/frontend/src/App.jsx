import { useEffect, useRef, useState } from "react";

const API = "/api";

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [health, setHealth] = useState(null);
  const inputRef = useRef(null);

  useEffect(() => {
    fetch(`${API}/health`)
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth({ status: "down", model_loaded: false }));
  }, []);

  function chooseFile(f) {
    if (!f) return;
    setFile(f);
    setResult(null);
    setError(null);
    setPreviewUrl(URL.createObjectURL(f));
  }

  async function analyze() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const body = new FormData();
      body.append("file", file);
      const res = await fetch(`${API}/predict`, { method: "POST", body });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `Request failed (${res.status})`);
      }
      setResult(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <header>
        <h1>
          Leaf<span className="accent">Lens</span>
        </h1>
        <p className="tagline">
          One model — segments the leaf <em>and</em> diagnoses it.
        </p>
        {health && !health.model_loaded && (
          <p className="warn">
            No trained model loaded on the server. Run{" "}
            <code>python -m app.train</code> first.
          </p>
        )}
      </header>

      <section
        className="dropzone"
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          chooseFile(e.dataTransfer.files?.[0]);
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          hidden
          onChange={(e) => chooseFile(e.target.files?.[0])}
        />
        {previewUrl ? (
          <img src={previewUrl} alt="selected leaf" className="thumb" />
        ) : (
          <p>Drop a leaf photo here, or click to choose one.</p>
        )}
      </section>

      <button className="primary" disabled={!file || loading} onClick={analyze}>
        {loading ? "Analyzing…" : "Analyze leaf"}
      </button>

      {error && <p className="error">{error}</p>}

      {result && (
        <section className="results">
          <div className="images">
            <figure>
              <img
                src={`data:image/png;base64,${result.overlay_png_base64}`}
                alt="mask overlay"
              />
              <figcaption>Segmented leaf (overlay)</figcaption>
            </figure>
            <figure>
              <img
                src={`data:image/png;base64,${result.mask_png_base64}`}
                alt="binary mask"
                className="mask"
              />
              <figcaption>Predicted mask</figcaption>
            </figure>
          </div>

          <div className="diagnosis">
            <h2>{result.predicted_class}</h2>
            <p className="confidence">
              {(result.confidence * 100).toFixed(1)}% confidence
            </p>
            <p className="coverage">
              Leaf covers {(result.leaf_coverage * 100).toFixed(0)}% of the frame
            </p>
            <ul className="topk">
              {result.top_k.map((t) => (
                <li key={t.label}>
                  <span className="bar" style={{ width: `${t.confidence * 100}%` }} />
                  <span className="label">{t.label}</span>
                  <span className="pct">{(t.confidence * 100).toFixed(1)}%</span>
                </li>
              ))}
            </ul>
          </div>
        </section>
      )}

      <footer>
        Multi-task MobileNetV2 — shared encoder, segmentation + classification heads.
      </footer>
    </div>
  );
}
