import { useRef, useState } from "react";
import { ingest } from "../api.js";

export default function UploadPanel({ onIngested, onError }) {
  const inputRef = useRef(null);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");

  async function handleFiles(files) {
    if (!files || files.length === 0) return;
    setBusy(true);
    onError("");
    try {
      for (const file of files) {
        setStatus(`Indexing ${file.name}…`);
        const res = await ingest(file);
        setStatus(res.message);
      }
      onIngested();
    } catch (err) {
      onError(err.message);
      setStatus("");
    } finally {
      setBusy(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  return (
    <div className="panel">
      <h2>1 · Add documents</h2>
      <label
        className={`dropzone ${busy ? "dropzone--busy" : ""}`}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          handleFiles(e.dataTransfer.files);
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.txt,.md,.markdown"
          multiple
          disabled={busy}
          onChange={(e) => handleFiles(e.target.files)}
        />
        <span className="dropzone__hint">
          {busy ? "Working…" : "Drop a PDF, .txt, or .md here — or click to browse"}
        </span>
      </label>
      {status && <p className="status">{status}</p>}
    </div>
  );
}
