import { useState } from "react";

export default function SearchBar({ onSearch, busy, disabled }) {
  const [value, setValue] = useState("");

  function submit(e) {
    e.preventDefault();
    const q = value.trim();
    if (q) onSearch(q);
  }

  return (
    <form className="searchbar" onSubmit={submit}>
      <input
        type="text"
        className="searchbar__input"
        placeholder={
          disabled
            ? "Upload a document first…"
            : "Ask a question about your documents…"
        }
        value={value}
        disabled={disabled || busy}
        onChange={(e) => setValue(e.target.value)}
      />
      <button
        type="submit"
        className="searchbar__btn"
        disabled={disabled || busy || !value.trim()}
      >
        {busy ? "Thinking…" : "Search"}
      </button>
    </form>
  );
}
