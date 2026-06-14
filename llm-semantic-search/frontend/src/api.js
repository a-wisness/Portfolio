// Thin wrapper around the backend API. Paths are same-origin (Vite proxies /api).

async function handle(res) {
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body.detail) detail = body.detail;
    } catch {
      // non-JSON error body; keep the default message
    }
    throw new Error(detail);
  }
  return res.json();
}

export async function ingest(file) {
  const form = new FormData();
  form.append("file", file);
  return handle(await fetch("/api/ingest", { method: "POST", body: form }));
}

export async function search(query, topK) {
  return handle(
    await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: topK ?? null }),
    })
  );
}

export async function getStats() {
  return handle(await fetch("/api/stats"));
}

export async function reset() {
  return handle(await fetch("/api/reset", { method: "POST" }));
}
