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

export async function getHealth() {
  return handle(await fetch("/api/health"));
}

export async function searchMovies(query, limit = 20) {
  const params = new URLSearchParams({ search: query, limit: String(limit) });
  return handle(await fetch(`/api/movies?${params}`));
}

export async function recommend(likedMovieIds, topK = 12) {
  return handle(
    await fetch("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ liked_movie_ids: likedMovieIds, top_k: topK }),
    })
  );
}

export async function userRecommendations(userId, topK = 12) {
  const params = new URLSearchParams({ top_k: String(topK) });
  return handle(await fetch(`/api/users/${userId}/recommendations?${params}`));
}

export async function similarMovies(movieId, topK = 12) {
  const params = new URLSearchParams({ top_k: String(topK) });
  return handle(await fetch(`/api/movies/${movieId}/similar?${params}`));
}
