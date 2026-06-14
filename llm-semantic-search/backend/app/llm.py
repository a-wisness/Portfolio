"""Claude answer synthesis.

Given a user question and the passages retrieved from the vector store, Claude
writes a single grounded answer that cites its sources as [n]. The model is
instructed to answer *only* from the provided context and to say so when the
context doesn't contain the answer — this is what keeps the system honest
instead of hallucinating.
"""

from __future__ import annotations

from functools import lru_cache

from .config import settings

SYSTEM_PROMPT = """You are a precise research assistant for a semantic search \
engine. Answer the user's question using ONLY the numbered context passages \
provided. Follow these rules:

- Ground every claim in the passages. Cite them inline with bracketed numbers \
like [1] or [2][3] that match the passage numbers.
- If the passages do not contain enough information to answer, say so plainly \
(e.g. "The documents don't appear to cover this.") rather than guessing.
- Be concise and direct. Lead with the answer, then any supporting detail.
- Do not invent sources, facts, or citation numbers that aren't in the context.
"""


@lru_cache(maxsize=1)
def _client():
    # Imported lazily so the module loads without the Anthropic SDK present
    # (e.g. in tests that stub out synthesis).
    import anthropic

    # Reads ANTHROPIC_API_KEY from the environment.
    return anthropic.Anthropic(api_key=settings.anthropic_api_key or None)


def _format_context(passages: list[dict]) -> str:
    blocks = []
    for i, p in enumerate(passages, start=1):
        blocks.append(
            f"[{i}] (source: {p['filename']})\n{p['text']}"
        )
    return "\n\n".join(blocks)


def synthesize_answer(query: str, passages: list[dict]) -> str:
    """Call Claude to produce a cited answer from the retrieved passages."""
    if not passages:
        return "I couldn't find anything relevant in the indexed documents."

    user_prompt = (
        f"Question: {query}\n\n"
        f"Context passages:\n\n{_format_context(passages)}\n\n"
        "Write a grounded, cited answer to the question."
    )

    response = _client().messages.create(
        model=settings.claude_model,
        max_tokens=settings.max_answer_tokens,
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    parts = [block.text for block in response.content if block.type == "text"]
    return "\n".join(parts).strip()
