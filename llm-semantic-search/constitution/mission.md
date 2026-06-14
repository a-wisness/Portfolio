# Mission

## Product

**Semantic Search Studio** is an LLM-powered semantic search engine over a
private document knowledge base. A user uploads documents (PDF, Markdown, or
plain text), and the system makes that corpus searchable by *meaning* rather
than by keyword. When the user asks a question in natural language, the engine
retrieves the most semantically relevant passages and uses **Claude** to
synthesize a single, grounded answer with inline citations back to the source
material.

## The problem it solves

Traditional keyword search (`Ctrl+F`, BM25, SQL `LIKE`) fails when the user's
wording doesn't match the document's wording. Someone searching "how do I get my
money back" won't find a paragraph titled "Refund Policy." Semantic search
closes that gap by matching on intent. Layering an LLM on top turns a list of
blue links into a direct, cited answer — the difference between *finding* and
*knowing*.

## Who it's for

- **Primary (portfolio context):** technical recruiters and hiring managers
  evaluating applied-AI / ML-engineering ability. The project is a clickable,
  end-to-end demonstration of a modern RAG (Retrieval-Augmented Generation)
  system.
- **Illustrative end user:** anyone with a pile of documents — research papers,
  contracts, internal wikis, product manuals — who wants to ask questions
  instead of skim.

## What success looks like

1. A visitor can upload a document and ask a question about it within 60 seconds
   of opening the app, with zero setup.
2. Answers are **grounded** — every claim traces to a retrieved passage, and the
   UI shows those passages so the answer is verifiable, not a black box.
3. The codebase reads as production-shaped: typed APIs, separated concerns
   (ingestion / embedding / retrieval / generation), and clear documentation.

## Guiding principles

- **Grounding over fluency.** The model answers *from the retrieved context* and
  says so when the context is insufficient. No confident hallucinations.
- **Transparency.** Always surface the sources behind an answer.
- **Sensible defaults, no keys to demo embeddings.** Embeddings run locally via
  open-source models; only answer synthesis calls the Claude API.
- **Portfolio-legible.** Favor clarity and readability over cleverness — a
  reviewer should understand the system in one pass.

## Explicit non-goals

- Not a general web search engine — it searches the user's uploaded corpus only.
- Not multi-tenant or auth-gated in v1 — it's a single-user demo.
- Not a fine-tuning / model-training project — it uses off-the-shelf models.
