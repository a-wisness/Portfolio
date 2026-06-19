# Austin's AI/ML Portfolio

Software developer with a background in machine learning, backend APIs, and full-stack applications. These projects demonstrate applied deep learning, LLM integration, and end-to-end system design — built from academic coursework and self-directed study.

## Skills

| Domain | Technologies |
|---|---|
| Machine Learning | TensorFlow/Keras, PyTorch, scikit-learn, Sentence Transformers |
| LLM Integration | Anthropic SDK (Claude), OpenAI SDK, ChromaDB, semantic search |
| Backend | Python 3.11+, FastAPI, Uvicorn, SQLModel, SQLite, Alembic |
| Frontend | React 18, Vite |
| Infrastructure | Docker, asyncio, aiosqlite |

---

## LeafLens — Multitask Plant Disease Analyzer

[Source](https://github.com/a-wisness/Portfolio/tree/main/leaf-disease-multitask) | [Original notebook](https://github.com/a-wisness/Portfolio/tree/main/LeafDiseaseDetection-DL)

LeafLens is a deep-learning plant-leaf analyzer built in **TensorFlow/Keras**. A user uploads a single crop-leaf photo and gets two answers from **one model in one forward pass**:

1. **Where the leaf is** — a pixel mask separating leaf from background (segmentation), shown as an overlay.
2. **What's wrong with it** — the predicted disease / leaf type out of 28 classes (classification), with a confidence score.

It ships as a Python training package, a FastAPI inference service, and a React UI (upload → mask overlay + disease label).

**Why this architecture?** The original project (`LeafDiseaseDetection-DL`, a senior Bachelor's project) used two completely separate models: a U-Net for segmentation and a MobileNetV2 classifier for disease. They shared nothing — not weights, not a pipeline, not an inference path. That is wasteful (two full backbones to train and serve) and misses an obvious synergy: knowing *where the leaf is* should help *diagnose it*. The multitask redesign trains a single shared backbone on both signals, reducing inference cost and encoding richer visual features from both tasks simultaneously.

**Results** (trained on deliberately disjoint datasets — segmentation images had no class labels; classification images had no masks):
- Segmentation: IoU **0.79**, Dice **0.86** over 705 validation images
- Classification: **57%** accuracy over 236 held-out test images across 28 classes

### Tech Stack
- ML/training — Python 3.11+ · TensorFlow/Keras · numpy · scikit-learn · Pydantic
- Backend — FastAPI · Uvicorn
- Frontend — React 18 · Vite
- Containerization — Docker

---

## CineMatch — Deep Learning Movie Recommender

[Source](https://github.com/a-wisness/Portfolio/tree/main/pytorch-recommender)

CineMatch is a deep-learning movie recommender built in **PyTorch**. It learns latent representations of users and movies from MovieLens-100k interaction data and serves personalized top-N recommendations through a FastAPI backend and a React UI. A visitor picks a few movies they like, and the system returns movies it predicts they will enjoy — with a transparent score and the movie's genres.

Built to explore the modern **collaborative filtering** approach: rather than hand-engineered rules, the model *learns* what "similar taste" means by training on implicit feedback signals (which users interacted with which movies). The project implements NeuMF (Neural Matrix Factorization) with both BCE and BPR training modes, leave-one-out evaluation, and a hyperparameter sweep.

**Results** (MovieLens-100k, NeuMF, 8 epochs, leave-one-out evaluation, catalog of 1,682 movies):
- HR@10: **0.73** · NDCG@10: **0.47**

### Tech Stack
- ML/training — Python 3.11+ · PyTorch · pandas · numpy · Pydantic
- Backend — FastAPI · Uvicorn
- Frontend — React 18 · Vite
- Containerization — Docker

---

## LLM Semantic Search

[Source](https://github.com/a-wisness/Portfolio/tree/main/llm-semantic-search)

**Semantic Search Studio** is an LLM-powered semantic search engine over a private document knowledge base. A user uploads documents (PDF, Markdown, or plain text), and the system makes that corpus searchable by *meaning* rather than by keyword. When the user asks a question in natural language, the engine retrieves the most semantically relevant passages and uses **Claude** to synthesize a single, grounded answer with inline citations back to the source material.

Traditional keyword search (`Ctrl+F`, BM25, SQL `LIKE`) fails when the user's wording doesn't match the document's wording — someone searching "how do I get my money back" won't find a paragraph titled "Refund Policy." Semantic search closes that gap by matching on intent. Layering an LLM on top turns a list of blue links into a direct, cited answer: the difference between *finding* and *knowing*.

### Tech Stack
- Backend — Python 3.11+ · FastAPI · Uvicorn · ChromaDB · Anthropic SDK · Sentence Transformers · pypdf · Pydantic
- Frontend — React 18 · Vite
- Containerization — Docker

---

## Discord AI Agent

[Source](https://github.com/a-wisness/Portfolio/tree/main/discord-ai-agent)

A self-hosted Discord bot that brings AI agents to community servers — enabling server managers to deploy customizable, LLM-powered assistants for Q&A, moderation, and channel management without requiring machine-learning expertise.

- **Accessible** — configure and run an AI agent with no ML background.
- **Customizable** — pick your LLM provider and model, write your own system prompt, and enable/disable modules per server.
- **Privacy-first** — your knowledge base and logs stay in *your* database; nothing leaves your host except the calls you make to your chosen LLM API.
- **Provider-agnostic** — Anthropic (Claude) and OpenAI are interchangeable behind a common `LLMProvider` interface.

Self-hosted only — not a SaaS product; does not train or fine-tune models.

### Tech Stack
- Python 3.12 · asyncio
- discord.py ≥ 2.4 (native slash commands)
- Anthropic (`claude-opus-4-8`) / OpenAI (`gpt-4o`) via a common `LLMProvider` protocol
- SQLModel · SQLite · aiosqlite · FTS5 search · Alembic migrations
- Docker

---

## Development Workflow

All projects except `LeafDiseaseDetection-DL` were built using **Spec-Driven Development** with Claude Code as an AI agent. Claude Sonnet 4.6 handled planning and scaffolding; Claude Opus 4.8 handled complex implementation and code review. Routing tasks between models by complexity kept token usage — and cost — materially lower than using a single model throughout.
