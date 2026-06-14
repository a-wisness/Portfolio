# Mission

Build an open-source, self-hosted Discord bot that brings AI agents to community servers — enabling server managers to deploy customizable, LLM-powered assistants for Q&A, moderation, and channel management without requiring machine-learning expertise.

## Core Goals

- **Accessibility** — any server manager can configure and run an AI agent with no ML background
- **Customizability** — operators choose the LLM provider and model, write their own system prompts, and enable/disable modules per server
- **Privacy-first** — all data (KB, logs) stays in the operator's own database; no data sent to third parties beyond the chosen LLM API
- **Extensibility** — the module system lets contributors add new capabilities without touching core logic

## Non-Goals

- Replacing Discord's built-in features (roles, channels, permissions)
- Building a SaaS product — this is self-hosted only
- Training or fine-tuning models
