# Tech Stack

## Language & Runtime
| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.12 | Mature async ecosystem; both Anthropic and OpenAI SDKs are Python-first |
| Runtime | `asyncio` | discord.py is async-native |

## Bot Framework
| Component | Library | Notes |
|---|---|---|
| Discord client | `discord.py >= 2.4` | Native slash commands via `app_commands`; maintained, async |
| Config | `pydantic-settings` | Typed env-var loading; `.env` support out of the box |

## LLM Providers
| Provider | SDK | Default model |
|---|---|---|
| Anthropic (default) | `anthropic >= 0.47` | `claude-opus-4-8` |
| OpenAI | `openai >= 1.0` | `gpt-4o` |

All LLM calls go through `LLMProvider` protocol — providers are interchangeable.

## Database
| Component | Library | Notes |
|---|---|---|
| ORM | `SQLModel` | Pydantic + SQLAlchemy; async via `aiosqlite` |
| Engine | SQLite (default) | Single file, zero infra; swap to Postgres via `DATABASE_URL` |
| FTS | SQLite FTS5 | Full-text search for the Q&A knowledge base |

## Deployment
| Component | Choice |
|---|---|
| Container | Docker + docker-compose |
| Secrets | `.env` file (never committed) |
| Persistence | `./data/` volume mounted into the container |

## Development
| Tool | Use |
|---|---|
| `pytest` + `pytest-asyncio` | Async tests |
| `mypy` | Type checking |
| `ruff` | Linting + formatting |
