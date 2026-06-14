# discord-ai-agent

An open-source, **self-hosted** Discord bot that brings AI agents to community servers. Deploy a customizable, LLM-powered assistant for **Q&A, auto-moderation, and role/channel management** — no machine-learning expertise required.

- **Accessible** — configure and run an AI agent with no ML background.
- **Customizable** — pick your LLM provider and model, write your own system prompt, and enable/disable modules per server.
- **Privacy-first** — your knowledge base and logs stay in *your* database; nothing leaves your host except the calls you make to your chosen LLM API.
- **Provider-agnostic** — Anthropic (Claude) and OpenAI are interchangeable behind a common interface.

> Self-hosted only. This is not a SaaS product and does not train or fine-tune models.

---

## Features

| Module | Slash commands | What it does |
|---|---|---|
| **Core config** | `/config`, `/ping` | Set the LLM provider/model per server. |
| **Agent** | `/agent prompt \| show \| reset \| clear` | Set the system prompt, inspect config, clear channel history. |
| **Conversation** | `/ask`, `@mention` | Ask the agent questions; it remembers recent context per channel. |
| **Knowledge base** | `/kb add \| search \| delete \| list` | A per-server Q&A knowledge base with SQLite FTS5 full-text search; the agent can search it automatically. |
| **Auto-moderation** | `/automod enable \| disable \| threshold \| logchannel \| status \| log` | LLM scores each message; deletes/flags violations above a threshold and logs the action. |
| **Management** | `/manage roles \| channels \| moderate \| enable \| disable \| status` | Agent-driven role/channel changes and moderation, with per-action opt-in. |

Built-in safeguards: prompt-injection hardening on the moderator, runtime permission checks on every admin command, per-user rate limiting, retry/backoff on LLM API errors, structured JSON logging with per-request guild context, and persisted conversation history that survives restarts.

---

## Tech stack

- **Python 3.12** + `asyncio`
- **discord.py ≥ 2.4** (native slash commands)
- **Anthropic** (`claude-opus-4-8` default) / **OpenAI** (`gpt-4o`) via a common `LLMProvider` protocol
- **SQLModel** + **SQLite** (`aiosqlite`), with **FTS5** search and **Alembic** migrations
- **Docker** + **docker-compose** for deployment

---

## Prerequisites

- **Python 3.12+** (for local runs) **or** **Docker** + **Docker Compose** (for containerized runs)
- A **Discord bot application** (see below)
- An API key for **Anthropic** and/or **OpenAI**

---

## 1. Create the Discord bot

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications) → **New Application**.
2. Open the **Bot** tab → **Add Bot**, then **Reset Token** and copy the token (this is your `DISCORD_BOT_TOKEN`).
3. Under **Privileged Gateway Intents**, enable **Message Content Intent** (required for `/ask` mentions and auto-moderation). Optionally enable **Server Members Intent** for full member search in `/manage`.
4. Open **OAuth2 → URL Generator**:
   - Scopes: `bot` and `applications.commands`
   - Bot Permissions: at minimum **Send Messages**, **Read Message History**; add **Manage Messages** (auto-mod deletes), **Manage Roles**, and **Manage Channels** if you want those modules.
5. Open the generated URL and invite the bot to your server.

---

## 2. Configure environment

Copy the example file and fill in your secrets:

```bash
cp .env.example .env
```

Edit `.env`:

```ini
# Required
DISCORD_BOT_TOKEN=your_discord_bot_token_here

# At least one is required to use LLM features
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Optional — defaults shown
# DATABASE_URL=sqlite+aiosqlite:///./data/bot.db
# DEFAULT_PROVIDER=anthropic
# DEFAULT_MODEL=claude-opus-4-8
# MAX_TOOL_ITERATIONS=10
# MAX_CONVERSATION_HISTORY=20
```

Additional tunables (rate limits, extended thinking, etc.) live in `bot/config.py` and can be overridden via matching env vars. Never commit `.env` — it is gitignored.

---

## 3. Run the bot

### Option A — Docker (recommended)

```bash
docker compose up --build -d
```

This builds the image, applies database migrations automatically on startup, and persists the SQLite database in the named `bot_data` volume. The container has a health check that reports unhealthy if the bot loses its connection to Discord.

View logs:

```bash
docker compose logs -f
```

### Option B — Local (Python)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m bot.main
```

Database migrations run automatically at startup; the SQLite file is created under `./data/`.

---

## 4. Register slash commands (first run)

Discord rate-limits global command sync to ~2/day, so the bot does **not** sync on every startup. Register (or re-register) the slash commands explicitly with the `--sync` flag whenever command definitions change:

```bash
# Local
python -m bot.main --sync

# Docker
docker compose run --rm bot python -m bot.main --sync
```

After syncing once, run normally (without `--sync`). Newly synced global commands can take a few minutes to appear in Discord.

---

## 5. Configure in Discord

Once the bot is online and commands are synced, an administrator can set it up:

```
/config provider:anthropic model:claude-opus-4-8
/agent prompt prompt:You are a friendly assistant for our gaming community.
/kb add title:Server rules content:Be kind. No spam. ...
/automod enable
/automod threshold value:0.8
```

Then anyone can talk to it:

```
/ask prompt:What are the server rules?
@YourBot how do I report someone?
```

---

## Development

Install dev dependencies and run the checks:

```bash
pip install -r requirements-dev.txt

pytest                 # run the test suite
ruff check .           # lint
mypy bot/              # type-check
```

Run tests with coverage (matches CI):

```bash
pytest --cov=bot --cov-report=term-missing
```

### Database migrations

The schema is managed with Alembic and applied automatically on startup. To create a new migration after changing a model in `bot/database.py`:

```bash
alembic revision --autogenerate -m "describe your change"
alembic upgrade head
```

### Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request: `ruff`, `mypy` (informational), `pytest` with a coverage floor, and a `docker build`.

---

## Project structure

```
bot/
  main.py            # entrypoint: logging, migrations, cog loading, heartbeat
  config.py          # pydantic-settings configuration
  database.py        # SQLModel tables + engine factory
  history.py         # persisted per-channel conversation history
  logging_config.py  # structured JSON logging + request context
  healthcheck.py     # container health check (Discord connectivity)
  db_migrate.py      # runs Alembic migrations at startup
  agents/            # the agentic tool loop
  llm/               # Anthropic + OpenAI providers behind LLMProvider
  commands/          # slash-command cogs
  modules/           # automod, qa (KB), manager logic
  tools/             # agent tool definitions
  services/          # guild config + agent construction
migrations/          # Alembic migration scripts
tests/               # pytest suite
constitution/        # mission, tech-stack, roadmap, code-review docs
```

---

## License

See the repository for license details.
