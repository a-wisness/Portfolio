# Roadmap

## Phase 0 — Foundation (current)
- [x] Project scaffold: `bot/`, `tests/`, `constitution/`
- [x] Config via pydantic-settings (`DISCORD_BOT_TOKEN`, API keys)
- [x] Multi-provider LLM abstraction (`LLMProvider` protocol)
- [x] SQLModel schema: `GuildConfig`, `KBEntry`, `ModLog`
- [x] Docker + docker-compose
- [ ] Bot connects to Discord and responds to a ping

## Phase 1 — Bot Core + Provider Router
- [x] `/config` slash command to set model provider and model name
- [x] `/agent prompt` / `/agent show` / `/agent reset` slash commands
- [x] Agentic loop in `bot/agents/base.py`
- [x] Provider router — `bot/services/guild.py` reads guild config from DB and routes to the right provider

## Phase 2 — Conversational Agent
- [x] `/ask` slash command for direct LLM queries
- [x] Conversation history per channel (last N turns, bounded deque in `bot/history.py`)
- [x] Mention handling (`@BotName <text>` → same path as `/ask`)
- [x] `/agent clear` to wipe channel history

## Phase 3 — Q&A Knowledge Base
- [x] `/kb add | search | delete | list` commands (`bot/commands/kb.py`)
- [x] FTS5 search via `kbentry_fts` content table (`bot/modules/qa.py`)
- [x] Agent has `search_knowledge_base` tool when `qa_enabled=True`
- [x] `/ask` arms the KB tool before each agent run; agent decides when to use it

## Phase 4 — Auto-Moderation
- [x] `/automod enable | disable | threshold | logchannel | status` commands
- [x] `on_message` classifier: guild's LLM scores each message; acts if `severity >= threshold`
- [x] Actions logged to `ModLog` table; posted to configured log channel
- [x] `/automod log [limit]` command to review recent moderation actions

## Phase 5 — Role & Channel Management
- [x] `/manage roles` — agent can suggest or apply role changes
- [x] `/manage channels` — agent can create/archive channels on request
- [x] Requires explicit operator opt-in per action type (`/manage enable <action>`)

---

## Remediation Plan — Code Review 2026-06-14

Addresses all 33 findings in `constitution/code-review.md`, phased by severity.
Each item is tagged with its review number (`CR#`). Do phases in order;
within a phase, items are independent unless noted.

### Phase R0 — Critical Hotfixes (do first) — ✅ done 2026-06-14
- [x] CR1 — Wrap untrusted `message.content` in `<message>` delimiters; system prompt now treats the tagged user turn as untrusted data, and `_wrap_untrusted()` escapes forged delimiters (`bot/modules/automod.py`)
- [x] CR2 — Added shared `require_permissions` / `require_admin` guards in `bot/utils.py`; wired a runtime check into every privileged handler in `admin.py`, `agent.py`, `automod.py`, `manage.py` (admin + manage_messages commands)
- [x] CR3 — Added `ALLOWED_MODELS` allowlist + `is_valid_provider` / `is_valid_model` in `llm/provider.py`; `/config` uses `app_commands.choices` for provider and validates the model before persisting (`commands/admin.py`)
- [x] CR4 — `/ask`, mention handler, and `/manage` now log the exception server-side and send a generic message; no raw `{exc}` reaches users (`commands/ask.py`, `commands/manage.py`)
- [x] CR5 — **No change needed:** the DB tests are already `async def` and genuinely execute (suite is 52 green, DB assertions real). The review described an earlier state; verified resolved.

### Phase R1 — Correctness & Architecture (this sprint) — ✅ done 2026-06-14
- [x] CR6 — Added `build_assistant_turn` / `build_tool_results_turn` to the `LLMProvider` protocol; both providers emit a neutral block format and OpenAI's `_to_api_messages` maps it to `tool_calls` + `role="tool"` messages. `agents/base.py` no longer does `isinstance` branching (`llm/provider.py`, `llm/anthropic.py`, `llm/openai.py`, `agents/base.py`)
- [x] CR7 — Thinking is now gated on `settings.anthropic_extended_thinking`, a per-call `extended_thinking` flag, and a `max_tokens >= 1024` floor; automod calls with `extended_thinking=False`, `max_tokens=512`, and warns on empty content (`llm/anthropic.py`, `modules/automod.py`, `config.py`)
- [x] CR8 — Anthropic `Response` now carries `thinking_blocks`; `build_assistant_turn` re-emits them ahead of text/tool_use blocks (`llm/anthropic.py`, `llm/provider.py`)
- [x] CR9 — Deleted `agents/registry.py` and all `registry.invalidate()` calls (`commands/admin.py`, `commands/agent.py`)
- [x] CR10 — Engine is now lazily built via `get_engine()`; `init_db(database_url=...)` can rebind it for tests (`database.py`)
- [x] CR11 — `get_or_create_config` wraps the insert in `try/except IntegrityError` and re-reads the winner on conflict (`services/guild.py`)
- [x] CR12 — Rewrote KB search to `JOIN kbentry_fts` + `WHERE guild_id` + `ORDER BY rank` + `LIMIT` last, so the guild filter and relevance both apply before truncation (`modules/qa.py`)
- [x] CR13 — `tree.sync()` now runs only with the `--sync` flag (`main.py`)
- [x] CR14 — Added `RateLimiter` (sliding window keyed on `(guild_id, user_id)`); wired into `/ask`, the mention handler, and automod `on_message`, with limits in `config.py` (`utils.py`, `commands/ask.py`, `commands/automod.py`)

### Phase R2 — Code Quality — ✅ done 2026-06-14
- [x] CR15 — Extracted `split_message()` + `MSG_LIMIT` into `bot/utils.py`; `commands/ask.py` and `commands/manage.py` import it (duplicate `_split` removed)
- [x] CR16 — Deleted unused `get_guild_agent()`; added `make_moderation_tools()` (real `warn_user` / `delete_message` impls that act + log to ModLog) and a new admin-only `/manage moderate` command that arms them (`services/guild.py`, `modules/manager.py`, `commands/manage.py`)
- [x] CR17 — `GuildConfig.updated_at` now uses a SQLAlchemy `onupdate` hook; advances on every mutation (`database.py`)
- [x] CR18 — `agents/base.py` uses `inspect.isawaitable(result)` instead of `hasattr(..., "__await__")`
- [x] CR19 — OpenAI tool-argument `json.loads` wrapped in `try/except`; logs and falls back to `{}` (`llm/openai.py`)
- [x] CR20 — Added a shared `retrying()` policy (tenacity: exp backoff, 4 attempts) on both providers' API calls for 429/5xx/connection errors; `tenacity>=8.0` added to `requirements.txt` (`llm/provider.py`, `llm/anthropic.py`, `llm/openai.py`)
- [x] CR21 — `GuildConfig.provider` / `model` use `default_factory` from `settings`, single-sourcing the default (`database.py`)

### Phase R3 — Testing — ✅ done 2026-06-14
- [x] CR23 — `tests/test_agent_loop.py` covers single-turn, multi-turn tool loop, max-iterations fallback, unknown-tool, tool-exception swallowing, and sync vs async tool fns (the provider-agnostic continuation is also exercised in `test_r1.py`)
- [x] CR24 — `tests/test_providers_contract.py` parametrizes both providers: shared `build_assistant_turn` / `build_tool_results_turn` format contract, plus `complete()` parsing with the SDK client mocked (`_create` patched)
- [x] CR25 — `conftest.py` uses `tempfile.mkstemp()` (race-free) and removes the file + disposes the engine after each test
- [x] CR26 — `_isolated_db` fixture is now function-scoped: each test gets a fresh DB via `init_db(url)`, eliminating cross-test pollution / reused-guild-ID collisions

### Phase R4 — Observability & Persistence — ✅ done 2026-06-14
- [x] CR28 — Both providers time each call and log `provider`/`model`/`input_tokens`/`output_tokens`/`latency_ms` via structured `extra=` (`llm/anthropic.py`, `llm/openai.py`)
- [x] CR29 — New `bot/logging_config.py`: `JsonFormatter` + `ContextFilter` + `request_context()` (contextvars) injecting `guild_id`/`request_id`; `main.py` calls `setup_logging()`; LLM-triggering handlers (`/ask`, mention, `/manage *`, automod `on_message`) wrap work in `request_context(...)`
- [x] CR30 — Conversation history now persists in a `ConversationMessage` table (async DB-backed `history.get/add_turn/clear`, bounded to `max_conversation_history` per channel); survives restarts. Callers in `ask.py`/`agent.py` updated to `await` (`database.py`, `history.py`)

### Phase R5 — Infrastructure & Migrations — ✅ done 2026-06-14
- [x] CR22 — Added Alembic (`alembic.ini`, `migrations/env.py` async→sync URL, `migrations/versions/0001_initial.py` baseline incl. the FTS5 table). `bot/db_migrate.run_migrations()` runs `upgrade head` at startup from `main.setup_hook` (replacing the prod `create_all`); tests still bootstrap via `init_db`. Verified: fresh-DB upgrade creates all tables + stamps version; re-run is a no-op
- [x] CR31 — `docker-compose.yml` uses named volume `bot_data:/app/data` (dropped the host bind mount and the obsolete `version` key)
- [x] CR32 — Bot touches a heartbeat file every 30s only while gateway-ready (`tasks.loop` in `main.py`); `bot/healthcheck.py` exits non-zero if it's missing/stale; Dockerfile `HEALTHCHECK` runs it. So a running-but-disconnected bot reports unhealthy
- [x] CR33 — `requirements.txt` pinned with upper bounds (`<3.0` etc.) on every dependency; `alembic>=1.13,<2.0` added; Dockerfile copies `alembic.ini` + `migrations/`

### CI — ✅ done 2026-06-14 (was deferred)
- [x] CR27 — `.github/workflows/ci.yml`: `lint-test` job (ruff → mypy → pytest+coverage) and a `docker-build` job. Blocking gates: `ruff check .`, `pytest --cov=bot --cov-fail-under=40` (currently 44%), and `docker build`. `mypy bot/` runs non-blocking (`continue-on-error`) over the existing discord.py/SQLModel typing friction — surfaced every run, tracked as tech debt. Added `pytest-cov` to `requirements-dev.txt`, `mypy.ini`, and gitignored `coverage.xml`. Verified locally: ruff clean, 91 tests pass at 44% coverage, and `docker build` + in-container migrate/healthcheck succeed
  - **Follow-up:** ratchet `--cov-fail-under` up over time; do a dedicated pass to clear the ~30 mypy errors, then flip mypy to blocking
