FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot/ bot/
COPY alembic.ini .
COPY migrations/ migrations/

RUN mkdir -p data

RUN useradd -m -u 1000 botuser && chown -R botuser:botuser /app
USER botuser

# Reports unhealthy if the bot process is up but disconnected from the Discord
# gateway (the heartbeat file goes stale). start-period covers login + migrations.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-m", "bot.healthcheck"]

CMD ["python", "-m", "bot.main"]
