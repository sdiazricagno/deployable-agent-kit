FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 AGENTKIT_DATA_DIR=/app/data

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY configs ./configs
COPY data ./data
COPY evals ./evals

RUN useradd --create-home agent && mkdir -p /app/traces && chown -R agent /app
USER agent

EXPOSE 8000
CMD ["sh", "-c", "agentkit serve ${AGENTKIT_CONFIG} --host 0.0.0.0 --port 8000"]
