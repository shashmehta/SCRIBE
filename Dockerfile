FROM python:3.10-slim

# g++ is required to compile annoy (a scanorama dependency) from source
RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
COPY scribe/ ./scribe/
RUN pip install --no-cache-dir -e .

COPY app.py .
COPY web/ ./web/

# Symlink bundled small cache to the path scribe/cache.py expects (~/.scribe/cache)
RUN mkdir -p /root/.scribe && ln -s /app/web/cache /root/.scribe/cache

# Point SCRIBE_OUTPUT_DIR to web/ so output/plots/ resolves to /app/web/plots/
ENV SCRIBE_OUTPUT_DIR=/app/web

# App dir for resolving bundled assets (poster) regardless of marimo's CWD
ENV SCRIBE_APP_DIR=/app

# Skip cache staleness check — no h5ad source files in the container
ENV SCRIBE_READ_ONLY=1

EXPOSE 7860
CMD ["marimo", "run", "app.py", "--host", "0.0.0.0", "--port", "7860", "--no-token"]
