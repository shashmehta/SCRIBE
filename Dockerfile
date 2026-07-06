FROM python:3.10-slim

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

# Skip cache staleness check — no h5ad source files in the container
ENV SCRIBE_READ_ONLY=1

# Pre-render default (grey) UMAP so the app skips matplotlib on startup
RUN python -c "from scribe.cache import generate_default_umap_plot; generate_default_umap_plot()"

EXPOSE 7860
CMD ["marimo", "run", "app.py", "--host", "0.0.0.0", "--port", "7860", "--no-token"]
