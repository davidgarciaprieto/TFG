# ── Dockerfile — NER Filter App ──────────────────────────────────────────────
# Imagen base: Python 3.11 slim (ligera, sin extras innecesarios)
FROM python:3.11-slim

# Metadatos
LABEL maintainer="davidgarciaprieto"
LABEL description="Aplicación de filtrado semiautomático de entidades nombradas"

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Evita que spaCy y HuggingFace pidan interacción
    SPACY_QUIET=1 \
    TRANSFORMERS_VERBOSITY=error \
    # Directorio de caché de HuggingFace dentro del contenedor
    HF_HOME=/app/.cache/huggingface \
    # Puerto de Flask
    FLASK_PORT=5000

WORKDIR /app

# ── 1. Dependencias del sistema ───────────────────────────────────────────────
# gcc y g++ son necesarios para compilar algunas extensiones de spaCy/stanza
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# ── 2. Dependencias Python ────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── 3. Modelos spaCy ──────────────────────────────────────────────────────────
# Solo descargamos el modelo large de español por defecto.
# Los modelos HuggingFace (BETO, SpanBERTa, XLM-RoBERTa) se descargan
# automáticamente la primera vez que se usan (se cachean en HF_HOME).
# Si quieres incluir más modelos en la imagen, añade líneas aquí:
#   RUN python -m spacy download en_core_web_lg
RUN python -m spacy download es_core_news_lg

# ── 4. Código de la aplicación ────────────────────────────────────────────────
COPY app.py \
     ner_model.py \
     entity_processor.py \
     entity_filter.py \
     entity_linker.py \
     ./
COPY templates/ ./templates/

# ── 5. Directorio de caché persistente ───────────────────────────────────────
# Los modelos HuggingFace y Stanza se guardarán aquí.
# Montar como volumen en producción para no re-descargarlos en cada arranque.
RUN mkdir -p /app/.cache/huggingface /app/.cache/stanza

# ── 6. Puerto ─────────────────────────────────────────────────────────────────
EXPOSE 5000

# ── 7. Comando de arranque ────────────────────────────────────────────────────
# Para producción usa gunicorn (más estable que el servidor de desarrollo Flask)
# Si no tienes gunicorn en requirements.txt, cambia a:
#   CMD ["python", "app.py"]
CMD ["python", "app.py"]
