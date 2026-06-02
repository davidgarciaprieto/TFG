# ── Dockerfile — NER Filter App (modelos pre-descargados, CPU only) ────────────
FROM python:3.11-slim

LABEL maintainer="davidgarciaprieto"
LABEL description="Aplicación de filtrado semiautomático de entidades nombradas"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SPACY_QUIET=1 \
    TRANSFORMERS_VERBOSITY=error \
    HF_HOME=/app/.cache/huggingface \
    STANZA_RESOURCES_DIR=/app/.cache/stanza \
    FLASK_PORT=5000

WORKDIR /app

# ── 1. Dependencias del sistema ───────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# ── 2. PyTorch CPU (sin CUDA, mucho más ligero: ~200 MB en vez de ~2 GB) ─────
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# ── 3. Resto de dependencias Python ──────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── 4. Modelos spaCy ──────────────────────────────────────────────────────────
RUN python -m spacy download es_core_news_lg

# ── 5. Modelos HuggingFace (BETO, SpanBERTa, XLM-RoBERTa) ───────────────────
RUN python -c "\
from transformers import AutoTokenizer, AutoModelForTokenClassification; \
print('Descargando BETO...'); \
AutoTokenizer.from_pretrained('mrm8488/bert-spanish-cased-finetuned-ner'); \
AutoModelForTokenClassification.from_pretrained('mrm8488/bert-spanish-cased-finetuned-ner'); \
print('Descargando SpanBERTa...'); \
AutoTokenizer.from_pretrained('MMG/xlm-roberta-large-ner-spanish'); \
AutoModelForTokenClassification.from_pretrained('MMG/xlm-roberta-large-ner-spanish'); \
print('Descargando XLM-RoBERTa...'); \
AutoTokenizer.from_pretrained('xlm-roberta-large-finetuned-conll03-english'); \
AutoModelForTokenClassification.from_pretrained('xlm-roberta-large-finetuned-conll03-english'); \
print('Modelos HuggingFace OK'); \
"

# ── 6. Modelo Stanza ──────────────────────────────────────────────────────────
RUN python -c "\
import stanza; \
print('Descargando Stanza ES...'); \
stanza.download('es', dir='/app/.cache/stanza'); \
print('Stanza OK'); \
"

# ── 7. Código de la aplicación ────────────────────────────────────────────────
COPY app.py \
     ner_model.py \
     entity_processor.py \
     entity_filter.py \
     entity_linker.py \
     ./
COPY templates/ ./templates/

EXPOSE 5000

CMD ["python", "app.py"]
