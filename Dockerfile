# ── Dockerfile — NER Filter App v3.0 (todos los modelos pre-descargados) ──────
FROM python:3.11-slim

LABEL maintainer="davidgarciaprieto"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SPACY_QUIET=1 \
    TRANSFORMERS_VERBOSITY=error \
    HF_HOME=/app/.cache/huggingface \
    STANZA_RESOURCES_DIR=/app/.cache/stanza \
    TRANSFORMERS_OFFLINE=0

WORKDIR /app

# ── 1. Sistema ────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# ── 2. PyTorch CPU ────────────────────────────────────────────────────────────
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# ── 3. Resto de dependencias ──────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── 4. Modelo spaCy ───────────────────────────────────────────────────────────
RUN python -m spacy download es_core_news_lg

# ── 5. Modelos HuggingFace ────────────────────────────────────────────────────
RUN python -c "\
from transformers import AutoTokenizer, AutoModelForTokenClassification; \
print('BETO...'); \
AutoTokenizer.from_pretrained('mrm8488/bert-spanish-cased-finetuned-ner'); \
AutoModelForTokenClassification.from_pretrained('mrm8488/bert-spanish-cased-finetuned-ner'); \
print('SpanBERTa...'); \
AutoTokenizer.from_pretrained('MMG/xlm-roberta-large-ner-spanish'); \
AutoModelForTokenClassification.from_pretrained('MMG/xlm-roberta-large-ner-spanish'); \
print('XLM-RoBERTa...'); \
AutoTokenizer.from_pretrained('xlm-roberta-large-finetuned-conll03-english'); \
AutoModelForTokenClassification.from_pretrained('xlm-roberta-large-finetuned-conll03-english'); \
print('HuggingFace OK'); \
"

# ── 6. Modelo Stanza ──────────────────────────────────────────────────────────
RUN python -c "\
import stanza, os; \
os.environ['STANZA_RESOURCES_DIR']='/app/.cache/stanza'; \
stanza.download('es'); \
print('Stanza OK'); \
"

# ── 7. Código ─────────────────────────────────────────────────────────────────
COPY app.py ner_model.py entity_processor.py entity_filter.py entity_linker.py ./
COPY templates/ ./templates/

EXPOSE 5000
CMD ["python", "app.py"]
