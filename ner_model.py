"""
ner_model.py
------------
Módulo NER multi-backend.
Soporta: spaCy, HuggingFace Transformers (BETO, SpanBERTa, XLM-RoBERTa), Stanza.
Todos devuelven la misma lista de objetos Entity.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional


# ─── Estructura de entidad (común a todos los backends) ───────────────────────

@dataclass
class Entity:
    text:        str            # Texto literal de la entidad
    label:       str            # Etiqueta NER  (PER, ORG, LOC …)
    label_desc:  str            # Descripción legible
    start_char:  int            # Posición de carácter de inicio
    end_char:    int            # Posición de carácter de fin
    sentence:    str            # Oración de contexto (si está disponible)
    score:       float  = 0.0  # Confianza del modelo (0‒1), si la da
    start_token: int    = 0
    end_token:   int    = 0
    kb_id:       str    = ""
    accepted: Optional[bool] = None  # None=pendiente | True=aceptada | False=rechazada
    # ── metadatos opcionales (rellenados solo por el ensamble) ──
    source:        str = ""             # qué backend produjo la entidad final
    tokenizer:     str = ""             # backend que produjo el span
    labeller:      str = ""             # backend que produjo la etiqueta final
    label_stanza:  str = ""             # etiqueta original de Stanza
    label_xlm:     str = ""             # etiqueta original de XLM
    rule:          str = ""             # regla aplicada para decidir la etiqueta

    def to_dict(self) -> dict:
        return asdict(self)

    def __repr__(self) -> str:
        estado = {True: "✓", False: "✗", None: "?"}[self.accepted]
        score  = f" [{self.score:.2f}]" if self.score else ""
        src    = f" <{self.source}>" if self.source else ""
        return f"[{estado}] {self.label:<6} {self.text!r:30s}{score}{src}  ({self.start_char}:{self.end_char})"


# ─── Descripciones de etiquetas ───────────────────────────────────────────────

LABEL_DESCRIPTIONS: dict[str, str] = {
    "PER": "Persona",       "PERSON": "Persona",
    "ORG": "Organización",
    "LOC": "Lugar",         "GPE": "Entidad geopolítica",
    "MISC": "Miscelánea",   "NORP": "Grupo nacional/religioso/político",
    "FAC": "Instalación",   "PRODUCT": "Producto",
    "EVENT": "Evento",      "WORK_OF_ART": "Obra de arte",
    "LAW": "Ley/norma",     "LANGUAGE": "Idioma",
    "DATE": "Fecha",        "TIME": "Tiempo",
    "PERCENT": "Porcentaje","MONEY": "Cantidad monetaria",
    "QUANTITY": "Cantidad", "ORDINAL": "Ordinal",
    "CARDINAL": "Cardinal",
}

def _desc(label: str) -> str:
    return LABEL_DESCRIPTIONS.get(label.upper(), label)


# ─── Catálogo de modelos disponibles ─────────────────────────────────────────

MODEL_CATALOG: dict[str, dict] = {
    "spacy_lg": {
        "name":    "spaCy es_core_news_lg",
        "backend": "spacy",
        "handle":  "es_core_news_lg",
        "lang":    "es",
        "desc":    "spaCy large · rápido · sin GPU",
        "install": "python -m spacy download es_core_news_lg",
    },
    "beto": {
        "name":    "BETO-NER",
        "backend": "hf",
        "handle":  "mrm8488/bert-spanish-cased-finetuned-ner",
        "lang":    "es",
        "desc":    "BERT español · alta precisión",
        "install": "pip install transformers torch",
    },
    "spanberta": {
        "name":    "SpanBERTa-NER",
        "backend": "hf",
        "handle":  "MMG/xlm-roberta-large-ner-spanish",
        "lang":    "es",
        "desc":    "RoBERTa español · muy preciso",
        "install": "pip install transformers torch",
    },
    "stanza_es": {
        "name":    "Stanza-ES",
        "backend": "stanza",
        "handle":  "es",
        "lang":    "es",
        "desc":    "Stanford Stanza · NLP completo",
        "install": "pip install stanza  →  stanza.download('es')",
    },
    "xlm_roberta": {
        "name":    "XLM-RoBERTa-large",
        "backend": "hf",
        "handle":  "xlm-roberta-large-finetuned-conll03-english",
        "lang":    "multi",
        "desc":    "XLM-RoBERTa large · multilingüe",
        "install": "pip install transformers torch",
    },
    "ensemble": {
        "name":    "Ensamble (Stanza tokeniza, XLM-RoBERTa etiqueta)",
        "backend": "ensemble",
        "handle":  "",
        "lang":    "es",
        "desc":    "Stanza tokeniza y etiqueta; XLM-RoBERTa arbitra ORG en spans cortos",
        "install": "requiere los dos modelos base (Stanza, XLM-RoBERTa)",
        "components": ["stanza_es", "xlm_roberta"],
    },
}

DEFAULT_MODEL = "spacy_lg"


# ─── Backends ─────────────────────────────────────────────────────────────────

class _SpacyBackend:
    def __init__(self, handle: str):
        import spacy
        try:
            self._nlp = spacy.load(handle, enable=["tok2vec", "ner", "senter"])
        except ValueError:
            self._nlp = spacy.load(handle, enable=["tok2vec", "ner"])
        except OSError:
            raise OSError(f"Modelo spaCy '{handle}' no encontrado.\nInstálalo con: python -m spacy download {handle}")

    def extract(self, text: str) -> list[Entity]:
        doc = self._nlp(text)
        sent_map: dict[int, str] = {}
        try:
            for s in doc.sents:
                for t in s:
                    sent_map[t.i] = s.text.strip()
        except ValueError:
            pass
        return [
            Entity(
                text=e.text, label=e.label_, label_desc=_desc(e.label_),
                start_char=e.start_char, end_char=e.end_char,
                sentence=sent_map.get(e.start, ""),
                start_token=e.start, end_token=e.end,
                kb_id=e.kb_id_ or "",
            )
            for e in doc.ents
        ]

    @property
    def available_labels(self) -> list[str]:
        return list(self._nlp.get_pipe("ner").labels)


class _HFBackend:
    """HuggingFace transformers pipeline — token-classification."""

    MAX_TOKENS = 512  # límite de BERT/RoBERTa

    def __init__(self, handle: str):
        try:
            from transformers import pipeline, AutoTokenizer
        except ImportError:
            raise ImportError("Instala transformers: pip install transformers torch")

        self._tokenizer = AutoTokenizer.from_pretrained(handle)
        self._pipe = pipeline(
            "ner",
            model=handle,
            tokenizer=self._tokenizer,
            aggregation_strategy="simple",   # agrupa tokens en entidades
            device=-1,                        # CPU; pon 0 para GPU
        )
        self._handle = handle
        # Inferir etiquetas disponibles desde el config del modelo
        try:
            self._labels = list(self._pipe.model.config.id2label.values())
        except Exception:
            self._labels = []

    def _split_chunks(self, text: str) -> list[tuple[str, int]]:
        """
        Divide el texto en fragmentos de máx. MAX_TOKENS tokens,
        respetando límites de oración cuando es posible.
        Devuelve lista de (chunk_text, offset_char).
        """
        # Reservamos 2 tokens para [CLS] y [SEP]
        max_tok = self.MAX_TOKENS - 2
        tokens  = self._tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids     = tokens["input_ids"]
        offsets = tokens["offset_mapping"]  # [(start_char, end_char), ...]

        if len(ids) <= max_tok:
            return [(text, 0)]

        chunks = []
        start  = 0
        while start < len(ids):
            end = min(start + max_tok, len(ids))
            chunk_start_char = offsets[start][0]
            chunk_end_char   = offsets[end - 1][1]
            chunks.append((text[chunk_start_char:chunk_end_char], chunk_start_char))
            start = end

        return chunks

    def extract(self, text: str) -> list[Entity]:
        if not text.strip():
            return []

        chunks  = self._split_chunks(text)
        entities = []

        for chunk_text, offset in chunks:
            results = self._pipe(chunk_text)
            for r in results:
                label = r["entity_group"].replace("B-", "").replace("I-", "")
                entities.append(Entity(
                    text       = r["word"].strip(),
                    label      = label,
                    label_desc = _desc(label),
                    start_char = r["start"] + offset,
                    end_char   = r["end"]   + offset,
                    sentence   = "",
                    score      = float(r.get("score", 0)),
                ))

        return entities

    @property
    def available_labels(self) -> list[str]:
        return self._labels


class _StanzaBackend:
    """Stanford Stanza NER."""

    def __init__(self, lang: str = "es"):
        try:
            import stanza
        except ImportError:
            raise ImportError("Instala stanza: pip install stanza")
        try:
            self._nlp = stanza.Pipeline(lang, processors="tokenize,ner", verbose=False)
        except Exception:
            import stanza
            stanza.download(lang)
            self._nlp = stanza.Pipeline(lang, processors="tokenize,ner", verbose=False)
        self._lang = lang

    def extract(self, text: str) -> list[Entity]:
        if not text.strip():
            return []
        doc = self._nlp(text)
        entities = []
        for sent in doc.sentences:
            sent_text = sent.text.strip()
            for e in sent.ents:
                entities.append(Entity(
                    text=e.text,
                    label=e.type,
                    label_desc=_desc(e.type),
                    start_char=e.start_char,
                    end_char=e.end_char,
                    sentence=sent_text,
                ))
        return entities

    @property
    def available_labels(self) -> list[str]:
        return ["PER", "ORG", "LOC", "MISC"]


class _EnsembleBackend:
    """
    Ensamble Stanza + XLM-RoBERTa derivado del análisis empírico de dos
    benchmarks (frases sueltas y párrafos por dominio).

    Diseño:
      - Stanza-ES es el motor principal: tokeniza y etiqueta.
        Tiene 0 errores de tokenización en ambos benchmarks y es el modelo
        más robusto al cambio de longitud de entrada (caída de F1 más
        pequeña al pasar de frases a párrafos).
      - XLM-RoBERTa actúa como árbitro condicional solo cuando se cumplen
        DOS condiciones simultáneamente:
            1. El span de Stanza tiene como máximo MAX_TOKENS tokens.
            2. La etiqueta de Stanza NO es ORG.
        Esto neutraliza los puntos débiles de XLM (fragmentación
        WordPiece en spans largos, falsos positivos espurios "gri",
        "oto", etc.) y conserva su única ventaja documentada: corregir
        ORG cortos que Stanza confunde con persona (Roche, Meta, Sanofi,
        Al Nassr, Ferrari).

    Regla de fusión:
      - Si XLM sí dice ORG sobre ese mismo span (IoU >= 0,4) y Stanza
        propuso otra etiqueta -> se adopta la propuesta de XLM.
      - En cualquier otro caso, se mantiene la propuesta de Stanza.

    Cada Entity producida lleva los campos `tokenizer`, `labeller`,
    `label_stanza`, `label_xlm` y `rule`, que permiten inspeccionar
    qué hizo cada parte del ensamble en cada decisión.
    """

    TOKENIZER  = "stanza_es"
    LABELLER   = "xlm_roberta"
    COMPONENTS = [TOKENIZER, LABELLER]
    MAX_TOKENS = 3   # umbral para activar el arbitraje de XLM

    _LABEL_NORM = {
        "PERSON": "PER", "PER": "PER",
        "ORG": "ORG", "ORGANIZATION": "ORG",
        "LOC": "LOC", "LOCATION": "LOC", "GPE": "LOC", "FAC": "LOC",
        "MISC": "MISC", "EVENT": "MISC", "PRODUCT": "MISC",
        "WORK_OF_ART": "MISC", "NORP": "MISC", "LANGUAGE": "MISC", "LAW": "MISC",
    }

    def __init__(self, model_resolver=None):
        """
        Args:
          model_resolver: callable opcional `key -> NERModel`. Si se proporciona,
            el ensamble reutiliza modelos ya cacheados (evita recargar Stanza
            y XLM-RoBERTa). Si es None, los carga él mismo.
        """
        self._resolver = model_resolver
        self._tok: Optional["NERModel"] = None
        self._lab: Optional["NERModel"] = None
        self._loaded = False

    def _lazy_load(self):
        if self._loaded:
            return
        if self._resolver is not None:
            self._tok = self._resolver(self.TOKENIZER)
            self._lab = self._resolver(self.LABELLER)
        else:
            self._tok = NERModel(self.TOKENIZER)
            self._lab = NERModel(self.LABELLER)
        self._loaded = True

    @classmethod
    def _norm(cls, label: str) -> str:
        return cls._LABEL_NORM.get(label.upper(), label.upper())

    @staticmethod
    def _iou(a: Entity, b: Entity) -> float:
        inter = max(0, min(a.end_char, b.end_char) - max(a.start_char, b.start_char))
        if inter == 0:
            return 0.0
        union = (a.end_char - a.start_char) + (b.end_char - b.start_char) - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _ntokens(e: Entity) -> int:
        return len(e.text.strip().split())

    def extract(self, text: str) -> list[Entity]:
        if not text.strip():
            return []
        self._lazy_load()

        # 1. Stanza tokeniza y etiqueta
        stanza_ents = self._tok.extract(text)
        for e in stanza_ents:
            e.label = self._norm(e.label)
            e.label_desc = _desc(e.label)

        # 2. XLM-RoBERTa extrae sus propias propuestas
        xlm_ents = self._lab.extract(text)
        for e in xlm_ents:
            e.label = self._norm(e.label)
            e.label_desc = _desc(e.label)

        result: list[Entity] = []
        for s in stanza_ents:
            final_label = s.label
            labeller    = self.TOKENIZER
            label_xlm   = ""
            rule        = "stanza_solo"

            if self._ntokens(s) <= self.MAX_TOKENS:
                # Buscar la propuesta de XLM para este span
                for x in xlm_ents:
                    if self._iou(s, x) >= 0.4:
                        label_xlm = x.label

                        if s.label != "ORG" and x.label == "ORG":
                            # Regla A: Stanza=PER/LOC/MISC y XLM=ORG
                            # → XLM corrige a ORG.
                            # Caso típico: Stanza=PER "Roche" → ORG
                            final_label = "ORG"
                            labeller    = self.LABELLER
                            rule        = "xlm_ORG_corrige_stanza"

                        elif s.label == "ORG" and x.label == "LOC":
                            # Regla B: Stanza=ORG y XLM=LOC
                            # → XLM corrige a LOC. XLM es más fiable para
                            # distinguir topónimos mal etiquetados como ORG.
                            # Casos documentados: "Estados Unidos", "Bernabéu".
                            final_label = "LOC"
                            labeller    = self.LABELLER
                            rule        = "xlm_LOC_corrige_ORG_de_stanza"

                        else:
                            # Sin corrección: se mantiene Stanza.
                            rule = "stanza_confirmado_por_xlm" if x.label == s.label \
                                   else "stanza_pese_a_xlm"
                        break

            result.append(Entity(
                text       = s.text,
                label      = final_label,
                label_desc = _desc(final_label),
                start_char = s.start_char,
                end_char   = s.end_char,
                sentence   = s.sentence,
                score      = s.score,
                start_token= s.start_token,
                end_token  = s.end_token,
                source     = "ensemble",
                tokenizer  = self.TOKENIZER,
                labeller   = labeller,
                label_stanza = s.label,
                label_xlm    = label_xlm,
                rule       = rule,
            ))

        result.sort(key=lambda e: e.start_char)
        return result

    @property
    def available_labels(self) -> list[str]:
        return ["PER", "ORG", "LOC", "MISC"]


# ─── Clase pública NERModel ───────────────────────────────────────────────────

class NERModel:
    """
    Interfaz unificada para todos los backends NER.

    Uso:
        model = NERModel("beto")
        entities = model.extract("Pedro Sánchez visitó Madrid.")
        for e in entities:
            print(e)
    """

    def __init__(self, model_key: str = DEFAULT_MODEL, model_resolver=None):
        """
        Args:
          model_key:      clave del catálogo MODEL_CATALOG
          model_resolver: callable opcional `key -> NERModel`, solo usado
                          cuando el backend es 'ensemble', para reutilizar
                          modelos cacheados.
        """
        if model_key not in MODEL_CATALOG:
            raise ValueError(f"Modelo '{model_key}' no encontrado. Disponibles: {list(MODEL_CATALOG)}")

        cfg = MODEL_CATALOG[model_key]
        self.model_key  = model_key
        self.model_name = cfg["name"]
        self.backend_id = cfg["backend"]
        self._cfg       = cfg

        if cfg["backend"] == "spacy":
            self._backend = _SpacyBackend(cfg["handle"])
        elif cfg["backend"] == "hf":
            self._backend = _HFBackend(cfg["handle"])
        elif cfg["backend"] == "stanza":
            self._backend = _StanzaBackend(cfg["handle"])
        elif cfg["backend"] == "ensemble":
            self._backend = _EnsembleBackend(model_resolver=model_resolver)
        else:
            raise ValueError(f"Backend desconocido: {cfg['backend']}")

    def extract(self, text: str) -> list[Entity]:
        """Extrae entidades y devuelve lista de Entity."""
        return self._backend.extract(text)

    def extract_as_dicts(self, text: str) -> list[dict]:
        """Extrae entidades como lista de diccionarios (JSON-serializable)."""
        return [e.to_dict() for e in self.extract(text)]

    def highlight_text(self, text: str, entities: list[Entity]) -> str:
        """Texto con entidades marcadas: [LABEL:texto]."""
        if not entities:
            return text
        for ent in sorted(entities, key=lambda e: e.start_char, reverse=True):
            text = text[:ent.start_char] + f"[{ent.label}:{ent.text}]" + text[ent.end_char:]
        return text

    @property
    def available_labels(self) -> list[str]:
        return self._backend.available_labels

    def __repr__(self) -> str:
        return f"NERModel(key='{self.model_key}', name='{self.model_name}')"


# ─── Demo rápida ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    texto = ("El presidente Pedro Sánchez se reunió con representantes de Google "
             "y Microsoft en Madrid el 15 de marzo de 2024.")

    for key in ["spacy_lg"]:   # añade más claves para probar otros backends
        print(f"\n{'='*60}\nModelo: {key}\n{'='*60}")
        try:
            m = NERModel(key)
            ents = m.extract(texto)
            for e in ents:
                print(e)
            print(json.dumps(m.extract_as_dicts(texto), ensure_ascii=False, indent=2))
        except Exception as ex:
            print(f"  ⚠ {ex}")
