"""
ner_model.py
------------
Módulo NER con spaCy — modelos LARGE.
Devuelve una lista de Entity procesable por entity_processor.py
"""

import spacy
from dataclasses import dataclass, asdict
from typing import Optional


# ─── Estructura de entidad ────────────────────────────────────────────────────

@dataclass
class Entity:
    text: str               # Texto literal de la entidad
    label: str              # Etiqueta NER  (PER, ORG, LOC …)
    label_desc: str         # Descripción legible ("Persona", "Organización" …)
    start_token: int        # Índice de token de inicio en el Doc
    end_token: int          # Índice de token de fin (exclusive) en el Doc
    start_char: int         # Posición de carácter de inicio en el texto
    end_char: int           # Posición de carácter de fin en el texto
    sentence: str           # Oración donde aparece la entidad
    kb_id: str = ""         # ID de base de conocimiento (si el modelo lo da)
    accepted: Optional[bool] = None  # None=pendiente | True=aceptada | False=rechazada

    def to_dict(self) -> dict:
        return asdict(self)

    def __repr__(self) -> str:
        estado = {True: "✓", False: "✗", None: "?"}[self.accepted]
        return f"[{estado}] {self.label:<6} {self.text!r:30s}  ({self.start_char}:{self.end_char})"


# ─── Catálogo de modelos ──────────────────────────────────────────────────────
#   Siempre se prefiere la variante 'lg' (large); si no está disponible
#   se puede bajar al 'md' o 'sm' cambiando la clave en SPACY_MODELS.

SPACY_MODELS = {
    # ── Español ──────────────────────────────────────────────────────────────
    "es_lg":  "es_core_news_lg",   # ← RECOMENDADO (large)
    "es_md":  "es_core_news_md",
    "es_sm":  "es_core_news_sm",
    # ── Inglés ───────────────────────────────────────────────────────────────
    "en_lg":  "en_core_web_lg",    # ← RECOMENDADO (large)
    "en_trf": "en_core_web_trf",   # transformer (máxima precisión, más lento)
    "en_md":  "en_core_web_md",
    "en_sm":  "en_core_web_sm",
    # ── Multilingüe ──────────────────────────────────────────────────────────
    "xx":     "xx_ent_wiki_sm",
}

# Modelo por defecto al arrancar la app
DEFAULT_MODEL = "es_lg"

# Etiquetas NER → descripción en castellano
LABEL_DESCRIPTIONS: dict[str, str] = {
    # Modelos es_core_news_*
    "PER":        "Persona",
    "ORG":        "Organización",
    "LOC":        "Lugar",
    "MISC":       "Miscelánea",
    # Modelos en_core_web_*
    "PERSON":     "Persona",
    "GPE":        "Entidad geopolítica",
    "NORP":       "Grupo nacional / religioso / político",
    "FAC":        "Instalación / edificio",
    "PRODUCT":    "Producto",
    "EVENT":      "Evento",
    "WORK_OF_ART":"Obra de arte",
    "LAW":        "Ley / norma",
    "LANGUAGE":   "Idioma",
    "DATE":       "Fecha",
    "TIME":       "Tiempo",
    "PERCENT":    "Porcentaje",
    "MONEY":      "Cantidad monetaria",
    "QUANTITY":   "Cantidad / medida",
    "ORDINAL":    "Número ordinal",
    "CARDINAL":   "Número cardinal",
}


# ─── Clase principal NER ──────────────────────────────────────────────────────

class NERModel:
    """
    Wrapper sobre spaCy para extracción de entidades nombradas.

    Uso rápido:
        model = NERModel("es_lg")
        entities = model.extract("Pedro Sánchez visitó Madrid ayer.")
        for e in entities:
            print(e)
    """

    def __init__(self, model_key: str = DEFAULT_MODEL):
        """
        Carga el modelo spaCy indicado.

        Args:
            model_key: Clave de SPACY_MODELS o nombre completo del modelo spaCy.
                       Ejemplos: "es_lg", "en_lg", "es_core_news_lg".
        """
        model_name = SPACY_MODELS.get(model_key, model_key)
        try:
            self.nlp = spacy.load(model_name, enable=["tok2vec", "ner", "senter"])
        except ValueError:
            # Algunos modelos no tienen senter; cargamos sin él
            self.nlp = spacy.load(model_name, enable=["tok2vec", "ner"])
        except OSError:
            raise OSError(
                f"Modelo '{model_name}' no encontrado.\n"
                f"Instálalo con:\n"
                f"  python -m spacy download {model_name}"
            )
        self.model_name = model_name
        self.model_key  = model_key

    # ── Extracción ────────────────────────────────────────────────────────────

    def extract(self, text: str) -> list[Entity]:
        """
        Extrae entidades nombradas y devuelve una lista de objetos Entity.

        La lista es directamente iterable y serializable:
            entities = model.extract(texto)
            for e in entities:
                print(e.text, e.label, e.start_char)

        Args:
            text: Cadena de texto a analizar.

        Returns:
            list[Entity] — lista vacía si el texto está vacío.
        """
        if not text or not text.strip():
            return []

        doc = self.nlp(text)

        # Mapa token → oración para enriquecer cada entidad
        # (algunos modelos large no tienen senter; se ignora en ese caso)
        sent_map: dict[int, str] = {}
        try:
            for sent in doc.sents:
                for tok in sent:
                    sent_map[tok.i] = sent.text.strip()
        except ValueError:
            pass  # el modelo no tiene sentencizer, sentence quedará vacío

        entities: list[Entity] = []
        for ent in doc.ents:
            entities.append(Entity(
                text        = ent.text,
                label       = ent.label_,
                label_desc  = LABEL_DESCRIPTIONS.get(ent.label_, ent.label_),
                start_token = ent.start,
                end_token   = ent.end,
                start_char  = ent.start_char,
                end_char    = ent.end_char,
                sentence    = sent_map.get(ent.start, ""),
                kb_id       = ent.kb_id_ or "",
            ))

        return entities

    def extract_as_dicts(self, text: str) -> list[dict]:
        """
        Extrae entidades y las devuelve como lista de diccionarios.
        Lista 100 % serializable a JSON.

        Ejemplo de elemento:
            {
              "text": "Pedro Sánchez",
              "label": "PER",
              "label_desc": "Persona",
              "start_char": 0,
              "end_char": 13,
              "sentence": "Pedro Sánchez visitó Madrid.",
              "accepted": null
            }
        """
        return [e.to_dict() for e in self.extract(text)]

    # ── Utilidades ────────────────────────────────────────────────────────────

    def highlight_text(self, text: str, entities: list[Entity]) -> str:
        """
        Devuelve el texto con las entidades marcadas como [LABEL:texto].
        Útil para depuración o previsualización en consola.

        Ejemplo:
            "[PER:Pedro Sánchez] visitó [LOC:Madrid] ayer."
        """
        if not entities:
            return text
        sorted_ents = sorted(entities, key=lambda e: e.start_char, reverse=True)
        result = text
        for ent in sorted_ents:
            marker = f"[{ent.label}:{ent.text}]"
            result = result[:ent.start_char] + marker + result[ent.end_char:]
        return result

    def get_label_description(self, label: str) -> str:
        """Descripción legible de una etiqueta NER."""
        return LABEL_DESCRIPTIONS.get(label, label)

    @property
    def available_labels(self) -> list[str]:
        """Etiquetas que reconoce el pipe NER del modelo cargado."""
        return list(self.nlp.get_pipe("ner").labels)

    def __repr__(self) -> str:
        return f"NERModel(model='{self.model_name}', labels={self.available_labels})"


# ─── Ejecución directa (demo rápida) ─────────────────────────────────────────

if __name__ == "__main__":
    import json

    texto = (
        "El presidente Pedro Sánchez se reunió con representantes de Google "
        "y Microsoft en Madrid el 15 de marzo de 2024. "
        "La cumbre fue organizada por la Unión Europea."
    )

    print("Cargando modelo es_core_news_lg …")
    model = NERModel("es_lg")
    print(f"Modelo: {model.model_name}")
    print(f"Etiquetas disponibles: {model.available_labels}\n")

    entities = model.extract(texto)

    print("── Entidades detectadas ──────────────────────────────────────")
    for e in entities:
        print(e)

    print("\n── Vista anotada ─────────────────────────────────────────────")
    print(model.highlight_text(texto, entities))

    print("\n── Como lista de diccionarios (procesable) ───────────────────")
    print(json.dumps(model.extract_as_dicts(texto), ensure_ascii=False, indent=2))
