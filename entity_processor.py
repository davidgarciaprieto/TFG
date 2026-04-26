"""
entity_processor.py
--------------------
Módulo de procesamiento de entidades extraídas por el modelo NER.
Permite filtrar, agrupar, exportar y aplicar reglas a las entidades.
"""

import json
import csv
import io
from collections import defaultdict
from typing import Callable
from ner_model import Entity


# ─── Procesador principal ─────────────────────────────────────────────────────

class EntityProcessor:
    def __init__(self, entities: list[Entity]):
        """
        Args:
            entities: Lista de entidades extraídas por NERModel.
        """
        self.entities = entities

    # ── Filtrado ──────────────────────────────────────────────────────────────

    def filter_by_label(self, labels: list[str]) -> list[Entity]:
        """Filtra entidades por etiqueta(s) NER."""
        return [e for e in self.entities if e.label in labels]

    def filter_accepted(self) -> list[Entity]:
        """Devuelve solo las entidades aceptadas por el usuario."""
        return [e for e in self.entities if e.accepted is True]

    def filter_rejected(self) -> list[Entity]:
        """Devuelve solo las entidades rechazadas."""
        return [e for e in self.entities if e.accepted is False]

    def filter_pending(self) -> list[Entity]:
        """Devuelve entidades sin clasificar (aceptadas=None)."""
        return [e for e in self.entities if e.accepted is None]

    def filter_by_custom(self, fn: Callable[[Entity], bool]) -> list[Entity]:
        """Filtra entidades con una función personalizada."""
        return [e for e in self.entities if fn(e)]

    # ── Agrupación ────────────────────────────────────────────────────────────

    def group_by_label(self) -> dict[str, list[Entity]]:
        """Agrupa entidades por etiqueta NER."""
        groups: dict[str, list[Entity]] = defaultdict(list)
        for e in self.entities:
            groups[e.label].append(e)
        return dict(groups)

    def unique_texts(self) -> list[str]:
        """Devuelve textos únicos de las entidades."""
        seen = set()
        result = []
        for e in self.entities:
            if e.text.lower() not in seen:
                seen.add(e.text.lower())
                result.append(e.text)
        return result

    # ── Actualización de estado ───────────────────────────────────────────────

    def accept(self, start_char: int) -> None:
        """Marca como aceptada la entidad con ese start_char."""
        for e in self.entities:
            if e.start_char == start_char:
                e.accepted = True

    def reject(self, start_char: int) -> None:
        """Marca como rechazada la entidad con ese start_char."""
        for e in self.entities:
            if e.start_char == start_char:
                e.accepted = False

    def accept_all(self) -> None:
        """Acepta todas las entidades."""
        for e in self.entities:
            e.accepted = True

    def reject_all(self) -> None:
        """Rechaza todas las entidades."""
        for e in self.entities:
            e.accepted = False

    def reset_all(self) -> None:
        """Resetea el estado de todas las entidades a pendiente."""
        for e in self.entities:
            e.accepted = None

    # ── Exportación ───────────────────────────────────────────────────────────

    def to_list(self, only_accepted: bool = False) -> list[dict]:
        """
        Convierte las entidades a lista de diccionarios.
        
        Args:
            only_accepted: Si True, solo incluye las aceptadas.
        """
        ents = self.filter_accepted() if only_accepted else self.entities
        return [e.to_dict() for e in ents]

    def to_json(self, only_accepted: bool = False, indent: int = 2) -> str:
        """Serializa las entidades a JSON."""
        return json.dumps(self.to_list(only_accepted), ensure_ascii=False, indent=indent)

    def to_csv(self, only_accepted: bool = False) -> str:
        """Exporta las entidades a formato CSV."""
        ents = self.filter_accepted() if only_accepted else self.entities
        if not ents:
            return ""

        output = io.StringIO()
        fieldnames = ["text", "label", "start_char", "end_char", "accepted"]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for e in ents:
            writer.writerow({
                "text": e.text,
                "label": e.label,
                "start_char": e.start_char,
                "end_char": e.end_char,
                "accepted": e.accepted,
            })
        return output.getvalue()

    # ── Estadísticas ──────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Devuelve estadísticas básicas del conjunto de entidades."""
        groups = self.group_by_label()
        return {
            "total": len(self.entities),
            "accepted": len(self.filter_accepted()),
            "rejected": len(self.filter_rejected()),
            "pending": len(self.filter_pending()),
            "by_label": {label: len(ents) for label, ents in groups.items()},
        }
