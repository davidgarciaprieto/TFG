"""
entity_filter.py
----------------
Pipeline de filtrado de entidades NER en español.

El filtrado se aplica en cascada sobre la lista de entidades producidas
por un modelo NER. Cada filtro puede marcar una entidad como eliminada
(kept=False) o conservarla (kept=True).  Las entidades eliminadas se
conservan en el resultado para auditoría, junto con el id del filtro
que las eliminó y la razón.

Filtros disponibles (en orden de aplicación):
  F0  Lista negra de usuario: términos que el usuario ha marcado
      explícitamente para ser eliminados (exacto, case-insensitive).
  F1  Longitud mínima: elimina entidades de texto demasiado corto
      (por defecto ≤ 1 carácter o solo dígitos).
  F2  Sintagmas sin nombre propio: elimina predicciones que comienzan
      por artículo + sustantivo común sin ninguna mayúscula interna
      (ej. "Los usuarios y empresas").
  F3  Subtérmino solapado: elimina la entidad B cuando su span de
      caracteres (start_char, end_char) está COMPLETAMENTE contenido
      dentro del span de otra entidad A en el MISMO texto y la entidad
      B no aparece además de forma INDEPENDIENTE en otro span fuera de A.
      Esto resuelve el caso clásico de falsos positivos: si el modelo
      extrae tanto "Plaza de España" como "España" del mismo texto, se
      comprueba si "España" aparece solo fuera del span de "Plaza de
      España". Si sí → ambas se conservan. Si no → "España" es
      redundante y se elimina.
  F4  Frecuencia mínima: en un corpus de múltiples textos, elimina
      entidades que aparecen menos veces que un umbral configurable.
      Útil para limpiar falsos positivos esporádicos.

Uso básico:
    ef = EntityFilter()
    results = ef.filter(entities)          # lista de dicts {"text":…, "label":…}
    stats   = ef.stats(results)

Uso con lista negra de usuario:
    ef = EntityFilter(user_blacklist=["IA", "software", "API"])
    results = ef.filter(entities)

Uso con JSON persistente de lista negra:
    ef = EntityFilter.from_json("blacklist.json")
    ef.add_to_blacklist("GPT")             # añadir en caliente
    ef.save_blacklist("blacklist.json")    # persistir
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional


# ─── Resultado de filtrado ────────────────────────────────────────────────────

@dataclass
class FilterResult:
    text:       str
    label:      str
    label_desc: str = ""
    frequency:  int = 1           # cuántas veces aparece en el lote de entrada
    kept:       bool = True       # ¿sobrevive al filtrado?
    filter_id:  str  = ""         # "F0"…"F4", vacío si kept=True
    reason:     str  = ""         # explicación legible del rechazo
    # posición en el texto (opcionales, útiles para F3)
    start_char: Optional[int] = None
    end_char:   Optional[int] = None
    sentence:   str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Constantes ───────────────────────────────────────────────────────────────

# Artículos y determinantes que indican un sintagma común (F2)
_ARTICLES = {"el", "la", "los", "las", "un", "una", "unos", "unas",
             "este", "esta", "estos", "estas", "ese", "esa", "esos",
             "esas", "aquel", "aquella", "su", "sus", "mi", "mis"}

# Palabras que NO deben ser entidades sueltas (F1 extendida)
_NOISE_TOKENS = {"el", "la", "los", "las", "de", "del", "en", "y", "o",
                 "a", "al", "con", "por", "para", "que", "se", "es",
                 "son", "fue", "era", "etc"}

# Patrón para detectar si una cadena tiene al menos una letra mayúscula
# después del primer carácter (marca de nombre propio interno)
_HAS_INTERNAL_UPPER = re.compile(r"[A-ZÁÉÍÓÚÀÈÌÒÙÄËÏÖÜÂÊÎÔÛÑÜ]", re.UNICODE)


def _normalize(text: str) -> str:
    """Normaliza texto para comparaciones: lower + strip."""
    return text.strip().lower()


def _has_proper_noun_char(text: str) -> bool:
    """True si el texto contiene al menos una letra mayúscula."""
    return bool(re.search(r"\p?[A-ZÁÉÍÓÚÑ]", text, re.UNICODE))


# ─── Clase principal ──────────────────────────────────────────────────────────

class EntityFilter:
    """
    Pipeline de filtrado de entidades NER.

    Parámetros
    ----------
    user_blacklist : list[str]
        Términos que el usuario quiere eliminar explícitamente.
        La comparación es case-insensitive y sin acentos.
        Se puede modificar en tiempo de ejecución con add_to_blacklist().
    min_length : int
        Longitud mínima de caracteres del texto de la entidad (F1).
        Por defecto 2.
    min_frequency : int
        Frecuencia mínima de aparición en el lote de entrada (F4).
        Por defecto 1 (desactiva el filtro de frecuencia).
    apply_f3 : bool
        Si True (por defecto), aplica el filtro de subtérminos solapados.
    """

    def __init__(
        self,
        user_blacklist:  list[str] | None = None,
        min_length:      int = 2,
        min_frequency:   int = 1,
        apply_f3:        bool = True,
    ):
        # La lista negra se guarda normalizada para comparación rápida
        self._blacklist_raw:  list[str] = list(user_blacklist or [])
        self._blacklist_norm: set[str]  = {
            self._norm_bl(t) for t in self._blacklist_raw
        }
        self.min_length    = min_length
        self.min_frequency = min_frequency
        self.apply_f3      = apply_f3

    # ── Gestión de la lista negra ─────────────────────────────────────────────

    @staticmethod
    def _norm_bl(text: str) -> str:
        """Normaliza para la lista negra: lower + quitar acentos + strip."""
        nfkd = unicodedata.normalize("NFKD", text.strip().lower())
        return "".join(c for c in nfkd if not unicodedata.combining(c))

    def add_to_blacklist(self, *terms: str) -> None:
        """Añade uno o varios términos a la lista negra del usuario."""
        for t in terms:
            if t not in self._blacklist_raw:
                self._blacklist_raw.append(t)
            norm = self._norm_bl(t)
            self._blacklist_norm.add(norm)

    def remove_from_blacklist(self, *terms: str) -> None:
        """Elimina términos de la lista negra."""
        for t in terms:
            if t in self._blacklist_raw:
                self._blacklist_raw.remove(t)
            self._blacklist_norm.discard(self._norm_bl(t))

    @property
    def blacklist(self) -> list[str]:
        """Lista negra actual (orden de inserción)."""
        return list(self._blacklist_raw)

    def save_blacklist(self, path: str | Path) -> None:
        """Guarda la lista negra en un fichero JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"blacklist": self._blacklist_raw}, f,
                      ensure_ascii=False, indent=2)

    @classmethod
    def load_blacklist(cls, path: str | Path) -> "EntityFilter":
        """Crea un EntityFilter cargando la lista negra desde JSON."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(user_blacklist=data.get("blacklist", []))

    @classmethod
    def from_dict(cls, config: dict) -> "EntityFilter":
        """Crea un EntityFilter desde un diccionario de configuración."""
        return cls(
            user_blacklist=config.get("blacklist", []),
            min_length=config.get("min_length", 2),
            min_frequency=config.get("min_frequency", 1),
            apply_f3=config.get("apply_f3", True),
        )

    # ── Filtros individuales ──────────────────────────────────────────────────

    def _f0_blacklist(self, text: str) -> tuple[bool, str]:
        """F0: lista negra del usuario."""
        if self._norm_bl(text) in self._blacklist_norm:
            return False, f"lista negra del usuario: '{text}'"
        return True, ""

    # Abreviaciones de tratamiento/título que son válidas aunque sean cortas
    _VALID_SHORT = {
        "dr","dra","sr","sra","mr","mrs","ms","prof","rev","gen",
        "col","mt","st","av","avda","cía","cta","dpto","apdo",
    }

    def _f1_length(self, text: str) -> tuple[bool, str]:
        """F1: longitud mínima, ruido y tokens malformados."""
        stripped = text.strip()

        # Cadena vacía
        if not stripped:
            return False, "cadena vacía"

        # Solo dígitos (fechas, números, porcentajes sueltos)
        if stripped.isdigit():
            return False, "cadena de solo dígitos"

        # Sin ninguna letra: puntuación pura, símbolos, guiones múltiples
        # Cubre: '--', '...', '.', ',', '[UE]', '&', '/', '()', etc.
        # Usamos una regex que busca al menos una letra unicode.
        if not re.search(r"[^\W\d_]", stripped, re.UNICODE):
            return False, f"sin ninguna letra (símbolo/puntuación): '{stripped}'"

        # Longitud insuficiente
        if len(stripped) < self.min_length:
            return False, f"longitud {len(stripped)} < mínimo {self.min_length}"

        # Tokens de ruido (palabras funcionales sueltas)
        if _normalize(stripped) in _NOISE_TOKENS:
            return False, f"token funcional: '{stripped}'"

        # Fragmentos WordPiece: empiezan por ##
        if stripped.startswith("##"):
            return False, f"fragmento WordPiece: '{stripped}'"

        # Token todo en minúsculas de ≤6 chars sin mayúsculas:
        # probable fragmento de tokenizador ('gri', 'oto', 'euro', 'occi').
        if (len(stripped) <= 6
                and stripped.islower()
                and stripped.isalpha()):
            return False, f"token corto en minúsculas (probable fragmento): '{stripped}'"

        # Token corto con mayúscula pero sin vocal completa:
        # cubre truncados como 'Tai', 'Biz', 'Suiz', 'Occi' que el
        # tokenizador parte a mitad de palabra.
        # Se preservan abreviaciones reconocidas (Dr, Sr, Mr, Prof, etc.)
        # y siglas puras (todas mayúsculas como XLM, GPU, ONU).
        words = stripped.split()
        if len(words) == 1:
            w = stripped
            w_lower = w.lower().rstrip(".")
            is_abbrev   = w_lower in self._VALID_SHORT
            is_acronym  = w.isupper() and len(w) >= 2       # GPU, ONU, ADN
            has_digit   = any(c.isdigit() for c in w)       # GPT-4, COVID-19
            has_hyphen  = "-" in w                          # XLM-RoBERTa
            if not (is_abbrev or is_acronym or has_digit or has_hyphen):
                # Detectar truncados: ≤4 chars con al menos una mayúscula
                # pero sin vocal (Bi, Biz, Tai, Occi sin 'o' standalone)
                VOWELS = set("aeiouáéíóúàèìòùäëïöüâêîôûAEIOUÁÉÍÓÚÀÈÌÒÙÄËÏÖÜÂÊÎÔÛ")
                has_vowel = any(c in VOWELS for c in w)
                if len(w) <= 4 and not has_vowel:
                    return False, f"token truncado sin vocal: '{w}'"
                # Token de 2-3 chars con mayúscula pero no acrónimo y sin vocal:
                # 'Bi', 'Bz', 'Fr' → probable fragmento
                if len(w) <= 3 and not has_vowel and not is_acronym:
                    return False, f"fragmento corto sin vocal: '{w}'"

        return True, ""

    def _f2_common_noun_phrase(self, text: str) -> tuple[bool, str]:
        """
        F2: sintagma nominal común sin nombre propio.
        Elimina predicciones como:
          'Los usuarios y empresas'
          'Esta estructura'
          'Ambas hebras son antiparalelas'
          'Estado heredero'
          'El entorno se sustenta fuertemente en tecnologías de virtualización'
        Conserva:
          'La Casa Blanca'    (Casa y Blanca tienen mayúscula)
          'Los Ángeles'       (Ángeles tiene mayúscula)
          'El País'           (País tiene mayúscula)
          'Estado Islámico'   (Islámico tiene mayúscula)
        """
        tokens = text.strip().split()
        if not tokens:
            return False, "cadena vacía"

        # Demostrativos y determinantes de sintagma común
        _COMMON_STARTS = (
            _ARTICLES
            | {"esta", "este", "estos", "estas", "ese", "esa", "esos", "esas",
               "ambas", "ambos", "dicha", "dicho", "toda", "todo", "todos",
               "todas", "cada", "cualquier", "ningún", "ninguna", "cierto",
               "cierta", "varios", "varias", "estado", "estados"}
        )

        first_lower = _normalize(tokens[0])

        # Si empieza por artículo, demostrativo, etc., mirar si hay
        # alguna mayúscula en el resto del texto
        if first_lower in _COMMON_STARTS and len(tokens) > 1:
            rest = " ".join(tokens[1:])
            if not re.search(r"[A-ZÁÉÍÓÚÀÈÌÒÙÄËÏÖÜÂÊÎÔÛÑÜ]", rest, re.UNICODE):
                return False, (
                    f"sintagma sin nombre propio "
                    f"({tokens[0]!r} + sustantivo/adjetivo común): '{text}'"
                )

        # Casos como 'Ambas hebras son antiparalelas': empieza con mayúscula
        # pero es una frase completa con verbo. Heurística: ≥4 tokens,
        # contiene verbo copulativo o auxiliar, sin nombre propio interno.
        _COPULAS = {"son", "es", "fue", "era", "ser", "están", "está",
                    "logró", "permite", "sustenta", "asegura", "permite",
                    "consiste", "caracterizó", "floreció", "perduró"}
        if len(tokens) >= 4:
            token_set = {_normalize(t) for t in tokens[1:]}   # skip primero
            if token_set & _COPULAS:
                # Frase con verbo: solo mantener si tiene nombre propio
                if not re.search(
                        r"[A-ZÁÉÍÓÚÀÈÌÒÙÄËÏÖÜÂÊÎÔÛÑÜ]",
                        " ".join(tokens[1:]),
                        re.UNICODE):
                    return False, (
                        f"frase con verbo sin nombre propio: '{text}'"
                    )

        # Si todos los tokens están en minúscula y hay >3 tokens, es falso positivo
        if len(tokens) > 3 and not re.search(
                r"[A-ZÁÉÍÓÚÀÈÌÒÙÄËÏÖÜÂÊÎÔÛÑÜ]", text, re.UNICODE):
            return False, f"frase sin mayúsculas ni nombre propio ({len(tokens)} tokens)"

        return True, ""

    def _f3_subspan(
        self,
        candidate: dict,
        all_entities: list[dict],
    ) -> tuple[bool, str]:
        """
        F3: subtérmino solapado — solo actúa cuando hay información de span.

        Elimina la entidad B (candidate) si y solo si:
          1. Ambas entidades (B y su contenedor A) tienen start_char/end_char.
          2. El span de B está COMPLETAMENTE dentro del span de A.
          3. B NO aparece de forma independiente en ningún otro span
             que no esté contenido en A.

        Sin información de span (start_char=None) el filtro NO actúa.
        La versión anterior tenía un fallback por comparación de texto que
        eliminaba entidades válidas como "Imperio romano" porque su texto
        está contenido en "Imperio romano de Oriente". Este fallback ha sido
        eliminado.
        """
        b_start = candidate.get("start_char")
        b_end   = candidate.get("end_char")

        # Sin información de span: no actuar
        if b_start is None or b_end is None:
            return True, ""

        b_text = _normalize(candidate.get("text", ""))

        for other in all_entities:
            if other is candidate:
                continue
            a_text = _normalize(other.get("text", ""))
            if a_text == b_text:
                continue

            a_start = other.get("start_char")
            a_end   = other.get("end_char")
            if a_start is None or a_end is None:
                continue

            # ¿B está completamente dentro de A?
            if a_start <= b_start and b_end <= a_end:
                has_independent = self._has_independent_occurrence(
                    candidate, all_entities
                )
                if not has_independent:
                    return (
                        False,
                        f"subtérmino de '{other.get('text', '')}' "
                        f"(span {b_start}:{b_end} ⊂ {a_start}:{a_end}) "
                        f"sin ocurrencia independiente"
                    )

        return True, ""

    @staticmethod
    def _has_independent_occurrence(
        candidate: dict,
        all_entities: list[dict],
    ) -> bool:
        """
        Devuelve True si `candidate` tiene al menos un span en `all_entities`
        que NO esté contenido dentro del span de ningún otro elemento.
        """
        c_text  = _normalize(candidate.get("text", ""))
        c_start = candidate.get("start_char")
        c_end   = candidate.get("end_char")

        for ent in all_entities:
            if ent is candidate:
                continue
            if _normalize(ent.get("text", "")) != c_text:
                continue
            # Hay otra ocurrencia del mismo texto; comprobar si esa
            # ocurrencia está dentro de algún A
            ent_start = ent.get("start_char")
            ent_end   = ent.get("end_char")
            if ent_start is None or ent_end is None:
                # Sin info de span: asumimos que es independiente
                return True
            # ¿Esta ocurrencia está dentro de algún contenedor?
            inside_some_a = False
            for other in all_entities:
                if other is ent or other is candidate:
                    continue
                a_start = other.get("start_char")
                a_end   = other.get("end_char")
                if a_start is None or a_end is None:
                    continue
                if a_start <= ent_start and ent_end <= a_end:
                    inside_some_a = True
                    break
            if not inside_some_a:
                return True   # Hay una ocurrencia independiente
        return False

    def _f4_frequency(
        self,
        text: str,
        freq: int,
    ) -> tuple[bool, str]:
        """F4: frecuencia mínima."""
        if freq < self.min_frequency:
            return False, (
                f"frecuencia {freq} < mínimo {self.min_frequency}"
            )
        return True, ""

    # ── Pipeline principal ────────────────────────────────────────────────────

    def filter(self, entities: list) -> list[FilterResult]:
        """
        Aplica el pipeline de filtrado a una lista de entidades.

        Acepta dos formatos de entrada (y mezclas de ambos):
          - str  → el texto de la entidad directamente (lo que envía
                   el frontend cuando el usuario pega términos línea a línea).
          - dict → {"text": str} con campos opcionales "label", "label_desc",
                   "start_char", "end_char", "sentence".
        """
        if not entities:
            return []

        # ── 1. Normalizar: convertir todo a dicts ────────────────────────────
        normed: list[dict] = []
        for e in entities:
            if isinstance(e, str):
                normed.append({"text": e.strip()})
            elif isinstance(e, dict):
                normed.append(e)
            else:
                normed.append({"text": str(e).strip()})

        # ── 2. Calcular frecuencias sobre la lista ya normalizada ─────────────
        freq: dict[str, int] = {}
        for e in normed:
            key = _normalize(e.get("text", ""))
            freq[key] = freq.get(key, 0) + 1

        # ── 3. Pre-calcular contenedores válidos para F3 ──────────────────────
        # Solo se consideran contenedores las entidades que pasan F0+F1+F2.
        # Evita que una entidad A ya marcada para eliminar impida que su
        # subtérmino B sea conservado.
        def _passes_f0_f1_f2(ent: dict) -> bool:
            t = ent.get("text", "").strip()
            ok, _ = self._f0_blacklist(t)
            if not ok:
                return False
            ok, _ = self._f1_length(t)
            if not ok:
                return False
            ok, _ = self._f2_common_noun_phrase(t)
            return ok

        valid_containers = [e for e in normed if _passes_f0_f1_f2(e)]

        # ── 4. Aplicar filtros en cascada ────────────────────────────────────
        results: list[FilterResult] = []

        for ent in normed:
            text       = ent.get("text", "").strip()
            label      = ent.get("label", "")
            label_desc = ent.get("label_desc", "")
            start      = ent.get("start_char")
            end        = ent.get("end_char")
            sentence   = ent.get("sentence", "")
            ent_freq   = freq.get(_normalize(text), 1)

            kept      = True
            filter_id = ""
            reason    = ""

            # F0 – lista negra del usuario
            if kept:
                kept, reason = self._f0_blacklist(text)
                if not kept:
                    filter_id = "F0"

            # F1 – longitud / tokens de ruido
            if kept:
                kept, reason = self._f1_length(text)
                if not kept:
                    filter_id = "F1"

            # F2 – sintagma sin nombre propio
            if kept:
                kept, reason = self._f2_common_noun_phrase(text)
                if not kept:
                    filter_id = "F2"

            # F3 – subtérmino solapado (solo contra contenedores válidos)
            if kept and self.apply_f3:
                kept, reason = self._f3_subspan(ent, valid_containers)
                if not kept:
                    filter_id = "F3"

            # F4 – frecuencia mínima
            if kept and self.min_frequency > 1:
                kept, reason = self._f4_frequency(text, ent_freq)
                if not kept:
                    filter_id = "F4"

            results.append(FilterResult(
                text       = text,
                label      = label,
                label_desc = label_desc,
                frequency  = ent_freq,
                kept       = kept,
                filter_id  = filter_id,
                reason     = reason,
                start_char = start,
                end_char   = end,
                sentence   = sentence,
            ))

        return results

    # ── Estadísticas ──────────────────────────────────────────────────────────

    def stats(self, results: list[FilterResult]) -> dict:
        total  = len(results)
        kept   = sum(1 for r in results if r.kept)
        by_filter: dict[str, int] = {}
        for r in results:
            if not r.kept:
                by_filter[r.filter_id] = by_filter.get(r.filter_id, 0) + 1
        return {
            "total":     total,
            "kept":      kept,
            "removed":   total - kept,
            "pct_kept":  round(kept / total * 100, 1) if total else 0,
            "by_filter": by_filter,
            "blacklist_size": len(self._blacklist_raw),
        }

    # ── Utilidades ────────────────────────────────────────────────────────────

    def filter_accepted(
        self, results: list[FilterResult]
    ) -> list[FilterResult]:
        return [r for r in results if r.kept]

    def filter_rejected(
        self, results: list[FilterResult]
    ) -> list[FilterResult]:
        return [r for r in results if not r.kept]

    def to_json(
        self,
        results: list[FilterResult],
        only_kept: bool = True,
    ) -> str:
        data = [r.to_dict() for r in results if not only_kept or r.kept]
        return json.dumps(data, ensure_ascii=False, indent=2)

    def to_csv(
        self,
        results: list[FilterResult],
        only_kept: bool = True,
    ) -> str:
        import csv, io
        buf = io.StringIO()
        fields = ["text", "label", "frequency", "kept", "filter_id", "reason",
                  "start_char", "end_char"]
        w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            if not only_kept or r.kept:
                w.writerow(r.to_dict())
        return buf.getvalue()


# ─── Demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = [
        {"text": "Pedro Sánchez",     "label": "PER", "start_char": 0,   "end_char": 13},
        {"text": "Madrid",            "label": "LOC", "start_char": 30,  "end_char": 36},
        {"text": "España",            "label": "LOC", "start_char": 50,  "end_char": 56},
        # Subtérmino: "España" dentro de "Plaza de España" — sin aparición independiente
        {"text": "Plaza de España",   "label": "LOC", "start_char": 60,  "end_char": 75},
        {"text": "España",            "label": "LOC", "start_char": 69,  "end_char": 75},
        # Sintagma sin nombre propio
        {"text": "Los usuarios y empresas", "label": "MISC", "start_char": 80, "end_char": 103},
        # Lista negra
        {"text": "IA",                "label": "ORG", "start_char": 110, "end_char": 112},
        # Ruido
        {"text": "3",                 "label": "MISC", "start_char": 120, "end_char": 121},
    ]

    ef = EntityFilter(user_blacklist=["IA", "software"])
    results = ef.filter(sample)
    stats   = ef.stats(results)

    print(f"\n{'='*60}")
    print(f"  Total: {stats['total']}  Conservadas: {stats['kept']}  "
          f"Eliminadas: {stats['removed']}")
    print(f"  Por filtro: {stats['by_filter']}")
    print(f"{'='*60}")
    for r in results:
        icon = "✓" if r.kept else "✗"
        print(f"  {icon} [{r.label}] {r.text!r:30s}",
              end="")
        if not r.kept:
            print(f"  → {r.filter_id}: {r.reason}")
        else:
            print()
