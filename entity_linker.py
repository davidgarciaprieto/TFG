"""
entity_linker.py
----------------
Enlaza entidades nombradas con Wikipedia/DBpedia usando varias fuentes
en cascada:
  1. DBpedia Spotlight  (anotación directa sobre texto, es/en)
  2. DBpedia Lookup     (búsqueda por nombre, devuelve varios candidatos)
  3. Wikipedia Search   (último recurso, devuelve varios candidatos)

Novedad respecto a versiones anteriores:
  - Cada entidad devuelve hasta N candidatos (por defecto 5) en el campo
    `candidates`, ordenados de mayor a menor relevancia.
  - La UI puede presentar los candidatos al usuario para que elija.
  - El campo `types` limpia los identificadores Wikidata (Q8054, Q206229)
    y los prefijos de namespace para mostrar solo el nombre del tipo.
"""

from __future__ import annotations
import logging
import re as _re
import requests
from dataclasses import dataclass, asdict, field
from typing import Optional

_TIMEOUT = 8
_HEADERS = {"User-Agent": "NERLinker/2.0 (research)"}

log = logging.getLogger("entity_linker")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[entity_linker] %(levelname)s %(message)s"))
    log.addHandler(h)
log.setLevel(logging.INFO)

MAX_CANDIDATES = 10


def _title_similarity(query: str, title: str) -> float:
    """
    Puntuación de similitud entre la consulta y un título candidato (0–1).
    Combina dos señales:
      - Coincidencia exacta o de prefijo: máxima puntuación.
      - Proporción de palabras de la consulta que aparecen en el título.
    Devuelve 1.0 para coincidencia exacta, valores menores para parciales.
    Mayor valor = más relevante = debe aparecer antes.
    """
    q = query.lower().strip()
    t = title.lower().strip()
    if not q or not t:
        return 0.0
    # Coincidencia exacta
    if q == t:
        return 1.0
    # Título empieza por la consulta
    if t.startswith(q):
        return 0.95
    # Consulta contenida en título
    if q in t:
        return 0.85
    # Proporción de palabras de la consulta presentes en el título
    q_words = set(q.split())
    t_words = set(t.split())
    if not q_words:
        return 0.0
    overlap = len(q_words & t_words) / len(q_words)
    return round(overlap * 0.8, 3)


# ─── Candidato individual ─────────────────────────────────────────────────────

@dataclass
class Candidate:
    """Un resultado de enlace candidato (puede haber varios por entidad)."""
    rank:       int             # 1 = mejor candidato
    uri:        str             # URI DBpedia
    wiki_url:   str             # URL Wikipedia
    wiki_title: str             # Título del artículo
    abstract:   str             # Extracto del artículo (≤400 chars)
    types:      list[str]       # tipos DBpedia limpios (sin prefijos ni Wikidata)
    score:      float           # confianza (0-1)
    score_type: str             # "similarity" | "popularity" | "none"
    source:     str             # "spotlight" | "dbpedia_lookup" | "wikipedia_api"

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Resultado de enlace ──────────────────────────────────────────────────────

@dataclass
class LinkedEntity:
    text:         str
    label:        str
    label_desc:   str
    # Mejor candidato (por compatibilidad con la versión anterior)
    uri:          Optional[str]  = None
    wiki_url:     Optional[str]  = None
    wiki_title:   Optional[str]  = None
    abstract:     Optional[str]  = None
    types:        list[str]      = field(default_factory=list)
    score:        float          = 0.0
    score_type:   str            = ""
    source:       str            = ""
    found:        bool           = False
    # Lista de candidatos para que el usuario elija
    candidates:   list[dict]     = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Utilidades ───────────────────────────────────────────────────────────────

def _dbpedia_uri_from_title(title: str, lang: str = "es") -> str:
    slug = title.replace(" ", "_")
    host = "es.dbpedia.org" if lang == "es" else "dbpedia.org"
    return f"http://{host}/resource/{requests.utils.quote(slug, safe='_')}"


def _title_from_dbpedia_uri(uri: str) -> str:
    raw = _re.sub(r"https?://(?:[a-z]+\.)?dbpedia\.org/resource/", "", uri)
    try:
        raw = requests.utils.unquote(raw)
    except Exception:
        pass
    return raw.replace("_", " ")


def _strip_html(text: str) -> str:
    """Elimina etiquetas HTML de un string (ej. '<B>ADN</B>' → 'ADN')."""
    return _re.sub(r"<[^>]+>", "", text).strip()


def _clean_types(raw_types) -> list[str]:
    """
    Extrae los tipos legibles de DBpedia, descartando todo lo que no sea
    de la ontología DBpedia o Schema.org.

    Se descartan:
      - Cualquier tipo del namespace Wikidata (prefijo 'Wikidata:' o URI
        'wikidata.org'): el sistema no usa Wikidata, sus IDs (Q8054,
        Q206229…) no son legibles ni relevantes para el usuario.
      - Términos genéricos de ontologías sin valor informativo
        (Thing, Agent, Resource, SocialObject, Place…).
      - Duplicados (mismo nombre con distinto prefijo/namespace).

    Formatos de entrada aceptados:
      - str separada por comas: "DBpedia:Hospital,Wikidata:Q16917,Schema:Hospital"
      - list de URIs:           ["http://dbpedia.org/ontology/Hospital", ...]
      - list de strings cortos: ["Organisation", "Hospital"]
    """
    _GENERIC = {
        "thing", "owl#thing", "resource", "agent", "entity",
        "socialobject", "object", "abstract", "event", "place",
        "timeperiod", "period", "work", "topicalconcept",
        "populated", "yago",
    }

    if isinstance(raw_types, str):
        parts = [t.strip() for t in raw_types.split(",") if t.strip()]
    elif isinstance(raw_types, list):
        parts = [str(t).strip() for t in raw_types if t]
    else:
        return []

    cleaned = []
    seen = set()
    for p in parts:
        # Descartar cualquier cosa de Wikidata (por prefijo o por URI)
        p_lower = p.lower()
        if "wikidata" in p_lower:
            continue

        # Extraer el nombre: la parte después del último ':', '/' o '#'
        name = _re.split(r"[:/\#]", p)[-1].strip()
        if not name or name.isdigit():
            continue
        if name.lower() in _GENERIC:
            continue
        if name not in seen:
            seen.add(name)
            cleaned.append(name)

    return cleaned[:5]


# ─── Clientes de API ─────────────────────────────────────────────────────────

class _DBpediaSpotlight:
    ENDPOINTS = {
        "es": [
            "https://api.dbpedia-spotlight.org/es/annotate",
            "https://api.dbpedia-spotlight.org/spanish/annotate",
        ],
        "en": [
            "https://api.dbpedia-spotlight.org/en/annotate",
            "https://api.dbpedia-spotlight.org/english/annotate",
        ],
    }

    def annotate(self, text: str, lang: str = "es",
                 confidence: float = 0.2) -> list[dict]:
        endpoints = self.ENDPOINTS.get(lang, self.ENDPOINTS["es"])
        for url in endpoints:
            try:
                res = requests.get(
                    url,
                    params={"text": text, "confidence": confidence},
                    headers={**_HEADERS, "Accept": "application/json"},
                    timeout=_TIMEOUT,
                )
                if res.status_code == 200:
                    return res.json().get("Resources", []) or []
                log.warning(f"Spotlight {url} -> {res.status_code}")
            except requests.exceptions.Timeout:
                log.warning(f"Spotlight {url} timeout")
            except Exception as e:
                log.warning(f"Spotlight {url} error: {e}")
        return []

    def candidates(self, text: str, lang: str = "es",
                   n: int = MAX_CANDIDATES) -> list[Candidate]:
        """Devuelve hasta n candidatos de Spotlight ordenados por score."""
        results = self.annotate(text, lang=lang, confidence=0.1)
        if not results:
            return []
        # Ordenar por similarityScore descendente
        results.sort(key=lambda r: float(r.get("@similarityScore", 0)), reverse=True)
        out = []
        for i, r in enumerate(results[:n]):
            uri   = r.get("@URI", "")
            title = _strip_html(_title_from_dbpedia_uri(uri))
            types = _clean_types(r.get("@types", ""))
            out.append(Candidate(
                rank       = i + 1,
                uri        = uri,
                wiki_url   = f"https://{lang}.wikipedia.org/wiki/{requests.utils.quote(title.replace(' ','_'))}",
                wiki_title = title,
                abstract   = "",  # se rellena en EntityLinker._enrich
                types      = types,
                score      = round(float(r.get("@similarityScore", 0)), 3),
                score_type = "similarity",
                source     = "spotlight",
            ))
        return out


class _DBpediaLookup:
    BASE = "https://lookup.dbpedia.org/api/search"

    def candidates(self, text: str, n: int = MAX_CANDIDATES) -> list[Candidate]:
        """Devuelve hasta n candidatos de DBpedia Lookup."""
        try:
            res = requests.get(
                self.BASE,
                params={"query": text, "format": "json", "maxResults": n},
                headers={**_HEADERS, "Accept": "application/json"},
                timeout=_TIMEOUT,
            )
            if res.status_code != 200:
                log.warning(f"DBpedia Lookup -> {res.status_code}")
                return []
            data = res.json()
            docs = data.get("docs") or data.get("results") or []
        except requests.exceptions.Timeout:
            log.warning("DBpedia Lookup timeout")
            return []
        except Exception as e:
            log.warning(f"DBpedia Lookup error: {e}")
            return []

        out = []
        for i, doc in enumerate(docs[:n]):
            uri_raw = doc.get("resource") or doc.get("uri") or ""
            uri = uri_raw[0] if isinstance(uri_raw, list) and uri_raw else uri_raw
            if not uri:
                continue
            label_raw = doc.get("label") or doc.get("name") or ""
            label = label_raw[0] if isinstance(label_raw, list) and label_raw else label_raw
            title = _strip_html(label or _title_from_dbpedia_uri(uri))
            types_raw = doc.get("type") or doc.get("types") or []
            types = _clean_types(types_raw)
            # Score: número de referencias entrantes (popularidad), normalizado
            raw_pop = doc.get("score", 0)
            if isinstance(raw_pop, list):
                raw_pop = raw_pop[0] if raw_pop else 0
            try:
                pop = float(raw_pop)
            except (TypeError, ValueError):
                pop = 0.0
            score = round(min(pop / 10000.0, 1.0), 3) if pop > 1 else 0.5
            out.append(Candidate(
                rank       = i + 1,
                uri        = uri,
                wiki_url   = f"https://es.wikipedia.org/wiki/{requests.utils.quote(title.replace(' ','_'))}",
                wiki_title = title,
                abstract   = "",
                types      = types,
                score      = score,
                score_type = "popularity",
                source     = "dbpedia_lookup",
            ))
        return out


class _WikipediaAPI:
    def candidates(self, text: str, lang: str = "es",
                   n: int = MAX_CANDIDATES) -> list[Candidate]:
        """Devuelve hasta n candidatos de Wikipedia Search."""
        search_url = f"https://{lang}.wikipedia.org/w/api.php"
        try:
            res = requests.get(search_url, params={
                "action": "query", "list": "search",
                "srsearch": text, "srlimit": n,
                "format": "json", "utf8": 1,
            }, headers=_HEADERS, timeout=_TIMEOUT)
            res.raise_for_status()
            items = res.json().get("query", {}).get("search", [])
        except Exception as e:
            log.warning(f"Wikipedia search error: {e}")
            return []

        out = []
        for i, item in enumerate(items[:n]):
            title = _strip_html(item.get("title", ""))
            slug  = title.replace(" ", "_")
            wiki_url = f"https://{lang}.wikipedia.org/wiki/{requests.utils.quote(slug)}"
            out.append(Candidate(
                rank       = i + 1,
                uri        = _dbpedia_uri_from_title(title, lang=lang),
                wiki_url   = wiki_url,
                wiki_title = title,
                abstract   = "",
                types      = [],
                score      = 0.0,
                score_type = "none",
                source     = "wikipedia_api",
            ))
        return out


# ─── Linker principal ─────────────────────────────────────────────────────────

class EntityLinker:
    """
    Enlaza entidades con Wikipedia/DBpedia devolviendo múltiples candidatos.

    Para cada entidad busca hasta MAX_CANDIDATES (5) resultados en cascada:
      1. DBpedia Spotlight
      2. DBpedia Lookup
      3. Wikipedia Search API

    El primer candidato es el mejor según cada fuente. La UI presenta todos
    los candidatos al usuario para que confirme o elija otro.
    """

    def __init__(self, lang: str = "es"):
        self.lang       = lang
        self._spotlight = _DBpediaSpotlight()
        self._lookup    = _DBpediaLookup()
        self._wiki      = _WikipediaAPI()

    def _fetch_abstract(self, title: str) -> tuple[str, str]:
        """Obtiene (abstract, wiki_url_definitiva), primero en self.lang y luego en 'en'."""
        slug = title.replace(" ", "_")
        default_url = f"https://{self.lang}.wikipedia.org/wiki/{requests.utils.quote(slug)}"
        for lang in (self.lang, "en"):
            try:
                url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(slug)}"
                r = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
                if r.ok:
                    data = r.json()
                    abstract = data.get("extract", "")[:400]
                    if abstract:
                        wiki_url = data.get("content_urls", {}).get(
                            "desktop", {}).get("page", default_url)
                        return abstract, wiki_url
            except Exception:
                continue
        return "", default_url

    def _enrich(self, candidates: list[Candidate]) -> list[Candidate]:
        """Rellena el abstract y wiki_url definitiva de todos los candidatos."""
        for c in candidates:
            abstract, wiki_url = self._fetch_abstract(c.wiki_title)
            c.abstract = abstract
            if wiki_url:
                c.wiki_url = wiki_url
        return candidates

    def link_one(self, entity: dict,
                 n: int = MAX_CANDIDATES) -> LinkedEntity:
        """
        Enlaza una entidad y devuelve hasta n candidatos.

        Estrategia: llama a todas las fuentes disponibles y combina los
        resultados, eliminando duplicados por URI. Así aunque Spotlight
        solo devuelva 1 match, Wikipedia Search puede aportar los demás.

        El campo `candidates` contiene todos; los campos de primer nivel
        (`uri`, `wiki_title`, etc.) corresponden al mejor candidato (rank=1).
        """
        text       = entity.get("text", "")
        label      = entity.get("label", "")
        label_desc = entity.get("label_desc", "")
        le = LinkedEntity(text=text, label=label, label_desc=label_desc)

        all_candidates: list[Candidate] = []
        seen_uris: set[str] = set()

        def _add(cands: list[Candidate]) -> None:
            for c in cands:
                if c.uri not in seen_uris:
                    seen_uris.add(c.uri)
                    all_candidates.append(c)

        # ── 1. Spotlight ─────────────────────────────────────────────────────
        try:
            sp = self._spotlight.candidates(text, lang=self.lang, n=n)
            if sp:
                _add(sp)
                log.info(f"[spotlight] '{text}' -> {len(sp)} candidatos")
        except Exception as e:
            log.warning(f"[spotlight] error: {e}")

        # ── 2. DBpedia Lookup ─────────────────────────────────────────────────
        try:
            lk = self._lookup.candidates(text, n=n)
            if lk:
                _add(lk)
                log.info(f"[dbpedia_lookup] '{text}' -> {len(lk)} candidatos")
        except Exception as e:
            log.warning(f"[dbpedia_lookup] error: {e}")

        # ── 3. Wikipedia Search (siempre, para completar candidatos) ──────────
        try:
            wp = self._wiki.candidates(text, lang=self.lang, n=n)
            if wp:
                _add(wp)
                log.info(f"[wikipedia_api] '{text}' -> {len(wp)} candidatos")
        except Exception as e:
            log.warning(f"[wikipedia_api] error: {e}")

        if not all_candidates:
            le.source = "none"
            le.found  = False
            log.info(f"[none] '{text}' sin resultados")
            return le

        # Ordenar todos los candidatos por similitud de título con el texto
        # buscado, independientemente de la fuente.
        all_candidates.sort(
            key=lambda c: _title_similarity(text, c.wiki_title),
            reverse=True
        )

        # Reasignar ranks y truncar a n
        for i, c in enumerate(all_candidates[:n]):
            c.rank = i + 1

        # Enriquecer abstracts de todos los candidatos
        candidates = self._enrich(all_candidates[:n])

        # Rellenar campos de primer nivel con el mejor candidato (rank=1)
        best = candidates[0]
        le.uri        = best.uri
        le.wiki_url   = best.wiki_url
        le.wiki_title = best.wiki_title
        le.abstract   = best.abstract
        le.types      = best.types
        le.score      = best.score
        le.score_type = best.score_type
        le.source     = best.source
        le.found      = True
        le.candidates = [c.to_dict() for c in candidates]
        return le

    def link(self, entities: list[dict],
             n: int = MAX_CANDIDATES) -> list[LinkedEntity]:
        """Enlaza una lista de entidades, deduplicando por texto."""
        seen:    dict[str, LinkedEntity] = {}
        results: list[LinkedEntity]      = []
        for ent in entities:
            text = ent.get("text", "").strip()
            if not text:
                results.append(LinkedEntity(text=text,
                               label=ent.get("label",""),
                               label_desc=ent.get("label_desc","")))
                continue
            if text not in seen:
                seen[text] = self.link_one(ent, n=n)
            results.append(seen[text])
        return results

    def stats(self, results: list[LinkedEntity]) -> dict:
        total = len(results)
        found = sum(1 for r in results if r.found)
        by_src: dict[str, int] = {}
        for r in results:
            by_src[r.source] = by_src.get(r.source, 0) + 1
        return {
            "total":     total,
            "found":     found,
            "not_found": total - found,
            "pct_found": round(found / total * 100, 1) if total else 0,
            "by_source": by_src,
        }


if __name__ == "__main__":
    sample = [
        {"text": "Hospital La Paz",  "label": "ORG"},
        {"text": "Pedro Sánchez",    "label": "PER"},
        {"text": "Madrid",           "label": "LOC"},
    ]
    linker = EntityLinker(lang="es")
    for ent in sample:
        le = linker.link_one(ent)
        print(f"\n[{le.label}] {le.text} — {len(le.candidates)} candidatos:")
        for c in le.candidates:
            print(f"  {c['rank']}. {c['wiki_title']!r:35s} [{c['source']}] score={c['score']}")
            if c['types']:
                print(f"      tipos: {', '.join(c['types'])}")
