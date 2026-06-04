# Sistema semiautomático de filtrado de Entidades Nombradas en español

TFG — Escuela Técnica Superior de Ingenieros Informáticos, UPM  
**David García Prieto** · Tutor: Mariano Rico

Sistema web para la extracción, filtrado y enlace a bases de conocimiento de entidades nombradas en español. Integra cinco modelos NER (spaCy, BETO-NER, SpanBERTa, Stanza-ES y XLM-RoBERTa) más un ensamble propio que supera al mejor modelo individual en el benchmark de evaluación.

---

## Inicio rápido con Docker

```bash
docker run -p 5000:5000 davidgarciaprieto/ner-filter:v3.0
```

O con la última versión:

```bash
docker run -p 5000:5000 davidgarciaprieto/ner-filter:latest
```

La primera ejecución descarga los pesos de los modelos (~4 GB). Una vez en marcha, abre [http://localhost:5000](http://localhost:5000).

Con docker-compose:

```bash
docker compose up
```

---

## Estructura del repositorio

```
TFG/
├── app.py                          # Servidor Flask — API REST y rutas
├── ner_model.py                    # Modelos NER + lógica del ensamble
├── entity_filter.py                # Pipeline de filtrado (F0–F4)
├── entity_linker.py                # Enlace a Wikipedia y DBpedia
├── entity_processor.py             # Preprocesamiento de entidades
├── templates/                      # Interfaz web (HTML5 + CSS + JS)
├── resultados benchmark aplicación/ # Resultados del benchmark propio 
├── resultados benchmark clásicos/    # Resultados del benchmark estadístico (TCL)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Módulos principales

**`ner_model.py`** — Carga y gestiona los cinco modelos NER. El ensamble combina Stanza-ES y XLM-RoBERTa mediante dos reglas de arbitraje derivadas del análisis empírico de errores: corrección de organizaciones cortas etiquetadas como PER y corrección de topónimos etiquetados como ORG.

**`entity_filter.py`** — Pipeline de cinco filtros en cascada: lista negra del usuario (F0), ruido léxico y fragmentos WordPiece (F1), sintagmas sin nombre propio (F2), subtérminos solapados (F3) y frecuencia mínima (F4).

**`entity_linker.py`** — Enlace a Wikipedia y DBpedia consultando en paralelo DBpedia Spotlight, DBpedia Lookup y Wikipedia Search API. Devuelve hasta 10 candidatos ordenados por similitud de título.

**`app.py`** — API REST con los endpoints `/api/ner`, `/api/filter`, `/api/link` y `/api/blacklist`. Sirve también la interfaz web.

---

## Ejecución sin Docker

```bash
pip install -r requirements.txt
python -m spacy download es_core_news_lg
python -m stanza.download es
python app.py
```

Requiere Python 3.10+ y ~8 GB de RAM para cargar todos los modelos.

---

## Evaluación

| Modelo | F₁ (frases) | F₁ (párrafos) |
|---|---|---|
| spaCy lg | 0,870 | 0,556 |
| BETO-NER | 0,867 | 0,788 |
| SpanBERTa | 0,868 | 0,824 |
| Stanza-ES | 0,921 | 0,865 |
| XLM-RoBERTa | 0,929 | 0,647 |
| **Combinado** | **≈ 0,955** | **0,865** |

El pipeline de filtrado alcanza p = r = F₁ = 1,000 sobre el benchmark de 100 términos en 10 bloques temáticos. Resultado acotado a los patrones de ruido documentados.
