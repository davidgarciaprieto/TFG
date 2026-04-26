# NER Filter — Filtrado Semiautomático de Entidades Nombradas

Aplicación web modular con spaCy **large models** para extraer, revisar
y exportar entidades nombradas de forma semiautomática.

---

## Estructura de ficheros

```
.
├── app.py                ← Servidor Flask + API REST
├── ner_model.py          ← Módulo NER (spaCy large): extracción y Entity
├── entity_processor.py   ← Filtrado, agrupación, exportación
├── requirements.txt
├── setup_venv.sh         ← Script de entorno virtual (Linux/macOS)
├── setup_venv.bat        ← Script de entorno virtual (Windows)
└── templates/
    └── index.html        ← Interfaz web interactiva
```

---

## Creación del entorno virtual

### Linux / macOS

```bash
chmod +x setup_venv.sh
./setup_venv.sh
```

### Windows

```bat
setup_venv.bat
```

### Manual (cualquier sistema)

```bash
# 1. Crear entorno
python -m venv .venv

# 2. Activar
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Descargar modelos large
python -m spacy download es_core_news_lg   # español  (~500 MB)
python -m spacy download en_core_web_lg    # inglés   (~560 MB)
python -m spacy download xx_ent_wiki_sm    # multilingüe
```

---

## Arrancar la aplicación

```bash
# Asegúrate de que el entorno está activo
source .venv/bin/activate

python app.py
# → http://localhost:5000
```

---

## Modelos disponibles

| Clave    | Modelo spaCy          | Idioma        | Tamaño |
|----------|-----------------------|---------------|--------|
| `es_lg`  | `es_core_news_lg`     | Español large | ~500 MB|
| `es_md`  | `es_core_news_md`     | Español medio | ~45 MB |
| `es_sm`  | `es_core_news_sm`     | Español small | ~12 MB |
| `en_lg`  | `en_core_web_lg`      | Inglés large  | ~560 MB|
| `en_trf` | `en_core_web_trf`     | Inglés transf.| ~440 MB|
| `en_md`  | `en_core_web_md`      | Inglés medio  | ~43 MB |
| `xx`     | `xx_ent_wiki_sm`      | Multilingüe   | ~12 MB |

---

## API REST

| Método | Ruta                   | Descripción                              |
|--------|------------------------|------------------------------------------|
| POST   | `/api/analyze`         | Analiza texto, devuelve entidades        |
| POST   | `/api/update`          | Acepta/rechaza/resetea una entidad       |
| POST   | `/api/bulk`            | Acción en bloque (+ filtro por etiqueta) |
| GET    | `/api/entities/list`   | **Lista procesable** (ver abajo)         |
| GET    | `/api/export/json`     | Descarga JSON                            |
| GET    | `/api/export/csv`      | Descarga CSV                             |

### `/api/entities/list` — lista procesable

Diseñada para ser consumida por otros scripts:

```
GET /api/entities/list                        # todas
GET /api/entities/list?filter=accepted        # solo aceptadas
GET /api/entities/list?filter=rejected        # solo rechazadas
GET /api/entities/list?label=PER,ORG          # filtrar por tipo
GET /api/entities/list?filter=accepted&label=PER
```

Ejemplo desde Python externo:

```python
import requests

r = requests.get("http://localhost:5000/api/entities/list?filter=accepted")
entities = r.json()   # lista de dicts

for e in entities:
    print(e["label"], e["text"], e["sentence"])
```

---

## Uso sin interfaz web (Python puro)

```python
from ner_model import NERModel
from entity_processor import EntityProcessor

model = NERModel("es_lg")   # carga es_core_news_lg
entities = model.extract("Pedro Sánchez visitó Madrid el 15 de marzo.")

# Iterar directamente
for e in entities:
    print(e)  # [?] PER    'Pedro Sánchez'   (0:13)

# Procesar
proc = EntityProcessor(entities)
proc.accept_all()
proc.reject(30)           # rechazar entidad por start_char

print(proc.to_json(only_accepted=True))
print(proc.stats())
# {'total': 3, 'accepted': 2, 'rejected': 1, 'pending': 0, 'by_label': {...}}
```

---

## Estructura de una entidad

```json
{
  "text":        "Pedro Sánchez",
  "label":       "PER",
  "label_desc":  "Persona",
  "start_token": 2,
  "end_token":   4,
  "start_char":  3,
  "end_char":    16,
  "sentence":    "El presidente Pedro Sánchez visitó Madrid.",
  "kb_id":       "",
  "accepted":    null
}
```

`accepted`: `null` = pendiente · `true` = aceptada · `false` = rechazada
