"""
app.py — NER multi-backend + filtrado + enlace Wikipedia/DBpedia
"""

from flask import Flask, render_template, request, jsonify, send_file
import io, json, traceback

from ner_model import NERModel, MODEL_CATALOG, DEFAULT_MODEL
from entity_processor import EntityProcessor
from entity_filter import EntityFilter
from entity_linker import EntityLinker

app = Flask(__name__)

_state: dict = {
    "text": "", "entities": [], "processor": None, "model_key": DEFAULT_MODEL,
}
_model_cache: dict[str, NERModel] = {}
# Lista negra de la sesión: persiste mientras el servidor está en marcha
_session_blacklist: list[str] = []

def get_model(model_key: str) -> NERModel:
    if model_key not in _model_cache:
        _model_cache[model_key] = NERModel(model_key)
    return _model_cache[model_key]


# ── NER ───────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", default=DEFAULT_MODEL)

@app.route("/api/models")
def list_models():
    return jsonify({k: {"name":v["name"],"desc":v["desc"],"backend":v["backend"]} for k,v in MODEL_CATALOG.items()})

@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        data      = request.get_json(force=True, silent=True) or {}
        text      = data.get("text", "").strip()
        model_key = data.get("model", DEFAULT_MODEL)
        if not text:
            return jsonify({"error": "El texto no puede estar vacío."}), 400
        if model_key not in MODEL_CATALOG:
            return jsonify({"error": f"Modelo '{model_key}' no existe."}), 400
        model     = get_model(model_key)
        entities  = model.extract(text)
        processor = EntityProcessor(entities)
        _state.update({"text": text, "entities": entities, "processor": processor, "model_key": model_key})
        return jsonify({"entities": processor.to_list(), "stats": processor.stats(),
                        "model": model.model_name, "available_labels": model.available_labels})
    except Exception as e:
        return jsonify({"error": str(e), "detail": traceback.format_exc()}), 500

@app.route("/api/update", methods=["POST"])
def update_entity():
    try:
        data       = request.get_json(force=True, silent=True) or {}
        start_char = data.get("start_char")
        action     = data.get("action")
        processor: EntityProcessor = _state.get("processor")
        if processor is None:
            return jsonify({"error": "No hay entidades cargadas."}), 400
        if   action == "accept": processor.accept(start_char)
        elif action == "reject": processor.reject(start_char)
        elif action == "reset":
            for e in processor.entities:
                if e.start_char == start_char: e.accepted = None
        else:
            return jsonify({"error": "Acción no válida."}), 400
        return jsonify({"entities": processor.to_list(), "stats": processor.stats()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/bulk", methods=["POST"])
def bulk_action():
    try:
        data    = request.get_json(force=True, silent=True) or {}
        action  = data.get("action")
        label   = data.get("label")
        processor: EntityProcessor = _state.get("processor")
        if processor is None:
            return jsonify({"error": "No hay entidades cargadas."}), 400
        targets = processor.filter_by_label([label]) if label else processor.entities
        for e in targets:
            if   action == "accept_all": e.accepted = True
            elif action == "reject_all": e.accepted = False
            elif action == "reset_all":  e.accepted = None
            else: return jsonify({"error": "Acción no válida."}), 400
        return jsonify({"entities": processor.to_list(), "stats": processor.stats()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/entities/list")
def entities_list():
    processor: EntityProcessor = _state.get("processor")
    if processor is None: return jsonify([])
    f = request.args.get("filter", "all")
    l = request.args.get("label", "")
    if   f == "accepted": ents = processor.filter_accepted()
    elif f == "rejected": ents = processor.filter_rejected()
    elif f == "pending":  ents = processor.filter_pending()
    else:                 ents = processor.entities
    if l:
        labels = [x.strip().upper() for x in l.split(",")]
        ents = [e for e in ents if e.label in labels]
    return jsonify([e.to_dict() for e in ents])

@app.route("/api/export/<fmt>")
def export_entities(fmt: str):
    try:
        processor: EntityProcessor = _state.get("processor")
        if processor is None: return jsonify({"error": "No hay entidades cargadas."}), 400
        only_accepted = request.args.get("only_accepted", "true").lower() == "true"
        if fmt == "json":
            content = processor.to_json(only_accepted=only_accepted)
            return send_file(io.BytesIO(content.encode()), mimetype="application/json",
                             as_attachment=True, download_name="entidades.json")
        elif fmt == "csv":
            content = processor.to_csv(only_accepted=only_accepted)
            return send_file(io.BytesIO(content.encode()), mimetype="text/csv",
                             as_attachment=True, download_name="entidades.csv")
        return jsonify({"error": "Formato no soportado."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Filtrado ──────────────────────────────────────────────────────────────────

@app.route("/api/filter", methods=["POST"])
def filter_entities():
    try:
        data     = request.get_json(force=True, silent=True) or {}
        entities = data.get("entities", [])
        if not entities:
            return jsonify({"error": "La lista de entidades está vacía."}), 400
        ef      = EntityFilter(user_blacklist=_session_blacklist)
        results = ef.filter(entities)
        stats   = ef.stats(results)
        return jsonify({
            "results":   [r.to_dict() for r in results],
            "stats":     stats,
            "blacklist": _session_blacklist,
        })
    except Exception as e:
        return jsonify({"error": str(e), "detail": traceback.format_exc()}), 500


@app.route("/api/blacklist", methods=["GET"])
def get_blacklist():
    """Devuelve la lista negra activa de la sesión."""
    return jsonify({"blacklist": _session_blacklist})


@app.route("/api/blacklist/add", methods=["POST"])
def add_to_blacklist():
    """
    Añade uno o varios términos a la lista negra de sesión.
    Body: { "terms": ["IA", "software"] }
    """
    data  = request.get_json(force=True, silent=True) or {}
    terms = data.get("terms", [])
    if isinstance(terms, str):
        terms = [terms]
    added = []
    for t in terms:
        t = t.strip()
        if t and t not in _session_blacklist:
            _session_blacklist.append(t)
            added.append(t)
    return jsonify({"added": added, "blacklist": _session_blacklist})


@app.route("/api/blacklist/remove", methods=["POST"])
def remove_from_blacklist():
    """
    Elimina uno o varios términos de la lista negra de sesión.
    Body: { "terms": ["IA"] }
    """
    data  = request.get_json(force=True, silent=True) or {}
    terms = data.get("terms", [])
    if isinstance(terms, str):
        terms = [terms]
    removed = []
    for t in terms:
        t = t.strip()
        if t in _session_blacklist:
            _session_blacklist.remove(t)
            removed.append(t)
    return jsonify({"removed": removed, "blacklist": _session_blacklist})


@app.route("/api/blacklist/clear", methods=["POST"])
def clear_blacklist():
    """Vacía la lista negra de sesión."""
    _session_blacklist.clear()
    return jsonify({"blacklist": _session_blacklist})

@app.route("/api/filter/export/<fmt>", methods=["POST"])
def export_filter(fmt: str):
    try:
        data        = request.get_json(force=True, silent=True) or {}
        results_raw = data.get("results", [])
        only_kept   = data.get("only_kept", True)
        rows = [r for r in results_raw if not only_kept or r.get("kept")]
        if fmt == "json":
            content = json.dumps(rows, ensure_ascii=False, indent=2)
            return send_file(io.BytesIO(content.encode()), mimetype="application/json",
                             as_attachment=True, download_name="filtrado.json")
        elif fmt == "csv":
            import csv
            output = io.StringIO()
            fields = ["text","label","frequency","kept","filter_id","reason"]
            w = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
            return send_file(io.BytesIO(output.getvalue().encode()), mimetype="text/csv",
                             as_attachment=True, download_name="filtrado.csv")
        return jsonify({"error": "Formato no soportado."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Enlace Wikipedia / DBpedia ────────────────────────────────────────────────

@app.route("/api/link", methods=["POST"])
def link_entities():
    try:
        data     = request.get_json(force=True, silent=True) or {}
        entities = data.get("entities", [])
        lang     = data.get("lang", "es")
        if not entities:
            return jsonify({"error": "La lista de entidades está vacía."}), 400
        linker  = EntityLinker(lang=lang)
        results = linker.link(entities)
        stats   = linker.stats(results)
        return jsonify({"results": [r.to_dict() for r in results], "stats": stats})
    except Exception as e:
        return jsonify({"error": str(e), "detail": traceback.format_exc()}), 500

@app.route("/api/link/export/<fmt>", methods=["POST"])
def export_link(fmt: str):
    try:
        data        = request.get_json(force=True, silent=True) or {}
        results_raw = data.get("results", [])
        only_found  = data.get("only_found", True)
        rows = [r for r in results_raw if not only_found or r.get("found")]
        if fmt == "json":
            content = json.dumps(rows, ensure_ascii=False, indent=2)
            return send_file(io.BytesIO(content.encode()), mimetype="application/json",
                             as_attachment=True, download_name="enlaces.json")
        elif fmt == "csv":
            import csv
            output = io.StringIO()
            fields = ["text","label","wiki_title","wiki_url","abstract","types","score","source","found"]
            w = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                row = dict(r); row["types"] = ", ".join(r.get("types") or [])
                w.writerow(row)
            return send_file(io.BytesIO(output.getvalue().encode()), mimetype="text/csv",
                             as_attachment=True, download_name="enlaces.csv")
        return jsonify({"error": "Formato no soportado."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
