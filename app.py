"""
app.py
------
Aplicación web Flask para el filtrado semiautomático de entidades nombradas.
Integra NERModel (large) y EntityProcessor con una interfaz interactiva.
"""

from flask import Flask, render_template, request, jsonify, send_file
import io
from ner_model import NERModel, SPACY_MODELS, DEFAULT_MODEL
from entity_processor import EntityProcessor

app = Flask(__name__)

_state: dict = {
    "text": "",
    "entities": [],
    "processor": None,
    "model_key": DEFAULT_MODEL,
}

def get_model(model_key: str = DEFAULT_MODEL) -> NERModel:
    """Carga o reutiliza el modelo NER (caché por clave)."""
    if not hasattr(app, "_ner_model") or app._ner_model_key != model_key:
        app._ner_model = NERModel(model_key)
        app._ner_model_key = model_key
    return app._ner_model


@app.route("/")
def index():
    return render_template("index.html", models=list(SPACY_MODELS.keys()), default=DEFAULT_MODEL)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "").strip()
    model_key = data.get("model", DEFAULT_MODEL)

    if not text:
        return jsonify({"error": "El texto no puede estar vacío."}), 400

    try:
        model = get_model(model_key)
        entities = model.extract(text)
        processor = EntityProcessor(entities)

        _state["text"] = text
        _state["entities"] = entities
        _state["processor"] = processor
        _state["model_key"] = model_key

        return jsonify({
            "entities": processor.to_list(),
            "stats": processor.stats(),
            "model": model.model_name,
            "available_labels": model.available_labels,
        })
    except OSError as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/update", methods=["POST"])
def update_entity():
    data = request.get_json()
    start_char = data.get("start_char")
    action = data.get("action")

    processor: EntityProcessor = _state.get("processor")
    if processor is None:
        return jsonify({"error": "No hay entidades cargadas."}), 400

    if action == "accept":
        processor.accept(start_char)
    elif action == "reject":
        processor.reject(start_char)
    elif action == "reset":
        for e in processor.entities:
            if e.start_char == start_char:
                e.accepted = None
    else:
        return jsonify({"error": "Accion no valida."}), 400

    return jsonify({"entities": processor.to_list(), "stats": processor.stats()})


@app.route("/api/bulk", methods=["POST"])
def bulk_action():
    data = request.get_json()
    action = data.get("action")
    label  = data.get("label")

    processor: EntityProcessor = _state.get("processor")
    if processor is None:
        return jsonify({"error": "No hay entidades cargadas."}), 400

    targets = processor.filter_by_label([label]) if label else processor.entities

    for e in targets:
        if   action == "accept_all": e.accepted = True
        elif action == "reject_all": e.accepted = False
        elif action == "reset_all":  e.accepted = None
        else: return jsonify({"error": "Accion no valida."}), 400

    return jsonify({"entities": processor.to_list(), "stats": processor.stats()})


@app.route("/api/entities/list")
def entities_list():
    """
    Lista procesable de entidades — diseñada para consumo externo.

    Parámetros opcionales:
      ?filter=accepted|rejected|pending|all   (defecto: all)
      ?label=PER,ORG,LOC                      (filtrar por etiquetas)

    Ejemplo externo:
        import requests
        ents = requests.get("http://localhost:5000/api/entities/list?filter=accepted").json()
        for e in ents:
            print(e["text"], e["label"])
    """
    processor: EntityProcessor = _state.get("processor")
    if processor is None:
        return jsonify([])

    filter_mode  = request.args.get("filter", "all")
    label_filter = request.args.get("label", "")

    if   filter_mode == "accepted": ents = processor.filter_accepted()
    elif filter_mode == "rejected": ents = processor.filter_rejected()
    elif filter_mode == "pending":  ents = processor.filter_pending()
    else:                           ents = processor.entities

    if label_filter:
        labels = [l.strip().upper() for l in label_filter.split(",")]
        ents = [e for e in ents if e.label in labels]

    return jsonify([e.to_dict() for e in ents])


@app.route("/api/export/<fmt>")
def export_entities(fmt: str):
    processor: EntityProcessor = _state.get("processor")
    if processor is None:
        return jsonify({"error": "No hay entidades cargadas."}), 400

    only_accepted = request.args.get("only_accepted", "true").lower() == "true"

    if fmt == "json":
        content = processor.to_json(only_accepted=only_accepted)
        return send_file(io.BytesIO(content.encode("utf-8")),
                         mimetype="application/json", as_attachment=True,
                         download_name="entidades.json")
    elif fmt == "csv":
        content = processor.to_csv(only_accepted=only_accepted)
        return send_file(io.BytesIO(content.encode("utf-8")),
                         mimetype="text/csv", as_attachment=True,
                         download_name="entidades.csv")
    else:
        return jsonify({"error": "Formato no soportado. Usa 'json' o 'csv'."}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
