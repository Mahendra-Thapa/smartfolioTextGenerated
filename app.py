"""
app.py
------
Flask REST API — Portfolio AI Backend
Compatible with Windows / Python 3.12
"""

import os
import sys
import socket

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask      import Flask, request, jsonify
from flask_cors import CORS

import lstm_model
import grammar
import rewriter

app = Flask(__name__)

# Allow ALL origins, ALL methods, ALL headers
# This fixes CORS for Next.js on localhost:3000 calling Flask on 127.0.0.1
CORS(app, 
     origins="*",
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "OPTIONS"],
     supports_credentials=False)

# Handle preflight OPTIONS requests explicitly
@app.after_request
def after_request(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# ════════════════════════════════════════════════════════════════════
# Health / Status
# ════════════════════════════════════════════════════════════════════

@app.route("/api/status", methods=["GET", "OPTIONS"])
def status():
    is_ready = lstm_model._ready.is_set()
    return jsonify({
        "ready":   is_ready,
        "message": "Model is ready." if is_ready else "Model is still training, please wait.",
    })


# ════════════════════════════════════════════════════════════════════
# Grammar & Spell Fix
# ════════════════════════════════════════════════════════════════════

@app.route("/api/grammar", methods=["POST", "OPTIONS"])
def api_grammar():
    """
    Request:  { "text": "i am a passionat softare enginr" }
    Response: { "original": "...", "corrected": "...", "changed": true }
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200

    body = request.get_json(silent=True) or {}
    text = body.get("text", "").strip()

    if not text:
        return jsonify({"error": "text field is required"}), 400

    return jsonify(grammar.correct(text))


# ════════════════════════════════════════════════════════════════════
# LSTM Autocomplete
# ════════════════════════════════════════════════════════════════════

@app.route("/api/autocomplete", methods=["POST", "OPTIONS"])
def api_autocomplete():
    """
    Request:  { "text": "I am a passionate", "top_k": 5, "num_words": 5 }
    Response: { "seed": "...", "completions": [...], "ready": true }
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200

    body      = request.get_json(silent=True) or {}
    text      = body.get("text", "").strip()
    top_k     = int(body.get("top_k",     lstm_model.TOP_K))
    num_words = int(body.get("num_words", lstm_model.NUM_WORDS))

    if not text:
        return jsonify({"error": "text field is required"}), 400

    if not lstm_model._ready.is_set():
        return jsonify({
            "seed":        text,
            "completions": [],
            "ready":       False,
            "message":     "Model is still training. Please try again shortly.",
        })

    completions = lstm_model.predict(text, top_k=top_k, num_words=num_words)
    return jsonify({"seed": text, "completions": completions, "ready": True})


# ════════════════════════════════════════════════════════════════════
# Text Rewriter
# ════════════════════════════════════════════════════════════════════

@app.route("/api/rewrite", methods=["POST", "OPTIONS"])
def api_rewrite():
    """
    Request:  { "text": "i worked on building a website", "field": "experience" }
    Response: { "original": "...", "rewritten": "...", "changed": true, "hint": "..." }
    Valid fields: bio, skills, experience, education, projects
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200

    body  = request.get_json(silent=True) or {}
    text  = body.get("text",  "").strip()
    field = body.get("field", "bio").strip().lower()

    if not text:
        return jsonify({"error": "text field is required"}), 400

    valid_fields = {"introduction", "bio", "skills", "projects", "experience", "additional_experience", "qualifications", "education"}
    if field not in valid_fields:
        return jsonify({"error": f"Invalid field. Must be one of: {', '.join(valid_fields)}"}), 400

    return jsonify(rewriter.rewrite(text, field))


# ════════════════════════════════════════════════════════════════════
# Retrain LSTM
# ════════════════════════════════════════════════════════════════════

@app.route("/api/retrain", methods=["POST", "OPTIONS"])
def api_retrain():
    """
    Request (optional): { "extra": ["sentence 1", "sentence 2"] }
    Response: { "message": "Retraining started in background." }
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200

    body  = request.get_json(silent=True) or {}
    extra = body.get("extra", [])

    if not isinstance(extra, list):
        return jsonify({"error": "extra must be a list of strings"}), 400

    lstm_model.retrain(extra_sentences=extra if extra else None)
    return jsonify({"message": "Retraining started in background."})


# ════════════════════════════════════════════════════════════════════
# Start Server (FIXED PORT)
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    PORT = 8003

    print(f"\n[API] Flask server running on http://127.0.0.1:{PORT}")
    print(f"[API] Test: http://127.0.0.1:{PORT}/api/status\n")

    app.run(
        host="127.0.0.1",
        port=PORT,
        debug=False,
        use_reloader=False,
        threaded=True,
    )

# ════════════════════════════════════════════════════════════════════
# Debug endpoint — shows vocabulary and tests prediction directly
# ════════════════════════════════════════════════════════════════════

@app.route("/api/debug", methods=["GET"])
def api_debug():
    """
    GET /api/debug
    Returns vocabulary size, sample words, and a test prediction.
    Use this to diagnose why autocomplete returns empty.
    """
    if lstm_model._tokenizer is None:
        return jsonify({"error": "Model not loaded yet"}), 503

    tok         = lstm_model._tokenizer
    vocab_size  = tok.vocab_size
    # first 30 words in vocabulary (most frequent)
    sample_words = [tok.index_word.get(i, "") for i in range(1, min(31, vocab_size))]

    # test prediction with known words
    test_result = lstm_model.predict("i am a", top_k=3, num_words=5)

    return jsonify({
        "vocab_size":   vocab_size,
        "max_seq_len":  lstm_model._max_seq_len,
        "sample_words": sample_words,
        "test_predict": test_result,
    })

