#!/usr/bin/env python3
"""Audio2Text - Transcribe audio files to text using OpenAI Whisper (local)."""

import json
import os
import queue
import threading
import time
import uuid
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from flask import (Flask, Response, jsonify, render_template, request,
                   send_from_directory)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".wma", ".aac", ".opus"}
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]

APP_VERSION = "1.0"

jobs: dict[str, dict] = {}


def _format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _generate_srt(segments: list[dict]) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        start = _format_srt_time(seg["start"])
        end = _format_srt_time(seg["end"])
        text = seg["text"].strip()
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m {s}s"


def _emit(job_id: str, event: str, data: dict):
    if job_id in jobs and "queue" in jobs[job_id]:
        jobs[job_id]["queue"].put({"event": event, "data": data})


def _transcribe_worker(job_id: str, audio_path: str, model_name: str, language: str):
    try:
        _emit(job_id, "status", {"msg": f"Carregant model Whisper '{model_name}'..."})
        import whisper
        model = whisper.load_model(model_name, device="cpu")

        _emit(job_id, "status", {"msg": "Transcrivint àudio (això pot trigar una estona)..."})
        t0 = time.time()

        transcribe_opts = {"verbose": False, "fp16": False}
        if language != "auto":
            transcribe_opts["language"] = language

        result = model.transcribe(audio_path, **transcribe_opts)
        elapsed = time.time() - t0

        detected_lang = result.get("language", "unknown")
        full_text = result["text"].strip()
        segments = result.get("segments", [])
        srt_content = _generate_srt(segments)

        job_dir = Path(audio_path).parent
        txt_path = job_dir / "transcript.txt"
        srt_path = job_dir / "transcript.srt"

        txt_path.write_text(full_text, encoding="utf-8")
        srt_path.write_text(srt_content, encoding="utf-8")

        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = {
            "text": full_text,
            "srt": srt_content,
            "language": detected_lang,
            "duration": _format_duration(elapsed),
            "chars": len(full_text),
            "segments": len(segments),
        }

        _emit(job_id, "done", jobs[job_id]["result"])

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        _emit(job_id, "error", {"msg": str(e)})


@app.route("/")
def index():
    return render_template("index.html", version=APP_VERSION, models=WHISPER_MODELS)


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "Cap fitxer d'àudio rebut"}), 400

    f = request.files["audio"]
    if not f.filename:
        return jsonify({"error": "Fitxer buit"}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Format no suportat: {ext}"}), 400

    model_name = request.form.get("model", "base")
    language = request.form.get("language", "auto")

    if model_name not in WHISPER_MODELS:
        model_name = "base"

    job_id = uuid.uuid4().hex[:12]
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    safe_name = f.filename.replace("/", "_").replace("\\", "_")
    audio_path = job_dir / safe_name
    f.save(str(audio_path))

    jobs[job_id] = {
        "status": "running",
        "filename": f.filename,
        "model": model_name,
        "language": language,
        "queue": queue.Queue(),
    }

    thread = threading.Thread(
        target=_transcribe_worker,
        args=(job_id, str(audio_path), model_name, language),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/stream/<job_id>")
def stream(job_id):
    if job_id not in jobs:
        return "Job not found", 404

    def generate():
        q = jobs[job_id]["queue"]
        while True:
            try:
                msg = q.get(timeout=120)
                yield f"event: {msg['event']}\ndata: {json.dumps(msg['data'], ensure_ascii=False)}\n\n"
                if msg["event"] in ("done", "error"):
                    break
            except queue.Empty:
                yield "event: ping\ndata: {}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/job/<job_id>")
def job_status(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Job no trobat"}), 404
    job = jobs[job_id]
    resp = {"status": job["status"], "filename": job.get("filename", "")}
    if job["status"] == "done":
        resp["result"] = job["result"]
    elif job["status"] == "error":
        resp["error"] = job.get("error", "Error desconegut")
    return jsonify(resp)


@app.route("/download/<job_id>/<fmt>")
def download(job_id, fmt):
    if job_id not in jobs or jobs[job_id]["status"] != "done":
        return "No disponible", 404
    job_dir = OUTPUT_DIR / job_id
    if fmt == "txt":
        return send_from_directory(str(job_dir), "transcript.txt", as_attachment=True)
    elif fmt == "srt":
        return send_from_directory(str(job_dir), "transcript.srt", as_attachment=True)
    return "Format no vàlid", 400


@app.route("/history")
def history():
    items = []
    for job_id, job in sorted(jobs.items(), reverse=True):
        if job["status"] == "done":
            items.append({
                "id": job_id,
                "filename": job.get("filename", ""),
                "model": job.get("model", ""),
                "language": job["result"].get("language", ""),
                "chars": job["result"].get("chars", 0),
                "duration": job["result"].get("duration", ""),
            })
    return jsonify(items)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5070, debug=True)
