#!/usr/bin/env python3
"""Audio2Text - Transcribe audio files to text.

Uses local Whisper when available, falls back to OpenAI Whisper API.
"""

import json
import os
import queue
import subprocess
import threading
import time
import uuid
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from dotenv import load_dotenv
from flask import (Flask, Response, jsonify, render_template, request,
                   send_from_directory)

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".wma", ".aac", ".opus"}
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

APP_VERSION = "1.2"

OPENAI_MAX_BYTES = 25 * 1024 * 1024  # 25 MB API limit

# Detect if local Whisper is available
HAS_LOCAL_WHISPER = False
try:
    import whisper
    HAS_LOCAL_WHISPER = True
except ImportError:
    pass

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


def _transcribe_local(job_id: str, audio_path: str, model_name: str, language: str):
    """Transcribe using local Whisper model."""
    _emit(job_id, "status", {"msg": f"Carregant model Whisper local '{model_name}'..."})
    model = whisper.load_model(model_name, device="cpu")

    _emit(job_id, "status", {"msg": "Transcrivint amb Whisper local..."})

    transcribe_opts = {"verbose": False, "fp16": False}
    if language != "auto":
        transcribe_opts["language"] = language

    result = model.transcribe(audio_path, **transcribe_opts)

    detected_lang = result.get("language", "unknown")
    full_text = result["text"].strip()
    segments = result.get("segments", [])
    srt_content = _generate_srt(segments)

    return full_text, srt_content, detected_lang, len(segments), "local"


def _compress_for_api(job_id: str, audio_path: str) -> str:
    """Compress audio to under 25 MB for the OpenAI API. Returns path to use."""
    file_size = os.path.getsize(audio_path)
    if file_size <= OPENAI_MAX_BYTES:
        return audio_path

    _emit(job_id, "status", {
        "msg": f"Fitxer massa gran per l'API ({file_size // 1024 // 1024} MB). Comprimint..."
    })

    compressed = str(Path(audio_path).parent / "compressed.mp3")
    target_bitrate = max(32, int(OPENAI_MAX_BYTES * 8 / (file_size / 48000) / 1000))
    target_bitrate = min(target_bitrate, 64)

    cmd = [
        "ffmpeg", "-y", "-i", audio_path,
        "-vn", "-ac", "1", "-ar", "16000",
        "-b:a", f"{target_bitrate}k",
        compressed,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {result.stderr[:200]}")

    new_size = os.path.getsize(compressed)
    _emit(job_id, "status", {
        "msg": f"Comprimit a {new_size // 1024 // 1024} MB. Enviant a l'API..."
    })
    return compressed


def _transcribe_api(job_id: str, audio_path: str, language: str):
    """Transcribe using OpenAI Whisper API."""
    import httpx

    audio_path = _compress_for_api(job_id, audio_path)
    _emit(job_id, "status", {"msg": "Transcrivint amb OpenAI Whisper API..."})

    # Transcription request
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    with open(audio_path, "rb") as af:
        files = {"file": (Path(audio_path).name, af, "audio/mpeg")}
        data = {
            "model": "whisper-1",
            "response_format": "verbose_json",
            "timestamp_granularities[]": "segment",
        }
        if language != "auto":
            data["language"] = language

        resp = httpx.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers=headers,
            files=files,
            data=data,
            timeout=600,
        )

    if resp.status_code != 200:
        raise RuntimeError(f"API error ({resp.status_code}): {resp.text[:300]}")

    result = resp.json()
    full_text = result.get("text", "").strip()
    detected_lang = result.get("language", "unknown")
    segments = result.get("segments", [])

    # Build SRT from API segments
    srt_lines = []
    for i, seg in enumerate(segments, 1):
        start = _format_srt_time(seg.get("start", 0))
        end = _format_srt_time(seg.get("end", 0))
        text = seg.get("text", "").strip()
        srt_lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    srt_content = "\n".join(srt_lines)

    return full_text, srt_content, detected_lang, len(segments), "api"


def _transcribe_worker(job_id: str, audio_path: str, model_name: str, language: str):
    try:
        t0 = time.time()

        if HAS_LOCAL_WHISPER:
            full_text, srt_content, detected_lang, num_segments, backend = \
                _transcribe_local(job_id, audio_path, model_name, language)
        elif OPENAI_API_KEY:
            full_text, srt_content, detected_lang, num_segments, backend = \
                _transcribe_api(job_id, audio_path, language)
        else:
            raise RuntimeError(
                "Ni Whisper local ni OPENAI_API_KEY disponibles. "
                "Instal·la openai-whisper o configura OPENAI_API_KEY."
            )

        elapsed = time.time() - t0

        job_dir = Path(audio_path).parent
        (job_dir / "transcript.txt").write_text(full_text, encoding="utf-8")
        (job_dir / "transcript.srt").write_text(srt_content, encoding="utf-8")

        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = {
            "text": full_text,
            "srt": srt_content,
            "language": detected_lang,
            "duration": _format_duration(elapsed),
            "chars": len(full_text),
            "segments": num_segments,
            "backend": backend,
        }

        _emit(job_id, "done", jobs[job_id]["result"])

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        _emit(job_id, "error", {"msg": str(e)})


@app.route("/")
def index():
    backend = "local" if HAS_LOCAL_WHISPER else ("api" if OPENAI_API_KEY else "none")
    return render_template(
        "index.html",
        version=APP_VERSION,
        models=WHISPER_MODELS,
        backend=backend,
        has_local=HAS_LOCAL_WHISPER,
    )


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
                "backend": job["result"].get("backend", ""),
            })
    return jsonify(items)


if __name__ == "__main__":
    print(f"Audio2Text v{APP_VERSION}")
    print(f"Backend: {'Whisper local' if HAS_LOCAL_WHISPER else 'OpenAI API' if OPENAI_API_KEY else 'CAP (configura Whisper o OPENAI_API_KEY)'}")
    app.run(host="0.0.0.0", port=5071, debug=False)
