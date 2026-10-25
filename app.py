#!/usr/bin/env python3
"""Audio2Text - Transcribe audio files to text.

Uses faster-whisper with GPU when available, falls back to OpenAI Whisper API.
"""

import json
import os
import queue
import subprocess
import threading
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from flask import (Flask, Response, jsonify, render_template, request,
                   send_file, send_from_directory)

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".wma", ".aac", ".opus"}
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

APP_VERSION = "1.7"

OPENAI_MAX_BYTES = 25 * 1024 * 1024  # 25 MB API limit

# Detect GPU and faster-whisper
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
HAS_LOCAL_WHISPER = False

try:
    from faster_whisper import WhisperModel
    import ctranslate2
    HAS_LOCAL_WHISPER = True
    if ctranslate2.get_cuda_device_count() > 0:
        DEVICE = "cuda"
        COMPUTE_TYPE = "float16"
except ImportError:
    pass

# Speed factors with faster-whisper + GPU (much faster than original whisper on CPU)
MODEL_SPEED_GPU = {
    "tiny": 120.0,
    "base": 80.0,
    "small": 40.0,
    "medium": 20.0,
    "large": 10.0,
}
MODEL_SPEED_CPU = {
    "tiny": 25.0,
    "base": 12.0,
    "small": 5.0,
    "medium": 2.5,
    "large": 1.0,
}
API_SPEED = 3.0

# faster-whisper model name mapping
FASTER_WHISPER_NAMES = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "medium": "medium",
    "large": "large-v3",
}

_loaded_models: dict[str, "WhisperModel"] = {}

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


def _get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", audio_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return float(data.get("format", {}).get("duration", 0))
    except Exception:
        pass
    return 0


def _emit(job_id: str, event: str, data: dict):
    if job_id in jobs and "queue" in jobs[job_id]:
        jobs[job_id]["queue"].put({"event": event, "data": data})


def _progress_ticker(job_id: str, audio_duration: float, speed_factor: float, phase: str):
    """Emit progress events based on estimated processing speed."""
    if audio_duration <= 0:
        return

    estimated_time = audio_duration / speed_factor
    t0 = time.time()
    last_pct = 0

    while job_id in jobs and jobs[job_id]["status"] == "running":
        elapsed = time.time() - t0
        pct = min(95, int(elapsed / estimated_time * 100))

        if pct > last_pct:
            last_pct = pct
            remaining = max(0, estimated_time - elapsed)
            eta = _format_duration(remaining) if remaining > 5 else "pocs segons"
            _emit(job_id, "progress", {
                "pct": pct,
                "msg": f"{phase}... {pct}% (queden ~{eta})",
            })

        time.sleep(2)


def _start_progress(job_id: str, audio_duration: float, speed_factor: float, phase: str):
    """Start a background progress ticker thread."""
    t = threading.Thread(
        target=_progress_ticker,
        args=(job_id, audio_duration, speed_factor, phase),
        daemon=True,
    )
    t.start()
    return t


def _preprocess_audio(job_id: str, audio_path: str, audio_duration: float = 0) -> str:
    """Preprocess audio: normalize volume, reduce noise, convert to 16kHz mono WAV."""
    preprocessed = str(Path(audio_path).parent / "preprocessed.wav")
    _emit(job_id, "status", {"msg": "Preprocessant àudio (normalitzant volum, reduint soroll)..."})
    timeout = max(180, int(audio_duration * 2) + 60)
    cmd = [
        "ffmpeg", "-y", "-i", audio_path,
        "-af", "highpass=f=80,lowpass=f=8000,afftdn=nf=-20,loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", "16000", "-ac", "1",
        preprocessed,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        return audio_path
    return preprocessed


def _get_model(model_name: str) -> "WhisperModel":
    """Load or return cached faster-whisper model."""
    if model_name not in _loaded_models:
        fw_name = FASTER_WHISPER_NAMES.get(model_name, model_name)
        _loaded_models[model_name] = WhisperModel(
            fw_name, device=DEVICE, compute_type=COMPUTE_TYPE,
        )
    return _loaded_models[model_name]


def _transcribe_local(job_id: str, audio_path: str, model_name: str, language: str,
                       audio_duration: float, initial_prompt: str = ""):
    """Transcribe using faster-whisper with GPU if available."""
    device_label = "GPU" if DEVICE == "cuda" else "CPU"
    _emit(job_id, "status", {"msg": f"Carregant model '{model_name}' ({device_label})..."})
    model = _get_model(model_name)

    processed_path = _preprocess_audio(job_id, audio_path, audio_duration)

    speed_table = MODEL_SPEED_GPU if DEVICE == "cuda" else MODEL_SPEED_CPU
    speed = speed_table.get(model_name, 5.0)
    _emit(job_id, "progress", {
        "pct": 0,
        "msg": f"Transcrivint amb faster-whisper {device_label} ({_format_duration(audio_duration)} d'àudio)...",
    })
    _start_progress(job_id, audio_duration, speed, "Transcrivint")

    transcribe_opts: dict = {
        "beam_size": 5,
        "vad_filter": True,
        "vad_parameters": {"min_silence_duration_ms": 300},
        "word_timestamps": True,
        "condition_on_previous_text": True,
        "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "repetition_penalty": 1.1,
    }
    if language != "auto":
        transcribe_opts["language"] = language
    if initial_prompt:
        transcribe_opts["initial_prompt"] = initial_prompt

    segments_iter, info = model.transcribe(processed_path, **transcribe_opts)

    detected_lang = info.language or "unknown"
    segments = []
    text_parts = []
    for seg in segments_iter:
        segments.append({"start": seg.start, "end": seg.end, "text": seg.text})
        text_parts.append(seg.text.strip())

    full_text = " ".join(text_parts)
    srt_content = _generate_srt(segments)

    return full_text, srt_content, detected_lang, len(segments), f"local-{device_label}"


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


def _transcribe_api(job_id: str, audio_path: str, language: str, audio_duration: float):
    """Transcribe using OpenAI Whisper API."""
    import httpx

    audio_path = _compress_for_api(job_id, audio_path)

    _emit(job_id, "progress", {
        "pct": 0,
        "msg": f"Enviant a l'API d'OpenAI ({_format_duration(audio_duration)} d'àudio)...",
    })
    _start_progress(job_id, audio_duration, API_SPEED, "Transcrivint via API")

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

    srt_lines = []
    for i, seg in enumerate(segments, 1):
        start = _format_srt_time(seg.get("start", 0))
        end = _format_srt_time(seg.get("end", 0))
        text = seg.get("text", "").strip()
        srt_lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    srt_content = "\n".join(srt_lines)

    return full_text, srt_content, detected_lang, len(segments), "api"


def _transcribe_worker(job_id: str, audio_path: str, model_name: str, language: str,
                       initial_prompt: str = ""):
    try:
        audio_duration = _get_audio_duration(audio_path)
        if audio_duration > 0:
            _emit(job_id, "info", {
                "audio_duration": _format_duration(audio_duration),
                "audio_seconds": round(audio_duration),
            })

        t0 = time.time()

        if HAS_LOCAL_WHISPER:
            full_text, srt_content, detected_lang, num_segments, backend = \
                _transcribe_local(job_id, audio_path, model_name, language, audio_duration,
                                  initial_prompt)
        elif OPENAI_API_KEY:
            full_text, srt_content, detected_lang, num_segments, backend = \
                _transcribe_api(job_id, audio_path, language, audio_duration)
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
        device=DEVICE.upper(),
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
    initial_prompt = request.form.get("prompt", "").strip()

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
        args=(job_id, str(audio_path), model_name, language, initial_prompt),
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
    elif fmt == "pdf":
        return _generate_pdf(job_id)
    return "Format no vàlid", 400


def _generate_pdf(job_id: str):
    from datetime import datetime

    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (HRFlowable, Paragraph, SimpleDocTemplate,
                                    Spacer, Table, TableStyle)

    job = jobs[job_id]
    result = job["result"]
    filename = job.get("filename", "audio")

    pdf_path = OUTPUT_DIR / job_id / "transcript.pdf"

    BRAND_BLUE = colors.HexColor("#2563eb")
    BRAND_DARK = colors.HexColor("#1e293b")
    BRAND_GRAY = colors.HexColor("#64748b")
    BRAND_LIGHT_BG = colors.HexColor("#f8fafc")
    BRAND_BORDER = colors.HexColor("#e2e8f0")
    BRAND_ACCENT = colors.HexColor("#0ea5e9")

    page_w, page_h = A4
    margin = 25 * mm

    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(BRAND_BLUE)
        canvas.rect(0, page_h - 8 * mm, page_w, 8 * mm, fill=True, stroke=False)
        canvas.setFillColor(BRAND_ACCENT)
        canvas.rect(0, page_h - 9 * mm, page_w, 1 * mm, fill=True, stroke=False)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(BRAND_GRAY)
        canvas.drawString(margin, 12 * mm, f"Audio2Text v{APP_VERSION}")
        canvas.drawRightString(page_w - margin, 12 * mm, f"Pàgina {doc.page}")
        canvas.setStrokeColor(BRAND_BORDER)
        canvas.line(margin, 16 * mm, page_w - margin, 16 * mm)
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        leftMargin=margin, rightMargin=margin,
        topMargin=margin + 10 * mm, bottomMargin=margin,
    )

    styles = getSampleStyleSheet()

    s_title = ParagraphStyle(
        "PDFTitle", parent=styles["Title"],
        fontSize=22, leading=28, textColor=BRAND_DARK,
        spaceAfter=4 * mm, alignment=TA_LEFT,
        fontName="Helvetica-Bold",
    )
    s_subtitle = ParagraphStyle(
        "PDFSubtitle", parent=styles["Normal"],
        fontSize=10, leading=14, textColor=BRAND_GRAY,
        spaceAfter=6 * mm, fontName="Helvetica",
    )
    s_section = ParagraphStyle(
        "PDFSection", parent=styles["Heading2"],
        fontSize=13, leading=18, textColor=BRAND_BLUE,
        spaceBefore=6 * mm, spaceAfter=3 * mm,
        fontName="Helvetica-Bold",
        borderWidth=0, borderPadding=0,
    )
    s_body = ParagraphStyle(
        "PDFBody", parent=styles["Normal"],
        fontSize=10, leading=16, textColor=BRAND_DARK,
        alignment=TA_JUSTIFY, fontName="Helvetica",
        spaceAfter=2 * mm,
    )
    s_srt_time = ParagraphStyle(
        "SRTTime", parent=styles["Normal"],
        fontSize=8, leading=11, textColor=BRAND_ACCENT,
        fontName="Helvetica-Bold",
    )
    s_srt_text = ParagraphStyle(
        "SRTText", parent=styles["Normal"],
        fontSize=9, leading=13, textColor=BRAND_DARK,
        fontName="Helvetica",
    )
    s_meta_label = ParagraphStyle(
        "MetaLabel", parent=styles["Normal"],
        fontSize=8.5, leading=12, textColor=BRAND_GRAY,
        fontName="Helvetica-Bold",
    )
    s_meta_val = ParagraphStyle(
        "MetaVal", parent=styles["Normal"],
        fontSize=9, leading=13, textColor=BRAND_DARK,
        fontName="Helvetica",
    )

    story = []

    base_name = Path(filename).stem
    story.append(Paragraph("Transcripció", s_title))
    story.append(Paragraph(
        f"{base_name} &middot; {datetime.now().strftime('%d/%m/%Y %H:%M')}",
        s_subtitle,
    ))

    story.append(HRFlowable(
        width="100%", thickness=0.5, color=BRAND_BORDER,
        spaceAfter=4 * mm, spaceBefore=0,
    ))

    lang_map = {
        "ca": "Català", "es": "Castellà", "en": "Anglès",
        "fr": "Francès", "de": "Alemany", "it": "Italià",
        "pt": "Portuguès", "ja": "Japonès", "zh": "Xinès",
    }
    detected = result.get("language", "?")
    lang_display = lang_map.get(detected, detected)

    backend_label = result.get("backend", "")
    if "GPU" in backend_label:
        backend_label = "Whisper local (GPU)"
    elif "CPU" in backend_label:
        backend_label = "Whisper local (CPU)"
    elif backend_label == "api":
        backend_label = "OpenAI API"

    meta_data = [
        [Paragraph("FITXER", s_meta_label), Paragraph(filename, s_meta_val),
         Paragraph("IDIOMA", s_meta_label), Paragraph(lang_display, s_meta_val)],
        [Paragraph("TEMPS", s_meta_label), Paragraph(result.get("duration", "-"), s_meta_val),
         Paragraph("BACKEND", s_meta_label), Paragraph(backend_label, s_meta_val)],
        [Paragraph("CARÀCTERS", s_meta_label), Paragraph(f"{result.get('chars', 0):,}", s_meta_val),
         Paragraph("SEGMENTS", s_meta_label), Paragraph(str(result.get("segments", 0)), s_meta_val)],
    ]

    col_w = (page_w - 2 * margin) / 4
    meta_table = Table(meta_data, colWidths=[col_w * 0.8, col_w * 1.2, col_w * 0.8, col_w * 1.2])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BRAND_LIGHT_BG),
        ("BOX", (0, 0), (-1, -1), 0.5, BRAND_BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, BRAND_BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROUNDEDCORNERS", [3, 3, 3, 3]),
    ]))
    story.append(meta_table)

    story.append(Paragraph("Transcripció completa", s_section))

    text = result.get("text", "")
    paragraphs = text.split("\n")
    for p in paragraphs:
        p = p.strip()
        if p:
            p_escaped = p.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(p_escaped, s_body))

    srt = result.get("srt", "")
    if srt.strip():
        story.append(Paragraph("Subtítols (SRT)", s_section))

        srt_rows = []
        blocks = srt.strip().split("\n\n")
        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) >= 3:
                timecode = lines[1]
                seg_text = " ".join(lines[2:])
                seg_text_escaped = seg_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                srt_rows.append([
                    Paragraph(timecode, s_srt_time),
                    Paragraph(seg_text_escaped, s_srt_text),
                ])

        if srt_rows:
            srt_table = Table(srt_rows, colWidths=[55 * mm, page_w - 2 * margin - 55 * mm])
            srt_table.setStyle(TableStyle([
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEBELOW", (0, 0), (-1, -2), 0.3, BRAND_BORDER),
                ("BACKGROUND", (0, 0), (0, -1), BRAND_LIGHT_BG),
            ]))
            story.append(srt_table)

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)

    download_name = f"{base_name}_transcripcio.pdf"
    return send_file(str(pdf_path), as_attachment=True, download_name=download_name)


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
