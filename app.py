#!/usr/bin/env python3
"""Audio2Text - Transcribe audio files to text.

Uses faster-whisper with GPU when available, falls back to OpenAI Whisper API.
"""

import json
import os
import queue
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from flask import (Flask, Response, jsonify, render_template, request,
                   send_file, send_from_directory)

import executive

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".wma", ".aac", ".opus", ".mp4", ".avi", ".mkv", ".mov"}
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]
# OpenAI API models: gpt-transcribe is more accurate but returns no timestamps
API_MODELS = ["gpt-transcribe", "whisper-1"]
API_CHUNK_SECONDS = 600  # gpt-transcribe: split long audio to avoid truncated output
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

APP_VERSION = "3.9"

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

# Speaker diarization with speechbrain (no token required)
HAS_DIARIZATION = False
_speaker_model = None

try:
    from speechbrain.inference.speaker import EncoderClassifier
    HAS_DIARIZATION = True
except ImportError:
    try:
        from speechbrain.pretrained import EncoderClassifier
        HAS_DIARIZATION = True
    except ImportError:
        pass


def _get_speaker_model():
    """Load or return cached speechbrain speaker embedding model."""
    global _speaker_model
    if _speaker_model is None:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            from speechbrain.pretrained import EncoderClassifier
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        _speaker_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(Path(__file__).parent / ".speechbrain_cache"),
            run_opts={"device": device},
        )
    return _speaker_model


def _load_audio_as_tensor(audio_path: str):
    """Load full audio file as mono tensor + sample rate using soundfile."""
    import soundfile as sf
    import torch
    data, sr = sf.read(audio_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return torch.from_numpy(data).unsqueeze(0), sr


def _extract_audio_segment_from_tensor(waveform, sr: int, start: float, end: float):
    """Extract a segment from a preloaded waveform tensor."""
    frame_start = int(start * sr)
    frame_end = int(end * sr)
    return waveform[:, frame_start:frame_end]


def _run_diarization(job_id: str, audio_path: str, audio_duration: float,
                     segments: list[dict]) -> list[dict]:
    """Run speaker diarization using speechbrain embeddings + clustering."""
    import torch
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering

    _emit(job_id, "status", {"msg": "Identificant interlocutors (extraient empremtes de veu)..."})
    _start_progress(job_id, audio_duration, 6.0, "Diaritzant")

    model = _get_speaker_model()
    embeddings = []
    valid_indices = []

    full_waveform, sr = _load_audio_as_tensor(audio_path)

    for i, seg in enumerate(segments):
        duration = seg["end"] - seg["start"]
        if duration < 0.3:
            continue
        try:
            chunk = _extract_audio_segment_from_tensor(full_waveform, sr, seg["start"], seg["end"])
            if chunk.shape[1] < 4800:
                continue
            with torch.no_grad():
                emb = model.encode_batch(chunk)
                embeddings.append(emb.squeeze().cpu().numpy())
                valid_indices.append(i)
        except Exception:
            continue

    if len(embeddings) < 2:
        return segments

    X = np.array(embeddings)
    from sklearn.preprocessing import normalize
    X = normalize(X)

    best_n = 2
    best_score = -1
    max_speakers = min(8, len(X) - 1)

    if len(X) >= 4:
        from sklearn.metrics import silhouette_score
        for n in range(2, max_speakers + 1):
            try:
                clustering = AgglomerativeClustering(
                    n_clusters=n, metric="cosine", linkage="average"
                )
                labels = clustering.fit_predict(X)
                if len(set(labels)) < 2:
                    continue
                score = silhouette_score(X, labels, metric="cosine")
                if score > best_score:
                    best_score = score
                    best_n = n
            except Exception:
                break

    clustering = AgglomerativeClustering(
        n_clusters=best_n, metric="cosine", linkage="average"
    )
    labels = clustering.fit_predict(X)

    label_map = {}
    counter = 0
    for idx, vi in enumerate(valid_indices):
        raw_label = int(labels[idx])
        if raw_label not in label_map:
            counter += 1
            label_map[raw_label] = f"Speaker {counter}"
        segments[vi]["speaker"] = label_map[raw_label]

    for seg in segments:
        if "speaker" not in seg:
            seg["speaker"] = ""

    _emit(job_id, "status", {"msg": f"Detectats {best_n} interlocutors"})
    return segments

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
NOTES_TIMEOUT = 1200  # seconds; reasoning models are slow on long transcripts

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


def _load_previous_jobs():
    """Load completed jobs from disk so history survives restarts."""
    from datetime import datetime

    for job_dir in OUTPUT_DIR.iterdir():
        if not job_dir.is_dir():
            continue
        txt_file = job_dir / "transcript.txt"
        srt_file = job_dir / "transcript.srt"
        if not txt_file.exists() or not srt_file.exists():
            continue

        job_id = job_dir.name
        text = txt_file.read_text(encoding="utf-8")
        srt = srt_file.read_text(encoding="utf-8")

        audio_files = [f for f in job_dir.iterdir()
                       if f.suffix.lower() in ALLOWED_EXTENSIONS
                       and f.stem != "preprocessed"]
        filename = audio_files[0].name if audio_files else "audio"

        mtime = txt_file.stat().st_mtime
        date_str = datetime.fromtimestamp(mtime).strftime("%d/%m/%Y %H:%M")

        meta_file = job_dir / "meta.json"
        meta = {}
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        notes = None
        notes_file = job_dir / "notes.json"
        if notes_file.exists():
            try:
                notes = json.loads(notes_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        srt_blocks = [b for b in srt.strip().split("\n\n") if b.strip()]

        jobs[job_id] = {
            "status": "done",
            "filename": filename,
            "model": meta.get("model", ""),
            "language": meta.get("language", ""),
            "date": date_str,
            "queue": queue.Queue(),
            "result": {
                "text": text,
                "srt": srt,
                "language": meta.get("detected_language", ""),
                "duration": meta.get("duration", ""),
                "chars": len(text),
                "segments": len(srt_blocks),
                "backend": meta.get("backend", ""),
                "diarization": meta.get("diarization", False),
                "num_speakers": meta.get("num_speakers", 0),
            },
        }
        if notes:
            jobs[job_id]["notes"] = notes
        exec_file = job_dir / "exec.json"
        if exec_file.exists():
            try:
                jobs[job_id]["exec"] = json.loads(exec_file.read_text(encoding="utf-8"))
            except Exception:
                pass


_load_previous_jobs()


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
        speaker = seg.get("speaker", "")
        if speaker:
            text = f"[{speaker}] {text}"
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def _generate_text_with_speakers(segments: list[dict]) -> str:
    """Generate text with speaker labels, grouping consecutive segments by same speaker."""
    if not segments or not segments[0].get("speaker"):
        return " ".join(seg["text"].strip() for seg in segments)

    parts = []
    current_speaker = None
    current_texts = []

    for seg in segments:
        speaker = seg.get("speaker", "")
        text = seg["text"].strip()
        if not text:
            continue

        if speaker != current_speaker:
            if current_texts:
                parts.append(f"[{current_speaker}]\n{' '.join(current_texts)}\n")
            current_speaker = speaker
            current_texts = [text]
        else:
            current_texts.append(text)

    if current_texts:
        parts.append(f"[{current_speaker}]\n{' '.join(current_texts)}\n")

    return "\n".join(parts)


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
    for seg in segments_iter:
        segments.append({"start": seg.start, "end": seg.end, "text": seg.text})

    return segments, detected_lang, processed_path, f"local-{device_label}"


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


def _api_post(data: dict, audio_path: str) -> dict:
    import httpx

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    with open(audio_path, "rb") as af:
        files = {"file": (Path(audio_path).name, af, "audio/mpeg")}
        resp = httpx.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers=headers,
            files=files,
            data=data,
            timeout=600,
        )
    if resp.status_code != 200:
        raise RuntimeError(f"API error ({resp.status_code}): {resp.text[:300]}")
    return resp.json()


def _split_sentences(text: str, start: float, end: float) -> list[dict]:
    """Split chunk text into sentences with timestamps estimated by text length."""
    parts = [p.strip() for p in re.split(r"(?<=[.!?…])\s+", text.strip()) if p.strip()]
    total = sum(len(p) for p in parts) or 1
    segments, t = [], start
    for p in parts:
        dt = (end - start) * len(p) / total
        segments.append({"start": t, "end": t + dt, "text": " " + p})
        t += dt
    return segments


def _transcribe_api(job_id: str, audio_path: str, language: str, audio_duration: float,
                    model_name: str = "gpt-transcribe", initial_prompt: str = ""):
    """Transcribe using the OpenAI transcription API."""
    _start_progress(job_id, audio_duration, API_SPEED, f"Transcrivint via API ({model_name})")

    if model_name == "whisper-1":
        audio_path = _compress_for_api(job_id, audio_path)
        _emit(job_id, "progress", {
            "pct": 0,
            "msg": f"Enviant a l'API d'OpenAI ({_format_duration(audio_duration)} d'àudio)...",
        })
        data = {
            "model": "whisper-1",
            "response_format": "verbose_json",
            "timestamp_granularities[]": "segment",
        }
        if language != "auto":
            data["language"] = language
        if initial_prompt:
            data["prompt"] = initial_prompt[:800]  # whisper-1 only uses ~224 tokens
        result = _api_post(data, audio_path)
        segments = [
            {"start": seg.get("start", 0), "end": seg.get("end", 0), "text": seg.get("text", "")}
            for seg in result.get("segments", [])
        ]
        return segments, result.get("language", language), audio_path, "api"

    # gpt-transcribe: mono 16 kHz chunks, previous text tail as context for continuity
    job_dir = Path(audio_path).parent
    _emit(job_id, "status", {"msg": "Preparant l'àudio en trossos per l'API..."})
    cmd = [
        "ffmpeg", "-y", "-i", audio_path, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k",
        "-f", "segment", "-segment_time", str(API_CHUNK_SECONDS),
        str(job_dir / "chunk_%03d.mp3"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=max(300, int(audio_duration) + 60))
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {result.stderr[:200]}")
    chunks = sorted(job_dir.glob("chunk_*.mp3"))

    segments, prev_text = [], ""
    for i, chunk in enumerate(chunks):
        _emit(job_id, "status", {"msg": f"Transcrivint tros {i + 1} de {len(chunks)}..."})
        prompt = initial_prompt
        if prev_text:
            prompt = (prompt + "\n\n" if prompt else "") + prev_text[-500:]
        data = {"model": model_name, "response_format": "json"}
        if language != "auto":
            data["language"] = language
        if prompt:
            data["prompt"] = prompt
        text = _api_post(data, str(chunk)).get("text", "").strip()
        start = i * API_CHUNK_SECONDS
        end = start + (_get_audio_duration(str(chunk)) or API_CHUNK_SECONDS)
        segments.extend(_split_sentences(text, start, end))
        prev_text = text
        chunk.unlink()

    return segments, language, audio_path, "api"


def _transcribe_worker(job_id: str, audio_path: str, model_name: str, language: str,
                       initial_prompt: str = "", diarize: bool = False):
    try:
        audio_duration = _get_audio_duration(audio_path)
        if audio_duration > 0:
            _emit(job_id, "info", {
                "audio_duration": _format_duration(audio_duration),
                "audio_seconds": round(audio_duration),
            })

        t0 = time.time()

        if HAS_LOCAL_WHISPER:
            segments, detected_lang, processed_path, backend = \
                _transcribe_local(job_id, audio_path, model_name, language, audio_duration,
                                  initial_prompt)
        elif OPENAI_API_KEY:
            segments, detected_lang, processed_path, backend = \
                _transcribe_api(job_id, audio_path, language, audio_duration,
                                model_name, initial_prompt)
        else:
            raise RuntimeError(
                "Ni Whisper local ni OPENAI_API_KEY disponibles. "
                "Instal·la openai-whisper o configura OPENAI_API_KEY."
            )

        num_speakers = 0
        if diarize and HAS_DIARIZATION and segments:
            try:
                diar_audio = processed_path if os.path.exists(processed_path) else audio_path
                segments = _run_diarization(job_id, diar_audio, audio_duration, segments)
                unique_speakers = set(s.get("speaker", "") for s in segments if s.get("speaker"))
                num_speakers = len(unique_speakers)
            except Exception as e:
                print(f"[DIARIZE] Error: {e}")
                _emit(job_id, "status", {"msg": f"Diarització fallida: {e}. Continuant sense speakers..."})

        has_speakers = any(seg.get("speaker") for seg in segments)
        if has_speakers:
            full_text = _generate_text_with_speakers(segments)
        else:
            full_text = " ".join(seg["text"].strip() for seg in segments)
        srt_content = _generate_srt(segments)

        elapsed = time.time() - t0

        from datetime import datetime

        job_dir = Path(audio_path).parent
        (job_dir / "transcript.txt").write_text(full_text, encoding="utf-8")
        (job_dir / "transcript.srt").write_text(srt_content, encoding="utf-8")

        meta = {
            "model": model_name,
            "language": language,
            "detected_language": detected_lang,
            "duration": _format_duration(elapsed),
            "backend": backend,
            "diarization": diarize and has_speakers,
            "num_speakers": num_speakers,
            "date": datetime.now().strftime("%d/%m/%Y %H:%M"),
        }
        (job_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        date_str = meta["date"]
        jobs[job_id]["date"] = date_str
        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = {
            "text": full_text,
            "srt": srt_content,
            "language": detected_lang,
            "duration": _format_duration(elapsed),
            "chars": len(full_text),
            "segments": len(segments),
            "backend": backend,
            "diarization": meta["diarization"],
            "num_speakers": num_speakers,
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
        models=WHISPER_MODELS if HAS_LOCAL_WHISPER else API_MODELS,
        backend=backend,
        has_local=HAS_LOCAL_WHISPER,
        device=DEVICE.upper(),
        has_diarization=HAS_DIARIZATION,
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

    allowed_models = WHISPER_MODELS if HAS_LOCAL_WHISPER else API_MODELS
    model_name = request.form.get("model", allowed_models[0])
    language = request.form.get("language", "auto")
    initial_prompt = request.form.get("prompt", "").strip()
    diarize = request.form.get("diarize", "false").lower() in ("true", "1", "on")

    if model_name not in allowed_models:
        model_name = "base" if HAS_LOCAL_WHISPER else API_MODELS[0]

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
        args=(job_id, str(audio_path), model_name, language, initial_prompt, diarize),
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

    num_spk = result.get("num_speakers", 0)
    spk_label = f"{num_spk} interlocutors" if num_spk else "No"

    meta_data = [
        [Paragraph("FITXER", s_meta_label), Paragraph(filename, s_meta_val),
         Paragraph("IDIOMA", s_meta_label), Paragraph(lang_display, s_meta_val)],
        [Paragraph("TEMPS", s_meta_label), Paragraph(result.get("duration", "-"), s_meta_val),
         Paragraph("BACKEND", s_meta_label), Paragraph(backend_label, s_meta_val)],
        [Paragraph("CARÀCTERS", s_meta_label), Paragraph(f"{result.get('chars', 0):,}", s_meta_val),
         Paragraph("SEGMENTS", s_meta_label), Paragraph(str(result.get("segments", 0)), s_meta_val)],
        [Paragraph("DIARITZACIÓ", s_meta_label), Paragraph(spk_label, s_meta_val),
         Paragraph("", s_meta_label), Paragraph("", s_meta_val)],
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

    SPEAKER_COLORS_PDF = [
        colors.HexColor("#2563eb"),
        colors.HexColor("#db2777"),
        colors.HexColor("#059669"),
        colors.HexColor("#d97706"),
        colors.HexColor("#7c3aed"),
        colors.HexColor("#ea580c"),
        colors.HexColor("#0d9488"),
        colors.HexColor("#c026d3"),
    ]

    s_speaker = ParagraphStyle(
        "PDFSpeaker", parent=styles["Normal"],
        fontSize=9, leading=13, textColor=BRAND_BLUE,
        fontName="Helvetica-Bold", spaceBefore=4 * mm, spaceAfter=1 * mm,
    )

    story.append(Paragraph("Transcripció completa", s_section))

    text = result.get("text", "")
    has_speakers = result.get("diarization", False)
    speaker_color_map = {}
    speaker_idx = 0

    if has_speakers:
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            import re
            m = re.match(r'^\[(.+?)\]$', line)
            if m:
                speaker = m.group(1)
                if speaker not in speaker_color_map:
                    speaker_color_map[speaker] = SPEAKER_COLORS_PDF[speaker_idx % len(SPEAKER_COLORS_PDF)]
                    speaker_idx += 1
                s_spk = ParagraphStyle(
                    f"Spk_{speaker}", parent=s_speaker,
                    textColor=speaker_color_map[speaker],
                )
                story.append(Paragraph(
                    speaker.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"),
                    s_spk,
                ))
            else:
                p_escaped = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                story.append(Paragraph(p_escaped, s_body))
    else:
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


NOTES_SYSTEM_PROMPT = """Ets un analista sènior que crea informes de KNOWLEDGE TRANSFER exhaustius.
La teva feina NO és resumir. La teva feina és EXTREURE I DOCUMENTAR absolutament tot el que es diu a la conversa perquè serveixi com a referència completa.

REGLES CRÍTIQUES:
- MAI INVENTS res. NOMÉS extreu informació que apareix EXPLÍCITAMENT a la transcripció
- Si un nom de tecnologia, empresa o persona NO es diu literalment al text, NO l'incloguis
- NO assumeixis detalls basant-te en el que és comú a la indústria. Si no es diu, no hi és
- Si no estàs 100% segur que alguna cosa es menciona, NO la posis
- Si es menciona un nom (persona, empresa, tecnologia, lloc), HA d'aparèixer al teu informe
- Si es menciona un número, dada, pressupost, quantitat, HA d'aparèixer
- Si s'explica una situació, problema o anècdota, HA d'aparèixer amb detall
- NO facis paràfrasis curtes. Explica cada punt amb el detall suficient perquè algú que no hi era ho entengui
- Cada secció ha de tenir TOTES les entrades que corresponguin, no un resum de les "principals"
- Les cites textuals han de ser frases reals i reveladores de la conversa, no genèriques
- La transcripció pot tenir errors de reconeixement de veu (noms mal transcrits). Transcriu-los tal qual

Genera els apunts en el MATEIX IDIOMA que la transcripció.

Retorna NOMÉS un JSON vàlid (sense markdown, sense ```json):
{
  "title": "Títol descriptiu",
  "summary": "Resum executiu complet (8-12 frases) que cobreixi TOTS els grans blocs temàtics tractats",
  "participants": [
    {"name": "Nom", "role": "Rol/càrrec", "context": "Context detallat: què fa, com es relaciona, què opina"}
  ],
  "people_mentioned": [
    {"name": "Nom", "role": "Rol", "context": "Per què es menciona, què es diu d'aquesta persona"}
  ],
  "topics": [
    {
      "title": "Tema",
      "summary": "Explicació DETALLADA de tot el que s'ha dit sobre aquest tema (mínim 5-8 frases). Inclou context, situació actual, problemes, opinions expressades i matisos",
      "subtopics": ["Subtema específic 1", "Subtema específic 2"],
      "key_facts": ["Fet concret i específic 1", "Fet concret i específic 2", "Fet concret 3"]
    }
  ],
  "platforms_and_technologies": [
    {"name": "Nom EXACTE", "category": "ERP/Cloud/Infra/Security/Comms/BI/Mail/etc", "context": "Descripció detallada: per a què serveix, qui l'usa, estat, problemes, plans de futur", "status": "actiu/en migració/pendent/problema/obsolet"}
  ],
  "providers_and_services": [
    {"name": "Nom proveïdor", "service": "Servei concret que dóna", "details": "Relació, contracte, qualitat, experiència, problemes", "sentiment": "positiu/neutre/negatiu/mixt"}
  ],
  "locations_and_sites": [
    {"name": "País/ciutat/planta/workcenter", "context": "Tot el que s'ha dit: mida, equip, infraestructura, problemes, particularitats"}
  ],
  "team_and_organization": [
    {"area": "Àrea o equip", "details": "Composició, responsables, estat, problemes, dinàmiques"}
  ],
  "strategies_and_plans": [
    {"strategy": "Estratègia o pla concret", "rationale": "Per què, motivació, context", "status": "en curs/planificat/ideat/urgent", "timeline": "Termini"}
  ],
  "problems_and_challenges": [
    {"problem": "Problema concret i específic", "impact": "Conseqüència real o potencial", "urgency": "alta/mitjana/baixa", "proposed_solution": "Solució proposada si n'hi ha"}
  ],
  "risks_and_warnings": [
    {"risk": "Risc o advertència mencionada", "context": "Detall i context"}
  ],
  "decisions": ["Decisió concreta 1"],
  "action_items": [
    {"task": "Tasca", "owner": "Responsable", "priority": "alta/mitjana/baixa", "deadline": "Termini"}
  ],
  "numbers_and_data": [
    {"metric": "Dada/xifra/pressupost exacte", "context": "En quin context, què significa"}
  ],
  "contracts_and_costs": [
    {"item": "Contracte/cost/pressupost", "details": "Import, termini, proveïdor, condicions"}
  ],
  "organizational_insights": [
    {"insight": "Observació sobre cultura, política interna, dinàmiques de poder, relacions entre àrees"}
  ],
  "quick_wins": [
    {"action": "Acció ràpida amb impacte", "rationale": "Per què és un quick win"}
  ],
  "quotes": ["Frase TEXTUAL reveladora de la conversa - han de ser frases reals dites, no paràfrasis"]
}

RECORDA:
- El teu objectiu és que aquest document sigui la REFERÈNCIA COMPLETA de tot el que s'ha parlat
- Cada menció compta. Si dubtes entre incloure o no alguna cosa que SÍ es diu, INCLOU-LA
- Però si dubtes de si alguna cosa es diu o t'ho estàs inventant, NO LA POSIS
- ZERO al·lucinacions. Preferible un informe incomplet que un informe amb informació falsa
- Quan no estiguis segur d'un nom propi, posa-ho entre [claudàtors] indicant que pot ser un error de transcripció"""


@app.route("/generate-notes/<job_id>", methods=["POST"])
def generate_notes(job_id):
    if job_id not in jobs or jobs[job_id]["status"] != "done":
        return jsonify({"error": "Job no trobat o no completat"}), 404

    if not OPENAI_API_KEY:
        return jsonify({"error": "OPENAI_API_KEY no configurada"}), 500

    text = jobs[job_id]["result"].get("text", "")
    if not text.strip():
        return jsonify({"error": "Transcripció buida"}), 400

    context = request.json.get("context", "") if request.is_json else ""

    user_msg = ""
    if context:
        user_msg = f"CONTEXT IMPORTANT (proporcionat per l'usuari, és informació fiable):\n{context}\n\nUtilitza aquest context per identificar correctament els participants, els seus càrrecs i el motiu de la conversa.\n\n"
    user_msg += f"TRANSCRIPCIÓ (pot contenir errors de reconeixement de veu):\n\n{text}"

    return _start_ai(job_id, "notes", NOTES_SYSTEM_PROMPT, user_msg)


@app.route("/generate-exec/<job_id>", methods=["POST"])
def generate_exec(job_id):
    if job_id not in jobs or jobs[job_id]["status"] != "done":
        return jsonify({"error": "Job no trobat o no completat"}), 404
    if not OPENAI_API_KEY:
        return jsonify({"error": "OPENAI_API_KEY no configurada"}), 500

    result = jobs[job_id]["result"]
    if not result.get("text", "").strip():
        return jsonify({"error": "Transcripció buida"}), 400

    context = request.json.get("context", "") if request.is_json else ""
    approx = jobs[job_id].get("model") == "gpt-transcribe"
    user_msg = ""
    if context:
        user_msg = f"CONTEXT DE L'USUARI (informació fiable):\n{context}\n\n"
    ends = re.findall(r"--> (\d+):(\d+):(\d+)", result.get("srt", ""))
    if ends:
        h, m, sec = map(int, ends[-1])
        user_msg += f"Durada de l'àudio: {h * 60 + m} min {sec} s\n"
    user_msg += ("Les marques [mm:ss] són APROXIMADES (±1 min); escriu els temps amb '~'.\n"
                 if approx else "Les marques [mm:ss] són exactes.\n")
    user_msg += ("\nTRANSCRIPCIÓ AMB MARQUES DE TEMPS (pot contenir errors de reconeixement de veu):\n\n"
                 + _timed_text(result.get("srt", ""), result.get("text", "")))
    return _start_ai(job_id, "exec", executive.EXEC_SYSTEM_PROMPT, user_msg)


def _timed_text(srt: str, fallback: str) -> str:
    """Transcript as paragraphs of ~45 s, each prefixed with [mm:ss]."""
    out, cur, cur_start = [], [], None
    for block in srt.strip().split("\n\n"):
        lines = block.strip().split("\n")
        if len(lines) < 3 or "-->" not in lines[1]:
            continue
        h, m, rest = lines[1].split(" --> ")[0].split(":")
        t = int(h) * 3600 + int(m) * 60 + float(rest.replace(",", "."))
        if cur_start is None:
            cur_start = t
        cur.append(" ".join(lines[2:]).strip())
        if t - cur_start >= 45:
            out.append(f"[{int(cur_start // 60):02d}:{int(cur_start % 60):02d}] " + " ".join(cur))
            cur, cur_start = [], None
    if cur:
        out.append(f"[{int(cur_start // 60):02d}:{int(cur_start % 60):02d}] " + " ".join(cur))
    return "\n".join(out) if out else fallback


def _start_ai(job_id: str, kind: str, system_prompt: str, user_msg: str):
    # Reasoning models can take several minutes: run in background, client polls
    if jobs[job_id].get(f"{kind}_status") != "running":
        jobs[job_id][f"{kind}_status"] = "running"
        jobs[job_id].pop(f"{kind}_error", None)
        threading.Thread(target=_ai_worker, args=(job_id, kind, system_prompt, user_msg),
                         daemon=True).start()
    return jsonify({"status": "running"})


@app.route("/notes-status/<job_id>")
def notes_status(job_id):
    return _ai_status(job_id, "notes")


@app.route("/exec-status/<job_id>")
def exec_status(job_id):
    return _ai_status(job_id, "exec")


def _ai_status(job_id: str, kind: str):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"status": "error", "error": "Job no trobat"}), 404
    status = job.get(f"{kind}_status") or ("done" if kind in job else "none")
    if status == "done":
        return jsonify({"status": "done", "notes": job[kind]})
    if status == "error":
        return jsonify({"status": "error", "error": job.get(f"{kind}_error", "Error desconegut")})
    return jsonify({"status": status})


def _ai_worker(job_id: str, kind: str, system_prompt: str, user_msg: str):
    """Run a JSON-producing LLM call in background; kind is 'notes' or 'exec'."""
    import httpx

    try:
        resp = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": "gpt-6.1-sol",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                "max_completion_tokens": 65000,
            },
            timeout=NOTES_TIMEOUT,
        )

        if resp.status_code != 200:
            print(f"[{kind.upper()}] API error {resp.status_code}: {resp.text[:500]}")
            raise RuntimeError(f"API error: {resp.text[:300]}")

        rjson = resp.json()
        job_dir = OUTPUT_DIR / job_id
        # Save full API response for debugging
        (job_dir / f"{kind}_api_response.json").write_text(
            json.dumps(rjson, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        msg = rjson["choices"][0]["message"]
        raw = (msg.get("content") or "").strip()
        print(f"[{kind.upper()}] content length={len(raw)}, keys={list(msg.keys())}")

        # Save raw response for debugging
        (job_dir / f"{kind}_raw.txt").write_text(raw, encoding="utf-8")

        # Extract JSON from response - reasoning models may wrap it
        content = raw
        if "```json" in content:
            content = content.split("```json", 1)[1].rsplit("```", 1)[0].strip()
        elif "```" in content:
            content = content.split("```", 1)[1].rsplit("```", 1)[0].strip()

        # Try to find JSON object if still not valid
        if not content.startswith("{"):
            start = content.find("{")
            if start != -1:
                end = content.rfind("}") + 1
                content = content[start:end]

        notes = json.loads(content)
        jobs[job_id][kind] = notes

        job_dir = OUTPUT_DIR / job_id
        (job_dir / f"{kind}.json").write_text(
            json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        jobs[job_id][f"{kind}_status"] = "done"

    except json.JSONDecodeError as e:
        raw = locals().get("raw", "")
        print(f"[{kind.upper()}] JSON parse error: {e}")
        print(f"[{kind.upper()}] Raw content (first 500): {raw[:500] if raw else 'EMPTY'}")
        jobs[job_id][f"{kind}_error"] = "L'IA no ha retornat un format vàlid. Torna a intentar-ho."
        jobs[job_id][f"{kind}_status"] = "error"
    except Exception as e:
        import traceback
        print(f"[{kind.upper()}] Exception: {traceback.format_exc()}")
        jobs[job_id][f"{kind}_error"] = str(e) or type(e).__name__
        jobs[job_id][f"{kind}_status"] = "error"



@app.route("/download/<job_id>/exec-pdf")
def download_exec_pdf(job_id):
    from datetime import datetime

    job = jobs.get(job_id, {})
    if "exec" not in job:
        return "No disponible", 404
    job_dir = OUTPUT_DIR / job_id
    stem = Path(job.get("filename", "audio")).stem
    footer = f"CONFIDENCIAL · {stem} · Informe generat amb Audio2Text a partir d'una transcripció automàtica · " \
             + datetime.now().strftime("%d/%m/%Y")
    executive.render_executive_pdf(job["exec"], str(job_dir / "exec.pdf"), footer)
    return send_from_directory(str(job_dir), "exec.pdf", as_attachment=True,
                               download_name=f"{stem} - Informe executiu CIO.pdf")


@app.route("/download/<job_id>/notes-pdf")
def download_notes_pdf(job_id):
    if job_id not in jobs or "notes" not in jobs.get(job_id, {}):
        return "No disponible", 404
    return _generate_notes_pdf(job_id)


def _generate_notes_pdf(job_id: str):
    from datetime import datetime

    from reportlab.lib import colors
    from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (HRFlowable, ListFlowable, ListItem,
                                    Paragraph, SimpleDocTemplate, Spacer,
                                    Table, TableStyle)

    job = jobs[job_id]
    notes = job["notes"]
    filename = job.get("filename", "audio")
    base_name = Path(filename).stem

    pdf_path = OUTPUT_DIR / job_id / "notes.pdf"

    BRAND = colors.HexColor("#7c3aed")
    BRAND_LIGHT = colors.HexColor("#8b5cf6")
    DARK = colors.HexColor("#1e293b")
    GRAY = colors.HexColor("#64748b")
    LIGHT_BG = colors.HexColor("#f8fafc")
    PURPLE_BG = colors.HexColor("#faf5ff")
    BORDER = colors.HexColor("#e2e8f0")
    GREEN = colors.HexColor("#059669")
    AMBER = colors.HexColor("#d97706")

    page_w, page_h = A4
    margin = 25 * mm

    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(BRAND)
        canvas.rect(0, page_h - 8 * mm, page_w, 8 * mm, fill=True, stroke=False)
        canvas.setFillColor(BRAND_LIGHT)
        canvas.rect(0, page_h - 9 * mm, page_w, 1 * mm, fill=True, stroke=False)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(GRAY)
        canvas.drawString(margin, 12 * mm, f"Audio2Text v{APP_VERSION} — Apunts generats amb IA")
        canvas.drawRightString(page_w - margin, 12 * mm, f"Pàgina {doc.page}")
        canvas.setStrokeColor(BORDER)
        canvas.line(margin, 16 * mm, page_w - margin, 16 * mm)
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        leftMargin=margin, rightMargin=margin,
        topMargin=margin + 10 * mm, bottomMargin=margin,
    )

    styles = getSampleStyleSheet()
    s_title = ParagraphStyle("NTitle", fontSize=20, leading=26, textColor=DARK,
                             spaceAfter=3*mm, fontName="Helvetica-Bold", alignment=TA_LEFT)
    s_subtitle = ParagraphStyle("NSub", fontSize=10, leading=14, textColor=GRAY,
                                spaceAfter=5*mm, fontName="Helvetica")
    s_section = ParagraphStyle("NSec", fontSize=12, leading=17, textColor=BRAND,
                               spaceBefore=6*mm, spaceAfter=3*mm, fontName="Helvetica-Bold")
    s_body = ParagraphStyle("NBody", fontSize=10, leading=15, textColor=DARK,
                            alignment=TA_JUSTIFY, fontName="Helvetica", spaceAfter=2*mm)
    s_bullet = ParagraphStyle("NBullet", fontSize=9.5, leading=14, textColor=DARK,
                              fontName="Helvetica", leftIndent=6*mm, spaceAfter=1.5*mm)
    s_quote = ParagraphStyle("NQuote", fontSize=9, leading=13, textColor=GRAY,
                             fontName="Helvetica-Oblique", leftIndent=8*mm, rightIndent=8*mm,
                             spaceAfter=2*mm, borderWidth=0.5, borderColor=BRAND_LIGHT,
                             borderPadding=4, borderRadius=2)

    def esc(t):
        return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    story = []

    title = notes.get("title", "Apunts de la reunió")
    story.append(Paragraph(esc(title), s_title))
    story.append(Paragraph(
        f"{base_name} &middot; {datetime.now().strftime('%d/%m/%Y %H:%M')}",
        s_subtitle,
    ))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER,
                             spaceAfter=4*mm, spaceBefore=0))

    TABLE_STYLE = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PURPLE_BG),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ])
    col_w = page_w - 2 * margin

    summary = notes.get("summary", "")
    if summary:
        story.append(Paragraph("Resum executiu", s_section))
        story.append(Paragraph(esc(summary), s_body))

    participants = notes.get("participants", [])
    if participants:
        story.append(Paragraph("Participants", s_section))
        for p in participants:
            if isinstance(p, str):
                story.append(Paragraph(f"• {esc(p)}", s_bullet))
            else:
                name = esc(p.get("name", ""))
                role = esc(p.get("role", ""))
                ctx = esc(p.get("context", ""))
                line = f"<b>{name}</b>"
                if role: line += f" — {role}"
                if ctx: line += f". {ctx}"
                story.append(Paragraph(line, s_bullet))

    people = notes.get("people_mentioned", [])
    if people:
        story.append(Paragraph("Persones mencionades", s_section))
        for p in people:
            if isinstance(p, str):
                story.append(Paragraph(f"• {esc(p)}", s_bullet))
            else:
                line = f"<b>{esc(p.get('name', ''))}</b>"
                if p.get("role"): line += f" — {esc(p['role'])}"
                if p.get("context"): line += f". {esc(p['context'])}"
                story.append(Paragraph(line, s_bullet))

    topics = notes.get("topics", [])
    if topics:
        story.append(Paragraph("Temes tractats", s_section))
        for t in topics:
            story.append(Paragraph(f"<b>{esc(t.get('title', ''))}</b>", s_bullet))
            story.append(Paragraph(esc(t.get("summary", "")), s_body))
            for fact in t.get("key_facts", []):
                story.append(Paragraph(f"  • {esc(fact)}", s_bullet))

    platforms = notes.get("platforms_and_technologies", [])
    if platforms:
        story.append(Paragraph("Plataformes i Tecnologies", s_section))
        rows = [[Paragraph("<b>Plataforma</b>", s_bullet),
                 Paragraph("<b>Context</b>", s_bullet),
                 Paragraph("<b>Estat</b>", s_bullet)]]
        for p in platforms:
            rows.append([Paragraph(esc(p.get("name", "")), s_bullet),
                         Paragraph(esc(p.get("context", "")), s_bullet),
                         Paragraph(esc(p.get("status", "")), s_bullet)])
        t = Table(rows, colWidths=[col_w*0.25, col_w*0.55, col_w*0.20])
        t.setStyle(TABLE_STYLE)
        story.append(t)

    providers = notes.get("providers_and_services", [])
    if providers:
        story.append(Paragraph("Proveïdors i Serveis", s_section))
        rows = [[Paragraph("<b>Proveïdor</b>", s_bullet),
                 Paragraph("<b>Detalls</b>", s_bullet),
                 Paragraph("<b>Valoració</b>", s_bullet)]]
        for p in providers:
            rows.append([Paragraph(esc(p.get("name", "")), s_bullet),
                         Paragraph(esc(p.get("details", "")), s_bullet),
                         Paragraph(esc(p.get("sentiment", "")), s_bullet)])
        t = Table(rows, colWidths=[col_w*0.25, col_w*0.55, col_w*0.20])
        t.setStyle(TABLE_STYLE)
        story.append(t)

    locations = notes.get("locations_and_sites", [])
    if locations:
        story.append(Paragraph("Ubicacions i Sites", s_section))
        for loc in locations:
            story.append(Paragraph(
                f"<b>{esc(loc.get('name', ''))}</b>: {esc(loc.get('context', ''))}",
                s_bullet))

    teams = notes.get("team_and_organization", [])
    if teams:
        story.append(Paragraph("Equip i Organització", s_section))
        for t in teams:
            story.append(Paragraph(
                f"<b>{esc(t.get('area', ''))}</b>: {esc(t.get('details', ''))}",
                s_bullet))

    strategies = notes.get("strategies_and_plans", [])
    if strategies:
        story.append(Paragraph("Estratègies i Plans", s_section))
        for s in strategies:
            line = f"<b>{esc(s.get('strategy', ''))}</b>: {esc(s.get('rationale', ''))}"
            tl = s.get("timeline", "")
            if tl: line += f" — <i>{esc(tl)}</i>"
            story.append(Paragraph(line, s_bullet))

    problems = notes.get("problems_and_challenges", [])
    if problems:
        story.append(Paragraph("Problemes i Reptes", s_section))
        for p in problems:
            line = f"<b>{esc(p.get('problem', ''))}</b> — {esc(p.get('impact', ''))}"
            sol = p.get("proposed_solution", "")
            if sol: line += f". Solució: {esc(sol)}"
            story.append(Paragraph(line, s_bullet))

    risks = notes.get("risks_and_warnings", [])
    if risks:
        story.append(Paragraph("Riscos i Advertències", s_section))
        for r in risks:
            story.append(Paragraph(
                f"<b>{esc(r.get('risk', ''))}</b>: {esc(r.get('context', ''))}",
                s_bullet))

    numbers = notes.get("numbers_and_data", [])
    if numbers:
        story.append(Paragraph("Dades i Xifres", s_section))
        for d in numbers:
            story.append(Paragraph(
                f"<b>{esc(d.get('metric', ''))}</b>: {esc(d.get('context', ''))}",
                s_bullet))

    contracts = notes.get("contracts_and_costs", [])
    if contracts:
        story.append(Paragraph("Contractes i Costos", s_section))
        for c in contracts:
            story.append(Paragraph(
                f"<b>{esc(c.get('item', ''))}</b>: {esc(c.get('details', ''))}",
                s_bullet))

    decisions = notes.get("decisions", [])
    if decisions:
        story.append(Paragraph("Decisions", s_section))
        for d in decisions:
            story.append(Paragraph(f"• {esc(d)}", s_bullet))

    actions = notes.get("action_items", [])
    if actions:
        story.append(Paragraph("Accions pendents", s_section))
        rows = [[Paragraph("<b>Tasca</b>", s_bullet),
                 Paragraph("<b>Responsable</b>", s_bullet),
                 Paragraph("<b>Prioritat</b>", s_bullet),
                 Paragraph("<b>Termini</b>", s_bullet)]]
        for a in actions:
            rows.append([Paragraph(esc(a.get("task", "")), s_bullet),
                         Paragraph(esc(a.get("owner", "-")), s_bullet),
                         Paragraph(esc(a.get("priority", "-")), s_bullet),
                         Paragraph(esc(a.get("deadline", "-")), s_bullet)])
        t = Table(rows, colWidths=[col_w*0.45, col_w*0.22, col_w*0.15, col_w*0.18])
        t.setStyle(TABLE_STYLE)
        story.append(t)

    insights = notes.get("organizational_insights", [])
    if insights:
        story.append(Paragraph("Observacions organitzatives", s_section))
        for i in insights:
            txt = i.get("insight", "") if isinstance(i, dict) else str(i)
            story.append(Paragraph(f"• {esc(txt)}", s_bullet))

    quick_wins = notes.get("quick_wins", [])
    if quick_wins:
        story.append(Paragraph("Quick Wins", s_section))
        for q in quick_wins:
            story.append(Paragraph(
                f"<b>{esc(q.get('action', ''))}</b>: {esc(q.get('rationale', ''))}",
                s_bullet))

    quotes = notes.get("quotes", [])
    if quotes:
        story.append(Paragraph("Cites rellevants", s_section))
        for q in quotes:
            story.append(Paragraph(f"«{esc(q)}»", s_quote))

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)

    download_name = f"{base_name}_apunts.pdf"
    return send_file(str(pdf_path), as_attachment=True, download_name=download_name)


@app.route("/history")
def history():
    from datetime import datetime

    items = []
    def _job_date(item):
        try:
            return datetime.strptime(item[1].get("date", ""), "%d/%m/%Y %H:%M")
        except ValueError:
            return datetime.min

    # Newest first (job ids are random, so sort by date)
    for job_id, job in sorted(jobs.items(), key=_job_date, reverse=True):
        if job["status"] == "done":
            items.append({
                "id": job_id,
                "filename": job.get("filename", ""),
                "model": job.get("model", ""),
                "language": job["result"].get("language", ""),
                "chars": job["result"].get("chars", 0),
                "duration": job["result"].get("duration", ""),
                "backend": job["result"].get("backend", ""),
                "date": job.get("date", ""),
            })
    return jsonify(items)


if __name__ == "__main__":
    print(f"Audio2Text v{APP_VERSION}")
    print(f"Backend: {'Whisper local' if HAS_LOCAL_WHISPER else 'OpenAI API' if OPENAI_API_KEY else 'CAP (configura Whisper o OPENAI_API_KEY)'}")
    app.run(host="0.0.0.0", port=5071, debug=False)
