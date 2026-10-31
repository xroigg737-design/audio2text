#!/usr/bin/env python3
"""CLI entry point for audio2text — transcribe audio files without the Flask server."""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# Re-use detection and utility functions from app.py without starting Flask.
# Importing app triggers Flask object creation and _load_previous_jobs(), both harmless.
from app import (
    ALLOWED_EXTENSIONS,
    COMPUTE_TYPE,
    DEVICE,
    FASTER_WHISPER_NAMES,
    HAS_DIARIZATION,
    HAS_LOCAL_WHISPER,
    OPENAI_API_KEY,
    _format_srt_time,
    _generate_srt,
    _generate_text_with_speakers,
    _get_audio_duration,
    _get_model,
    _preprocess_audio,
    _run_diarization,
)

# Stub _emit so diarization progress messages are printed to stderr
_original_emit = None
try:
    import app as _app_module
    _original_emit = _app_module._emit

    def _cli_emit(job_id, event, data):
        msg = data.get("msg", "")
        if msg:
            print(f"  [{event}] {msg}", file=sys.stderr)

    _app_module._emit = _cli_emit
    _app_module._start_progress = lambda *a, **k: None
except Exception:
    pass


def _build_segments_v1(segments: list[dict], job_id: str, audio_file: str,
                       model_name: str, language: str, duration: float,
                       num_speakers: int) -> dict:
    """Build the v1 JSON segment contract."""
    speakers = sorted(set(s.get("speaker", "") for s in segments if s.get("speaker")))
    participants = [
        {"speaker_id": spk, "name": None, "role": None}
        for spk in speakers
    ] if speakers else []

    segs = []
    for i, seg in enumerate(segments, 1):
        entry = {
            "id": i,
            "speaker_id": seg.get("speaker") or None,
            "start_ms": int(seg["start"] * 1000),
            "end_ms": int(seg["end"] * 1000),
            "text": seg["text"].strip(),
        }
        segs.append(entry)

    return {
        "schema_version": "1",
        "meeting_id": job_id,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "participants": participants,
        "segments": segs,
        "audio_ref": Path(audio_file).name,
        "quality": {
            "model": model_name,
            "language": language,
            "duration": duration,
            "num_speakers": num_speakers,
            "device": DEVICE,
        },
    }


def transcribe_file(audio_path: str, output_dir: str, model_name: str = "large",
                     language: str = "auto", diarize: bool = False) -> dict:
    """Transcribe a single audio file and write all outputs to output_dir."""
    audio_path = str(Path(audio_path).resolve())
    ext = Path(audio_path).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Format no suportat: {ext}. Permesos: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Fitxer no trobat: {audio_path}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    job_id = uuid.uuid4().hex[:12]
    print(f"Job {job_id}: transcrivint {Path(audio_path).name}", file=sys.stderr)

    audio_duration = _get_audio_duration(audio_path)
    if audio_duration > 0:
        mins = int(audio_duration // 60)
        secs = int(audio_duration % 60)
        print(f"  Durada: {mins}m {secs}s", file=sys.stderr)

    t0 = time.time()

    if not HAS_LOCAL_WHISPER:
        raise RuntimeError(
            "CLI requereix faster-whisper instal·lat (backend local). "
            "Per usar l'API d'OpenAI, fes servir la interfície web."
        )

    fw_name = FASTER_WHISPER_NAMES.get(model_name, model_name)
    device_label = "GPU" if DEVICE == "cuda" else "CPU"
    print(f"  Model: {fw_name} ({device_label})", file=sys.stderr)

    # Preprocess
    print("  Preprocessant àudio...", file=sys.stderr)
    processed_path = _preprocess_audio(job_id, audio_path, audio_duration)

    # Transcribe
    print("  Transcrivint...", file=sys.stderr)
    model = _get_model(model_name)
    transcribe_opts = {
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

    segments_iter, info = model.transcribe(processed_path, **transcribe_opts)
    detected_lang = info.language or "unknown"
    segments = [{"start": seg.start, "end": seg.end, "text": seg.text} for seg in segments_iter]
    backend = f"local-{device_label}"

    print(f"  {len(segments)} segments, idioma detectat: {detected_lang}", file=sys.stderr)

    # Diarize
    num_speakers = 0
    if diarize and HAS_DIARIZATION and segments:
        print("  Diaritzant...", file=sys.stderr)
        try:
            diar_audio = processed_path if os.path.exists(processed_path) else audio_path
            segments = _run_diarization(job_id, diar_audio, audio_duration, segments)
            unique_speakers = set(s.get("speaker", "") for s in segments if s.get("speaker"))
            num_speakers = len(unique_speakers)
            print(f"  {num_speakers} parlants detectats", file=sys.stderr)
        except Exception as e:
            print(f"  Diarització fallida: {e}", file=sys.stderr)

    # Generate outputs
    has_speakers = any(seg.get("speaker") for seg in segments)
    if has_speakers:
        full_text = _generate_text_with_speakers(segments)
    else:
        full_text = " ".join(seg["text"].strip() for seg in segments)
    srt_content = _generate_srt(segments)

    elapsed = time.time() - t0

    # Write transcript.txt
    (out / "transcript.txt").write_text(full_text, encoding="utf-8")

    # Write transcript.srt
    (out / "transcript.srt").write_text(srt_content, encoding="utf-8")

    # Write meta.json
    meta = {
        "model": model_name,
        "language": language,
        "detected_language": detected_lang,
        "duration": f"{int(elapsed // 60)}m {int(elapsed % 60)}s",
        "backend": backend,
        "diarization": diarize and has_speakers,
        "num_speakers": num_speakers,
        "date": datetime.now().strftime("%d/%m/%Y %H:%M"),
    }
    (out / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Write segments.json (v1 contract)
    v1 = _build_segments_v1(
        segments, job_id, audio_path, model_name,
        detected_lang, audio_duration, num_speakers,
    )
    (out / "segments.json").write_text(
        json.dumps(v1, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
    print(f"  Completat en {elapsed_str}", file=sys.stderr)
    print(f"  Sortida: {out}", file=sys.stderr)

    return {
        "job_id": job_id,
        "output_dir": str(out),
        "segments": len(segments),
        "detected_language": detected_lang,
        "duration": elapsed_str,
        "num_speakers": num_speakers,
        "files": ["transcript.txt", "transcript.srt", "meta.json", "segments.json"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Audio2Text CLI — transcriu àudios sense servidor Flask",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Exemple:\n  python cli.py reunio.mp3 --output-dir ./sortida --diarize --language ca",
    )
    parser.add_argument("audio_file", help="Fitxer d'àudio a transcriure")
    parser.add_argument("--output-dir", "-o", required=True,
                        help="Directori de sortida")
    parser.add_argument("--language", "-l", default="auto",
                        help="Idioma (ca, es, en, auto). Per defecte: auto")
    parser.add_argument("--diarize", "-d", action="store_true",
                        help="Activar diarització de parlants")
    parser.add_argument("--model", "-m", default="large",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Model Whisper (per defecte: large)")
    parser.add_argument("--json", action="store_true",
                        help="Imprimeix resultat en JSON a stdout")

    args = parser.parse_args()

    try:
        result = transcribe_file(
            audio_path=args.audio_file,
            output_dir=args.output_dir,
            model_name=args.model,
            language=args.language,
            diarize=args.diarize,
        )
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"\nTranscripció completada: {result['segments']} segments, "
                  f"idioma: {result['detected_language']}, "
                  f"parlants: {result['num_speakers']}, "
                  f"temps: {result['duration']}")
            print(f"Fitxers: {', '.join(result['files'])}")
            print(f"Directori: {result['output_dir']}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
