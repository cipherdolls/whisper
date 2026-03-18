import logging
import os
import tempfile
from contextlib import asynccontextmanager
from enum import Enum

from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import PlainTextResponse, RedirectResponse

from app.config import settings
from app.model_manager import get_model, get_model_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Task(str, Enum):
    transcribe = "transcribe"
    translate = "translate"


class OutputFormat(str, Enum):
    json = "json"
    txt = "txt"
    vtt = "vtt"
    srt = "srt"
    tsv = "tsv"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Pre-warming model on startup...")
    get_model()
    logger.info("Model ready")
    yield


app = FastAPI(
    title="Whisper ASR Web Service",
    description="OpenAI Whisper ASR for RTX 5090 (Blackwell sm_120)",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health():
    return get_model_info()


@app.post("/asr")
async def asr(
    audio_file: UploadFile = File(...),
    encode: bool = Query(True, description="Encode audio with ffmpeg"),
    task: Task = Query(Task.transcribe),
    language: str | None = Query(None, description="Language code (e.g. en, de, fr)"),
    output: OutputFormat = Query(OutputFormat.json),
    word_timestamps: bool = Query(False),
    vad_filter: bool = Query(True),
):
    suffix = os.path.splitext(audio_file.filename or ".wav")[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(await audio_file.read())
        tmp.flush()

        model = get_model()
        segments_gen, info = model.transcribe(
            tmp.name,
            task=task.value,
            language=language,
            word_timestamps=word_timestamps,
            vad_filter=vad_filter,
        )

        segments = []
        full_text_parts = []
        for i, seg in enumerate(segments_gen):
            words = []
            if seg.words:
                words = [
                    {"start": w.start, "end": w.end, "word": w.word, "probability": w.probability}
                    for w in seg.words
                ]
            segment_dict = {
                "id": i,
                "seek": seg.seek,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "tokens": seg.tokens,
                "temperature": seg.temperature,
                "avg_logprob": seg.avg_logprob,
                "compression_ratio": seg.compression_ratio,
                "no_speech_prob": seg.no_speech_prob,
                "words": words,
            }
            segments.append(segment_dict)
            full_text_parts.append(seg.text)

    detected_language = info.language

    if output == OutputFormat.json:
        return {
            "text": "".join(full_text_parts).strip(),
            "segments": segments,
            "language": detected_language,
        }

    if output == OutputFormat.txt:
        return PlainTextResponse("".join(full_text_parts).strip())

    if output == OutputFormat.vtt:
        return PlainTextResponse(_format_vtt(segments), media_type="text/vtt")

    if output == OutputFormat.srt:
        return PlainTextResponse(_format_srt(segments))

    if output == OutputFormat.tsv:
        return PlainTextResponse(_format_tsv(segments), media_type="text/tab-separated-values")


@app.post("/detect-language")
async def detect_language(audio_file: UploadFile = File(...)):
    suffix = os.path.splitext(audio_file.filename or ".wav")[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(await audio_file.read())
        tmp.flush()

        model = get_model()
        _, info = model.transcribe(tmp.name)

    return {
        "language_code": info.language,
        "language_probability": info.language_probability,
    }


def _format_timestamp(seconds: float, use_comma: bool = False) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    sep = "," if use_comma else "."
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _format_vtt(segments: list[dict]) -> str:
    lines = ["WEBVTT", ""]
    for seg in segments:
        start = _format_timestamp(seg["start"])
        end = _format_timestamp(seg["end"])
        lines.append(f"{start} --> {end}")
        lines.append(seg["text"].strip())
        lines.append("")
    return "\n".join(lines)


def _format_srt(segments: list[dict]) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        start = _format_timestamp(seg["start"], use_comma=True)
        end = _format_timestamp(seg["end"], use_comma=True)
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(seg["text"].strip())
        lines.append("")
    return "\n".join(lines)


def _format_tsv(segments: list[dict]) -> str:
    lines = ["start\tend\ttext"]
    for seg in segments:
        start = int(seg["start"] * 1000)
        end = int(seg["end"] * 1000)
        lines.append(f"{start}\t{end}\t{seg['text'].strip()}")
    return "\n".join(lines)


def start():
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port)
