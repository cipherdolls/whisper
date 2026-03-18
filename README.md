# Whisper ASR Web Service (RTX 5090 / Blackwell)

OpenAI Whisper-compatible ASR web service built for NVIDIA RTX 5090 (sm_120 / Blackwell).

Uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with CTranslate2 on CUDA 12.8 — no PyTorch required.

## Quick Start

```bash
docker compose up --build
```

Swagger UI: http://localhost:9000/docs

## API

### `POST /asr`

Transcribe or translate audio. Compatible with [onerahmet/whisper-asr-webservice](https://github.com/ahmetoner/whisper-asr-webservice) API.

Query params:
- `task` — `transcribe` (default) or `translate`
- `language` — language code (e.g. `en`, `de`, `fr`), auto-detected if omitted
- `output` — `json` (default), `txt`, `vtt`, `srt`, `tsv`
- `word_timestamps` — `true` / `false`
- `encode` — `true` / `false`
- `vad_filter` — `true` (default) / `false`

Body: `audio_file` (multipart form upload)

```bash
curl -X POST "http://localhost:9000/asr?task=transcribe&output=json&language=en&word_timestamps=true" \
  -F "audio_file=@recording.wav"
```

### `POST /detect-language`

Detect the language of an audio file.

### `GET /health`

Returns model status, device, and compute type.

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `ASR_MODEL` | `large-v3-turbo` | Whisper model size |
| `ASR_MODEL_PATH` | `/root/.cache/huggingface` | Model cache directory |
| `ASR_DEVICE` | `cuda` | `cuda` or `cpu` |
| `COMPUTE_TYPE` | `float16` | `float16` recommended (INT8 broken on sm_120) |
| `MODEL_IDLE_TIMEOUT` | `0` | Seconds before unloading idle model (0 = never) |

## Why Custom?

No pre-built Whisper Docker image supports RTX 5090 / Blackwell (sm_120). This image uses:
- `nvidia/cuda:12.8.0-cudnn9-runtime-ubuntu22.04` for sm_120 support
- `float16` compute type (INT8 tensor cores on Blackwell require padding fixes not yet in stable CTranslate2)
- No PyTorch dependency — faster-whisper uses CTranslate2 directly
