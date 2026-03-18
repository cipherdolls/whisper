import logging
import threading
import time

from faster_whisper import WhisperModel

from app.config import settings

logger = logging.getLogger(__name__)

_model: WhisperModel | None = None
_lock = threading.Lock()
_last_used: float = 0.0
_timer: threading.Timer | None = None


def get_model() -> WhisperModel:
    global _model, _last_used, _timer

    with _lock:
        if _model is None:
            logger.info(
                "Loading model %s on %s (compute_type=%s)",
                settings.asr_model,
                settings.asr_device,
                settings.compute_type,
            )
            _model = WhisperModel(
                settings.asr_model,
                device=settings.asr_device,
                compute_type=settings.compute_type,
                download_root=settings.asr_model_path,
            )
            logger.info("Model loaded successfully")

        _last_used = time.monotonic()
        _schedule_idle_check()
        return _model


def unload_model() -> None:
    global _model, _timer

    with _lock:
        if _model is not None:
            logger.info("Unloading model due to idle timeout")
            del _model
            _model = None
        if _timer is not None:
            _timer.cancel()
            _timer = None


def _schedule_idle_check() -> None:
    global _timer

    if settings.model_idle_timeout <= 0:
        return

    if _timer is not None:
        _timer.cancel()

    def _check_idle():
        elapsed = time.monotonic() - _last_used
        if elapsed >= settings.model_idle_timeout:
            unload_model()

    _timer = threading.Timer(settings.model_idle_timeout, _check_idle)
    _timer.daemon = True
    _timer.start()


def get_model_info() -> dict:
    return {
        "model": settings.asr_model,
        "device": settings.asr_device,
        "compute_type": settings.compute_type,
        "loaded": _model is not None,
    }
