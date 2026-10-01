import os
import tempfile
import logging

logger = logging.getLogger(__name__)

# Выбор бэкенда: faster-whisper (быстрее) или openai-whisper (проще)
# pip install faster-whisper  ИЛИ  pip install openai-whisper

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")  # tiny | base | small | medium | large
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")           # cpu | cuda
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "ru")        # ru | en | None (авто)

_model = None


def get_whisper_model():
    global _model
    if _model is not None:
        return _model

    try:
        from faster_whisper import WhisperModel
        logger.info(f"Loading faster-whisper model: {WHISPER_MODEL_SIZE} on {WHISPER_DEVICE}")
        _model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type="int8")
        logger.info("faster-whisper loaded successfully")
        return _model
    except ImportError:
        pass

    try:
        import whisper
        logger.info(f"Loading openai-whisper model: {WHISPER_MODEL_SIZE}")
        _model = whisper.load_model(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE)
        logger.info("openai-whisper loaded successfully")
        return _model
    except ImportError:
        raise RuntimeError(
            "Whisper не установлен. Выполни:\n"
            "  pip install faster-whisper\n"
            "  ИЛИ pip install openai-whisper"
        )


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    """Транскрибирует аудио через Whisper. Возвращает распознанный текст."""
    model = get_whisper_model()

    ext = os.path.splitext(filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        # faster-whisper
        try:
            from faster_whisper import WhisperModel
            if isinstance(model, WhisperModel):
                segments, info = model.transcribe(
                    tmp_path,
                    language=WHISPER_LANGUAGE or None,
                    beam_size=5,
                )
                text = " ".join(seg.text for seg in segments).strip()
                logger.info(f"Transcribed ({info.language}): {text[:100]}")
                return text
        except ImportError:
            pass

        # openai-whisper
        import whisper_utils
        result = model.transcribe(
            tmp_path,
            language=WHISPER_LANGUAGE or None,
        )
        text = result["text"].strip()
        logger.info(f"Transcribed: {text[:100]}")
        return text

    finally:
        os.unlink(tmp_path)
