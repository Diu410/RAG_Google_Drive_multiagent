"""Optional local CosyVoice adapter used by the voice-first interface."""
from __future__ import annotations

import io
import re
import sys
import threading
import wave
from pathlib import Path

from settings import (COSYVOICE_MODEL_DIR, COSYVOICE_PROMPT_AUDIO, COSYVOICE_PROMPT_TEXT,
                      COSYVOICE_ROOT, COSYVOICE_SPEAKER, COSYVOICE_SPEED)


class TTSUnavailable(RuntimeError):
    pass


def speech_text(text: str) -> str:
    text = re.sub(r"```[\s\S]*?```", " фрагмент кода ", text)
    text = re.sub(r"[`*_#>|]", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


class CosyVoiceProvider:
    def __init__(self):
        self._model = None
        self._error = None
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def status(self) -> dict:
        configured = bool(COSYVOICE_ROOT and COSYVOICE_MODEL_DIR)
        return {
            "provider": "cosyvoice",
            "configured": configured,
            "ready": self._model is not None,
            "fallback": "browser-speech-synthesis" if not self._model else None,
            "error": self._error,
        }

    def _load(self):
        if self._model is not None:
            return self._model
        if not COSYVOICE_ROOT or not COSYVOICE_MODEL_DIR:
            raise TTSUnavailable("CosyVoice не настроен")
        root, model_dir = Path(COSYVOICE_ROOT).resolve(), Path(COSYVOICE_MODEL_DIR).resolve()
        if not root.is_dir() or not model_dir.is_dir():
            raise TTSUnavailable("Путь CosyVoice или каталог модели не найден")
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                if str(root) not in sys.path:
                    sys.path.insert(0, str(root))
                if str(root / "third_party" / "Matcha-TTS") not in sys.path:
                    sys.path.insert(0, str(root / "third_party" / "Matcha-TTS"))
                from cosyvoice.cli.cosyvoice import AutoModel
                self._model = AutoModel(model_dir=str(model_dir))
            except Exception as exc:
                self._error = f"{type(exc).__name__}: {exc}"
                raise TTSUnavailable("Не удалось загрузить CosyVoice") from exc
        return self._model

    @staticmethod
    def _wav(chunks, sample_rate: int) -> bytes:
        import numpy as np
        arrays = []
        for item in chunks:
            tensor = item["tts_speech"].detach().cpu().float().reshape(-1)
            arrays.append(tensor.numpy())
        if not arrays:
            raise TTSUnavailable("CosyVoice не вернул аудио")
        samples = np.clip(np.concatenate(arrays), -1, 1)
        pcm = (samples * 32767).astype("<i2").tobytes()
        output = io.BytesIO()
        with wave.open(output, "wb") as target:
            target.setnchannels(1)
            target.setsampwidth(2)
            target.setframerate(sample_rate)
            target.writeframes(pcm)
        return output.getvalue()

    def synthesize(self, text: str) -> bytes:
        chunks = list(self.stream_pcm(text))
        if not chunks:
            raise TTSUnavailable("CosyVoice не вернул аудио")
        return self._wav(({"tts_speech": chunk} for _, chunk in chunks), chunks[0][0])

    def stream_pcm(self, text: str):
        """Yield sample rate and tensors as soon as CosyVoice produces them."""
        model = self._load()
        text = speech_text(text)
        if not text:
            raise ValueError("Нет текста для озвучивания")
        with self._inference_lock:
            if COSYVOICE_PROMPT_AUDIO and COSYVOICE_PROMPT_TEXT:
                chunks = model.inference_zero_shot(
                    text, COSYVOICE_PROMPT_TEXT, COSYVOICE_PROMPT_AUDIO,
                    stream=True, speed=COSYVOICE_SPEED,
                )
            elif COSYVOICE_SPEAKER:
                chunks = model.inference_sft(
                    text, COSYVOICE_SPEAKER, stream=True, speed=COSYVOICE_SPEED,
                )
            else:
                raise TTSUnavailable(
                    "Укажите COSYVOICE_PROMPT_AUDIO и COSYVOICE_PROMPT_TEXT либо COSYVOICE_SPEAKER"
                )
            for item in chunks:
                yield model.sample_rate, item["tts_speech"]


tts_provider = CosyVoiceProvider()
