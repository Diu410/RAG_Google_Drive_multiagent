# Live mode

Live mode is a continuous voice conversation. While it is enabled, the microphone stays
active, cumulative audio is transcribed approximately every 1.3 seconds, and the partial
phrase is shown in the conversation. About 1.1 seconds of silence completes the phrase
and sends it automatically. Conversational and library tokens are rendered as they are
generated and completed phrases are passed to streaming TTS. Workspace answers are streamed
after their source review, without speaking intermediate drafts. Listening resumes after playback finishes.

## Use

1. Start the desktop application normally.
2. Enable `LIVE` next to the microphone.
3. Start speaking. The current transcription appears as a temporary Live message.
4. Pause briefly to send the phrase automatically. Answers are intentionally limited to
   one or two conversational phrases.
5. Use `Stop` to cancel the current generation, or `Завершить Live-разговор` to stop the
   microphone and all queued audio. A 60-second per-phrase safety limit remains active.

The UI shows the active voice backend. If CosyVoice is unavailable, it reports the switch
to the local operating-system voice.

## CosyVoice

The official CosyVoice repository recommends a dedicated Python 3.10 environment. Clone
CosyVoice with its submodules, install its official requirements, and download a model,
for example `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`. Then configure `api/.env`:

```dotenv
COSYVOICE_ROOT=C:/AI/CosyVoice
COSYVOICE_MODEL_DIR=C:/AI/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B
COSYVOICE_PROMPT_AUDIO=C:/AI/voices/assistant.wav
COSYVOICE_PROMPT_TEXT=Точный текст, произнесённый в образце голоса.
COSYVOICE_SPEED=1.0
```

The prompt audio and its transcript must match. An SFT model can instead use
`COSYVOICE_SPEAKER`. Restart the backend after changing these settings.

Relevant endpoints:

- `POST /assistant/live` — NDJSON stages, final-answer tokens and result;
- `POST /voice/stream` — NDJSON PCM audio chunks;
- `POST /voice/synthesize` — complete WAV compatibility endpoint;
- `GET /voice/status` — active TTS provider and fallback state.
