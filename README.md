<div align="center">

<img src="docs/logo.png" alt="Agentic" width="288">

**Agentic XTTS: a fast Coqui XTTS-v2 text-to-speech server with voice cloning and streaming. It powers Agentic's voices.**

[![License: GPL v3](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)
![Python 3.10–3.13](https://img.shields.io/badge/python-3.10%E2%80%933.13-3776AB)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688)

[Setup](#setup) · [Configuration](#configuration) · [API](#api) · [Performance](#performance) · [Use with Agentic](#use-with-agentic) · [Agentic server](https://github.com/smurflabs/Agentic)

</div>

---

## What is it?

Agentic XTTS serves [Coqui XTTS-v2](https://huggingface.co/coqui/XTTS-v2) text to speech over HTTP with [FastAPI](https://fastapi.tiangolo.com). It is built to be called by LLM agents and apps. The [Agentic server](https://github.com/smurflabs/Agentic) uses it to read answers aloud, to speak in Dictation mode, and for its **Audio** tab.

- **Voice cloning** from one or more reference clips. Each voice's fingerprint is computed once at upload and reused for every request.
- **Streaming** (`/tts/stream`): the first audio arrives in about 0.5 s, as WAV or raw PCM.
- **Every XTTS generation setting:** temperature, top-k, top-p, repetition and length penalty, speed, and clone-tuning options.
- **17 languages** and 58 built-in speakers.
- **Requests in parallel:** one model copy per worker (`XTTS_WORKERS`).
- **A CUDA-graph decoder** ([`fast_gpt.py`](fast_gpt.py)) that makes generation about 4× faster when the CPU is the bottleneck.
- **API-key auth,** and uploaded voices that expire automatically.
- **[`LLM_PROMPT.md`](LLM_PROMPT.md):** a ready-made tool prompt that teaches an LLM to use every endpoint.

## Performance

Measured on a GTX 1660 Ti (6 GB) with a Xeon E5-2670 v2 (2013). RTF is the seconds of compute per second of audio; lower is better.

| | Stock XTTS | `fast_gpt.py` |
|---|---|---|
| `/tts` RTF | 1.36 | **0.34** |
| `/tts/stream` RTF | 1.44 | **0.43** |
| Stream, time to first audio | 1.4 s | **0.5 s** |
| GPU utilization | 27% | ~90% |

XTTS generates about 21 audio tokens per second of speech, one at a time. With Hugging Face's `generate()`, each token costs hundreds of small kernel launches plus Python overhead, so on a slow CPU the GPU sits idle waiting for them.

`fast_gpt.py` removes that overhead:
- **Fixed shapes:** it allocates GPT-2's attention memory (the KV cache) once at full length, so every decode step has identical tensor shapes.
- **One graph per step:** it records the whole step as a single CUDA graph (embeddings, 30 transformer layers, repetition penalty, temperature, top-k, top-p and sampling). Each token is then one graph replay.
- **Fewer stop checks:** it checks for the stop token every 8 steps instead of every step.
- **Small footprint:** it replaces only `gpt.generate` and `gpt.get_generator`, so XTTS's own `inference()` and `inference_stream()` run unchanged on top of it.

The output matches stock XTTS:
- **Greedy decoding** produces the same tokens, checked on the test sentences used during development.
- **Sampling** produces the same probability distribution as Hugging Face's logits processors; see [`tests/test_sampling.py`](tests/test_sampling.py).

Set `XTTS_FAST=0` to fall back to stock generation.

## Setup

You need an NVIDIA GPU with 3 GB or more of VRAM per worker, and Python 3.10–3.13 (tested on 3.12).

```powershell
git clone https://github.com/smurflabs/Agentic_xtts
cd Agentic_xtts
py -3.12 -m venv venv
venv\Scripts\python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
venv\Scripts\python -m pip install -r requirements.txt
```

**FFmpeg:** torchcodec needs FFmpeg's *shared* DLLs (versions 4–8); a static `ffmpeg.exe` is not enough.
- **Windows:** download a `win64-gpl-shared` build, for example `ffmpeg-n8.1-latest-win64-gpl-shared-8.1.zip` from [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds/releases). Extract it so that `ffmpeg\bin\avcodec-*.dll` sits next to `server.py`.
- **Linux:** install FFmpeg from your package manager.

**Configure and run:**

```powershell
copy start_server.example.cmd start_server.cmd   # then edit it: set XTTS_API_KEY, the host and the workers
start_server.cmd
```

The first run downloads the model (about 1.8 GB). `server.py` sets `COQUI_TOS_AGREED=1`, which accepts the model license on your behalf, so read the [license notes](#license-notes) first.

Interactive API docs are at `http://<host>:8020/docs`.

**At boot:** register a scheduled task that runs `start_server.cmd` as SYSTEM. In the launcher, uncomment the `TTS_HOME` and `HF_HOME` lines so SYSTEM finds the model.

## Configuration

| Variable | Default | What it does |
|---|---|---|
| `XTTS_HOST` | `127.0.0.1` | `0.0.0.0` listens on the network. Set an API key and a firewall rule first |
| `XTTS_PORT` | `8020` | |
| `XTTS_WORKERS` | `2` | Model copies, which is the number of requests handled in parallel. About 2.3 GB of VRAM each |
| `XTTS_API_KEY` | none | When set, every request needs the header `X-API-Key: <key>` |
| `XTTS_VOICE_TTL_DAYS` | `30` | Uploaded voices are deleted this many days after upload. `0` keeps them forever |
| `XTTS_FAST` | `1` | `1` uses the CUDA-graph decoder; `0` uses stock Hugging Face generation |

## API

| Endpoint | What it does |
|---|---|
| `POST /tts` | JSON body → a complete WAV (24 kHz mono, 16-bit) |
| `POST /tts/stream?format=wav\|pcm` | The same body → audio streamed as it is generated |
| `POST /voices` | multipart: `name`, one or more `files`, plus optional clone tuning → clones a voice |
| `GET /voices/{name}`, `DELETE /voices/{name}` | Show a voice's metadata and expiry, or delete it |
| `GET /speakers`, `GET /languages` | The built-in speakers and cloned voices, and the supported languages |

**`/tts` and `/tts/stream` body fields:**
- `text` (required)
- `language`
- `speaker` (a built-in speaker) or `voice` (a cloned voice)
- `speed`
- generation settings: `temperature`, `top_k`, `top_p`, `repetition_penalty`, `length_penalty`, `enable_text_splitting`
- streaming only: `stream_chunk_size`

**`/voices` tuning fields:** `gpt_cond_len`, `gpt_cond_chunk_len`, `max_ref_length`, `trim_silence_db`, `normalize`.

[`LLM_PROMPT.md`](LLM_PROMPT.md) lists every field with its range, when to use it, and examples.

```bash
curl -X POST http://localhost:8020/tts -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"text":"Hello there!","speaker":"Damien Black"}' -o hello.wav

curl -X POST http://localhost:8020/voices -H "X-API-Key: $KEY" \
  -F name=narrator -F files=@clip1.wav -F files=@clip2.wav -F trim_silence_db=30
```

## Use with Agentic

1. Run this server with an `XTTS_API_KEY`. If Agentic runs on another machine, also set `XTTS_HOST=0.0.0.0` and a firewall rule.
2. In Agentic's `appsettings.json`, set:

   ```json
   "TextToSpeech": { "BaseUrl": "http://<host>:8020", "MaxConcurrency": 2 }
   ```

   Set `MaxConcurrency` to the same number as `XTTS_WORKERS`. Agentic queues further requests itself.
3. In the Agentic web app, open **Audio → Settings** and paste the key. It is kept in Agentic's encrypted vault and never reaches a browser. You can also set the environment variable `AGENTIC_TextToSpeech__ApiKey`.

Agentic then uses this server in these places:
- Chat's **Read aloud**, and Dictation mode's spoken summaries.
- The **Audio** tab: speak any text with every setting, clone voices, and give each voice a personality.
- The voices in the CLI, the desktop app and VS Code.

## License notes

The code in this repository is free software under the [GNU General Public License v3.0](LICENSE).

The XTTS-v2 **model weights** are under the [Coqui Public Model License](https://coqui.ai/cpml), which allows **non-commercial use only**.

Only clone voices you have permission to use.
