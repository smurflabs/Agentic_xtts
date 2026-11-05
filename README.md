# Agentic XTTS

A FastAPI server for [Coqui XTTS-v2](https://huggingface.co/coqui/XTTS-v2) text-to-speech, built to be called by LLM agents and apps. It provides voice cloning, streaming, and a CUDA-graph decoder that makes generation about 4× faster on hardware where the CPU is the bottleneck.

- **Voice cloning** from one or more reference clips. Each voice's fingerprint is computed once at upload and reused for every request.
- **Streaming** (`/tts/stream`): first audio in about 0.5 s, as WAV or raw PCM.
- **All XTTS generation settings**: temperature, top-k, top-p, repetition and length penalty, speed, plus clone-tuning options.
- **17 languages** and 58 built-in speakers.
- **Concurrent requests**: one model copy per worker (`XTTS_WORKERS`).
- **API-key auth**, plus automatic expiry of uploaded voices.
- **[`LLM_PROMPT.md`](LLM_PROMPT.md)**: a ready-made tool prompt that teaches an LLM to use every endpoint.

## Performance

Measured on a GTX 1660 Ti (6 GB) with a Xeon E5-2670 v2 (2013). RTF is seconds of compute per second of audio; lower is better.

| | stock XTTS | `fast_gpt.py` |
|---|---|---|
| `/tts` RTF | 1.36 | **0.34** |
| `/tts/stream` RTF | 1.44 | **0.43** |
| Stream, time to first audio | 1.4 s | **0.5 s** |
| GPU utilization | 27% | ~90% |

XTTS generates about 21 audio tokens per second of speech, one at a time. With Hugging Face `generate()`, each token costs hundreds of small kernel launches plus Python overhead. On a slow CPU, the GPU sits idle waiting for them.

[`fast_gpt.py`](fast_gpt.py) allocates GPT-2's attention memory (the KV cache) once at full length, so every decode step has identical tensor shapes. It records the whole step as one CUDA graph: embeddings, 30 transformer layers, repetition penalty, temperature, top-k, top-p and sampling. Each token is then a single graph replay. The decoder checks for the stop token every 8 steps rather than every step.

It replaces only `gpt.generate` and `gpt.get_generator`, so XTTS's own `inference()` and `inference_stream()` run unchanged on top of it.

The output is the same as stock XTTS:
- **Greedy decoding** produces the same tokens as stock XTTS, checked on the test sentences used during development.
- **Sampling** produces the same probability distribution as Hugging Face's logits processors. See [`tests/test_sampling.py`](tests/test_sampling.py).

Set `XTTS_FAST=0` to fall back to stock generation.

## Setup (Windows)

Requirements: an NVIDIA GPU with 3 GB or more of VRAM per worker, and Python 3.10–3.13. This was tested on Python 3.12.

```powershell
git clone https://github.com/smurflabs/Agentic_xtts
cd Agentic_xtts
py -3.12 -m venv venv
venv\Scripts\python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
venv\Scripts\python -m pip install -r requirements.txt
```

**FFmpeg:** torchcodec needs FFmpeg's *shared* DLLs (versions 4–8); a static `ffmpeg.exe` is not enough. Download a `win64-gpl-shared` build, for example `ffmpeg-n8.1-latest-win64-gpl-shared-8.1.zip` from [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds/releases). Extract it so that `ffmpeg\bin\avcodec-*.dll` sits next to `server.py`. On Linux, install FFmpeg from your package manager instead.

**Configure and run:**

```powershell
copy start_server.example.cmd start_server.cmd   # then edit: set XTTS_API_KEY, host, workers
start_server.cmd
```

The first run downloads the model (about 1.8 GB) and asks you to accept the license. `server.py` sets `COQUI_TOS_AGREED=1`, which accepts it on your behalf; read the [license notes](#license-notes) first.

Interactive API docs are served at `http://<host>:8020/docs`.

To start the server at boot, register a scheduled task that runs `start_server.cmd` as SYSTEM, and uncomment the `TTS_HOME` and `HF_HOME` lines in the launcher.

## Configuration

| Variable | Default | |
|---|---|---|
| `XTTS_HOST` | `127.0.0.1` | Use `0.0.0.0` to listen on the network. Set an API key and a firewall rule first. |
| `XTTS_PORT` | `8020` | |
| `XTTS_WORKERS` | `2` | Number of model copies, which is the number of requests handled in parallel. About 2.3 GB of VRAM each. |
| `XTTS_API_KEY` | none | If set, every request needs the header `X-API-Key: <key>`. |
| `XTTS_VOICE_TTL_DAYS` | `30` | Uploaded voices are deleted this many days after upload. `0` keeps them forever. |
| `XTTS_FAST` | `1` | `1` uses the CUDA-graph decoder; `0` uses stock Hugging Face generation. |

## API

| Endpoint | |
|---|---|
| `POST /tts` | JSON body → complete WAV (24 kHz mono, 16-bit) |
| `POST /tts/stream?format=wav\|pcm` | Same body → audio streamed as it's generated |
| `POST /voices` | multipart: `name`, one or more `files`, plus optional clone tuning → clones a voice |
| `GET /voices/{name}` / `DELETE /voices/{name}` | Show a voice's metadata and expiry / delete it |
| `GET /speakers`, `GET /languages` | List built-in speakers and cloned voices / supported languages |

`/tts` and `/tts/stream` body fields:
- `text` (required)
- `language`
- `speaker` (a built-in speaker) or `voice` (a cloned voice)
- `speed`
- generation settings: `temperature`, `top_k`, `top_p`, `repetition_penalty`, `length_penalty`, `enable_text_splitting`
- streaming only: `stream_chunk_size`

`/voices` tuning fields: `gpt_cond_len`, `gpt_cond_chunk_len`, `max_ref_length`, `trim_silence_db`, `normalize`.

See [`LLM_PROMPT.md`](LLM_PROMPT.md) for every field, its range, guidance on when to use each one, and examples.

```bash
curl -X POST http://localhost:8020/tts -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"text":"Hello there!","speaker":"Damien Black"}' -o hello.wav

curl -X POST http://localhost:8020/voices -H "X-API-Key: $KEY" \
  -F name=narrator -F files=@clip1.wav -F files=@clip2.wav -F trim_silence_db=30
```

## License notes

The code in this repository is licensed under the [GNU GPL v3.0](LICENSE). The XTTS-v2 **model weights** are licensed under the [Coqui Public Model License](https://coqui.ai/cpml), which allows **non-commercial use only**.

Only clone voices you have permission to use.
