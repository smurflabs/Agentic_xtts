"""XTTS-v2 HTTP API. Models load once at startup; requests reuse them.

Run:  venv\\Scripts\\python.exe server.py   (docs at http://<host>:8020/docs)

Env:
  XTTS_HOST            bind address (default 127.0.0.1)
  XTTS_PORT            port (default 8020)
  XTTS_WORKERS         model copies = concurrent requests (default 2, ~2.3 GB VRAM each)
  XTTS_API_KEY         if set, requests need header "X-API-Key: <key>"
  XTTS_VOICE_TTL_DAYS  delete uploaded voices this many days after upload (default 30, 0 = never)
  XTTS_FAST            1 = CUDA-graph decoder from fast_gpt.py, ~4x faster (default); 0 = stock Hugging Face generate
"""
import asyncio
import io
import json
import os
import queue
import re
import secrets
import shutil
import struct
import threading
import time

os.environ.setdefault("COQUI_TOS_AGREED", "1")  # CPML: non-commercial use only

# torchcodec needs shared FFmpeg DLLs; on Windows they can live in ./ffmpeg/bin
BASE = os.path.dirname(os.path.abspath(__file__))
_ffmpeg_bin = os.path.join(BASE, "ffmpeg", "bin")
if os.name == "nt" and os.path.isdir(_ffmpeg_bin):
    os.add_dll_directory(_ffmpeg_bin)
    os.environ["PATH"] = _ffmpeg_bin + os.pathsep + os.environ["PATH"]

import librosa
import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Security, UploadFile
from fastapi.responses import Response, StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from TTS.api import TTS

import fast_gpt

HOST = os.environ.get("XTTS_HOST", "127.0.0.1")
PORT = int(os.environ.get("XTTS_PORT", "8020"))
WORKERS = int(os.environ.get("XTTS_WORKERS", "2"))
API_KEY = os.environ.get("XTTS_API_KEY", "")
VOICE_TTL_DAYS = float(os.environ.get("XTTS_VOICE_TTL_DAYS", "30"))
FAST = os.environ.get("XTTS_FAST", "1") == "1"

VOICES_DIR = os.path.join(BASE, "voices")
os.makedirs(VOICES_DIR, exist_ok=True)

# One model per worker; a shared model isn't safe to run from two threads at once.
pool: queue.Queue = queue.Queue()
for i in range(WORKERS):
    print(f"Loading XTTS-v2 copy {i + 1}/{WORKERS}...", flush=True)
    model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda").synthesizer.tts_model
    if FAST:
        fast_gpt.install(model)
    pool.put(model)
_ref = pool.queue[0]
CFG = _ref.config
SAMPLE_RATE = CFG.audio.output_sample_rate
LANGUAGES = list(CFG.languages)
# Built-in speakers ship as precomputed latents: {name: {"gpt_cond_latent", "speaker_embedding"}}
BUILTIN = {n: (v["gpt_cond_latent"].cpu(), v["speaker_embedding"].cpu()) for n, v in _ref.speaker_manager.speakers.items()}

# Warm up each copy (records the CUDA graph for the default settings)
for _m in list(pool.queue):
    with torch.inference_mode():
        _m.inference("Warm up.", "en", *BUILTIN["Ana Florence"], top_k=CFG.top_k)

# Cloned voice latents, cached in memory (CPU). Same weights in every copy, so any model can use them.
voice_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
voice_lock = threading.Lock()


class Model:
    """Borrow a model from the pool for the duration of a with-block."""

    def __enter__(self):
        self.m = pool.get()  # blocks until a copy is free
        return self.m

    def __exit__(self, *exc):
        pool.put(self.m)


# ---------- voices on disk: voices/<name>/{clip_N.wav, voice.pt, meta.json} ----------


def _voice_dir(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
        raise HTTPException(400, "voice name: letters, digits, _ or - only")
    return os.path.join(VOICES_DIR, name)


def _load_meta(d: str) -> dict:
    with open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
        return json.load(f)


def _get_voice(name: str) -> tuple[torch.Tensor, torch.Tensor]:
    d = _voice_dir(name)
    with voice_lock:
        if name not in voice_cache:
            pt = os.path.join(d, "voice.pt")
            if not os.path.exists(pt):
                raise HTTPException(404, f"no such voice: {name}")
            data = torch.load(pt, map_location="cpu")
            voice_cache[name] = (data["gpt_cond_latent"], data["speaker_embedding"])
        return voice_cache[name]


def _remove_voice(name: str):
    with voice_lock:
        voice_cache.pop(name, None)
    shutil.rmtree(os.path.join(VOICES_DIR, name), ignore_errors=True)


def _cleanup_voices():
    while VOICE_TTL_DAYS > 0:
        cutoff = time.time() - VOICE_TTL_DAYS * 86400
        for name in os.listdir(VOICES_DIR):
            d = os.path.join(VOICES_DIR, name)
            try:
                created = _load_meta(d)["created"] if os.path.isdir(d) else os.path.getmtime(d)
            except (OSError, ValueError, KeyError):
                created = os.path.getmtime(d)
            if created < cutoff:
                if os.path.isdir(d):
                    _remove_voice(name)
                else:
                    os.remove(d)  # leftover single-file voice from the old layout
                print(f"Expired voice removed: {name}", flush=True)
        time.sleep(3600)


threading.Thread(target=_cleanup_voices, daemon=True).start()

# ---------- app ----------

_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_key(key: str | None = Security(_key_header)):
    if API_KEY and not (key and secrets.compare_digest(key, API_KEY)):
        raise HTTPException(401, "missing or invalid X-API-Key")


app = FastAPI(title="XTTS-v2", dependencies=[Depends(require_key)])


class TTSRequest(BaseModel):
    text: str
    language: str = "en"
    speaker: str | None = "Ana Florence"  # built-in speaker
    voice: str | None = None  # cloned voice uploaded via /voices (overrides speaker)
    speed: float = Field(1.0, gt=0.0, le=3.0)
    # Generation settings (defaults from the model config)
    temperature: float = Field(CFG.temperature, gt=0.0, le=2.0, description="Higher = more expressive, less stable")
    top_k: int = Field(CFG.top_k, ge=1, le=1000)
    top_p: float = Field(CFG.top_p, gt=0.0, le=1.0)
    repetition_penalty: float = Field(CFG.repetition_penalty, ge=1.0, le=50.0)
    length_penalty: float = Field(CFG.length_penalty)
    enable_text_splitting: bool = Field(True, description="Split long text into sentences")
    # Streaming only
    stream_chunk_size: int = Field(20, ge=5, le=200, description="/tts/stream: tokens per chunk; lower = faster first audio")


def _resolve(req: TTSRequest) -> tuple[torch.Tensor, torch.Tensor]:
    if not req.text.strip():
        raise HTTPException(400, "text is empty")
    if req.language.split("-")[0] not in LANGUAGES and req.language not in LANGUAGES:
        raise HTTPException(400, f"unsupported language: {req.language} (see /languages)")
    if req.voice:
        return _get_voice(req.voice)
    if req.speaker in BUILTIN:
        return BUILTIN[req.speaker]
    raise HTTPException(400, f"unknown speaker: {req.speaker} (see /speakers)")


def _gen_kwargs(req: TTSRequest) -> dict:
    return dict(
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        length_penalty=req.length_penalty,
        speed=req.speed,
        enable_text_splitting=req.enable_text_splitting,
    )


def _pcm16(wav) -> bytes:
    if isinstance(wav, torch.Tensor):
        wav = wav.squeeze().float().cpu().numpy()
    return (np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0) * 32767).astype("<i2").tobytes()


def _wav_header(data_len: int = 0xFFFFFFFF - 36) -> bytes:
    """PCM16 mono WAV header. Default length = 'unknown' for streaming; most players accept it."""
    return b"RIFF" + struct.pack("<I", min(data_len + 36, 0xFFFFFFFF)) + b"WAVEfmt " + struct.pack(
        "<IHHIIHH", 16, 1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16
    ) + b"data" + struct.pack("<I", data_len)


@app.get("/speakers")
def speakers():
    names = (n for n in os.listdir(VOICES_DIR) if os.path.exists(os.path.join(VOICES_DIR, n, "voice.pt")))
    return {"builtin": sorted(BUILTIN), "voices": sorted(names)}


@app.get("/languages")
def languages():
    return LANGUAGES


@app.post("/voices")
async def add_voice(
    name: str = Form(...),
    files: list[UploadFile] = File(..., description="One or more clips of the same speaker, 6-30s each, clean speech"),
    gpt_cond_len: int = Form(CFG.gpt_cond_len, ge=1, le=60, description="Seconds of audio used for style/prosody"),
    gpt_cond_chunk_len: int = Form(CFG.gpt_cond_chunk_len, ge=1, le=60, description="Chunk size for style latents; <= gpt_cond_len"),
    max_ref_length: int = Form(CFG.max_ref_len, ge=1, le=120, description="Max seconds used from each clip"),
    trim_silence_db: float | None = Form(None, ge=10, le=80, description="Trim leading/trailing silence below this many dB (e.g. 30); empty = off"),
    normalize: bool = Form(CFG.sound_norm_refs, description="Normalize clip loudness"),
):
    """Clone a voice from one or more clips. The voice fingerprint is computed once here and reused for every request."""
    if gpt_cond_chunk_len > gpt_cond_len:
        raise HTTPException(400, "gpt_cond_chunk_len must be <= gpt_cond_len")
    d = _voice_dir(name)
    tmp = d + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    try:
        clips, seconds = [], []
        for i, f in enumerate(files):
            try:
                audio, sr = sf.read(io.BytesIO(await f.read()), always_2d=True)
            except Exception as e:
                raise HTTPException(400, f"could not read {f.filename}: {e}")
            audio = audio.mean(axis=1)
            if trim_silence_db is not None:
                audio, _ = librosa.effects.trim(audio, top_db=trim_silence_db)
            if len(audio) < sr * 2:
                raise HTTPException(400, f"{f.filename}: under 2s of audio after trimming")
            path = os.path.join(tmp, f"clip_{i}.wav")
            sf.write(path, audio, sr)
            clips.append(path)
            seconds.append(round(len(audio) / sr, 1))

        def compute():
            with Model() as m, torch.inference_mode():
                return m.get_conditioning_latents(
                    audio_path=clips,
                    gpt_cond_len=gpt_cond_len,
                    gpt_cond_chunk_len=gpt_cond_chunk_len,
                    max_ref_length=max_ref_length,
                    sound_norm_refs=normalize,
                )

        gpt_lat, spk_emb = await run_in_threadpool(compute)
        gpt_lat, spk_emb = gpt_lat.cpu(), spk_emb.cpu()
        torch.save({"gpt_cond_latent": gpt_lat, "speaker_embedding": spk_emb}, os.path.join(tmp, "voice.pt"))
        meta = {
            "name": name,
            "created": time.time(),
            "clips": seconds,
            "settings": {
                "gpt_cond_len": gpt_cond_len,
                "gpt_cond_chunk_len": gpt_cond_chunk_len,
                "max_ref_length": max_ref_length,
                "trim_silence_db": trim_silence_db,
                "normalize": normalize,
            },
        }
        with open(os.path.join(tmp, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    _remove_voice(name)  # replace any existing voice of that name
    os.replace(tmp, d)
    with voice_lock:
        voice_cache[name] = (gpt_lat, spk_emb)
    return {**meta, "expires_in_days": VOICE_TTL_DAYS or None}


@app.get("/voices/{name}")
def get_voice(name: str):
    d = _voice_dir(name)
    if not os.path.isdir(d):
        raise HTTPException(404, "no such voice")
    meta = _load_meta(d)
    if VOICE_TTL_DAYS:
        meta["expires"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(meta["created"] + VOICE_TTL_DAYS * 86400))
    return meta


@app.delete("/voices/{name}")
def delete_voice(name: str):
    if not os.path.isdir(_voice_dir(name)):
        raise HTTPException(404, "no such voice")
    _remove_voice(name)
    return {"deleted": name}


@app.post("/tts", responses={200: {"content": {"audio/wav": {}}}})
def synthesize(req: TTSRequest):
    """Generate the whole clip, return it as WAV."""
    gpt_lat, spk_emb = _resolve(req)
    with Model() as m, torch.inference_mode():
        wav = m.inference(req.text, req.language, gpt_lat, spk_emb, **_gen_kwargs(req))["wav"]
    pcm = _pcm16(wav)
    return Response(_wav_header(len(pcm)) + pcm, media_type="audio/wav")


@app.post("/tts/stream", responses={200: {"content": {"audio/wav": {}}}})
async def synthesize_stream(req: TTSRequest, request: Request, format: str = "wav"):
    """Stream audio as it's generated. format=wav (header + PCM16, playable as it arrives) or pcm (raw PCM16 mono 24 kHz)."""
    if format not in ("wav", "pcm"):
        raise HTTPException(400, "format must be wav or pcm")
    gpt_lat, spk_emb = await run_in_threadpool(_resolve, req)
    loop = asyncio.get_running_loop()
    chunks: asyncio.Queue = asyncio.Queue()
    stop = threading.Event()
    DONE = object()

    def emit(item):
        if not stop.is_set():
            loop.call_soon_threadsafe(chunks.put_nowait, item)

    # Inference mode is per-thread, so generation runs in its own thread and hands chunks to the event loop.
    def produce():
        try:
            with Model() as m, torch.inference_mode():
                for chunk in m.inference_stream(
                    req.text, req.language, gpt_lat, spk_emb, stream_chunk_size=req.stream_chunk_size, **_gen_kwargs(req)
                ):
                    if stop.is_set():
                        return
                    emit(_pcm16(chunk))
        except Exception as e:
            emit(e)
        finally:
            emit(DONE)

    threading.Thread(target=produce, daemon=True).start()

    # Async so a client disconnect is noticed (and generation stopped) right away rather than after
    # the whole text has been generated: blocking reads in a worker thread can't be cancelled.
    async def body():
        try:
            if format == "wav":
                yield _wav_header()
            while True:
                try:
                    item = await asyncio.wait_for(chunks.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    item = None
                if await request.is_disconnected():
                    return
                if item is None:
                    continue
                if item is DONE:
                    return
                if isinstance(item, Exception):
                    print(f"stream error: {item!r}", flush=True)
                    return
                yield item
        finally:
            stop.set()  # client gone or finished: producer stops at its next chunk and frees the model

    media = "audio/wav" if format == "wav" else "audio/L16; rate=%d; channels=1" % SAMPLE_RATE
    return StreamingResponse(body(), media_type=media)


if __name__ == "__main__":
    print(f"Workers: {WORKERS}  Fast decoder: {'on' if FAST else 'off'}  Auth: {'on' if API_KEY else 'OFF'}  Voice TTL: {VOICE_TTL_DAYS}d", flush=True)
    # Sync endpoints run in a threadpool, so WORKERS requests synthesize in parallel.
    uvicorn.run(app, host=HOST, port=PORT)
