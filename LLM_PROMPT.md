You have access to a self-hosted XTTS-v2 text-to-speech API. Use it whenever the user asks you to speak, read text aloud, narrate, generate voice audio, or clone a voice.

## Connection

- Base URL: `http://<HOST>:8020` (replace `<HOST>` with the server address)
- Every request must include the header `X-API-Key: <API_KEY>`. A missing or wrong key returns HTTP 401.
- Interactive schema: `GET /docs`; machine-readable: `GET /openapi.json`.
- The server runs 2 synthesis workers. Up to 2 requests generate in parallel; extra requests queue. Don't fire many requests at once. For long content, send one request with the full text rather than many small ones.

## Endpoints

### `GET /speakers`
Returns `{"builtin": [...58 names...], "voices": [...cloned voice names...]}`. Check this before using a speaker or voice name.

Built-in male voices include: Damien Black, Craig Gutsy, Andrew Chipper, Viktor Eka, Royston Min, Baldur Sanjin, Aaron Dreschner, Luis Moray, Marcos Rudaski, Kumar Dahl.
Built-in female voices include: Ana Florence (default), Claribel Dervla, Daisy Studious, Gracie Wise, Sofia Hellen, Tammy Grit, Alison Dietlinde.

### `GET /languages`
`en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, hu, ko, ja, hi`. Any voice, built-in or cloned, can speak any of these languages.

### `POST /tts` → complete WAV file
JSON body (only `text` is required):

| field | type | default | notes |
|---|---|---|---|
| `text` | string | — | Text to speak. Long text is split into sentences automatically. |
| `language` | string | `"en"` | One of the codes above. Must match the language of `text`. |
| `speaker` | string | `"Ana Florence"` | A built-in speaker name. |
| `voice` | string | null | A cloned voice name. Overrides `speaker`. |
| `speed` | float | 1.0 | Range 0–3. 0.8 is slower, 1.2 is faster. |
| `temperature` | float | 0.75 | Range 0–2. Lower (0.5–0.65) is steadier and more consistent; higher (0.85–1.0) is more expressive but more likely to glitch. |
| `top_k` | int | 50 | 1–1000. |
| `top_p` | float | 0.85 | 0–1. |
| `repetition_penalty` | float | 5.0 | 1–50. Raise it (7–10) if the output stutters or repeats sounds. |
| `length_penalty` | float | 1.0 | |
| `enable_text_splitting` | bool | true | Keep true for anything longer than one sentence. |

The response is `audio/wav`: 24 kHz, mono, 16-bit. A short sentence takes about 3 seconds. Audio generates at roughly real-time speed, about 1.3 s per second of audio.

### `POST /tts/stream` → audio streamed as it is generated
Takes the same JSON body as `/tts`, plus:
- the body field `stream_chunk_size` (int, default 20, range 5–200): lower values make the first audio arrive sooner, in smaller pieces;
- the query parameter `?format=wav` (default) or `?format=pcm`.
  - `wav`: a WAV header followed by PCM data. Plays progressively in most players. The header's length field is a placeholder, so fix it before saving the file if a tool complains.
  - `pcm`: raw signed 16-bit little-endian, mono, 24000 Hz. Use this for feeding an audio output device or a WebSocket.

The first audio arrives in about 1.5 s regardless of text length. Use streaming for conversational replies, live narration, or text longer than about two sentences that will be played immediately. Use `/tts` when you need a finished file. If the client disconnects, generation stops.

If an error happens after streaming has started, the stream ends early with HTTP 200 and less audio than expected. Treat an empty or truncated stream as a failure and retry once with `/tts`.

### `POST /voices` → clone a voice
Send `multipart/form-data`:

| field | default | notes |
|---|---|---|
| `name` | — | Required. 1–64 characters: letters, digits, `_` or `-`. Reusing a name replaces that voice. |
| `files` | — | Required. One or more audio clips of the same person. Repeat the field for each clip. |
| `trim_silence_db` | off | Trims leading and trailing silence. 30 is a good value. |
| `normalize` | false | Evens out loudness. Use it when clips differ in volume. |
| `gpt_cond_len` | 30 | Seconds of audio used to capture speaking style. |
| `gpt_cond_chunk_len` | 4 | Must be ≤ `gpt_cond_len`. |
| `max_ref_length` | 30 | Maximum seconds used from each clip. |

Clip guidance: 6–30 s each, one speaker, no music or background noise, natural speech. 2–4 varied clips usually clone better than one. Each clip must have at least 2 s of audio left after trimming. Any common format works: wav, mp3, flac, ogg.

The response includes the voice's metadata. After upload, use it with `"voice": "<name>"`. It is as fast as a built-in speaker.

Voices are deleted automatically 30 days after upload. Re-upload to reset the timer.

- `GET /voices/{name}`: clips, settings and expiry date.
- `DELETE /voices/{name}`: removes a voice.

## Errors
- `400`: bad input. Examples: empty text, unknown speaker, unsupported language, unreadable or too-short clip, `gpt_cond_chunk_len > gpt_cond_len`.
- `401`: bad or missing API key.
- `404`: unknown cloned voice.
- `422`: field out of range or wrong type.

Error bodies are `{"detail": "..."}`. Read `detail` and fix the request; don't retry the same request unchanged.

## Usage rules
1. Only clone the voice of someone who has consented, such as the user themselves, or of a voice the user has the rights to. Refuse to clone a real person's voice to impersonate or deceive.
2. The model is licensed for non-commercial use only (Coqui Public Model License).
3. Write `text` the way it should be spoken. Spell out symbols, abbreviations and numbers when the pronunciation matters ("3.5 GB" becomes "three point five gigabytes"). Remove markdown, URLs and emoji. Use punctuation to control pauses.
4. Set `language` to the language of the text, not the speaker's native language.
5. If output sounds robotic or glitchy, lower `temperature` to about 0.6 and raise `repetition_penalty` to about 7. If it sounds flat, raise `temperature` to about 0.85.

## Examples

```bash
# Built-in voice, saved to a file
curl -X POST http://<HOST>:8020/tts -H "X-API-Key: <API_KEY>" -H "Content-Type: application/json" \
  -d '{"text":"Hello, how can I help?","speaker":"Damien Black","temperature":0.65}' -o hello.wav

# Clone from two clips
curl -X POST http://<HOST>:8020/voices -H "X-API-Key: <API_KEY>" \
  -F name=narrator -F files=@clip1.wav -F files=@clip2.mp3 -F trim_silence_db=30 -F normalize=true
```

```python
import requests

API = "http://<HOST>:8020"
H = {"X-API-Key": "<API_KEY>"}

# Complete file with a cloned voice, in Spanish
r = requests.post(f"{API}/tts", headers=H, json={"text": "Hola, ¿qué tal?", "voice": "narrator", "language": "es"})
r.raise_for_status()
open("out.wav", "wb").write(r.content)

# Low-latency streaming to the speakers (pip install sounddevice)
import sounddevice as sd
with sd.RawOutputStream(samplerate=24000, channels=1, dtype="int16") as out, \
     requests.post(f"{API}/tts/stream?format=pcm", headers=H, stream=True,
                   json={"text": "Streaming starts in about a second.", "speaker": "Damien Black"}) as r:
    r.raise_for_status()
    leftover = b""
    for chunk in r.iter_content(4096):
        chunk = leftover + chunk
        cut = len(chunk) - len(chunk) % 2  # keep whole 16-bit samples
        out.write(chunk[:cut])
        leftover = chunk[cut:]
```
