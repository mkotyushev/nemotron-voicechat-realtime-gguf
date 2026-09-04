# NemotronLabs VoiceChat 11B — Realtime GGUF server

Run [`nvidia/NVIDIA-NemotronLabs-VoiceChat-11B`](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B)
as Q8_0 GGUF behind an OpenAI Realtime-compatible WebSocket API. The complete
speech-to-speech model fits on a 16 GB GPU.

The companion client is
[`voicechat-desktop-client`](https://github.com/mkotyushev/voicechat-desktop-client).

## Attribution

- NVIDIA publishes the model under OpenMDW-1.1.
- [`sansamour/llama-voicechat.cpp`](https://github.com/sansamour/llama-voicechat.cpp)
  provides the MIT-licensed GGUF runtime.
- This repository adds the Realtime bridge, deployment scripts, and four small
  runtime/converter patches.

Weights are not stored here. `download.sh` fetches the community GGUF container
and the tokenizer files needed by the converters. See [NOTICE](NOTICE).

## Layout

| Path | Purpose |
|---|---|
| `bridge/` | OpenAI Realtime-compatible WebSocket server |
| `patches/` | streaming, duplex, prompt-prefill, and Q8_0 converter patches |
| `download.sh` | fetch the source GGUF and tokenizer |
| `convert.sh` | split the source into the four runtime files |
| `convert_asr_to_mmproj.py` | build the perception file from another compatible ASR checkpoint |
| `docker-compose.yml` | build and run the server |
| `MODEL_DIR` | downloaded and converted model files; `./models` by default |

## Bring it up

You need Linux, Git, Docker Compose, NVIDIA Container Toolkit, a recent NVIDIA
driver, and [`uv`](https://docs.astral.sh/uv/).

```bash
cp .env.example .env
./setup.sh
./download.sh
./convert.sh
docker compose build
docker compose up -d
```

The first build compiles `llama-voicechat` for `CUDA_ARCH` from `.env`. The
default is Ampere `sm_86`; change it before building for another GPU.

Check readiness with:

```bash
docker compose ps
curl -s http://127.0.0.1:9070/v1/realtime/health
```

The model loads before the health check becomes ready. There is no separate
warm-up step.

Configuration changes are applied with `docker compose up -d`, not
`docker compose restart`. Changing `VC_REF` also requires rebuilding the image
and rerunning `convert.sh` so the converters match the runtime.

## Why GGUF

NVIDIA's official container uses the original fp32 checkpoint and targets much
larger accelerators. The community GGUF packs the same five model components
into one quantized source file.

That source file is not directly runnable. The fork's converters split it into:

```text
nemotron_voicechat_11b-stt-llm-Q8_0.gguf
nemotron_voicechat_11b-stt-llm-Q8_0-function-head.gguf
mmproj-voicechat-perception-Q8_0.gguf
voicechat-tts-Q8_0.gguf
```

Keep all four files together under their generated names. The runtime finds the
function head beside the language model.

Q8_0 is the supported default. Q4_0 also converts, but this deployment has been
tuned and measured with Q8_0.

## API

| Endpoint | Purpose |
|---|---|
| `ws://<server-host>:9070/v1/realtime` | Realtime session |
| `ws://<server-host>:9070/realtime` | alias |
| `http://<server-host>:9070/v1/realtime/health` | readiness |
| `http://<server-host>:9070/` | discovery |

Audio is PCM16 mono at 24 kHz in both directions, sent as base64 inside JSON.
The server accepts 80 ms input frames and streams transcript and speech events
while the microphone stream continues.

One WebSocket is one conversation. Conversation state lives in the model's
timeline and is discarded when the socket closes.

Only one session can be active at a time. A second connection receives
`session_in_use` instead of waiting in a queue.

The voice endpoint has no authentication. The checked-in configuration binds
to `127.0.0.1`; expose it only on a trusted network or behind an authenticated
proxy or firewall.

## Realtime behavior

VoiceChat runs on a continuous 12.5 Hz timeline. The model can start answering
before the speaker stops and can react to speech while it is talking.

The bridge keeps the perception encoder and language model caches alive across
frames. Generated audio is returned in short chunks instead of waiting for the
complete turn.

`session.update` can install instructions and tools. Tool results return through
`conversation.item.create`, after which generation continues on the same
timeline.

The model has no typed-turn input channel. Every answer is spoken, and the model
answers in English.

## Deployment patches

The image applies four patches to the pinned runtime:

| Patch | Purpose |
|---|---|
| `q8_0-converters.patch` | read and preserve a Q8_0 source container |
| `stream-audio.patch` | emit speech chunks during generation |
| `system-prefill.patch` | batch the system prompt without changing timeline positions |
| `full-duplex.patch` | cache-aware perception and a persistent duplex timeline |

The patches remain separate so the exact delta from the fork stays reviewable.
`VC_REF` pins the source commit they apply to.

## Measured on this box

The reference measurements use an RTX 3090; smaller 16 GB cards should fit the
same Q8_0 model, although latency depends on the GPU.

| Metric | Value |
|---|---|
| Model load to healthy | 4–10 s |
| Voice server VRAM | 14.6 GB |
| Image size | 5.9 GB |
| Build time for one CUDA architecture | about 13 min |
| Conversion time | about 2 min |
| First text in the paced speech test | about 1.2 s |
| First audio in the paced speech test | about 1.9 s |

Tested: streamed English speech in and out, barge-in, repeated sessions, system
instructions, tool calls, bounded disconnect cleanup, and rejection of a second
client.

Not tested: hours-long sessions, behavior near the timeline/context limits, or
extended tool use during continuous overlapping speech.

## Configuration

Edit `.env`, then run `docker compose up -d`.

| Variable | Default | Meaning |
|---|---|---|
| `SRC_QUANT` | `Q8_0` | source and output quantization |
| `CUDA_ARCH` | `86` | CUDA architecture compiled into the image |
| `N_GPU_LAYERS` | `99` | layers offloaded to the GPU |
| `SESSION_SECONDS` | `180` | maximum model timeline |
| `STREAM_FRAMES` | `8` | generated speech frames per streamed chunk |
| `VAD_SILENCE_MS` | `700` | silence before the UI reports speech stopped |
| `VAD_MARGIN` | `3.5` | speech threshold above the measured noise floor |
| `TOOL_TIMEOUT_S` | `60` | maximum wait for a tool result |
| `BIND_ADDR` | `127.0.0.1` | host interface exposed by Docker |
| `PORT` | `9070` | host port |

VAD only drives lifecycle events and response-settle hints. It never withholds
audio from the model or delays response start.

## Using another ASR encoder

The default `ASR_MODEL=container` uses the perception encoder shipped inside
VoiceChat. A compatible streaming FastConformer checkpoint can replace only
that component while the language model, function head, speech generator, and
codec stay unchanged.

Set a name and checkpoint directory in `.env`:

```bash
ASR_MODEL=my-encoder
ASR_DIR=/path/to/my-encoder
```

The directory must contain `config.json` and `model.safetensors`.
`convert_asr_to_mmproj.py` converts it into
`mmproj-asr-my-encoder-Q8_0.gguf`; `convert.sh` runs that converter and the
server selects the resulting file through the same `ASR_MODEL` value.

If the checkpoint does not carry VoiceChat's `proj` and mel-featurizer tensors,
the converter takes them from the source VoiceChat container. This makes the
swap possible, but does not guarantee that the language model understands the
new encoder's output space.

The unfinished multilingual alignment experiments now live in
[`nemotron-voicechat-asr-multilingual`](https://github.com/mkotyushev/nemotron-voicechat-asr-multilingual).
They are research artifacts, not a recommended server default.

## License

The original code in this repository is MIT-licensed. The model weights retain
their upstream license and notices; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
