#!/usr/bin/env python3
"""An OpenAI Realtime WebSocket in front of `llama-voicechat --serve`.

NVIDIA ships NemotronLabs VoiceChat as a Triton/vLLM container that speaks the
OpenAI Realtime protocol over a WebSocket and wants ~66 GB of VRAM. This box has
24, so the model runs here as a GGUF under `llama-voicechat` instead — which is
a command line tool that reads json lines on stdin and writes wav files to disk.

This bridge puts the first interface back on top of the second, so that clients
are written against the protocol NVIDIA documents rather than against a local
quirk, and so that moving to the real container later is a URL change.

What matches the NVIDIA container:

  GET  /                       service discovery
  GET  /v1/realtime/health     readiness
  WS   /v1/realtime            the protocol, and /realtime as an alias
  PCM16 mono 24 kHz both directions, 80 ms chunks, base64 in json
  session.update / input_audio_buffer.append / conversation.item.create
  the response.* and conversation.item.* event stream, tool calls included

What still differs from NVIDIA's container:

  * There is no user-side transcript: the RNN-T decoder is in the checkpoint
    but no converter carries it into the GGUF set, so
    conversation.item.input_audio_transcription.* is never sent.
  * Speech start/stop events use a small energy gate. Every microphone frame
    still reaches the model and response starts never wait for it. The same
    epochs keep a completed response from reopening on a quantized-head echo.
  * One session at a time, server-wide — the model has one timeline.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import signal
import sys
import tempfile
import time
import uuid
from http import HTTPStatus
from pathlib import Path

import numpy as np
import websockets
from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.http11 import Response

sys.path.insert(0, str(Path(__file__).parent))

from audio import (  # noqa: E402
    CLIENT_RATE,
    MODEL_IN_RATE,
    StreamResampler,
    TurnDetector,
    float_to_pcm16,
    read_wav,
    resample,
    write_wav,
)
from engine import EngineError, VoiceChatEngine  # noqa: E402

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("bridge")

SERVICE = "nemotron-voicechat"
VERSION = "0.3.0"
MODEL_NAME = "nemotron-voicechat"

# 80 ms of 24 kHz PCM16, the chunk size the NVIDIA client uses.
OUT_CHUNK = int(CLIENT_RATE * 0.08)
MODEL_CHUNK = int(MODEL_IN_RATE * 0.08)


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default) or default)


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default) or default)


def mmproj_name(asr_model: str, quant: str) -> str:
    """File name of the perception encoder ASR_MODEL selects.

    The container's encoder keeps the name the fork's converter has always
    written. Everything else is a checkpoint directory -- the two NVIDIA
    publishes, or a compatible custom checkpoint -- and gets a file named after it, so
    several can sit side by side in /models and switching is a restart rather
    than a reconvert. convert.sh derives the same name from the same variable;
    if the two ever disagree the container starts and cannot find its encoder.

    No list of allowed names is kept here on purpose. A new encoder should be a
    directory and a rebuild, not an edit to the bridge.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", asr_model):
        raise SystemExit(
            f"ASR_MODEL={asr_model!r} is not a usable name: it names a file, so "
            "letters, digits, dot, dash and underscore only"
        )
    if asr_model == "container":
        return f"mmproj-voicechat-perception-{quant}.gguf"
    return f"mmproj-asr-{asr_model}-{quant}.gguf"


class Config:
    def __init__(self) -> None:
        self.host = os.environ.get("BRIDGE_HOST", "0.0.0.0")
        self.port = env_int("BRIDGE_PORT", 8080)
        self.binary = os.environ.get("VC_BINARY", "/app/llama-voicechat")
        model_dir = Path(os.environ.get("VC_MODEL_DIR", "/models"))
        quant = os.environ.get("SRC_QUANT", "Q8_0")
        self.asr_model = os.environ.get("ASR_MODEL", "container")
        self.model = str(model_dir / f"nemotron_voicechat_11b-stt-llm-{quant}.gguf")
        self.mmproj = str(model_dir / mmproj_name(self.asr_model, quant))
        self.tts = str(model_dir / f"voicechat-tts-{quant}.gguf")
        self.n_gpu_layers = env_int("N_GPU_LAYERS", 99)
        self.session_seconds = env_int("SESSION_SECONDS", 180)
        self.extra_decoding_seconds = env_int("EXTRA_DECODING_SECONDS", 50)
        self.temp = env_float("TEMP", 0.0)
        # Frames of speech per streamed chunk; 0 waits for the whole turn.
        self.stream_frames = env_int("STREAM_FRAMES", 8)
        self.tool_timeout = env_float("TOOL_TIMEOUT_S", 60.0)
        self.vad_silence_ms = env_int("VAD_SILENCE_MS", 700)
        self.vad_margin = env_float("VAD_MARGIN", 3.5)
        self.vad_min_speech_ms = env_int("VAD_MIN_SPEECH_MS", 350)
        self.vad_max_turn_ms = env_int("VAD_MAX_TURN_MS", 30000)
        self.scratch = Path(os.environ.get("VC_SCRATCH", tempfile.gettempdir()))


# --------------------------------------------------------------------- tools


TOOL_PREAMBLE = "You can use the following tools to assist the user if required:"
TOOL_FORMAT = (
    "If you decide to call any tool(s), use the following format:\n"
    '<TOOLCALL>[{"name": "tool_name1", "arguments": "tool_args1"}, '
    '{"name": "tool_name2", "arguments": "tool_args2"}]</TOOLCALL>\n\n'
    "The user will execute tool-calls and return responses from tool(s) in "
    "this format:\n"
    '<TOOL_RESPONSE>[{"tool_response1"}, {"tool_response2"}]</TOOL_RESPONSE>\n\n'
    "Based on the tool responses, you can call additional tools if needed, "
    "correct tool calls if any errors are found, or just respond to the user."
)


def render_system_prompt(instructions: str | None, tools: list[dict] | None) -> str:
    """Build the one text the model gets, in Nemotron-Nano-9B-v2's own format.

    VoiceChat has no text input channel and no chat template, so the driver has
    to render the tool list itself; `llama-voicechat --system` writes it into
    the perception channel one token per timeline frame. The engine batches the
    conditioning compute, but the prompt still consumes one KV position per
    token and therefore reduces the timeline left for conversation.

    NVIDIA's docs require system prompts and tool responses to be ASCII, so
    non-ASCII is stripped rather than left to produce a silent misparse.
    """
    parts: list[str] = []
    if instructions:
        parts.append(instructions.strip())
    if tools:
        listed = ", ".join(json.dumps(t, ensure_ascii=True) for t in tools)
        parts.append(f"{TOOL_PREAMBLE}\n<AVAILABLE_TOOLS>[{listed}]</AVAILABLE_TOOLS>")
        parts.append(TOOL_FORMAT)
    text = "\n\n".join(parts)
    return text.encode("ascii", "ignore").decode("ascii")


def parse_tool_call(text: str) -> list[dict]:
    """Turn the function channel's text into {name, arguments} dicts.

    The channel carries what the model wrote between the sotc and eotc markers,
    so usually a bare json array. It is model output, so it is parsed
    defensively: a call we cannot read is reported with the raw text as its
    arguments rather than dropped, which at least lets the driver see it.
    """
    raw = text.strip()
    for marker in ("<TOOLCALL>", "</TOOLCALL>"):
        raw = raw.replace(marker, "")
    raw = raw.strip()

    parsed = None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("["), raw.rfind("]")
        if start != -1 and end > start:
            try:
                parsed = json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                parsed = None

    if parsed is None:
        return [{"name": "unknown", "arguments": raw}]
    if isinstance(parsed, dict):
        parsed = [parsed]

    calls = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        args = item.get("arguments", {})
        # The Realtime protocol carries arguments as a json *string*.
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=True)
        calls.append({"name": str(item.get("name", "unknown")), "arguments": args})
    return calls or [{"name": "unknown", "arguments": raw}]


# ------------------------------------------------------------------- session


class Session:
    """One WebSocket connection, which is one conversation."""

    def __init__(self, ws: ServerConnection, engine: VoiceChatEngine, cfg: Config):
        self.ws = ws
        self.engine = engine
        self.cfg = cfg
        self.id = f"sess_{uuid.uuid4().hex[:16]}"
        self.instructions: str | None = None
        self.tools: list[dict] = []
        self.configured = False
        self.turn_task: asyncio.Task | None = None
        self.audio_task: asyncio.Task | None = None
        self.started = time.time()
        self.n_turns = 0
        self.detector = TurnDetector(
            rate=CLIENT_RATE,
            silence_ms=cfg.vad_silence_ms,
            margin=cfg.vad_margin,
            min_speech_ms=cfg.vad_min_speech_ms,
            max_turn_ms=cfg.vad_max_turn_ms,
        )
        self._call_ids: dict[str, str] = {}
        self._audio_queue: asyncio.Queue[tuple[bytes, int, bool]] = asyncio.Queue(maxsize=64)
        self._speech_epoch = 0
        self._input_speaking = False
        self._in = StreamResampler(CLIENT_RATE, MODEL_IN_RATE)
        self._model_buf = np.zeros(0, dtype=np.float32)
        self._response_id: str | None = None
        self._item_id: str | None = None
        self._response_opened_by_tool = False
        self._transcript_parts: list[str] = []
        self._closed = False
        self._tool_active = False
        self._dropped_audio = 0
        # One per turn: the codec's 22.05 kHz has to reach the client at 24 kHz,
        # and the chunks are pieces of one continuous utterance, so the
        # resampler has to carry its state across them.
        self._out: StreamResampler | None = None

    # ---------------------------------------------------------------- output

    async def send(self, event_type: str, **fields) -> None:
        payload = {"type": event_type, "event_id": f"event_{uuid.uuid4().hex[:16]}"}
        payload.update(fields)
        await self.ws.send(json.dumps(payload))

    async def error(self, message: str, code: str = "server_error") -> None:
        log.warning("session %s: %s", self.id, message)
        await self.send("error", error={"type": "invalid_request_error",
                                        "code": code, "message": message})

    # ----------------------------------------------------------------- input

    async def run(self) -> None:
        await self.send(
            "session.created",
            session={
                "type": "realtime",
                "id": self.id,
                "model": MODEL_NAME,
                "modalities": ["audio"],
                "audio": {
                    "input": {"format": {"type": "audio/pcm", "rate": CLIENT_RATE}},
                    "output": {"format": {"type": "audio/pcm", "rate": CLIENT_RATE}},
                },
                "instructions": self.instructions,
            },
        )

        try:
            async for raw in self.ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await self.error("message is not json", "invalid_json")
                    continue

                kind = msg.get("type")
                if kind == "input_audio_buffer.append":
                    await self._on_audio(msg)
                elif kind == "session.update":
                    await self._on_session_update(msg)
                elif kind == "conversation.item.create":
                    await self._on_item_create(msg)
                elif kind == "session.close":
                    break
                else:
                    await self.error(f"unsupported event type {kind!r}", "unknown_event")
        finally:
            await self.close()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for task in (self.audio_task, self.turn_task):
            if task is None or task is asyncio.current_task():
                continue
            if not task.done():
                task.cancel()
            try:
                await asyncio.wait_for(task, timeout=5)
            except asyncio.CancelledError:
                pass
            except asyncio.TimeoutError:
                log.error("session %s: worker did not stop in 5s", self.id)
            except Exception:
                log.exception("session %s: worker failed during close", self.id)
        self._tool_active = False
        try:
            await self.send(
                "session.end",
                session={
                    "id": self.id,
                    "turns": self.n_turns,
                    "duration_seconds": round(time.time() - self.started, 1),
                },
            )
        except (websockets.ConnectionClosed, RuntimeError, OSError):
            pass  # the client hung up first; nothing to tell it

    async def _on_session_update(self, msg: dict) -> None:
        session = msg.get("session") or {}
        if "instructions" in session:
            self.instructions = session.get("instructions")
        if "tools" in session:
            self.tools = session.get("tools") or []

        # The prompt goes in once, before the first turn, because that is the
        # only place the model has for one. A later session.update can still
        # change what we report, but not what the model was told.
        if not self.configured:
            prompt = render_system_prompt(self.instructions, self.tools)
            if prompt:
                # The exact token count comes back in system_start. This rough
                # count is only a useful indication of timeline consumption;
                # wall time now comes from the logical prefill batch.
                est_frames = round(len(prompt.split()) * 1.3)
                log.info("session %s: system prompt, roughly %d timeline frames",
                         self.id, est_frames)
                try:
                    await self.engine.set_system(prompt)
                except EngineError as e:
                    await self.error(f"system prompt rejected: {e}")
                    return
            try:
                await self.engine.start_duplex()
            except EngineError as e:
                await self.error(f"duplex start rejected: {e}")
                return
            self.configured = True
            self.audio_task = asyncio.create_task(self._audio_loop())

        await self.send(
            "session.updated",
            session={
                "audio": {
                    "input": {"format": {"type": "audio/pcm", "rate": CLIENT_RATE}},
                    "output": {"format": {"type": "audio/pcm", "rate": CLIENT_RATE}},
                },
                "instructions": self.instructions,
                "tools": self.tools,
            },
        )

    async def _on_item_create(self, msg: dict) -> None:
        item = msg.get("item") or {}
        if item.get("type") != "function_call_output":
            await self.error("only function_call_output items are supported",
                             "unsupported_item")
            return
        if not self.tools:
            await self.error("no tools were registered in session.update",
                             "tools_not_set")
            return
        output = item.get("output", "")
        if not isinstance(output, str):
            output = json.dumps(output, ensure_ascii=True)
        # The model was trained on ASCII tool responses.
        output = output.encode("ascii", "ignore").decode("ascii")
        if not self.engine.answer_tool_call(output):
            await self.error("no tool call is waiting for a result",
                             "no_pending_call")

    async def _on_audio(self, msg: dict) -> None:
        try:
            pcm = base64.b64decode(msg.get("audio", ""))
        except (ValueError, TypeError):
            await self.error("audio is not valid base64", "invalid_audio")
            return
        if not pcm:
            return

        chunk = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        event, _ = self.detector.push(chunk)

        if event == "started":
            self._speech_epoch += 1
            self._input_speaking = True
            await self.send(
                "input_audio_buffer.speech_started",
                audio_start_ms=0,
                item_id=f"item_{uuid.uuid4().hex[:16]}",
            )
        elif event == "stopped":
            self._input_speaking = False
            await self.send(
                "input_audio_buffer.speech_stopped",
                audio_end_ms=0,
                item_id=f"item_{uuid.uuid4().hex[:16]}",
            )
        if not self.configured:
            await self.error("session.update must precede audio", "session_not_configured")
            return

        # Resampling is continuous across WebSocket chunks. Feed the model in
        # exact 80 ms blocks even if a client sends a different packet size.
        converted = self._in.push(chunk)
        if self._tool_active:
            self._model_buf = np.zeros(0, dtype=np.float32)
            return
        if converted.size:
            self._model_buf = np.concatenate([self._model_buf, converted])
        while self._model_buf.size >= MODEL_CHUNK:
            frame = self._model_buf[:MODEL_CHUNK]
            self._model_buf = self._model_buf[MODEL_CHUNK:]
            queued = (float_to_pcm16(frame), self._speech_epoch, self._input_speaking)
            try:
                self._audio_queue.put_nowait(queued)
            except asyncio.QueueFull:
                # Realtime input cannot back-pressure the socket reader. Keep
                # the newest microphone frame and discard stale queued audio.
                self._audio_queue.get_nowait()
                self._audio_queue.put_nowait(queued)
                self._dropped_audio += 1
                if self._dropped_audio == 1 or self._dropped_audio % 25 == 0:
                    log.warning("session %s: dropped %d stale microphone frames",
                                self.id, self._dropped_audio)

    # --------------------------------------------------------------- duplex

    async def _audio_loop(self) -> None:
        """Advance the model at 12.5 Hz, using silence on an input underrun."""
        silence = bytes(MODEL_CHUNK * 2)
        deadline = time.monotonic()
        try:
            while True:
                delay = deadline - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                try:
                    pcm, speech_epoch, input_speaking = self._audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pcm = silence
                    speech_epoch = self._speech_epoch
                    input_speaking = False

                await self.engine.frame(
                    pcm,
                    self._on_engine_event,
                    speech_epoch=speech_epoch,
                    input_speaking=input_speaking,
                )
                # process_audio_frame does not return until any tool splice is
                # complete, including the bounded no-eotr fallback.
                self._tool_active = False
                deadline += 0.08
                now = time.monotonic()
                if now - deadline > 0.5:
                    log.warning("session %s: duplex clock is %.0f ms behind",
                                self.id, 1000 * (now - deadline))
                    deadline = now
        except asyncio.CancelledError:
            raise
        except EngineError as e:
            await self.error(f"duplex stream failed: {e}", "duplex_failed")
            await self.ws.close(1011, "duplex stream failed")
        finally:
            self._tool_active = False

    async def _begin_response(self, ev: dict, *, opened_by_tool: bool = False) -> None:
        if self._response_id is not None:
            await self._finish_response({"reason": "superseded", "t": ev.get("t")})

        self._response_id = f"resp_{uuid.uuid4().hex[:16]}"
        self._item_id = f"item_{uuid.uuid4().hex[:16]}"
        self._response_opened_by_tool = opened_by_tool
        self._transcript_parts = []
        self._out = None
        self.n_turns += 1

        await self.send(
            "response.created",
            response={"id": self._response_id, "object": "realtime.response",
                      "status": "in_progress", "status_details": None, "output": []},
        )
        await self.send(
            "response.output_item.added", response_id=self._response_id, output_index=0,
            item={"id": self._item_id, "object": "realtime.item",
                  "type": "message", "role": "assistant"},
        )
        await self.send(
            "response.content_part.added", response_id=self._response_id,
            item_id=self._item_id, output_index=0, content_index=0,
            part={"type": "audio"},
        )

    async def _finish_response(self, ev: dict) -> None:
        response_id, item_id = self._response_id, self._item_id
        if response_id is None or item_id is None:
            return

        transcript = ev.get("text") or "".join(self._transcript_parts)
        await self.send(
            "response.output_audio_transcript.done",
            response_id=response_id, item_id=item_id,
            output_index=0, content_index=0, transcript=transcript,
        )
        await self._flush_audio(response_id, item_id)
        await self.send(
            "response.output_audio.done", response_id=response_id,
            item_id=item_id, output_index=0, content_index=0,
        )
        await self.send(
            "response.content_part.done", response_id=response_id, item_id=item_id,
            output_index=0, content_index=0, part={"type": "audio"},
        )
        await self.send(
            "response.output_item.done", response_id=response_id, output_index=0,
            item={"id": item_id, "object": "realtime.item",
                  "type": "message", "role": "assistant"},
        )
        reason = ev.get("reason", "completed")
        status = "cancelled" if reason == "superseded" else "completed"
        await self.send(
            "response.done",
            response={
                "id": response_id, "object": "realtime.response",
                "status": status, "status_details": None, "output": [],
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": int(ev.get("frames", 0)),
                    "total_tokens": int(ev.get("frames", 0)),
                    "input_token_details": {"cached_tokens": 0},
                    "output_token_details": {
                        "text_tokens": 0,
                        "audio_tokens": int(ev.get("spoken", 0)),
                    },
                },
                "metrics": {
                    "frames": ev.get("frames"),
                    "timeline_frame": ev.get("t"),
                    "reason": reason,
                },
            },
        )
        self._response_id = None
        self._item_id = None
        self._response_opened_by_tool = False
        self._transcript_parts = []

    async def _on_engine_event(self, ev: dict) -> None:
        kind = ev.get("kind")
        if kind == "tool_call_start":
            self._tool_active = True
            self._model_buf = np.zeros(0, dtype=np.float32)
            dropped = 0
            while True:
                try:
                    self._audio_queue.get_nowait()
                    dropped += 1
                except asyncio.QueueEmpty:
                    break
            log.info("session %s: tool call started; discarded %d queued frames",
                     self.id, dropped)
        elif kind == "response_start":
            if self._response_id is not None and self._response_opened_by_tool:
                self._response_opened_by_tool = False
            else:
                await self._begin_response(ev)
        elif kind == "assistant_text_delta":
            if self._response_id and self._item_id:
                delta = ev.get("delta", "")
                self._transcript_parts.append(delta)
                await self.send(
                    "response.output_audio_transcript.delta",
                    response_id=self._response_id, item_id=self._item_id,
                    output_index=0, content_index=0, delta=delta,
                )
        elif kind == "audio_delta":
            if self._response_id and self._item_id:
                await self._send_audio(ev.get("path", ""),
                                       self._response_id, self._item_id)
        elif kind == "tool_call":
            if self._response_id is None or self._item_id is None:
                await self._begin_response(ev, opened_by_tool=True)
            log.info("session %s: relaying tool call: %s",
                     self.id, ev.get("text", ""))
            for call in parse_tool_call(ev.get("text", "")):
                call_id = f"call_{uuid.uuid4().hex[:16]}"
                await self.send(
                    "response.function_call_arguments.done",
                    response_id=self._response_id, item_id=self._item_id,
                    output_index=0, call_id=call_id,
                    name=call["name"], arguments=call["arguments"],
                )
        elif kind == "tool_response":
            log.info("session %s: injecting tool response (%s tokens)",
                     self.id, ev.get("tokens"))
        elif kind == "tool_response_end":
            self._tool_active = False
            log.info("session %s: tool response splice complete", self.id)
        elif kind == "response_end":
            await self._finish_response(ev)
        elif kind == "warning":
            log.warning("session %s at frame %s: %s",
                        self.id, ev.get("t"), ev.get("message"))

    # ----------------------------------------------------- legacy turn mode

    async def _run_turn(self, utterance: np.ndarray) -> None:
        response_id = f"resp_{uuid.uuid4().hex[:16]}"
        item_id = f"item_{uuid.uuid4().hex[:16]}"
        self.n_turns += 1
        n = self.n_turns

        wav_in = str(self.cfg.scratch / f"{self.id}-{n}-in.wav")
        # With streaming on, the speech arrives in `audio_delta` events during
        # the turn and there is no final wav to ask for — requesting one would
        # only make the engine decode the whole turn a second time.
        wav_out = (
            "" if self.cfg.stream_frames > 0
            else str(self.cfg.scratch / f"{self.id}-{n}-out.wav")
        )

        # The encoder is fixed at 16 kHz; the client speaks 24.
        write_wav(wav_in, resample(utterance, CLIENT_RATE, MODEL_IN_RATE), MODEL_IN_RATE)

        await self.send(
            "response.created",
            response={"id": response_id, "object": "realtime.response",
                      "status": "in_progress", "status_details": None, "output": []},
        )
        await self.send(
            "response.output_item.added", response_id=response_id, output_index=0,
            item={"id": item_id, "object": "realtime.item",
                  "type": "message", "role": "assistant"},
        )
        await self.send(
            "response.content_part.added", response_id=response_id, item_id=item_id,
            output_index=0, content_index=0, part={"type": "audio"},
        )

        transcript_parts: list[str] = []
        self._out = None

        async def on_event(ev: dict) -> None:
            kind = ev.get("kind")
            if kind == "assistant_text_delta":
                delta = ev.get("delta", "")
                transcript_parts.append(delta)
                # Text runs ahead of speech here — it is generated as the
                # timeline advances, while the wav only exists once the turn
                # is over. Clients get the words first and the voice after.
                await self.send(
                    "response.output_audio_transcript.delta",
                    response_id=response_id, item_id=item_id,
                    output_index=0, content_index=0, delta=delta,
                )
            elif kind == "audio_delta":
                # Speech for part of the turn, decoded while the rest is still
                # being generated. This is what lets the client start playing
                # before the model has finished talking.
                await self._send_audio(ev.get("path", ""), response_id, item_id)
            elif kind == "tool_call":
                for call in parse_tool_call(ev.get("text", "")):
                    call_id = f"call_{uuid.uuid4().hex[:16]}"
                    await self.send(
                        "response.function_call_arguments.done",
                        response_id=response_id, item_id=item_id, output_index=0,
                        call_id=call_id, name=call["name"], arguments=call["arguments"],
                    )
            elif kind == "warning":
                log.warning("turn %d: %s", n, ev.get("message"))
            elif kind == "stream_compare":
                log.info("streaming perception comparison: %s", ev)

        try:
            end = await self.engine.turn(wav_in, wav_out, on_event)
        except asyncio.CancelledError:
            raise
        except EngineError as e:
            await self.error(f"turn failed: {e}", "turn_failed")
            await self.send(
                "response.done",
                response={"id": response_id, "object": "realtime.response",
                          "status": "failed",
                          "status_details": {"error": {"message": str(e)}},
                          "output": []},
            )
            return
        finally:
            for path in (wav_in,):
                try:
                    os.unlink(path)
                except OSError:
                    pass

        transcript = end.get("text") or "".join(transcript_parts)
        await self.send(
            "response.output_audio_transcript.done",
            response_id=response_id, item_id=item_id,
            output_index=0, content_index=0, transcript=transcript,
        )

        # When streaming, every frame has already gone out as an audio_delta
        # and the engine wrote no final wav.
        if wav_out:
            await self._send_audio(wav_out, response_id, item_id)
        await self._flush_audio(response_id, item_id)

        await self.send(
            "response.output_audio.done", response_id=response_id,
            item_id=item_id, output_index=0, content_index=0,
        )
        await self.send(
            "response.content_part.done", response_id=response_id, item_id=item_id,
            output_index=0, content_index=0, part={"type": "audio"},
        )
        await self.send(
            "response.output_item.done", response_id=response_id, output_index=0,
            item={"id": item_id, "object": "realtime.item",
                  "type": "message", "role": "assistant"},
        )
        await self.send(
            "response.done",
            response={
                "id": response_id, "object": "realtime.response",
                "status": "completed", "status_details": None, "output": [],
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": int(end.get("frames", 0)),
                    "total_tokens": int(end.get("frames", 0)),
                    "input_token_details": {"cached_tokens": 0},
                    "output_token_details": {
                        "text_tokens": 0,
                        "audio_tokens": int(end.get("spoken", 0)),
                    },
                },
                # Not in NVIDIA's schema; the generation time is the one number
                # worth having on a box where a turn takes seconds, not ms.
                "metrics": {
                    "generation_ms": end.get("ms"),
                    "frames": end.get("frames"),
                    "timeline_frame": end.get("t"),
                },
            },
        )

        # The timeline is capped by SESSION_SECONDS; tell the client how close
        # it is rather than letting the turn that overruns it simply warn.
        used = end.get("t")
        if isinstance(used, int):
            cap = self.cfg.session_seconds * 12.5
            if used > cap * 0.85:
                log.warning("session %s: %d/%d frames of timeline used",
                            self.id, used, int(cap))

    async def _send_audio(self, wav_out: str, response_id: str, item_id: str) -> None:
        """Read one wav of speech, resample it to the client rate, chunk it out.

        Used both for a streamed audio_delta and for the whole-turn wav when
        streaming is off; the file is unlinked either way.
        """
        if not wav_out:
            return
        try:
            speech, rate = read_wav(wav_out)
        except (OSError, ValueError) as e:
            log.warning("no output wav: %s", e)
            return
        finally:
            try:
                os.unlink(wav_out)
            except OSError:
                pass

        if speech.size == 0:
            return

        if self._out is None:
            self._out = StreamResampler(rate, CLIENT_RATE)
        await self._emit_pcm(self._out.push(speech), response_id, item_id)

    async def _flush_audio(self, response_id: str, item_id: str) -> None:
        """Release the resampler's tail at the end of the turn."""
        if self._out is None:
            return
        await self._emit_pcm(self._out.flush(), response_id, item_id)
        self._out = None

    async def _emit_pcm(self, out, response_id: str, item_id: str) -> None:
        for i in range(0, out.size, OUT_CHUNK):
            await self.send(
                "response.output_audio.delta",
                response_id=response_id, item_id=item_id,
                output_index=0, content_index=0,
                delta=base64.b64encode(float_to_pcm16(out[i : i + OUT_CHUNK])).decode(),
            )


# -------------------------------------------------------------------- server


class Bridge:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.engine = VoiceChatEngine(
            binary=cfg.binary,
            model=cfg.model,
            mmproj=cfg.mmproj,
            tts=cfg.tts,
            n_gpu_layers=cfg.n_gpu_layers,
            session_seconds=cfg.session_seconds,
            extra_decoding_seconds=cfg.extra_decoding_seconds,
            temp=cfg.temp,
            stream_frames=cfg.stream_frames,
            tool_timeout=cfg.tool_timeout,
            scratch=str(cfg.scratch),
        )
        self.active: Session | None = None
        self.total_turns = 0
        self.failures = 0

    # --------------------------------------------------------------- routing

    def http(self, connection: ServerConnection, request) -> Response | None:
        path = request.path.split("?")[0].rstrip("/") or "/"

        if path in ("/v1/realtime", "/realtime"):
            return None  # let the WebSocket handshake proceed

        if path == "/":
            return self._json(HTTPStatus.OK, {
                "service": SERVICE,
                "version": VERSION,
                "websocket": "/v1/realtime",
                "health": "/v1/realtime/health",
                "loopback_mode": False,
                "backend": "llama-voicechat",
                "model_name": MODEL_NAME,
                # Which speech encoder is loaded decides whether this server
                # understands anything but English, and nothing else in the
                # protocol says so.
                "asr_model": self.cfg.asr_model,
            })

        if path == "/v1/realtime/health":
            ready = self.engine.ready.is_set() and self.engine.alive
            body = {
                "status": "ok" if ready else "error",
                "service": "nemotron-voicechat-websocket-server",
                # NVIDIA reports "triton" here. Saying so would be a lie that
                # a client could act on, so this one says what it is.
                "mode": "llama-voicechat",
                "backend_status": "ready" if ready else "loading",
                "model_inference_stats": {
                    "success_count": self.total_turns,
                    "fail_count": self.failures,
                },
                "session_active": self.active is not None,
            }
            if not self.engine.alive and self.engine.failed:
                body["error"] = self.engine.failed
            return self._json(
                HTTPStatus.OK if ready else HTTPStatus.SERVICE_UNAVAILABLE, body
            )

        return self._json(HTTPStatus.NOT_FOUND, {"error": f"no route {path}"})

    @staticmethod
    def _json(status: HTTPStatus, payload: dict) -> Response:
        body = (json.dumps(payload) + "\n").encode()
        headers = Headers({
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        })
        return Response(status.value, status.phrase, headers, body)

    # -------------------------------------------------------------- sessions

    async def handle(self, ws: ServerConnection) -> None:
        if not (self.engine.ready.is_set() and self.engine.alive):
            await ws.close(1013, "model is not loaded yet")
            return

        # One timeline, so one session. Refusing is better than interleaving
        # two conversations into one KV cache, which is what sharing would be.
        if self.active is not None:
            await ws.send(json.dumps({
                "type": "error",
                "event_id": f"event_{uuid.uuid4().hex[:16]}",
                "error": {
                    "type": "invalid_request_error",
                    "code": "session_in_use",
                    "message": "another session is active; this server holds one "
                               "conversation at a time",
                },
            }))
            await ws.close(1013, "session in use")
            return

        session = Session(ws, self.engine, self.cfg)
        self.active = session
        log.info("session %s open from %s", session.id, ws.remote_address)
        try:
            await session.run()
        except websockets.ConnectionClosed:
            log.info("session %s: client disconnected", session.id)
        except Exception:
            self.failures += 1
            log.exception("session %s failed", session.id)
        finally:
            await session.close()
            self.total_turns += session.n_turns
            if self.active is session:
                self.active = None
            log.info("session %s closed after %d turns", session.id, session.n_turns)
            # A conversation ends with the socket, so the next one starts from
            # a clean timeline rather than inheriting this one's state.
            try:
                if self.engine.alive:
                    await self.engine.reset()
            except EngineError as e:
                log.error("reset after session failed: %s", e)


async def main() -> int:
    cfg = Config()

    log.info("asr encoder: %s (%s)", cfg.asr_model, Path(cfg.mmproj).name)

    for path in (cfg.binary, cfg.model, cfg.mmproj, cfg.tts):
        if not Path(path).exists():
            log.error("missing: %s", path)
            log.error("the four converted files are made by convert.sh; see README")
            if path == cfg.mmproj:
                log.error("ASR_MODEL=%s was not the one convert.sh last built",
                          cfg.asr_model)
            return 1

    cfg.scratch.mkdir(parents=True, exist_ok=True)
    bridge = Bridge(cfg)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    async with serve(
        bridge.handle,
        cfg.host,
        cfg.port,
        process_request=bridge.http,
        # Audio deltas are small, but a long turn's transcript plus the
        # handshake should not be able to trip a default cap.
        max_size=16 * 1024 * 1024,
        ping_interval=20,
        ping_timeout=60,
    ):
        log.info("listening on %s:%d", cfg.host, cfg.port)
        # Eager load: by the time the health endpoint says ok, the weights are
        # resident and the first caller does not pay for the load.
        try:
            await bridge.engine.start()
        except (EngineError, OSError) as e:
            log.error("llama-voicechat failed to start: %s", e)
            return 1
        log.info("model loaded, ready for sessions")
        await stop.wait()
        log.info("shutting down")
        await bridge.engine.stop()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
