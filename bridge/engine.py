"""Drives `llama-voicechat --serve`.

The tool speaks one JSON object per line on stdin and one event per line on
stdout; every log line stays on stderr. This wraps that in something asyncio
can await.

Two properties of the model shape this whole file:

**One timeline.** VoiceChat has no chat history to replay — there is a single
12.5 Hz timeline and the state the model's answer leaves behind is the state
the next question starts from. So there is exactly one process, exactly one
conversation in it, and `reset` is the only way back to the beginning.

**Frames are serial.** In duplex mode each command carries the next 80 ms of
PCM. The lock covers one frame, not a whole utterance, so the bridge can keep
accepting microphone audio while text and speech events come back.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from typing import Any, Callable

log = logging.getLogger("engine")


class EngineError(RuntimeError):
    pass


class VoiceChatEngine:
    def __init__(
        self,
        binary: str,
        model: str,
        mmproj: str,
        tts: str,
        *,
        n_gpu_layers: int = 99,
        session_seconds: int = 180,
        extra_decoding_seconds: int = 50,
        temp: float = 0.0,
        stream_frames: int = 8,
        tool_timeout: float = 60.0,
        scratch: str = "/tmp",
        env: dict[str, str] | None = None,
    ) -> None:
        self.stream_frames = stream_frames
        self.tool_timeout = tool_timeout
        self.argv = [
            binary,
            "-m", model,
            "--mmproj", mmproj,
            "--tts", tts,
            "-ngl", str(n_gpu_layers),
            "--session-seconds", str(session_seconds),
            "--extra-decoding-seconds", str(extra_decoding_seconds),
            "--temp", str(temp),
            "--serve",
        ]
        if stream_frames > 0:
            # Speech arrives as `audio_delta` events during the turn instead of
            # one wav at the end. Needs patches/stream-audio.patch, which the
            # image applies at build time.
            self.argv += ["--stream-audio", str(stream_frames),
                          "--stream-dir", scratch]
        self.env = {**os.environ, **(env or {})}

        self.proc: asyncio.subprocess.Process | None = None
        self.ready = asyncio.Event()
        self.failed: str | None = None

        self._events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._lock = asyncio.Lock()
        self._system_sent = False
        self._duplex_started = False
        self._frame_seq = 0
        self._turns = 0
        # Set while a turn is waiting for the driver to answer a tool call.
        self._pending_tool: asyncio.Future[str] | None = None

    # ------------------------------------------------------------------ life

    async def start(self) -> None:
        log.info("starting: %s", " ".join(self.argv))
        self.proc = await asyncio.create_subprocess_exec(
            *self.argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self.env,
        )
        asyncio.create_task(self._pump_stdout())
        asyncio.create_task(self._pump_stderr())

        # The model is loaded eagerly, before the port is reported healthy, so
        # that health means "ready to talk" rather than "the process exists".
        ev = await self._next_event(timeout=1800)
        if ev.get("kind") != "ready":
            raise EngineError(f"expected 'ready', got {ev!r}")
        log.info("ready: %s", {k: v for k, v in ev.items() if k != "kind"})
        self.ready.set()

    async def stop(self) -> None:
        if self.proc is None or self.proc.returncode is not None:
            return
        try:
            await self._send({"cmd": "quit"})
            await asyncio.wait_for(self.proc.wait(), timeout=10)
        except (asyncio.TimeoutError, ConnectionError, BrokenPipeError):
            self.proc.kill()
            await self.proc.wait()

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    @property
    def turns(self) -> int:
        return self._turns

    # ------------------------------------------------------------------ pipes

    async def _pump_stdout(self) -> None:
        assert self.proc and self.proc.stdout
        while True:
            line = await self.proc.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", "replace").strip()
            if not text:
                continue
            try:
                await self._events.put(json.loads(text))
            except json.JSONDecodeError:
                # stdout is documented as json-only in serve mode, so anything
                # else is a bug worth seeing rather than swallowing.
                log.warning("non-json on stdout: %s", text[:200])
        self.failed = self.failed or "llama-voicechat exited"
        await self._events.put({"kind": "error", "message": self.failed})

    async def _pump_stderr(self) -> None:
        assert self.proc and self.proc.stderr
        while True:
            line = await self.proc.stderr.readline()
            if not line:
                break
            log.info("vc: %s", line.decode("utf-8", "replace").rstrip())

    async def _send(self, obj: dict[str, Any]) -> None:
        if not self.alive:
            raise EngineError(self.failed or "llama-voicechat is not running")
        assert self.proc and self.proc.stdin
        self.proc.stdin.write((json.dumps(obj) + "\n").encode())
        await self.proc.stdin.drain()

    async def _next_event(self, timeout: float | None = None) -> dict[str, Any]:
        try:
            return await asyncio.wait_for(self._events.get(), timeout=timeout)
        except asyncio.TimeoutError as e:
            raise EngineError(f"no event from llama-voicechat in {timeout}s") from e

    # ----------------------------------------------------------------- driving

    async def set_system(self, text: str) -> None:
        """Write the system prompt into the perception channel.

        Only valid before the first turn. Every prompt token still occupies one
        80 ms timeline/KV position, but the patched engine conditions those
        positions as a logical prefill batch rather than generation-style one at
        a time.
        """
        async with self._lock:
            if self._system_sent:
                raise EngineError("system prompt already set; reset first")
            await self._send({"cmd": "system", "text": text})
            while True:
                ev = await self._next_event(timeout=1800)
                kind = ev.get("kind")
                if kind == "system":
                    self._system_sent = True
                    return
                if kind == "error":
                    raise EngineError(ev.get("message", "system prompt failed"))

    async def reset(self) -> None:
        """Start a new conversation on the same loaded model."""
        async with self._lock:
            await self._send({"cmd": "reset"})
            while True:
                ev = await self._next_event(timeout=120)
                if ev.get("kind") == "reset":
                    self._system_sent = False
                    self._duplex_started = False
                    self._frame_seq = 0
                    self._turns = 0
                    return
                if ev.get("kind") == "error":
                    raise EngineError(ev.get("message", "reset failed"))

    def answer_tool_call(self, output: str) -> bool:
        """Hand a tool result to the turn that is waiting for one."""
        fut = self._pending_tool
        if fut is None or fut.done():
            return False
        fut.set_result(output)
        return True

    async def _skip_tool_call(self) -> None:
        """Release the serve loop if a pending client/tool path goes away."""
        if not self.alive:
            return
        task = asyncio.create_task(self._send({"cmd": "tool_skip"}))
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=2)
        except Exception as e:
            task.cancel()
            log.warning("could not skip pending tool call: %s", e)

    async def _handle_tool_call(
        self,
        ev: dict[str, Any],
        on_event: Callable[[dict[str, Any]], Any],
    ) -> None:
        """Relay one call, then answer or release the nested serve reader."""
        loop = asyncio.get_running_loop()
        pending = loop.create_future()
        self._pending_tool = pending
        response_sent = False
        try:
            # Install the future first. A fast tool can answer from the event
            # callback before it returns.
            await on_event(ev)
            try:
                output = await asyncio.wait_for(pending, timeout=self.tool_timeout)
            except asyncio.TimeoutError as e:
                raise EngineError(
                    f"tool result did not arrive in {self.tool_timeout:g}s"
                ) from e
            await self._send({"cmd": "tool_response", "text": output})
            response_sent = True
        except asyncio.CancelledError:
            if not response_sent:
                await self._skip_tool_call()
            raise
        except Exception:
            if not response_sent:
                await self._skip_tool_call()
            raise
        finally:
            if self._pending_tool is pending:
                self._pending_tool = None

    async def start_duplex(self) -> None:
        """Start the persistent 12.5 Hz microphone timeline."""
        async with self._lock:
            if self._duplex_started:
                return
            await self._send({"cmd": "duplex_start"})
            while True:
                ev = await self._next_event(timeout=120)
                kind = ev.get("kind")
                if kind == "duplex_start":
                    self._duplex_started = True
                    return
                if kind == "error":
                    raise EngineError(ev.get("message", "duplex start failed"))

    async def frame(
        self,
        pcm16: bytes,
        on_event: Callable[[dict[str, Any]], Any],
        speech_epoch: int = 0,
        input_speaking: bool = False,
    ) -> dict[str, Any]:
        """Advance the live timeline with one block of 16 kHz PCM16."""
        async with self._lock:
            if not self._duplex_started:
                raise EngineError("duplex mode has not started")
            seq = self._frame_seq
            self._frame_seq += 1
            await self._send({
                "cmd": "audio_frame",
                "seq": seq,
                "audio": base64.b64encode(pcm16).decode("ascii"),
                "speech_epoch": speech_epoch,
                "input_speaking": input_speaking,
            })

            tool_started = False
            try:
                while True:
                    ev = await self._next_event(timeout=3600)
                    kind = ev.get("kind")
                    if kind == "error":
                        raise EngineError(ev.get("message", "duplex frame failed"))

                    if kind == "tool_call_start":
                        tool_started = True
                    if kind == "tool_call":
                        # _handle_tool_call owns cancellation from this point.
                        tool_started = False
                        await self._handle_tool_call(ev, on_event)
                    else:
                        await on_event(ev)

                    if kind == "duplex_frame" and ev.get("seq") == seq:
                        if ev.get("failed"):
                            raise EngineError("duplex frame failed")
                        return ev
            except asyncio.CancelledError:
                if tool_started:
                    await self._skip_tool_call()
                raise

    async def turn(
        self,
        wav_in: str,
        wav_out: str,
        on_event: Callable[[dict[str, Any]], Any],
    ) -> dict[str, Any]:
        """Run one turn. Awaits `on_event` for every event, returns `turn_end`.

        A tool call is handled inline: `on_event` is expected to relay the
        `tool_call` to the driver, which calls `answer_tool_call`. The model's
        clock is frozen over that region — it stops listening while it consults
        the tool — so there is no rush, but there is also no way to run the
        turn without answering.
        """
        async with self._lock:
            await self._send({"cmd": "turn", "audio": wav_in, "out": wav_out})
            self._turns += 1

            while True:
                ev = await self._next_event(timeout=3600)
                kind = ev.get("kind")

                if kind == "error":
                    raise EngineError(ev.get("message", "turn failed"))

                if kind == "tool_call":
                    await self._handle_tool_call(ev, on_event)
                else:
                    await on_event(ev)

                if kind == "turn_end":
                    return ev
