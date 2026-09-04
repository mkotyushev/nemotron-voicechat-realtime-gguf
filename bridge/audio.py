"""Sample-rate conversion and WAV I/O for the VoiceChat bridge.

Three rates meet here and none of them agree:

    24000  the client's rate, in both directions (what the NVIDIA container
           negotiates, so what our clients already speak)
    16000  the FastConformer encoder's rate, fixed
    22050  the codec's output rate, fixed at 12.5 Hz x 1764 samples

numpy only. scipy would give us `resample_poly` but it is 100 MB of wheel for
one function, and this one is short enough to read.
"""

from __future__ import annotations

import wave
from collections import deque

import numpy as np

CLIENT_RATE = 24000
MODEL_IN_RATE = 16000
MODEL_OUT_RATE = 22050


def resample(x: np.ndarray, sr_in: int, sr_out: int, taps: int = 16) -> np.ndarray:
    """Band-limited resample by windowed-sinc interpolation.

    Cost is O(n_out * taps), not O(n * lcm(rates)) — which matters because
    22050 -> 24000 is 160/147, and the textbook upsample-filter-decimate would
    zero-stuff a second of audio out to 3.5 M samples before filtering it back
    down.

    When downsampling, the sinc is stretched by the rate ratio so its first
    null lands on the *output* Nyquist rather than the input one; that stretch
    is the anti-alias filter. Upsampling needs no such thing, so `cutoff` is
    clamped at 1.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if sr_in == sr_out or x.size == 0:
        return x.astype(np.float32)

    ratio = sr_out / sr_in
    n_out = int(round(x.size * ratio))
    if n_out == 0:
        return np.zeros(0, dtype=np.float32)

    cutoff = min(1.0, ratio)
    half = int(np.ceil(taps / cutoff))

    # Where each output sample sits in input-sample coordinates.
    t = np.arange(n_out, dtype=np.float64) / ratio
    base = np.floor(t).astype(np.int64)

    padded = np.pad(x, (half, half))
    out = np.zeros(n_out, dtype=np.float64)

    for k in range(-half + 1, half + 1):
        n = base + k
        d = t - n
        # Hann window over the kernel span; zero outside it, which np.sinc
        # would not give us on its own.
        w = 0.5 + 0.5 * np.cos(np.pi * np.clip(d / half, -1.0, 1.0))
        out += np.sinc(d * cutoff) * w * padded[n + half]

    return (out * cutoff).astype(np.float32)


class StreamResampler:
    """Resample a stream block by block, with no seam between blocks.

    Calling `resample` on each block separately does not work once the speech
    is streamed: it zero-pads both edges of every block, so each boundary loses
    the kernel's worth of real context (a ~0.7 ms tick, once per chunk), and
    each block's length is rounded independently, so the pieces do not add up
    to the resampled whole.

    This keeps the kernel's left/right context and the fractional read position
    across calls, so the concatenated output is the same as resampling the
    whole stream at once. It costs one block of latency at the tail, which
    `flush` releases.
    """

    def __init__(self, sr_in: int, sr_out: int, taps: int = 16) -> None:
        self.sr_in = sr_in
        self.sr_out = sr_out
        self.ratio = sr_out / sr_in
        self.cutoff = min(1.0, self.ratio)
        self.half = int(np.ceil(taps / self.cutoff))
        # The true stream starts with silence. Keeping that left padding in the
        # buffer avoids negative numpy indices wrapping around to the newest
        # samples during the first block.
        self._buf = np.zeros(self.half, dtype=np.float64)
        self._base = -self.half  # global input index of _buf[0]
        self._k = 0       # next output index to produce
        self._n_input = 0

    def _run(self, limit: int) -> np.ndarray:
        """Produce output samples whose kernels fit entirely inside _buf."""
        if limit <= self._k:
            return np.zeros(0, dtype=np.float32)

        k = np.arange(self._k, limit, dtype=np.float64)
        t = k / self.ratio - self._base
        base = np.floor(t).astype(np.int64)

        out = np.zeros(t.size, dtype=np.float64)
        for j in range(-self.half + 1, self.half + 1):
            n = base + j
            d = t - n
            w = 0.5 + 0.5 * np.cos(np.pi * np.clip(d / self.half, -1.0, 1.0))
            out += np.sinc(d * self.cutoff) * w * self._buf[n]

        self._k = limit
        return (out * self.cutoff).astype(np.float32)

    def push(self, x: np.ndarray) -> np.ndarray:
        """Feed a block, get back every output sample that is now certain."""
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size:
            self._buf = np.concatenate([self._buf, x])
            self._n_input += x.size

        # The newest output we can finish is the one whose kernel still ends
        # inside what we have.
        last_in = self._base + self._buf.size - 1 - self.half
        limit = int(np.floor(last_in * self.ratio)) + 1
        out = self._run(max(self._k, limit))

        # Drop input the next output can no longer reach back to.
        need = int(np.floor(self._k / self.ratio)) - self.half
        drop = max(0, need - self._base)
        if drop > 0:
            self._buf = self._buf[drop:]
            self._base += drop
        return out

    def flush(self) -> np.ndarray:
        """Release the tail, zero-padding only at the true end of the stream."""
        self._buf = np.concatenate([self._buf, np.zeros(self.half + 1)])
        limit = int(round(self._n_input * self.ratio))
        out = self._run(max(self._k, limit))
        self._buf = np.zeros(self.half, dtype=np.float64)
        self._base = -self.half
        self._k = 0
        self._n_input = 0
        return out


def pcm16_to_float(buf: bytes) -> np.ndarray:
    """Little-endian signed 16-bit PCM -> float32 in [-1, 1)."""
    return np.frombuffer(buf, dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm16(x: np.ndarray) -> bytes:
    """float32 -> little-endian signed 16-bit PCM, clipped rather than wrapped."""
    y = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0 - 1.0 / 32768.0)
    return (y * 32768.0).astype("<i2").tobytes()


def write_wav(path: str, x: np.ndarray, rate: int) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(float_to_pcm16(x))


def read_wav(path: str) -> tuple[np.ndarray, int]:
    """Read a mono/stereo 16-bit WAV as float32. Returns (samples, rate)."""
    with wave.open(path, "rb") as w:
        rate = w.getframerate()
        channels = w.getnchannels()
        width = w.getsampwidth()
        raw = w.readframes(w.getnframes())

    if width != 2:
        raise ValueError(f"{path}: expected 16-bit PCM, got {width * 8}-bit")

    x = pcm16_to_float(raw)
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    return x, rate


class TurnDetector:
    """Tracks speech epochs for UI and response-lifecycle hints.

    The NVIDIA container gets these events from the model's own ASR channel.
    The GGUF conversion omits that decoder, so the bridge supplies a small
    energy gate instead. It never buffers, drops, or withholds an input frame
    and it never decides when a response may start.

    Energy gate against a noise floor, the same shape as the client's own
    barge-in gate: a fixed absolute threshold does not survive a change of
    microphone. Two details are load-bearing, and both were found by getting
    them wrong first.

    **The floor is a rolling minimum, not an average.** An exponential average
    measured only while quiet stops adapting the moment the gate opens, so on a
    noisy stream the gate opens on the first chunk and then never closes. The
    minimum over a few seconds is the silence between words, whatever the gate
    currently believes.

    **`min_level` is a floor on the threshold, not on the signal.** With no
    calibration window a stream that opens on speech would otherwise set an
    impossibly high bar for itself. A live client streams room tone before
    anyone talks, but a wav fed from a file does not.
    """

    def __init__(
        self,
        rate: int = CLIENT_RATE,
        silence_ms: int = 700,
        margin: float = 3.5,
        min_speech_ms: int = 350,
        max_turn_ms: int = 30000,
        preroll_ms: int = 300,
        min_level: float = 0.01,
        floor_window_ms: int = 3000,
    ) -> None:
        self.rate = rate
        self.silence_ms = silence_ms
        self.margin = margin
        self.min_speech_ms = min_speech_ms
        self.max_turn_ms = max_turn_ms
        self.preroll = int(rate * preroll_ms / 1000)
        self.min_level = min_level
        # The floor is the quietest chunk in the last few seconds. A rolling
        # minimum tracks a noisy room without ever being dragged up by speech,
        # which an exponential average is: once the gate opens on the first
        # chunk of a noisy stream it stops adapting and never closes again.
        self._levels: deque[float] = deque(
            maxlen=max(1, int(floor_window_ms / (1000.0 * 0.08)))
        )
        self._speaking = False
        self._silence_run = 0.0
        self._speech_ms = 0.0
        self._buf: list[np.ndarray] = []
        self._buf_ms = 0.0
        self._preroll: list[np.ndarray] = []

    @property
    def speaking(self) -> bool:
        return self._speaking

    @property
    def floor(self) -> float:
        return min(self._levels) if self._levels else 0.0

    @property
    def threshold(self) -> float:
        return max(self.floor * self.margin, self.min_level)

    def reset(self) -> None:
        """Drop the current utterance but keep the measured noise floor."""
        self._speaking = False
        self._silence_run = 0.0
        self._speech_ms = 0.0
        self._buf.clear()
        self._buf_ms = 0.0
        self._preroll.clear()

    def push(self, chunk: np.ndarray) -> tuple[str | None, np.ndarray | None]:
        """Feed one chunk. Returns (event, utterance).

        event is "started", "stopped" or None. utterance is set only with
        "stopped", and only if the utterance was long enough to be worth
        sending; a too-short one reports "stopped" with None so the client's
        indicator still settles.
        """
        ms = 1000.0 * chunk.size / self.rate
        rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0

        # Measured in both states, so a noisy room is still tracked through a
        # long utterance: the minimum over the window is the gap between words.
        self._levels.append(rms)
        loud = rms > self.threshold

        if not self._speaking:
            # Hold a rolling pre-roll so the utterance does not start clipped:
            # the gate opens a chunk or two after the onset, always.
            self._preroll.append(chunk)
            keep = 0
            total = 0
            for i in range(len(self._preroll) - 1, -1, -1):
                total += self._preroll[i].size
                keep += 1
                if total >= self.preroll:
                    break
            if keep < len(self._preroll):
                self._preroll = self._preroll[-keep:]

            if loud:
                self._speaking = True
                self._buf = list(self._preroll)
                self._buf_ms = sum(1000.0 * c.size / self.rate for c in self._buf)
                self._preroll = []
                self._silence_run = 0.0
                self._speech_ms = ms
                return "started", None

            return None, None

        self._buf.append(chunk)
        self._buf_ms += ms

        if loud:
            self._silence_run = 0.0
            self._speech_ms += ms
        else:
            self._silence_run += ms

        done = self._silence_run >= self.silence_ms or self._buf_ms >= self.max_turn_ms
        if not done:
            return None, None

        utterance = np.concatenate(self._buf) if self._buf else np.zeros(0, np.float32)
        long_enough = self._speech_ms >= self.min_speech_ms
        self.reset()
        return "stopped", (utterance if long_enough else None)
