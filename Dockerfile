# llama-voicechat.cpp (sansamour fork) — runs NemotronLabs VoiceChat 11B, plus
# the Realtime WebSocket bridge that puts an NVIDIA-compatible API in front of
# it.
#
# Build the Linux runtime from the pinned fork. NVIDIA's official container uses
# the original fp32 checkpoint; this image runs the converted Q8_0 files.

ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=24.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build

# Pinned commit — see VC_REF in .env for how to move it.
ARG VC_REF
# 86 = Ampere sm_86. Building one architecture keeps compile time manageable.
ARG CUDA_ARCH=86

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build build-essential gcc-14 g++-14 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# nvcc 12.8 refuses gcc-15; Ubuntu 24.04 defaults to gcc-13, which is too old
# for parts of the tree. gcc-14 satisfies both.
ENV CC=gcc-14 CXX=g++-14 CUDAHOSTCXX=g++-14

WORKDIR /src
RUN git clone --filter=blob:none -b voicechat \
        https://github.com/sansamour/llama-voicechat.cpp.git . \
    && git checkout ${VC_REF}

# Incremental speech output. Without it a turn's wav is written once, at
# turn_end, which is after the tts drain — the longest phase of a turn — so a
# client hears nothing for several seconds and then the whole answer at once.
# See patches/stream-audio.patch for the measurements and the mechanism.
COPY patches/stream-audio.patch /tmp/
RUN git apply /tmp/stream-audio.patch

# The fork conditions the system prompt in the right place, but does so through
# its single-frame generation batch.  Preserve the same timeline/KV positions
# while submitting the conditioning prefix as a logical llama.cpp prefill batch.
COPY patches/system-prefill.patch /tmp/
RUN git apply /tmp/system-prefill.patch

# Cache-aware perception encoding and the persistent 12.5 Hz duplex timeline.
# This stays as a deployment patch so the fork pin remains explicit and easy to
# compare with upstream.
COPY patches/full-duplex.patch /tmp/
RUN git apply /tmp/full-duplex.patch

# GGML_CUDA_CUB_3DOT2 fetches CCCL 3.2, which a CUDA 12 toolkit does not bundle
# and the build needs; CUDA 13 already has it, so drop the flag if CUDA_VERSION
# moves to 13. LLAMA_CURL=OFF because nothing here downloads models at runtime —
# convert.sh does that on the host. GGML_NATIVE=OFF keeps this host_s CPU flags
# out of the image; that matters more here than it does for a pure-GPU server,
# because the TTS stage keeps three hot loops on the CPU (see README), but a
# portable image is still worth more than the few percent.
#
# --allow-shlib-undefined is not optional: libggml-cuda.so calls the CUDA
# *driver* API (cuMemCreate, cuMemAddressReserve, ...) which lives in
# libcuda.so.1, shipped with the host driver rather than the toolkit. It is
# absent at build time and injected by nvidia-container-toolkit at run time.
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
        -DGGML_CUDA_GRAPHS=ON \
        -DGGML_CUDA_CUB_3DOT2=ON \
        -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_UI=OFF \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    && cmake --build build --target llama-voicechat -j"$(nproc)"

RUN mkdir -p /out/lib \
    && find build -name "*.so*" -exec cp -P {} /out/lib \;

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl ca-certificates python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=build /src/build/bin/llama-voicechat /app/
COPY --from=build /out/lib/ /app/lib/
ENV LD_LIBRARY_PATH=/app/lib

# The bridge. Pure stdlib except websockets + numpy: it owns the Realtime
# protocol, the turn detector and the resamplers, and drives llama-voicechat
# --serve as a subprocess over its json line protocol.
RUN python3 -m venv /app/venv \
    && /app/venv/bin/pip install --no-cache-dir "websockets>=13" "numpy>=1.26"
COPY bridge/ /app/bridge/

EXPOSE 8080
ENTRYPOINT ["/app/venv/bin/python", "-u", "/app/bridge/server.py"]
