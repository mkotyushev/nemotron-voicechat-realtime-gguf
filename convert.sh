#!/usr/bin/env bash
# Split the VoiceChat container into the four files llama-voicechat loads.
#
# The published GGUF is one file holding five models under NeMo tensor names,
# with no llama.cpp KV metadata and no tokenizer. llama.cpp cannot open it:
#
#     unknown model architecture: 'nemotron_voicechat'
#
# Three converters in the fork split it into pieces llama.cpp already
# understands — a nemotron_h model, an mtmd audio projector, and two side GGUFs
# the tool reads itself. Nothing is requantized; quantized tensors are copied
# block for block and stay bit-identical.
#
# Runs on the host, not in the container: it is pure numpy plus the repo's own
# gguf-py, and keeping it out of the image means the image does not have to
# carry a 12 GB source file it never reads at run time.
#
# ~15 minutes and ~9 GB of output for Q8_0. Idempotent — it skips a stage whose
# output is already there. Pass --force to redo them.
set -euo pipefail

cd "$(dirname "$0")"
# shellcheck disable=SC1091
source .env

HERE=$(pwd)
PY="$HERE/.venv/bin/python"
DEST=${MODEL_DIR:-$HERE/models}
Q=${SRC_QUANT:-Q8_0}
ASR=${ASR_MODEL:-container}
# Any name is allowed. It picks the mmproj file and, unless ASR_DIR says
# otherwise, the checkpoint directory beside the other models -- which is how a
# locally built encoder is used without renaming anything. `container` is the
# one special case: it means the encoder inside the VoiceChat checkpoint, which
# is not a directory at all.
if [ "$ASR" != container ]; then
    ASR_DIR=${ASR_DIR:-$DEST/asr-${ASR}}
fi

# The checkout lives on the bulk disk with everything else large.
WORK=${CONVERT_WORK:-$HERE/.cache/llama-voicechat.cpp}

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

SRC="$DEST/nemotron_voicechat_11b-${Q}.gguf"
if [ ! -f "$SRC" ]; then
    echo "missing $SRC — run ./download.sh first" >&2
    exit 1
fi

# ------------------------------------------------------------------ the fork
#
# Pinned to VC_REF, the same commit the image is built from. The converters and
# the runtime have to agree on the gguf layout, so they must not drift apart.
if [ ! -d "$WORK/.git" ]; then
    echo "== cloning llama-voicechat.cpp @ ${VC_REF:0:12} =="
    mkdir -p "$(dirname "$WORK")"
    git clone --filter=blob:none -b voicechat \
        https://github.com/sansamour/llama-voicechat.cpp.git "$WORK"
fi

git -C "$WORK" checkout --quiet --force "$VC_REF"
git -C "$WORK" clean -qfd tools/voicechat

# See patches/q8_0-converters.patch for what this does and why. It is a no-op
# for a Q4_0 source, so it is applied either way rather than conditionally.
echo "== patching converters for ${Q} =="
git -C "$WORK" apply "$HERE/patches/q8_0-converters.patch"

# ------------------------------------------------------------------- convert
LLM="$DEST/nemotron_voicechat_11b-stt-llm-${Q}.gguf"
TTS="$DEST/voicechat-tts-${Q}.gguf"
REF="$DEST/ref_nano9b"
# One name per ASR_MODEL, so switching back and forth does not rebuild anything
# and bridge/server.py can find the right one from the same variable. The
# container's keeps the name it has always had.
if [ "$ASR" = container ]; then
    MMPROJ="$DEST/mmproj-voicechat-perception-${Q}.gguf"
else
    MMPROJ="$DEST/mmproj-asr-${ASR}-${Q}.gguf"
fi
# bridge/server.py builds the same name from the same variable. If the two ever
# disagree the container starts and then cannot find its encoder.

run_stage() {
    local out=$1 name=$2; shift 2
    if [ -f "$out" ] && [ "$FORCE" -eq 0 ]; then
        echo "== $name: $out exists, skipping (--force to redo) =="
        return
    fi
    echo "== $name =="
    "$PY" "$@"
}

# Stage 1 also writes "<output>-function-head.gguf" beside its output: the
# turn-taking / tool-call head, which llama.cpp would reject inside a
# nemotron_h file. llama-voicechat finds it by name, so do not rename either.
run_stage "$LLM" "stage 1: STT language model" \
    "$WORK/tools/voicechat/convert_voicechat_to_nemotron_h.py" \
    "$SRC" --ref-dir "$REF" -o "$LLM"

# Stage 2 has two sources. The container's own encoder goes through the fork's
# converter; a standalone streaming ASR checkpoint goes through ours, which
# writes the same mmproj layout but takes the encoder from safetensors and keeps
# only `proj` and the mel featurizer from the container. See ASR_MODEL in .env
# and the converter's docstring.
if [ "$ASR" = container ]; then
    run_stage "$MMPROJ" "stage 2: perception encoder (container)" \
        "$WORK/tools/voicechat/convert_voicechat_perception_to_mmproj.py" \
        "$SRC" -o "$MMPROJ"
else
    if [ ! -f "$ASR_DIR/model.safetensors" ]; then
        echo "missing $ASR_DIR/model.safetensors —" >&2
        echo "  published encoders come from ./download.sh; custom checkpoints use ASR_DIR" >&2
        exit 1
    fi
    # A checkpoint that carries its own proj needs nothing from the container,
    # so only pass one when it does not. The converter says so too, but deciding
    # it here keeps the 12 GB file off the command line entirely.
    ASR_ARGS=()
    grep -qa '"proj.weight"' "$ASR_DIR/model.safetensors" || ASR_ARGS+=(--container "$SRC")
    run_stage "$MMPROJ" "stage 2: perception encoder (${ASR})" \
        "$HERE/convert_asr_to_mmproj.py" \
        --asr-dir "$ASR_DIR" --quant "$Q" \
        --work "$WORK" "${ASR_ARGS[@]}" -o "$MMPROJ"
fi

run_stage "$TTS" "stage 3: speech generator + codec" \
    "$WORK/tools/voicechat/convert_voicechat_tts_to_gguf.py" \
    "$SRC" --ref-dir "$REF" -o "$TTS"

echo
echo "== done =="
ls -la "$DEST"/*"${Q}"*.gguf

missing=0
for f in "$LLM" "${LLM%.gguf}-function-head.gguf" "$MMPROJ" "$TTS"; do
    [ -f "$f" ] || { echo "MISSING: $f" >&2; missing=1; }
done
[ "$missing" -eq 0 ] || exit 1

echo
echo "All four files present. Next: docker compose up -d"
