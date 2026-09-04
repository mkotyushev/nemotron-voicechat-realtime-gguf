#!/usr/bin/env bash
# Fetch the VoiceChat source container, the tokenizer the converters need, and
# the standalone ASR encoder if one is selected.
#
# Nothing downloaded here is what the server actually loads:
#
#  1. nemotron_voicechat_11b-${SRC_QUANT}.gguf  — one container holding five
#     models under NeMo tensor names, with no llama.cpp KV metadata and no
#     tokenizer. llama.cpp cannot open it. convert.sh splits it into the four
#     files the runtime wants.
#  2. ref_nano9b/ — three small json files from NVIDIA-Nemotron-Nano-9B-v2.
#     VoiceChat ships no tokenizer; it was trained from that base model, so the
#     vocab comes from there. Weights are NOT downloaded, only the tokenizer.
#  3. asr-${ASR_MODEL}/ — the standalone streaming ASR checkpoint whose encoder
#     replaces the container's, unless ASR_MODEL=container. ~2.4 GB of F32
#     safetensors, of which convert.sh keeps the encoder and drops the RNN-T
#     decoder, the joint and the language prompt projector.
set -euo pipefail

cd "$(dirname "$0")"
# shellcheck disable=SC1091
source .env

export HF_HOME=${HF_HOME:-$(pwd)/.cache/huggingface}
export HF_HUB_ENABLE_HF_TRANSFER=1

HF="$(dirname "$0")/.venv/bin/hf"
SRC_REPO=hoidhxd/NVIDIA-NemotronLabs-VoiceChat-11B-GGUF
REF_REPO=nvidia/NVIDIA-Nemotron-Nano-9B-v2
DEST=${MODEL_DIR:-$(pwd)/models}
Q=${SRC_QUANT:-Q8_0}
ASR=${ASR_MODEL:-container}

case "$ASR" in
    multilingual) ASR_REPO=nvidia/nemotron-3.5-asr-streaming-0.6b ;;
    en)           ASR_REPO=nvidia/nemotron-speech-streaming-en-0.6b ;;
    container)    ASR_REPO= ;;
    # Anything else is a checkpoint built here rather than published --
    # Any other name is a local checkpoint, so there is nothing to fetch.
    *)            ASR_REPO=
                  echo "ASR_MODEL=$ASR is not published; expecting it at ${ASR_DIR:-$DEST/asr-$ASR}" ;;
esac

echo "Source : $SRC_REPO  (nemotron_voicechat_11b-${Q}.gguf)"
echo "ASR    : ${ASR_REPO:-none, using the encoder inside the container}"
echo "Dest   : $DEST"

mkdir -p "$DEST"

"$HF" download "$SRC_REPO" "nemotron_voicechat_11b-${Q}.gguf" --local-dir "$DEST"

# Tokenizer only — three json files, no weights. The 9B base model repo is
# ~18 GB and none of it is needed here.
"$HF" download "$REF_REPO" config.json tokenizer.json tokenizer_config.json \
    --local-dir "$DEST/ref_nano9b"

# The standalone encoder. convert.sh reads its hyper-parameters out of
# config.json; processor_config.json is not read, but it is the record of the
# featurizer settings and, for the multilingual model, of the language prompt
# ids. The .nemo and .gguf in the same repos are other runtimes packaging the
# same weights and are not fetched.
if [ -n "$ASR_REPO" ]; then
    "$HF" download "$ASR_REPO" config.json model.safetensors processor_config.json \
        --local-dir "$DEST/asr-${ASR}"
fi

echo
echo "Done. Next: ./convert.sh"
ls -la "$DEST"
