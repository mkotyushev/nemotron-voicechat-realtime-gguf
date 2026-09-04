#!/usr/bin/env python3
"""
Build the perception mmproj from a *standalone* NVIDIA streaming ASR checkpoint
instead of from the VoiceChat container.

Why this exists
---------------
The fork's stage 2 (`convert_voicechat_perception_to_mmproj.py`) lifts
`stt_model.perception` out of the 12 GB VoiceChat container. That encoder is a
causal FastConformer, and NVIDIA also publishes the same encoder standing on its
own, as the front half of a streaming RNN-T ASR model:

    nvidia/nemotron-speech-streaming-en-0.6b     English
    nvidia/nemotron-3.5-asr-streaming-0.6b       40 language-locales, its successor

Their `config.json` files agree with VoiceChat's perception config on every
number that shapes the graph — 24 layers, d_model 1024, 8 heads, ffn 4096, 128
mel bins, subsampling factor 8, conv kernel 9, no biases anywhere — and the
tensors map one to one onto the ones stage 2 writes, with the same shapes and
the same numpy orientation. So the ASR half can be swapped for the multilingual
one without touching the language model, the TTS or the codec, which is all this
converter does.

Two things do NOT come from the ASR checkpoint, because it does not have them:

  * `proj`, the 1024 -> 4480 linear into the STT LLM's embedding space. That is
    VoiceChat's, not the ASR model's — the ASR model's own `encoder_projector`
    is 1024 -> 640 into an RNN-T joint and is not the same thing. It is copied
    out of the container, block for block.
  * `preprocessor.featurizer.fb` and `.window`, the mel filterbank and the STFT
    window. The HF repos compute those in the processor rather than shipping
    them, and both models' `processor_config.json` agree with VoiceChat's
    featurizer exactly (128 mel, n_fft 512, win 400, hop 160, 16 kHz), so the
    container's are correct for either. Also copied from the container.

What is different about the swapped-in encoder, measured
--------------------------------------------------------
VoiceChat's perception encoder is not a copy of the published English one; it is
a fine-tune of it. Cosine similarity, container vs the two published encoders:

                                                VC~EN    VC~ML    EN~ML
    layers.0.self_attn.q_proj.weight            0.855    0.742    0.737
    layers.12.self_attn.v_proj.weight           0.909    0.717    0.686
    layers.23.feed_forward1.linear1.weight      0.934    0.630    0.621
    subsampling.linear.weight                   0.966    0.930    0.944

So `--asr-dir` pointing at the English model is *not* a way to reproduce the
container's own encoder — use the fork's stage 2 for that (`ASR_MODEL=container`).
And `proj`, which is kept from the container, was trained against the container's
encoder, whose output space these have drifted from. Whether the drift is small
enough for VoiceChat to still understand the multilingual encoder is an
empirical question and the reason this is a switch rather than a replacement.

The language prompt is dropped
------------------------------
The multilingual model conditions on a language ID: a 128-wide one-hot is
concatenated to each 1024-wide encoder frame and run through `prompt_projector`
(1152 -> 2048 -> 1024) before the RNN-T decoder sees it. That MLP sits *after*
the encoder, so it is not part of what VoiceChat's `proj` consumes, and the
`voicechat` graph in clip.cpp has nowhere to put it. It is left out, which means
the encoder runs unconditioned — the equivalent of the model's own `auto`
prompt, minus the projection. Folding it in for a fixed language is possible
(the one-hot collapses into `linear_1`'s bias) but needs two matmuls and a ReLU
added to the graph, so it is a C++ change, not a converter one.

Attention window
----------------
`encoder_config.sliding_window` counts the frame itself, and clip.cpp's
`attn_window_size` counts only the frames to the left, so this writes
`sliding_window - 1`: 70 for the English model, matching VoiceChat's
att_context_size [70, 0], and 56 for the multilingual one.

Usage
-----
    python convert_asr_to_mmproj.py \
        --asr-dir   /srv/.../asr-multilingual \
        --container /srv/.../nemotron_voicechat_11b-Q8_0.gguf \
        --quant     Q8_0 \
        -o          /srv/.../mmproj-asr-multilingual-Q8_0.gguf

The published safetensors are F32; `--quant` says what the 2-D weights are
written as, so Q8_0 gives a file the same size and shape as the one stage 2
produces. Everything the graph needs dense — norms, biases, the depthwise conv,
the two position biases — stays F32, and the conv2d kernels stay F16, because
that is what ggml_conv_2d wants. Same rules as stage 2.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import struct
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger("asr-mmproj")

# The fork's checkout: gguf-py, and vc_gguf for reading the container. convert.sh
# has already applied patches/q8_0-converters.patch to it, which is what lets
# GGUFSource.f32() read the Q8_0 tensors this needs dense.
DEFAULT_WORK = Path.home() / ".cache" / "llama-voicechat.cpp"


class SafeTensors:
    """Minimal safetensors reader: an 8-byte length, a JSON header, then data.

    Same reasoning as vc_gguf.GGUFSource — the file is 2.4 GB of F32 and only
    one tensor is needed at a time, so a plain seek/read keeps peak RSS at one
    tensor and avoids adding a dependency to the deployment's venv.
    """

    def __init__(self, path: Path):
        self.path = path
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.header = json.loads(self.f.read(n))
        self.header.pop("__metadata__", None)
        self.data_start = 8 + n

    def __contains__(self, name: str) -> bool:
        return name in self.header

    def f32(self, name: str) -> np.ndarray:
        """Read a tensor as F32 in numpy (PyTorch) order.

        That is the same orientation vc_gguf.GGUFSource.f32() returns — it
        reverses the GGUF dims — so every tensor here goes to the writer exactly
        the way stage 2 sends its own, and a Linear's (out, in) becomes ggml
        {in, out} on write.
        """
        m = self.header.get(name)
        if m is None:
            raise SystemExit(f"missing tensor in {self.path.name}: {name}")
        start, end = m["data_offsets"]
        self.f.seek(self.data_start + start)
        raw = self.f.read(end - start)
        if len(raw) != end - start:
            raise SystemExit(f"{name}: short read")
        dtype = m["dtype"]
        if dtype == "F32":
            a = np.frombuffer(raw, dtype=np.float32)
        elif dtype == "F16":
            a = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        elif dtype == "BF16":
            # bf16 is the top half of an f32; widen rather than lose it
            u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
            a = u.view(np.float32)
        else:
            raise SystemExit(f"{name}: unhandled safetensors dtype {dtype}")
        return a.reshape(tuple(m["shape"]))


# The five real convs inside the subsampling stack, in the order stage 2 numbers
# them: the gaps (1, 4) are ReLU. NeMo's `pre_encode.conv.{i}` is one flat
# Sequential; the HF port splits it into a stem plus two depthwise/pointwise
# pairs. Same five kernels, same shapes.
PRE_ENCODE_CONVS = (
    (0, "encoder.subsampling.conv_in"),
    (2, "encoder.subsampling.layers.0.depthwise_conv"),
    (3, "encoder.subsampling.layers.0.pointwise_conv"),
    (5, "encoder.subsampling.layers.1.depthwise_conv"),
    (6, "encoder.subsampling.layers.1.pointwise_conv"),
)

# encoder block: (mmproj name, ASR checkpoint name). Everything else in a block
# keeps its NeMo name in the HF port and is handled by the loops below.
ATTN_LINEARS = (
    ("attn_q.weight", "self_attn.q_proj.weight"),
    ("attn_k.weight", "self_attn.k_proj.weight"),
    ("attn_v.weight", "self_attn.v_proj.weight"),
    ("attn_out.weight", "self_attn.o_proj.weight"),
    # NeMo calls it linear_pos; the HF port calls the same thing relative_k_proj
    ("linear_pos.weight", "self_attn.relative_k_proj.weight"),
)

NORMS = (
    ("ln1", "norm_self_att"),
    ("ln2", "norm_out"),
    ("ffn_norm", "norm_feed_forward1"),
    ("ffn_norm_1", "norm_feed_forward2"),
    ("norm_conv", "norm_conv"),
)

FFN_LINEARS = (
    ("ffn_up.weight", "feed_forward1.linear1.weight"),
    ("ffn_down.weight", "feed_forward1.linear2.weight"),
    ("ffn_up_1.weight", "feed_forward2.linear1.weight"),
    ("ffn_down_1.weight", "feed_forward2.linear2.weight"),
)

LAYER_NORM_EPS = 1e-5
PROJ_DIM = 4480

FILE_TYPE = {
    "Q8_0": "MOSTLY_Q8_0",
    "Q4_0": "MOSTLY_Q4_0",
    "F16": "MOSTLY_F16",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asr-dir", type=Path, required=True,
                    help="directory holding the ASR model's config.json and model.safetensors")
    ap.add_argument("--container", type=Path, default=None,
                    help="nemotron_voicechat_11b-*.gguf, for proj and the mel featurizer. "
                         "Not needed when --asr-dir already carries them")
    ap.add_argument("--quant", default="Q8_0", choices=sorted(FILE_TYPE),
                    help="what the 2-D weights are written as (default: Q8_0)")
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK,
                    help="llama-voicechat.cpp checkout, for gguf-py and vc_gguf")
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")

    sys.path.insert(0, str(args.work / "gguf-py"))
    sys.path.insert(0, str(args.work / "tools" / "voicechat"))
    import gguf  # noqa: E402
    from gguf import quants  # noqa: E402
    from vc_gguf import GGML_BLOCK, GGUFSource  # noqa: E402

    qtype = None if args.quant == "F16" else getattr(gguf.GGMLQuantizationType, args.quant)

    # ------------------------------------------------------------ hyper-params
    cfg = json.loads((args.asr_dir / "config.json").read_text())
    enc = cfg["encoder_config"]

    n_layer = enc["num_hidden_layers"]
    n_embd = enc["hidden_size"]
    n_head = enc["num_attention_heads"]
    n_ff = enc["intermediate_size"]
    n_mel = enc["num_mel_bins"]
    subsampling = enc["subsampling_factor"]
    conv_kernel = enc["conv_kernel_size"]
    # clip.cpp masks on `q - k <= attn_window_size`, so it counts left context
    # only; sliding_window counts the frame itself as well.
    att_left = enc["sliding_window"] - 1

    # The `voicechat` graph in clip.cpp is built around these; a checkpoint that
    # disagreed would load and then be silently wrong, so say so here instead.
    if subsampling != 8:
        raise SystemExit(f"subsampling_factor is {subsampling}, the voicechat graph needs 8")
    if enc.get("attention_bias", False) or enc.get("convolution_bias", False):
        raise SystemExit("this encoder has linear/conv biases; the voicechat graph has none")
    if enc.get("scale_input", False):
        raise SystemExit("scale_input is set; the voicechat graph does not scale the input")

    logger.info("asr model : %s", args.asr_dir)
    logger.info("            %s, %d layers, d_model %d, %d mel, left context %d",
                cfg.get("model_type", "?"), n_layer, n_embd, n_mel, att_left)
    logger.info("2-D weights: %s", args.quant)

    st = SafeTensors(args.asr_dir / "model.safetensors")
    SRC = "stt_model.perception."
    proj_dim = PROJ_DIM

    # Where proj and the mel featurizer come from. A custom checkpoint may carry
    # both, in which case this converter needs nothing else. A published
    # checkpoint carries neither, so the container has to supply them, and the
    # proj it supplies was trained against the container's own encoder: that
    # mismatch is what makes an unaligned swap answer as if it heard silence.
    self_contained = "proj.weight" in st
    src = None
    if self_contained:
        proj_dim = int(st.header["proj.weight"]["shape"][0])
        logger.info("proj      : %s (self-contained, 1024 -> %d)", args.asr_dir, proj_dim)
        for name in ("proj.bias", "preprocessor.featurizer.fb", "preprocessor.featurizer.window"):
            if name not in st:
                raise SystemExit(
                    f"{args.asr_dir} has proj.weight but not {name}; it is half an "
                    "custom checkpoint; fix it or pass --container."
                )
    else:
        if args.container is None:
            raise SystemExit(
                f"{args.asr_dir} carries no proj, so --container is needed for it and "
                "for the mel featurizer. A custom checkpoint may carry both."
            )
        logger.info("container : %s (proj and the mel featurizer)", args.container)
        src = GGUFSource(args.container)

    enc = st.f32

    # ------------------------------------------------------------------ header
    w = gguf.GGUFWriter(path=None, arch="clip")

    w.add_type(gguf.GGUFType.MMPROJ)
    w.add_name(f"VoiceChat perception ({args.asr_dir.name})")
    w.add_description(
        f"Causal FastConformer encoder from {args.asr_dir.name}, on VoiceChat's "
        f"mel featurizer and 1024 -> {proj_dim} projection into the STT LLM "
        "embedding space. Language prompt conditioning is not included."
        + (" The projection came with the checkpoint rather than from the "
           "container; see its alignment.json." if self_contained else "")
    )

    w.add_clip_has_audio_encoder(True)
    # Must stay "voicechat": it is what selects clip_graph_voicechat, and the
    # causal downsampling, layer-norm conv and [left, 0] attention it implements
    # are exactly what these encoders are.
    w.add_clip_projector_type("voicechat")

    w.add_audio_embedding_length(n_embd)
    w.add_audio_feed_forward_length(n_ff)
    w.add_audio_block_count(n_layer)
    w.add_audio_head_count(n_head)
    w.add_audio_projection_dim(proj_dim)
    w.add_audio_attention_layernorm_eps(LAYER_NORM_EPS)
    w.add_audio_num_mel_bins(n_mel)
    w.add_audio_subsampling_factor(subsampling)
    w.add_audio_conv_kernel_size(conv_kernel)
    w.add_audio_window_size(att_left)

    w.add_file_type(getattr(gguf.LlamaFileType, FILE_TYPE[args.quant]))

    n_quant = 0
    n_copied = 0
    n_conv = 0

    def put(dst: str, arr: np.ndarray, src_name: str, dtype=np.float32) -> None:
        """Write a tensor dense, as F32 unless the graph wants F16."""
        nonlocal n_conv
        arr = np.ascontiguousarray(arr, dtype=dtype)
        w.add_tensor(dst, arr)
        n_conv += 1
        logger.debug("dense   %-52s -> %-30s %s %s", src_name, dst,
                     np.dtype(dtype).name, arr.shape)

    def put_weight(dst: str, arr: np.ndarray, src_name: str) -> None:
        """Write a 2-D weight at --quant.

        Stage 2 copies these out of the container without touching them because
        they arrive already quantized. Here the source is F32, so they are
        quantized on the way out to land at the same size.
        """
        nonlocal n_quant
        if arr.ndim != 2:
            raise SystemExit(f"{src_name}: expected a 2-D weight, got {arr.shape}")
        if qtype is None:
            put(dst, arr, src_name, dtype=np.float16)
            return
        block = gguf.GGML_QUANT_SIZES[qtype][0]
        if arr.shape[-1] % block:
            raise SystemExit(f"{src_name}: row of {arr.shape[-1]} is not a multiple of {block}")
        data = quants.quantize(np.ascontiguousarray(arr, dtype=np.float32), qtype)
        w.add_tensor(dst, data, raw_shape=data.shape, raw_dtype=qtype)
        n_quant += 1
        logger.debug("quant   %-52s -> %-30s %s %s", src_name, dst, args.quant, arr.shape)

    def copy_container(src_name: str, dst: str) -> None:
        """Move a 2-D tensor out of the container untouched, blocks and all."""
        nonlocal n_copied
        t = src.take(src_name)
        if len(t["dims"]) != 2:
            raise SystemExit(f"{src_name}: expected a 2-D tensor, got {t['dims']}")
        block, size = GGML_BLOCK[t["ty"]]
        row_elems = t["dims"][0]
        n_rows = t["elements"] // row_elems
        data = np.frombuffer(src.raw(t), dtype=np.uint8).reshape(n_rows, row_elems // block * size)
        w.add_tensor(dst, data, raw_shape=data.shape,
                     raw_dtype=getattr(gguf.GGMLQuantizationType, t["ty"]))
        n_copied += 1
        logger.debug("copy    %-52s -> %-30s %s %s", src_name, dst, t["ty"], t["dims"])

    # ------------------------------------------------------------ featurizer
    # {1, 128, 257} in numpy order; clip reads it flat as [mel][fft_bin]. An
    # aligned checkpoint carries a copy of the container's, because the HF repos
    # compute theirs in the processor and ship neither.
    if self_contained:
        put("a.mel_filters", st.f32("preprocessor.featurizer.fb").reshape(n_mel, -1),
            args.asr_dir.name + " preprocessor.featurizer.fb")
        put("a.window", st.f32("preprocessor.featurizer.window"),
            args.asr_dir.name + " preprocessor.featurizer.window")
    else:
        put("a.mel_filters", src.f32(SRC + "preprocessor.featurizer.fb").reshape(n_mel, -1),
            "container " + SRC + "preprocessor.featurizer.fb")
        put("a.window", src.f32(SRC + "preprocessor.featurizer.window"),
            "container " + SRC + "preprocessor.featurizer.window")

    # ------------------------------------------------------- conv subsampling
    for i, name in PRE_ENCODE_CONVS:
        # ggml_conv_2d / ggml_conv_2d_dw_direct want an F16 kernel, shape as is
        put(f"a.conv1d.{i}.weight", enc(name + ".weight"), name + ".weight",
            dtype=np.float16)
        # added to a {freq, time, channel} tensor, so it has to broadcast on ne2:
        # numpy (C, 1, 1) -> ne {1, 1, C}
        put(f"a.conv1d.{i}.bias", enc(name + ".bias").reshape(-1, 1, 1), name + ".bias")

    put_weight("a.pre_encode.out.weight", enc("encoder.subsampling.linear.weight"),
               "encoder.subsampling.linear.weight")
    put("a.pre_encode.out.bias", enc("encoder.subsampling.linear.bias"),
        "encoder.subsampling.linear.bias")

    # ----------------------------------------------------------- projector
    # A published ASR checkpoint has no equivalent -- its own encoder_projector
    # goes to a 640-wide RNN-T joint, not to the STT LLM's 4480-wide embedding --
    # so it borrows the container's, which was trained against the container's
    # own encoder and does not fit this one.
    #
    # An aligned checkpoint carries its own: VoiceChat's projection composed with
    # the map fitted between the two encoders' outputs. That composition is the
    # whole mechanism. The encoder ends in a LayerNorm and `proj` is the next
    # operation, so a linear correction on the encoder output is exactly a
    # different `proj`, and the graph never learns that anything changed.
    if self_contained:
        put_weight("mm.a.proj.weight", st.f32("proj.weight"), args.asr_dir.name + " proj.weight")
        put("mm.a.proj.bias", st.f32("proj.bias"), args.asr_dir.name + " proj.bias")
    else:
        copy_container(SRC + "proj.weight", "mm.a.proj.weight")
        put("mm.a.proj.bias", src.f32(SRC + "proj.bias"), "container " + SRC + "proj.bias")

    # ---------------------------------------------------------------- layers
    for il in range(n_layer):
        p = f"encoder.layers.{il}."
        b = f"a.blk.{il}."

        for dst, name in ATTN_LINEARS:
            put_weight(b + dst, enc(p + name), p + name)

        # {d_head, n_head}; dense because ggml_add needs a float tensor
        put(b + "pos_bias_u", enc(p + "self_attn.bias_u"), p + "self_attn.bias_u")
        put(b + "pos_bias_v", enc(p + "self_attn.bias_v"), p + "self_attn.bias_v")

        for dst, name in NORMS:
            put(b + dst + ".weight", enc(p + name + ".weight"), p + name + ".weight")
            put(b + dst + ".bias", enc(p + name + ".bias"), p + name + ".bias")

        for dst, name in FFN_LINEARS:
            put_weight(b + dst, enc(p + name), p + name)

        # pointwise convs are k=1, so they are matmuls
        put(b + "conv_pw1.weight", enc(p + "conv.pointwise_conv1.weight").squeeze(-1),
            p + "conv.pointwise_conv1.weight", dtype=np.float16)
        put(b + "conv_pw2.weight", enc(p + "conv.pointwise_conv2.weight").squeeze(-1),
            p + "conv.pointwise_conv2.weight", dtype=np.float16)

        # depthwise conv runs through ggml_ssm_conv, which wants {kernel, channels} F32
        put(b + "conv_dw.weight", enc(p + "conv.depthwise_conv.weight").squeeze(1),
            p + "conv.depthwise_conv.weight")

        # conv_norm_type=layer_norm. The container calls this `batch_norm`
        # because NeMo reuses the attribute name; the HF port calls it `norm`.
        put(b + "conv_norm.weight", enc(p + "conv.norm.weight"), p + "conv.norm.weight")
        put(b + "conv_norm.bias", enc(p + "conv.norm.bias"), p + "conv.norm.bias")

    logger.info("tensors: %d quantized, %d copied from the container, %d dense",
                n_quant, n_copied, n_conv)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    w.open_output_file(args.output)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=True)
    w.close()

    logger.info("wrote %s (%.1f MiB)", args.output, os.path.getsize(args.output) / 1024 ** 2)


if __name__ == "__main__":
    main()
