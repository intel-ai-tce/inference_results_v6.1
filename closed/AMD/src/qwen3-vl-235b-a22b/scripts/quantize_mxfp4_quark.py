"""Quantize Qwen3-VL-235B-A22B-Instruct (BF16) to MXFP4 with AMD Quark (w4a4 / w4a6 / w4a8; w4a16 via --weight-only).

Calibration follows MLPerf Inference v6.1 (mlcommons/inference#2600): the OFFICIAL 20-sample set from
Shopify/product-catalogue (see PR2600_CALIBRATION_INDICES) -- exactly those 20 rows, not a first-N
subset. Every scheme shares 4-bit MXFP4 group-32 WEIGHTS; --scheme selects the ACTIVATION precision:
  - mxfp4            : w4a4 (dynamic MXFP4 activations). Weight-scales only, so calibration needs
                       no forward (CPU-feasible via --max-gpu-mem cpu).
  - mxfp4_mxfp6_e2m3 : w4a6 (dynamic MXFP6-e2m3 activations). Also weight-scales only / no forward
                       (CPU-feasible) like w4a4 -- just 6-bit activations for more headroom.
  - mxfp4_fp8        : w4a8 (STATIC per-tensor FP8 activations). vLLM's W4A8 path requires static
                       per-tensor FP8, so calibration runs a forward over the Shopify set to observe
                       activation ranges -> GPU sharding required.
  - (w4a16)          : `--scheme mxfp4 --weight-only` -- MXFP4 weights, activations kept BF16.

Usage (file: quantize_mxfp4_quark.py):
  # w4a8 (static FP8 acts -> needs GPU calibration forward):
  HF_TOKEN=hf_xxx python quantize_mxfp4_quark.py --scheme mxfp4_fp8 \
      --output-name Qwen3-VL-235B-A22B-Instruct-MXFP4-W4A8-quark --max-gpu-mem 250GiB
  # w4a6 (dynamic MXFP6 acts -> weight-only, no forward, CPU-feasible):
  python quantize_mxfp4_quark.py --scheme mxfp4_mxfp6_e2m3 \
      --output-name Qwen3-VL-235B-A22B-Instruct-MXFP4-W4A6-quark
  # w4a4 (default) / w4a16 (add --weight-only).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
from io import BytesIO

import torch
from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download
from openai.types import ResponseFormatJSONSchema
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ConfigDict, field_validator
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextExperts

# Keep the full Quark recipe version-controlled instead of shelling out.
from quark.torch import LLMTemplate, ModelQuantizer, export_safetensors
# quark >=0.12 MoE-prep entry point (native transformers-v5 Qwen3-VL-MoE unfuse). This script
# assumes quark >=0.12; older quark (which only had prepare_for_moe_quant) is not supported.
from quark.torch.utils.llm import preprocess_for_quantization
from quark.torch.quantization import AutoSmoothQuantConfig, RotationConfig, SmoothQuantConfig
# Block-scale FP8 specs for --fp8-last-n-act block128/mxfp8. quark has no preset *string* for
# per-group FP8, so we build the QLayerConfig by hand (see build_blockscale_fp8_qlayer_config)
# and inject it into layer_quant_config.
from quark.torch.quantization import (
    FP8E4M3PerGroupSpec,
    OCP_MXFP8E4M3Spec,
    QLayerConfig,
)


def _drop_fused_moe_params(model: torch.nn.Module) -> None:
    """Delete the original fused expert tensors after the MoE unfuse + quantization.

    preprocess_for_quantization replaces each fused Qwen3VLMoeTextExperts with per-expert nn.Linear
    (which then get quantized to 4-bit), but quark's own cleanup of the now-dead fused
    `gate_up_proj`/`down_proj` parameters does NOT fire on this path -- so without this they are
    exported as BF16 alongside the quantized per-expert weights, ~3x bloating the checkpoint
    (582GB instead of ~128GB) and confusing the loader. The unfused forward uses the per-expert
    Linears, so the fused params are dead weight and safe to remove.
    """
    for module in model.modules():
        if isinstance(module, Qwen3VLMoeTextExperts):
            for attr in ("gate_up_proj", "down_proj"):
                if hasattr(module, attr):
                    delattr(module, attr)

# Defaults: a bare run reproduces the requested checkpoint name.
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-235B-A22B-Instruct"
DEFAULT_OUTPUT_NAME = "Qwen3-VL-235B-A22B-Instruct-MXFP4-mlperf6.1-closed"
DEFAULT_HF_REPO_ID = "amd/Qwen3-VL-235B-A22B-Instruct-MXFP4-mlperf6.1-closed"
# MLPerf Inference v6.1 official calibration set (mlcommons/inference#2600):
# https://github.com/mlcommons/inference/pull/2600/changes
PR2600_CALIBRATION_INDICES = [
    20232, 21162, 33584, 46825, 45190, 46143, 14189, 16658, 26406, 9565,
    33733, 31057, 47465, 33503, 42293, 7768, 1962, 39746, 13568, 22527,
]
DEFAULT_NUM_CALIBRATION_SAMPLES = len(PR2600_CALIBRATION_INDICES)
DEFAULT_MAX_SEQUENCE_LENGTH = 65536  # >= the PR's max ISL (61566, at index 22527)

# Quark uses fnmatch globs for excluded layers.
QUARK_EXCLUDE = ["*lm_head", "*visual*", "*mlp.gate"]


def build_q3vl_rotation_config() -> RotationConfig:
    """Offline-R1 (SpinQuant/QuaRot) Hadamard rotation mapping for Qwen3-VL-MoE."""
    L = "model.language_model.layers"
    attn_qkv = [f"{L}.layer_id.self_attn.{p}" for p in ("q_proj", "k_proj", "v_proj")]
    moe_in = [
        f"{L}.layer_id.mlp.experts.*.gate_proj",
        f"{L}.layer_id.mlp.experts.*.up_proj",
    ]
    scaling_layers = {
        "first_layer": [
            {
                "prev_modules": [
                    "model.language_model.embed_tokens",
                    "model.visual.merger.linear_fc2",
                ],
                "norm_module": f"{L}.layer_id.input_layernorm",
                "next_modules": attn_qkv,
            },
            {
                "prev_modules": [f"{L}.layer_id.self_attn.o_proj"],
                "norm_module": f"{L}.layer_id.post_attention_layernorm",
                "next_modules": moe_in,
            },
        ],
        "middle_layers": [
            {
                "prev_modules": [f"{L}.pre_layer_id.mlp.experts.*.down_proj"],
                "norm_module": f"{L}.layer_id.input_layernorm",
                "next_modules": attn_qkv,
            },
            {
                "prev_modules": [f"{L}.layer_id.self_attn.o_proj"],
                "norm_module": f"{L}.layer_id.post_attention_layernorm",
                "next_modules": moe_in,
            },
        ],
        "last_layer": [
            {
                "prev_modules": [f"{L}.layer_id.mlp.experts.*.down_proj"],
                "norm_module": "model.language_model.norm",
                "next_modules": ["lm_head"],
            },
        ],
    }

    return RotationConfig(
        r1=True,
        r2=False,
        r3=False,
        r4=False,
        random_r1=False,
        online_r1_rotation=False,
        backbone="model.language_model",
        model_decoder_layers="model.language_model.layers",
        scaling_layers=scaling_layers,
    )


def _q3vl_sq_scaling_layers(n_experts: int = 128) -> list:
    """Shared Q3VL-MoE SmoothQuant/AutoSmoothQuant scaling-point mapping (rationale in
    build_q3vl_smoothquant_config): attention in/out + each expert's down_proj input."""
    layers = [
        {"prev_op": "input_layernorm",
         "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
         "inp": "self_attn.q_proj", "module2inspect": "self_attn"},
        {"prev_op": "self_attn.v_proj", "layers": ["self_attn.o_proj"], "inp": "self_attn.o_proj"},
    ]
    for i in range(n_experts):
        layers.append({
            "prev_op": f"mlp.experts.{i}.up_proj",
            "layers": [f"mlp.experts.{i}.down_proj"],
            "inp": f"mlp.experts.{i}.down_proj",
        })
    return layers


def build_q3vl_smoothquant_config(n_experts: int = 128, alpha: float = 0.5) -> SmoothQuantConfig:
    """SmoothQuant mapping for Qwen3-VL-MoE"""
    return SmoothQuantConfig(
        alpha=alpha,
        scaling_layers=_q3vl_sq_scaling_layers(n_experts),
        model_decoder_layers="model.language_model.layers",
    )


def build_q3vl_autosmoothquant_config(n_experts: int = 128) -> AutoSmoothQuantConfig:
    """AutoSmoothQuant: per-scaling-point alpha AUTO-SEARCH (min-MSE) over the SAME Q3VL-MoE mapping as
    build_q3vl_smoothquant_config -- replaces the fixed global alpha=0.5 with a per-layer/per-expert
    alpha chosen to minimize post-smoothing MSE. Still weight-folded => all-fp4, zero runtime cost,
    serves as-is on native W4A4. Needs a calibration forward, same as --smoothquant."""
    return AutoSmoothQuantConfig(
        scaling_layers=_q3vl_sq_scaling_layers(n_experts),
        model_decoder_layers="model.language_model.layers",
        compute_scale_loss="MSE",
    )


def build_smoothquant_text_calib(processor, token: str | None, num_samples: int = 16, callen: int = 256):
    """Fixed-length TEXT-only calibration for SmoothQuant (the required workaround for Q3VL)."""
    tok = processor.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ds = load_shopify_dataset(token=token).select(PR2600_CALIBRATION_INDICES[:num_samples])
    samples = []
    for r in ds:
        txt = (
            f"Title: {r['product_title']}\n"
            f"Description: {str(r['product_description'])[:1500]}\n"
            f"Categories: {r['potential_product_categories']}"
        )
        ids = processor.apply_chat_template(
            [{"role": "user", "content": txt}],
            tokenize=True, add_generation_prompt=False, return_tensors="pt",
        )
        enc = tok.pad({"input_ids": ids[0][:callen]}, padding="max_length", max_length=callen, return_tensors="pt")
        samples.append({"input_ids": enc["input_ids"].view(1, -1), "attention_mask": enc["attention_mask"].view(1, -1)})
    print(f"--smoothquant: built {len(samples)} fixed-len({callen}) text-only calib samples.")
    return samples


def build_smoothquant_multimodal_calib(processor, token: str | None, num_samples: int = 16,
                                       img_size: int = 392, total_len: int = 640):
    """Fixed-length MULTIMODAL calibration for SmoothQuant -> image-driven activation scales."""
    tok = processor.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ds = load_shopify_dataset(token=token).select(PR2600_CALIBRATION_INDICES[:num_samples])
    samples, grids = [], set()
    for r in ds:
        img = r["product_image"].convert("RGB").resize((img_size, img_size))  # PIL Image in the dataset
        msg = [
            {"role": "system", "content": "Classify the product image into a category and return a JSON object."},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Title: " + str(r["product_title"])[:400]}]},
        ]
        enc = processor.apply_chat_template(msg, tokenize=True, add_generation_prompt=False,
                                            return_dict=True, return_tensors="pt")
        grids.add(tuple(enc["image_grid_thw"][0].tolist()))
        ids = enc["input_ids"][0]
        L = ids.shape[0]
        if L >= total_len:
            ids2, am = ids[:total_len], torch.ones(total_len, dtype=torch.long)
        else:
            pad = total_len - L
            ids2 = torch.cat([ids, torch.full((pad,), tok.pad_token_id, dtype=ids.dtype)])
            am = torch.cat([torch.ones(L, dtype=torch.long), torch.zeros(pad, dtype=torch.long)])
        samples.append({"input_ids": ids2.view(1, -1), "attention_mask": am.view(1, -1),
                        "pixel_values": enc["pixel_values"].to(torch.bfloat16),
                        "image_grid_thw": enc["image_grid_thw"]})
    assert len(grids) == 1, f"image_grid_thw not identical across calib (image resize inconsistent): {grids}"
    print(f"--smoothquant: built {len(samples)} MULTIMODAL calib samples (img {img_size}, len {total_len}, grid {grids}).")
    return samples


def rotate_offline_residual_extras(model: torch.nn.Module) -> None:
    """Apply the offline R1 to residual-touching modules quark's norm-fold mapping can't reach.

    Two kinds, both bf16 (excluded from quant), so these are lossless weight edits with the SAME
    deterministic Hadamard quark used for R1 (``get_rotation_matrix(hidden, random=False)``):

    1. DEEPSTACK mergers -- features are ADDED straight into the residual at layers [8,16,24]
       (``Qwen3VLMoeModel._deepstack_process``: ``h[mask] += visual_embeds``), bypassing every
       RMSNorm, so ``scaling_layers`` never touches them. Left unrotated they land in the UNrotated
       basis while the residual is rotated -> image tokens corrupted, F1 (an image task) collapses.
       OUTPUT rotation (their output enters the residual): ``W' = R1^T @ W``, ``b' = b @ R1``.
    2. ROUTER gates (``mlp.gate``) -- ``Qwen3VLMoeTextTopKRouter`` holds a raw weight Parameter
       (``F.linear(h, weight)``), not an nn.Linear, so quark's next_modules can't rotate it. It
       READS the rotated residual, so INPUT rotation: ``W' = W @ R1``. Routing is unchanged
       (identity), but the weight must absorb R1 or logits are computed in the wrong basis.

    R1's size is the residual dim = the merger ``out_features`` (== text hidden), matching quark's R1.
    """
    from quark.torch.algorithm.rotation.rotation_utils import get_rotation_matrix

    visual = model.model.visual
    mergers = list(getattr(visual, "deepstack_merger_list", []) or [])
    if not mergers:
        print("[warn] no deepstack mergers found; skipping deepstack rotation")
        return
    hidden = mergers[0].linear_fc2.out_features
    R1 = get_rotation_matrix(hidden, device="cpu", random=False)

    n_ds = 0
    for m in mergers:
        fc2 = m.linear_fc2
        W = fc2.weight.data
        R = R1.to(dtype=W.dtype, device=W.device)
        fc2.weight.data = R.T @ W          # output rotation: y' = y @ R1
        if fc2.bias is not None:
            fc2.bias.data = fc2.bias.data @ R
        n_ds += 1

    # Rotate every MoE router gate on INPUT (it reads the rotated residual).
    n_rt = 0
    for name, mod in model.named_modules():
        if name.endswith(".mlp.gate") and hasattr(mod, "weight") and not isinstance(mod, torch.nn.Linear):
            W = mod.weight.data            # [num_experts, hidden]
            R = R1.to(dtype=W.dtype, device=W.device)
            mod.weight.data = W @ R        # input rotation: (h @ R1) @ (W @ R1)^T == h @ W^T
            n_rt += 1
    print(f"Rotated {n_ds} deepstack merger linear_fc2 (output) + {n_rt} router gates (input) "
          f"by offline R1 (hidden={hidden}).")


def _num_text_layers(model: torch.nn.Module) -> int:
    """Number of text-decoder blocks (Qwen3-VL-MoE stores it under text_config)."""
    cfg = model.config
    tc = getattr(cfg, "text_config", None)
    n = getattr(tc, "num_hidden_layers", None) if tc is not None else None
    if n is None:
        n = getattr(cfg, "num_hidden_layers", None)
    if not n:
        raise SystemExit(
            "could not determine num_hidden_layers to place --fp8-last-n-layers"
        )
    return int(n)


def build_fp8_last_n_layer_config(model: torch.nn.Module, n: int, scheme: str = "fp8") -> dict[str, str]:
    """Map the EXPERTS of the LAST n decoder blocks to quark's 'fp8' (W8A8) scheme.

    Returns a quark `layer_config` dict {glob: scheme}. quark matches the glob against module
    names via fnmatch; `*language_model.layers.{i}.mlp.experts*` selects the per-expert
    gate/up/down Linears of block i -- NOT its attention.

    NOTE the trailing `experts*` (no dot): the same dict is re-read by vLLM at load time, where
    it fnmatches against the FusedMoE MODULE prefix `language_model.model.layers.{i}.mlp.experts`
    (no trailing component). A `.mlp.experts.*` glob (trailing dot) does NOT match that bare
    prefix, so vLLM would fall back to the global (MXFP4) method and build FP4-packed expert
    buffers -> a 2048-vs-4096 shape crash when the FP8 weights load. `experts*` matches both the
    bare module prefix and the per-expert Linears. See docs/mxfp4-mixed-precision.md S5b.

    Why experts-only (not the whole block): vLLM's quark path has no FP8-weight-only scheme, so
    FP8 weights force FP8 *activations* (W8A8). Static per-tensor FP8 on ATTENTION activations
    collapses accuracy (docs/mxfp4-mixed-precision.md S3.1: F1 -> 0.0), whereas expert
    activations are robust to low precision (expert-act FP4 cost ~0.0009). So we upgrade only the
    expert *weights* (where ~90% of the bytes and the weight-quant residual live) to FP8, keep the
    static FP8 activations confined to the experts, and leave attention on the global (W4A16)
    scheme. The router `mlp.gate` is excluded via QUARK_EXCLUDE and stays BF16. Anchored with a
    trailing '.' so `layers.4.` never matches `layers.40.`.
    """
    total = _num_text_layers(model)
    n = max(0, min(n, total))
    start = total - n
    lc = {f"*language_model.layers.{i}.mlp.experts*": scheme for i in range(start, total)}
    print(
        f"--fp8-last-n-layers {n} (scheme={scheme}): promoting the EXPERTS of blocks "
        f"[{start}..{total - 1}] to FP8 (W8A8); attention + blocks [0..{start - 1}] stay on the "
        "global (MXFP4) scheme."
    )
    return lc


def build_fp8_first_n_layer_config(model: torch.nn.Module, n: int, scheme: str = "fp8") -> dict[str, str]:
    """Map the EXPERTS of the FIRST n decoder blocks to quark's FP8 (W8A8) scheme.

    Mirror of build_fp8_last_n_layer_config but anchored at the TOP of the stack (blocks [0..n-1]):
    tests whether promoting the INITIAL layers' experts to FP8 clears the accuracy gate at fewer
    layers than the last-N variant. Same glob semantics (trailing `experts*` so vLLM's FusedMoE
    module prefix also matches).
    """
    total = _num_text_layers(model)
    n = max(0, min(n, total))
    lc = {f"*language_model.layers.{i}.mlp.experts*": scheme for i in range(0, n)}
    print(
        f"--fp8-first-n-layers {n} (scheme={scheme}): promoting the EXPERTS of blocks "
        f"[0..{n - 1}] to FP8 (W8A8); attention + blocks [{n}..{total - 1}] stay on the "
        "global (MXFP4) scheme."
    )
    return lc


def build_blockscale_fp8_qlayer_config(kind: str) -> QLayerConfig:
    """Build a per-group (block-scale) FP8 QLayerConfig for the FP8 experts.

    quark exposes per-group FP8 only as Spec objects (FP8E4M3PerGroupSpec / OCP_MXFP8E4M3Spec),
    NOT as a preset scheme *string* -- and template.get_config(layer_config=...) accepts only
    strings. So we construct the QLayerConfig directly and the caller injects it into
    quant_config.layer_quant_config for the last/first-N expert globs.

    - kind='block128': DeepSeek-style per_1x128 FP8, FP32 scales. Weight = STATIC per-group-128
      (computed from weights at quant time, no forward); activation = DYNAMIC per-group-128
      (runtime scale). This is the layout vLLM's block-scale FP8 MoE oracle expects
      (kFp8Static128BlockSym weight + kFp8Dynamic128Sym act -> aiter fmoe_fp8_blockscale_g1u1).
      Requires vLLM patch 0028 (QuarkW8A8Fp8MoEMethod otherwise rejects per_group).
    - kind='mxfp8': OCP MXFP8, per_1x32 with E8M0 scales (fallback; different kernel path).

    Both are forward-free for the FP8 experts themselves (dynamic activations); a calibration
    forward is still required only when --smoothquant is set (for the SQ scales), same as ptpc_fp8.
    """
    if kind == "block128":
        weight = FP8E4M3PerGroupSpec(
            ch_axis=-1, group_size=128, scale_format="float32", is_dynamic=False
        ).to_quantization_spec()
        act = FP8E4M3PerGroupSpec(
            ch_axis=-1, group_size=128, scale_format="float32", is_dynamic=True
        ).to_quantization_spec()
    elif kind == "mxfp8":
        weight = OCP_MXFP8E4M3Spec(ch_axis=-1, is_dynamic=False).to_quantization_spec()
        act = OCP_MXFP8E4M3Spec(ch_axis=-1, is_dynamic=True).to_quantization_spec()
    else:
        raise ValueError(f"build_blockscale_fp8_qlayer_config: unexpected kind {kind!r}")
    return QLayerConfig(
        input_tensors=act, output_tensors=None, weight=weight, bias=None
    )


class ProductMetadata(BaseModel):
    """Expected JSON schema for the VLM response."""

    category: str
    """Full product category, e.g. "Clothing & Accessories > Clothing > Shirts"."""

    brands: list[str]
    """Product brands, e.g. ["giorgio armani", "hugo boss"]."""

    is_secondhand: bool
    """Whether the product is second-hand."""


class BaseModelWithAttributeDescriptionsFromDocstrings(BaseModel):
    """Base model that turns attribute docstrings into schema descriptions."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")


class LoadedSample(BaseModelWithAttributeDescriptionsFromDocstrings):
    """LoadGen-ready request payload."""

    messages: list[ChatCompletionMessageParam]
    """Chat messages sent to the VLM inference endpoint."""

    response_format: ResponseFormatJSONSchema | None = None
    """Optional guided-decoding response format."""

    @field_validator("messages", mode="after")
    @classmethod
    def ensure_content_is_list(
        cls,
        messages: list[ChatCompletionMessageParam],
    ) -> list[ChatCompletionMessageParam]:
        """Convert Pydantic's ValidatorIterator content back to a list."""
        for message in messages:
            if (
                "content" in message
                and message["content"].__class__.__module__
                == "pydantic_core._pydantic_core"
                and message["content"].__class__.__name__ == "ValidatorIterator"
            ):
                message["content"] = list(message["content"])  # type: ignore[arg-type]
        return messages


def load_shopify_dataset(
    repo_id: str = "Shopify/product-catalogue",  # the PR #2600 perf/calibration dataset (== the-catalogue-public-beta)
    splits: tuple[str, ...] = ("train", "test"),
    token: str | None = None,
):
    """Load Shopify dataset splits from Hugging Face."""
    return load_dataset(
        repo_id,
        token=token,
        split="+".join(splits),
        revision="main",
    )


def process_sample_to_vllm_messages(
    sample: dict,
    use_guided_decoding: bool = False,
) -> LoadedSample:
    """Convert a Shopify sample into the VLM request format."""
    image_file = BytesIO()
    image_format = sample["product_image"].format
    sample["product_image"].save(image_file, format=image_format)
    image_bytes = image_file.getvalue()
    image_base64 = base64.b64encode(image_bytes)
    image_base64_string = image_base64.decode("utf-8")
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": f"""Please analyze the product from the user prompt
and provide the following fields in a valid JSON object:
- category
- brand
- is_secondhand

You must choose only one, which is the most appropriate, correct, and specifc
category out of the list of possible product categories.

The description of the product sometimes contains various types of source code
(e.g., JavaScript, CSS, HTML, etc.), where useful product information is embedded
somewhere inside the source code. For this task, you should extract the useful
product information from the source code and leverage it, and discard the
programmatic parts of the source code.

Your response should only contain a valid JSON object and nothing more, e.g.,
you should not fence the JSON object inside a ```json code block.
The JSON object should match the followng JSON schema:
```json
{json.dumps(ProductMetadata.model_json_schema(), indent=2)}
```
""",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""The title of the product is the following:
```text
{sample['product_title']}
```

The description of the product is the following:
```text
{sample['product_description']}
```

The following are the possible product categories:
```json
{sample['potential_product_categories']}
```
""",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_format};base64,"
                        f"{image_base64_string}",
                    },
                },
            ],
        },
    ]

    return LoadedSample(
        messages=messages,
        response_format=(
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "product_metadata",
                    "schema": ProductMetadata.model_json_schema(),
                    "strict": True,
                },
            }
            if use_guided_decoding
            else None
        ),
    )


def build_calibration_dataset(
    processor,
    max_sequence_length: int,
    token: str | None,
    num_samples: int | None = None,
    require_images: bool = True,
):
    """Load and tokenize the OFFICIAL (PR mlcommons/inference#2600) multimodal Shopify calibration set.

    Selects EXACTLY the 20 PR2600_CALIBRATION_INDICES from the concatenated train+test dataset (NOT
    the first-N rows) BEFORE tokenization, so we only decode/encode the handful we calibrate on. The
    submission-correct call uses all 20 (num_samples=None/20); a smaller num_samples takes a prefix of
    that fixed set for quick tests and is NOT PR-compliant.

    require_images: assert that pixel_values made it through tokenization. Required for activation
    calibration (the vision tower must see images), but irrelevant for weight-only MXFP4 (no forward),
    where it is relaxed so the run works under transformers 4.57.x (which drops base64 images in the
    single-step apply_chat_template).
    """
    ds = load_shopify_dataset(token=token)
    idxs = PR2600_CALIBRATION_INDICES
    if num_samples is not None:
        idxs = idxs[: min(num_samples, len(idxs))]
    ds = ds.select(idxs)

    def preprocess_function(example):
        messages = process_sample_to_vllm_messages(example).messages
        return processor.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=max_sequence_length,
            tokenize=True,
            add_special_tokens=False,
            return_dict=True,
            add_generation_prompt=False,
        )

    ds = ds.map(preprocess_function, batched=False, remove_columns=ds.column_names)
    # VERIFY: apply_chat_template(tokenize=True) must emit pixel_values for the
    # base64 image_url content. If images are silently dropped, the vision tower
    # calibrates on no visual activations and the checkpoint is quietly wrong.
    # (Skipped for weight-only, which runs no forward -- see require_images.)
    if require_images:
        assert "pixel_values" in ds.column_names, (
            "calibration samples have no pixel_values -- images were dropped by "
            "apply_chat_template; the vision tower would not see visual inputs"
        )
    return ds


def quark_collate(batch):
    """Collate one multimodal sample into model-forward inputs."""
    assert len(batch) == 1
    item = batch[0]
    return {
        key: (
            torch.tensor(value)
            if key != "pixel_values"
            else torch.tensor(value, dtype=torch.bfloat16).squeeze(0)
        )
        for key, value in item.items()
    }


def assert_vision_tower_unquantized(model) -> None:
    """Guard the invariant that the excluded vision tower stayed in float (BF16).

    Catches a regression where QUARK_EXCLUDE stops matching the vision tower and it
    gets silently quantized -- which the run would otherwise complete without error,
    surfacing only as a much later F1 drop.
    """
    visual = getattr(getattr(model, "model", None), "visual", None)
    if visual is None:
        print("[warn] could not locate model.model.visual to verify it stayed BF16")
        return
    bad = sorted(
        {
            str(p.dtype)
            for p in visual.parameters()
            if p.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        }
    )
    if bad:
        raise SystemExit(
            f"Vision tower appears quantized (param dtypes {bad}); it must stay in "
            "float. Check the QUARK_EXCLUDE patterns."
        )
    print("OK: vision tower (model.model.visual) retained float precision.")


def assert_vision_tower_fp8(model) -> None:
    """--vit-fp8 counterpart of assert_vision_tower_unquantized: verify the ViT Linears were actually
    picked up for quantization. Catches a silent glob miss ('*visual*' not matching) that would leave
    the vision tower BF16 -- which would otherwise complete without error and only surface as no
    throughput change / a wrong precision map.

    NOTE: quark stores frozen weights as FAKE-quantized values in the ORIGINAL dtype (bf16); the real
    fp8 tensor is only materialized at export_safetensors (see quantize_linear.py: "After freeze,
    self.weight contains final fake-quantized values"). So we CANNOT check for float8 param dtype
    here -- instead verify the ViT nn.Linear were replaced by quark QuantLinear (which only happens
    when a non-excluded layer matched an fp8 layer_quant_config / the global scheme).
    """
    from quark.torch.quantization.nn.modules.quantize_linear import QuantLinear

    visual = getattr(getattr(model, "model", None), "visual", None)
    if visual is None:
        print("[warn] could not locate model.model.visual to verify fp8 quant")
        return
    n_q = sum(1 for _, m in visual.named_modules() if isinstance(m, QuantLinear))
    if n_q == 0:
        raise SystemExit(
            "--vit-fp8 set but the vision tower has no QuantLinear modules; the '*visual*' layer "
            "scheme did not match. Check the quark layer_config / exclude for the ViT."
        )
    print(f"OK: vision tower quantized ({n_q} QuantLinear modules); exports as fp8.")


def copy_missing_processor_files(model_id: str, save_dir: str, token: str | None) -> None:
    """Ensure the full processor/tokenizer sits next to the quantized weights.

    vLLM serves Q3VL as a VLM and needs the image processor (preprocessor_config.json),
    tokenizer, and chat template alongside the weights. processor.save_pretrained does NOT
    reliably emit preprocessor_config.json across transformers versions (5.x writes a merged
    processor_config.json instead), so vLLM fails to load. These files are unchanged by
    quantization, so copy any missing ones from the source model. config.json is never
    overwritten (it carries the quantization_config); weights/indexes are skipped.
    """
    token = token or None  # an empty-string token sends an illegal "Bearer " auth header
    src = (
        model_id
        if os.path.isdir(model_id)
        else snapshot_download(
            model_id,
            token=token,
            allow_patterns=["*.json", "*.jinja", "merges.txt", "vocab.json"],
        )
    )
    for fn in os.listdir(src):
        if fn == "config.json" or fn.endswith((".safetensors", ".index.json")):
            continue
        srcpath = os.path.join(src, fn)
        # Skip subdirectories (e.g. HF's `original/` consolidated-weights dir): shutil.copy
        # raises IsADirectoryError on them, and none of them are processor/tokenizer files.
        if os.path.isdir(srcpath):
            continue
        dst = os.path.join(save_dir, fn)
        if not os.path.exists(dst):
            shutil.copy(srcpath, dst)


def add_router_gate_excludes(save_dir: str) -> None:
    """Record the MoE router gate as excluded in the exported config, using vLLM's naming.

    quark keeps the router (``mlp.gate``) in BF16 but does NOT write it into the checkpoint's
    ``quantization_config.exclude``, so on load vLLM tries to quantize it and dies on a shape
    mismatch ([num_experts, hidden] BF16 vs a packed param). vLLM matches ``exclude`` by EXACT
    module-prefix (not glob) against its OWN module names, where the hf_to_vllm_mapper reorders
    ``model.language_model.*`` -> ``language_model.model.*``. So write the exact per-layer names
    vLLM will check. Without this the served quark checkpoint fails to load.
    """
    cfg_path = os.path.join(save_dir, "config.json")
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    qc = cfg.get("quantization_config")
    if not qc:
        return
    text_cfg = cfg.get("text_config") or {}
    n_layers = text_cfg.get("num_hidden_layers") or cfg.get("num_hidden_layers")
    if not n_layers:
        print("[warn] num_hidden_layers not found; skipping router-gate excludes")
        return
    exclude = qc.setdefault("exclude", [])
    added = 0
    for i in range(n_layers):
        name = f"language_model.model.layers.{i}.mlp.gate"
        if name not in exclude:
            exclude.append(name)
            added += 1
    with open(cfg_path, "w") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"Recorded {added} router-gate exclude entries (vLLM naming) in config.json")


def add_attention_excludes_vllm_naming(save_dir: str) -> None:
    """Record attention q/k/v/o projections as excluded, in vLLM's module naming.

    For --attention-bf16 we exclude `*self_attn*` at quant time (quark keeps attention BF16). But
    vLLM matches `quantization_config.exclude` against its OWN module names
    (`language_model.model.layers.{i}.self_attn.{q,k,v,o}_proj`, with the hf_to_vllm_mapper
    reorder), so — exactly like the router gate (add_router_gate_excludes) — we append the explicit
    vLLM-named entries so vLLM reliably skips attention on load instead of trying to quantize the
    BF16 weights. Harmless if redundant. No-op when there's no quantization_config.
    """
    cfg_path = os.path.join(save_dir, "config.json")
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    qc = cfg.get("quantization_config")
    if not qc:
        return
    text_cfg = cfg.get("text_config") or {}
    n_layers = text_cfg.get("num_hidden_layers") or cfg.get("num_hidden_layers")
    if not n_layers:
        print("[warn] num_hidden_layers not found; skipping attention excludes")
        return
    exclude = qc.setdefault("exclude", [])
    added = 0
    for i in range(n_layers):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            name = f"language_model.model.layers.{i}.self_attn.{proj}"
            if name not in exclude:
                exclude.append(name)
                added += 1
    with open(cfg_path, "w") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"Recorded {added} attention-projection exclude entries (vLLM naming) in config.json")


def remap_layer_quant_config_to_vllm(save_dir: str) -> None:
    """Rewrite per-layer quant-config keys from transformers naming to vLLM naming.

    quark matches `layer_quant_config` globs against the *transformers* module tree at quant time
    (`model.language_model.layers.{i}...`), so --fp8-last-n-layers uses `*language_model.layers.{i}.*`.
    But vLLM re-reads the SAME dict at load time and fnmatches it against its OWN module names,
    where hf_to_vllm_mapper reorders `model.language_model.*` -> `language_model.model.*`. The glob
    then never matches, vLLM silently falls back to the global (MXFP4) scheme for those blocks, and
    the FusedMoE buffer is sized for FP4-packed weights (half width) -> a 2048-vs-4096 shape crash
    when the FP8 expert tensors load. Rewriting the keys to `*language_model.model.layers.{i}.*`
    (the same reorder add_router_gate_excludes applies to excludes) makes vLLM pick the FP8 scheme.
    Idempotent; a no-op when there is no layer_quant_config.
    """
    cfg_path = os.path.join(save_dir, "config.json")
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    qc = cfg.get("quantization_config") or {}
    lq = qc.get("layer_quant_config") or {}
    if not lq:
        return
    remapped = {
        k.replace("language_model.layers.", "language_model.model.layers."): v
        for k, v in lq.items()
    }
    if remapped == lq:
        return
    qc["layer_quant_config"] = remapped
    with open(cfg_path, "w") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"Remapped {len(remapped)} layer_quant_config keys to vLLM naming in config.json")


def strip_rotation_algo_config(save_dir: str) -> None:
    """Null out ``quantization_config.algo_config`` in the exported config for serving.

    With --rotation, quark serializes the RotationConfig into ``quantization_config.algo_config``
    (a list of dicts). vLLM's quark loader (``apply_vllm_mapper``) walks every quant_config value
    as a list of module-name STRINGS to remap, so it hits the dict elements of algo_config and
    dies with ``'dict' object has no attribute 'endswith'``. Rotation is already folded into the
    weights -- the algo_config is quant-time-only metadata -- so set it to null (matching a
    non-rotated checkpoint). Idempotent; no-op when already null / no quantization_config.
    """
    cfg_path = os.path.join(save_dir, "config.json")
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    qc = cfg.get("quantization_config")
    if not qc or qc.get("algo_config") is None:
        return
    qc["algo_config"] = None
    with open(cfg_path, "w") as fh:
        json.dump(cfg, fh, indent=2)
    print("Stripped quantization_config.algo_config (rotation metadata) for vLLM serving.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize Qwen3-VL-235B-A22B to MXFP4 weights with AMD Quark "
        "(w4a4 / w4a6 / w4a8 via --scheme; w4a16 via --weight-only).",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"BF16 source checkpoint (HF id or local path). Default: {DEFAULT_MODEL_ID}",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory under which the checkpoint folder is written. Default: cwd",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Checkpoint folder name. Default: {DEFAULT_OUTPUT_NAME}",
    )
    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=DEFAULT_NUM_CALIBRATION_SAMPLES,
        help="How many of the 20 OFFICIAL PR mlcommons/inference#2600 calibration indices to use "
        "(a prefix of that fixed set). Default 20 = the full PR-exact set (submission-correct); a "
        "smaller value is a quick-test subset and is NOT PR-compliant.",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=DEFAULT_MAX_SEQUENCE_LENGTH,
    )
    parser.add_argument(
        "--scheme",
        default="mxfp4",
        choices=["mxfp4", "mxfp4_mxfp6_e2m3", "mxfp4_fp8"],
        help="Quark quant scheme (all share 4-bit MXFP4 weights; the activation precision varies). "
        "'mxfp4' = w4a4 (dynamic MXFP4 acts; weight-scales only, CPU-calibratable). "
        "'mxfp4_mxfp6_e2m3' = w4a6 (dynamic MXFP6-e2m3 acts; also weight-scales only / no forward, "
        "CPU-calibratable). 'mxfp4_fp8' = w4a8 (STATIC per-tensor FP8 acts; needs a GPU calibration "
        "forward, which vLLM's W4A8 path requires). For w4a16 use '--scheme mxfp4 --weight-only'. "
        "Default: mxfp4.",
    )
    parser.add_argument(
        "--weight-only",
        action="store_true",
        help="Quantize weights to MXFP4 but keep activations at 16-bit (W4A16) instead of "
        "w4a4. Served on ROCm via AITER_MXFP4_BF16. Only valid with --scheme mxfp4.",
    )
    parser.add_argument(
        "--fp8-last-n-layers",
        type=int,
        default=0,
        metavar="N",
        help="Mixed precision: promote the EXPERTS of the LAST N text-decoder blocks from MXFP4 "
        "to FP8 W8A8 (e4m3, static per-tensor), while attention and the first (num_layers - N) "
        "blocks stay MXFP4. Implemented via quark's per-layer `layer_config` (matched on "
        "`*language_model.layers.{i}.mlp.experts*`). vLLM's quark path has no FP8-weight-only "
        "scheme, so FP8 weights require FP8 activations; we confine those to the experts (robust "
        "to low precision) and keep attention BF16 to avoid the FP8-attention collapse "
        "(docs/mxfp4-mixed-precision.md S3.1). ALWAYS needs a GPU calibration forward (static FP8 "
        "expert activations). Combine with --weight-only for a W4A16 base (BF16 activations "
        "everywhere except the FP8 experts). Only valid with --scheme mxfp4. Default 0 (off).",
    )
    parser.add_argument(
        "--fp8-first-n-layers",
        type=int,
        default=0,
        metavar="N",
        help="Like --fp8-last-n-layers but promotes the EXPERTS of the FIRST N text-decoder blocks "
        "(blocks [0..N-1]) to FP8 W8A8, with the rest (blocks [N..) MXFP4. Mutually exclusive with "
        "--fp8-last-n-layers. Tests whether initial-layer FP8 clears the gate at fewer layers than "
        "last-N. Uses the same --fp8-last-n-act granularity. Only valid with --scheme mxfp4. Default 0.",
    )
    parser.add_argument(
        "--fp8-last-n-act",
        default="per_tensor",
        choices=["per_tensor", "per_token", "block128", "mxfp8", "bf16"],
        help="Activation-scale granularity for the --fp8-last-n-layers FP8 experts. 'per_tensor' "
        "(default) = STATIC per-tensor FP8 (one scale/tensor, needs a calibration forward). "
        "'per_token' = DYNAMIC per-token FP8 (runtime per-token scale, no calibration). vLLM's FP8 "
        "MoE requires MATCHED weight/act granularity, so per_token maps to quark 'ptpc_fp8' "
        "(per-channel weight + per-token act). WARNING: on 0.25+post3 / a7e14c9d5 the per-token FP8 "
        "MoE kernel (fmoe_bf16_pertokenFp8) GPU-faults under FULL cudagraph -- the resulting "
        "checkpoint currently does NOT serve; kept as a diagnostic/A-B knob. per_tensor is the "
        "usable path. per_token is forward-free (no calibration). "
        "'block128' = DYNAMIC per_1x128 block-scale FP8 (DeepSeek-style, FP32 scales) on BOTH weight "
        "(static, per-group-128) and activation (dynamic, per-group-128) -- routes to aiter's "
        "fmoe_fp8_blockscale_g1u1 (needs vLLM patch 0028; ~89%% MFMA eff vs ~34%% for ptpc). "
        "'mxfp8' = per_1x32 OCP MXFP8 (E8M0 scales) fallback. Both are forward-free for the FP8 "
        "experts (dynamic act); a calibration forward is still needed only if --smoothquant is set. "
        "Built via a custom QLayerConfig injected into layer_quant_config (quark has no preset for "
        "per-group FP8), see build_blockscale_fp8_qlayer_config. "
        "'bf16' = KEEP the last-N experts in BF16 (add their globs to EXCLUDES) -- Stage 1 of Path A: "
        "SmoothQuant still folds into them (algo runs before the exclude), so they export as post-SQ "
        "BF16, ready for external 128x128 block-FP8 quantization by scripts/blockquant_last_n_experts.py "
        "(quark 0.12 cannot emit 128x128 2D-block FP8 -- PerBlock2DMinMaxObserver is export-only). "
        "The DeepSeek fmoe_fp8_blockscale_g1u1 kernel needs 128x128 weight blocks; this two-stage "
        "path produces them from BF16 without an fp8 round-trip.",
    )
    parser.add_argument(
        "--attention-bf16",
        action="store_true",
        help="Throughput variant: keep ALL attention projections (q/k/v/o) in BF16 (excluded from "
        "quantization) while leaving the MoE EXPERTS on the base scheme (W4A4 by default, i.e. FP4 "
        "activations — the fast path where most compute lives). This is the attention-BF16 base "
        "(F1~0.7782 alone) meant to recover throughput vs a full W4A16 base (all-BF16 activations). "
        "Combine with --fp8-last-n-layers to add FP8 expert weights on the last N blocks. Distinct "
        "from --weight-only (which makes EXPERT activations BF16 too).",
    )
    parser.add_argument(
        "--max-gpu-mem",
        default=None,
        help=(
            "Per-GPU memory cap for sharding, e.g. '90GiB'. Forces an even device_map "
            "and leaves headroom for quark calibration buffers (avoids fragmentation OOM "
            "on large MoE models). Default: unset (device_map='auto' decides)."
        ),
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="HF token (defaults to $HF_TOKEN) for dataset access and upload.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload the saved checkpoint to --hf-repo-id (private). Off by default.",
    )
    parser.add_argument(
        "--hf-repo-id",
        default=DEFAULT_HF_REPO_ID,
        help=f"Target HF repo for upload. Default: {DEFAULT_HF_REPO_ID}",
    )
    parser.add_argument(
        "--rotation",
        action="store_true",
        help="Apply offline-R1 Hadamard rotation (SpinQuant/QuaRot) before quantization to "
        "reduce fp4 activation-quant loss. Weight-folded (zero runtime cost); serves as-is on "
        "the native W4A4 path. Includes the Q3VL-MoE experts, router, main vision merger, and a "
        "post-step rotating the 3 deepstack mergers. Most useful for --scheme mxfp4 (W4A4).",
    )
    parser.add_argument(
        "--smoothquant",
        action="store_true",
        help="Apply SmoothQuant (per-channel activation-outlier migration into weights) before "
        "quantization. Weight-folded (zero runtime cost); serves as-is on native W4A4. Needs a "
        "calibration forward (GPU sharding required) -- uses a FIXED-LENGTH TEXT-ONLY calib because "
        "quark's SmoothQuant forward mishandles Q3VL multimodal M-RoPE. Most useful for --scheme mxfp4.",
    )
    parser.add_argument(
        "--sq-alpha",
        type=float,
        default=0.5,
        help="SmoothQuant migration strength (0..1). Higher = migrate more outlier magnitude from "
        "activations into weights. Default 0.5. Only used with --smoothquant.",
    )
    parser.add_argument(
        "--sq-multimodal-calib",
        action="store_true",
        help="Use fixed-length MULTIMODAL calibration for SmoothQuant (image-driven scales) instead "
        "of the text-only fallback. Every calib image is resized to a fixed square so all samples "
        "share grid_thw+length (required for quark's cached-position-embeddings forward). More "
        "representative for the image task. Only used with --smoothquant / --auto-smoothquant.",
    )
    parser.add_argument(
        "--auto-smoothquant",
        action="store_true",
        help="Apply AutoSmoothQuant (per-layer/per-expert alpha AUTO-SEARCH, min-MSE) instead of the "
        "fixed global --sq-alpha SmoothQuant. Same Q3VL-MoE mapping + calibration forward; targets the "
        "fp4->fp8 weight-quant gap while staying all-fp4. Mutually exclusive with --smoothquant.",
    )
    parser.add_argument(
        "--vit-fp8",
        action="store_true",
        help="Quantize the VISION TOWER's Linear weights to FP8 (e4m3) in THIS pass (one-step: no "
        "separate post-process). Drops '*visual*' from the BF16 exclude and gives the ViT the "
        "'ptpc_fp8' scheme (per-channel weight + DYNAMIC per-token activation). quark's Linear-only "
        "scheme hits exactly the ViT nn.Linear (attn qkv/proj, mlp fc1/fc2, main+deepstack mergers); "
        "the patch-embed Conv3d, pos-embed, norms and biases are not nn.Linear -> stay BF16. Weights "
        "are quantized data-free and activations are dynamic, so this adds NO calibration requirement. "
        "SERVE NOTE: the fp8 ViT MLP weights require VLLM_Q3VL_FUSE_VIT_GELU_FC1=0 -- the GELU->fc1 "
        "epilogue fusion runs a raw torch._addmm_activation on the weight, bypassing the quant method "
        "(bf16-act x fp8-weight dtype crash). Combines with any --scheme / --smoothquant / "
        "--auto-smoothquant / --fp8-last-n-layers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = os.path.join(args.output_dir, args.output_name)

    # Fail fast on a missing upload token before hours of quantization.
    if args.push_to_hub and not args.hf_token:
        raise SystemExit(
            "--push-to-hub requires an HF token (set $HF_TOKEN or --hf-token)."
        )

    if args.weight_only and args.scheme != "mxfp4":
        raise SystemExit("--weight-only (W4A16) only applies to --scheme mxfp4; drop it for mxfp4_fp8.")

    if args.fp8_last_n_layers and args.fp8_first_n_layers:
        raise SystemExit("--fp8-first-n-layers and --fp8-last-n-layers are mutually exclusive.")
    _fp8_n = args.fp8_last_n_layers or args.fp8_first_n_layers
    if _fp8_n and args.scheme not in ("mxfp4", "mxfp4_fp8"):
        raise SystemExit("--fp8-{first,last}-n-layers builds on an MXFP4 base; only valid with --scheme mxfp4 or mxfp4_fp8.")

    # Load the 235B BF16 model sharded across available GPUs.
    # device_map="auto" can place the shards unevenly; combined with the per-param
    # quantizer buffers quark allocates during calibration (and the transient weight
    # doubling while fused experts are unfused), one GPU can fill up and fail a tiny
    # allocation via fragmentation. Capping max_memory forces an even shard with ample
    # headroom for those buffers.
    # --max-gpu-mem cpu: load entirely on host RAM and calibrate on CPU. MXFP4 uses dynamic
    # (runtime) activation quant, so quantize_model only computes weight scales -- no forward
    # pass -- which is CPU-feasible for the 235B model and sidesteps the GPU-allocator
    # fragmentation that OOMs quark calibration on this 8xMI350X box despite free VRAM.
    # mxfp4_fp8 uses STATIC per-tensor FP8 activations -> quantize_model must run a forward over
    # the calibration set to observe activation ranges (unlike mxfp4's dynamic/weight-only path).
    # A 235B forward on CPU is impractical, so static schemes require GPU sharding.
    # The FP8 experts (--fp8-last-n-layers) are STATIC per-tensor W8A8 -> their activation
    # observers need a calibration forward, JUST like mxfp4_fp8. This holds even with
    # --weight-only (which only relaxes the MXFP4 *base* to W4A16; the FP8 experts stay W8A8,
    # since vLLM has no FP8-weight-only scheme).
    # per-token FP8 activations are DYNAMIC (runtime scale) -> no calibration forward needed.
    # Only static per-tensor FP8 experts require the forward.
    fp8_static_layers = bool(_fp8_n) and args.fp8_last_n_act == "per_tensor"
    static_act = (args.scheme == "mxfp4_fp8") or fp8_static_layers
    cpu_mode = args.max_gpu_mem == "cpu"
    if args.smoothquant and args.auto_smoothquant:
        raise SystemExit("--smoothquant and --auto-smoothquant are mutually exclusive (pick one).")
    if (static_act or args.smoothquant or args.auto_smoothquant) and cpu_mode:
        needs = ("--auto-smoothquant (calibration forward)" if args.auto_smoothquant else
                 "--smoothquant (calibration forward)" if args.smoothquant else
                 "--scheme mxfp4_fp8" if args.scheme == "mxfp4_fp8" else "--fp8-{first,last}-n-layers (static W8A8 experts)")
        raise SystemExit(
            f"{needs} needs a calibration forward; "
            "--max-gpu-mem cpu would run a 235B forward on CPU (impractical). Use GPU sharding "
            "(omit '--max-gpu-mem cpu'; e.g. '--max-gpu-mem 250GiB' for an even shard)."
        )
    if cpu_mode:
        from_kwargs: dict = {"torch_dtype": "auto", "device_map": "cpu"}
    else:
        from_kwargs = {"torch_dtype": "auto", "device_map": "auto"}
        if args.max_gpu_mem:
            n_gpus = torch.cuda.device_count()
            from_kwargs["max_memory"] = {i: args.max_gpu_mem for i in range(n_gpus)}
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(args.model_id, **from_kwargs)
    model.eval()

    # Qwen3-VL-MoE packs all experts into a single fused Qwen3VLMoeTextExperts module (batched
    # gate_up_proj/down_proj tensors). quark's Linear-targeting mxfp4 scheme does NOT match those
    # fused tensors, so they MUST be unfused into per-expert nn.Linear first -- else the ~90% expert
    # bulk stays BF16 (checkpoint ~3x too large, won't dispatch to FP4).
    #
    # quark >=0.12's preprocess_for_quantization is the supported entry point; its registry handler
    # (replace_qwen3vlmoe_experts_with_linear) does the Qwen3VLMoe transpose + gate/up split +
    # weight-sync + forward-override NATIVELY for both transformers <5 and >=5. Do NOT apply the old
    # 0.11.2-era custom fixups or docker/patches/0004 on top -- they would DOUBLE-TRANSFORM this
    # native handler and corrupt the experts.
    preprocess_for_quantization(model)

    processor = AutoProcessor.from_pretrained(args.model_id)

    if args.smoothquant or args.auto_smoothquant:
        # SmoothQuant / AutoSmoothQuant calib: multimodal (image-driven scales, fixed-length so quark's cached
        # position-embeddings forward works) if requested, else the text-only fallback. Either way
        # the served checkpoint runs full multimodal inference.
        # Forward --num-calibration-samples so SmoothQuant/AutoSmoothQuant calibrate on the FULL
        # official PR2600 set (default 20), not the builders' internal 16-sample default. The
        # per-layer alpha search is calibration-defined, so the compliant 20 samples must be used.
        sq_calib = (
            build_smoothquant_multimodal_calib(processor, args.hf_token, num_samples=args.num_calibration_samples)
            if args.sq_multimodal_calib
            else build_smoothquant_text_calib(processor, args.hf_token, num_samples=args.num_calibration_samples)
        )
        calib_dataloader = DataLoader(sq_calib, batch_size=1, collate_fn=lambda b: b[0])
    else:
        # Build a DataLoader of multimodal calibration inputs (subset selected first).
        ds = build_calibration_dataset(
            processor,
            args.max_sequence_length,
            args.hf_token,
            args.num_calibration_samples,
            # Only a static-activation scheme (mxfp4_fp8) runs a calibration forward that needs the
            # vision tower to see images. mxfp4 (dynamic activations) and --weight-only run no forward,
            # so the calib data is unused -- relax the image guardrail (also lets it run under tf-4.57,
            # which drops base64 images in the single-step apply_chat_template).
            require_images=static_act,
        )
        calib_dataloader = DataLoader(ds, batch_size=1, collate_fn=quark_collate)

    # Verified against amd-quark 0.11.2: model_type "qwen3_vl_moe" is a registered
    # LLMTemplate, "mxfp4" is in get_supported_schemes(), and the excluded-layers
    # kwarg on get_config() is `exclude_layers` (NOT `exclude`).
    template = LLMTemplate.get(model.config.model_type)
    # --fp8-last-n-layers: per-layer override map for the last N blocks; the rest keep the global
    # (mxfp4) scheme. quark folds these into quant_config.layer_quant_config.
    #   per_token  => quark 'ptpc_fp8' (per-channel weight + dynamic per-token act)
    #   per_tensor => quark 'fp8'      (static per-tensor)
    # These two are PRESET STRINGS passed via layer_config= (quark resolves them).
    #   block128 / mxfp8 => per-group (block-scale) FP8, which quark has NO preset string for. We
    #   pass layer_config=None here and INJECT a hand-built QLayerConfig (below) into
    #   quant_config.layer_quant_config for the same expert globs. vLLM requires MATCHED weight/act
    #   granularity; block128 = per_1x128 weight+act (needs vLLM patch 0028).
    _block_act = args.fp8_last_n_act in ("block128", "mxfp8")
    _bf16_act = args.fp8_last_n_act == "bf16"  # Path A Stage 1: keep last-N experts post-SQ BF16
    _fp8_scheme = "ptpc_fp8" if args.fp8_last_n_act == "per_token" else "fp8"
    # Expert-glob list is identical regardless of scheme -- reuse the builders (they also print the
    # informative block-range line); for block/bf16 schemes we take only their keys.
    _lc_for_globs = (
        build_fp8_first_n_layer_config(model, args.fp8_first_n_layers, _fp8_scheme)
        if args.fp8_first_n_layers
        else build_fp8_last_n_layer_config(model, args.fp8_last_n_layers, _fp8_scheme)
        if args.fp8_last_n_layers
        else None
    )
    # block128/mxfp8 -> inject custom QLayerConfig (below); bf16 -> exclude (keep BF16); else -> string scheme.
    layer_config = None if (_block_act or _bf16_act) else _lc_for_globs
    # --attention-bf16: exclude all attention projections so they stay BF16 (throughput variant;
    # experts remain on the base scheme = FP4 activations). '*self_attn*' catches q/k/v/o proj.
    excludes = list(QUARK_EXCLUDE)
    if args.attention_bf16:
        excludes.append("*self_attn*")
        print("--attention-bf16: excluding *self_attn* (all attention projections kept BF16).")
    if _bf16_act and _lc_for_globs:
        # Path A Stage 1: exclude the last/first-N experts from quark quantization so they stay BF16.
        # SmoothQuant (an algo_config pass in quantize_model) still folds into them -> post-SQ BF16.
        excludes.extend(list(_lc_for_globs.keys()))
        print(
            f"--fp8-last-n-act bf16: EXCLUDING {len(_lc_for_globs)} expert glob(s) from quantization "
            "(kept post-SQ BF16 for external 128x128 block-FP8 quant via blockquant_last_n_experts.py)."
        )
    # --vit-fp8: fold the vision tower into this quark pass (one submission script, no separate
    # post-process). Remove the ViT from the BF16 exclude and give it the 'ptpc_fp8' layer scheme
    # (per-channel weight + dynamic per-token act) -- the same served config validated for the
    # standalone fp8-ViT checkpoint. quark applies it only to the ViT nn.Linear; the patch-embed
    # Conv3d, pos-embed, norms and biases (not nn.Linear) stay BF16. '*visual*' matches at BOTH quant
    # time (transformers names, e.g. model.visual.blocks.0.attn.qkv) and serve time (vLLM names,
    # visual.blocks.0.attn.qkv, via the hf_to_vllm_mapper) -> no remap needed. ptpc_fp8 uses dynamic
    # activations, so no ViT calibration is required even under text-only SmoothQuant calib.
    if args.vit_fp8:
        excludes = [e for e in excludes if e != "*visual*"]
        # Keep the ViT position embedding BF16: it is an nn.Embedding, which quark weight-quantizes
        # (emitting a stray '<...>.pos_embed._weight_scale'), but vLLM's ViT pos_embed is a plain
        # nn.Embedding with no scale param -> load fails ("no parameter named pos_embed._weight_scale").
        # It is the only quantizable non-Linear in the tower (patch_embed is Conv3d, absent from
        # quark's LAYER_TO_QUANT_LAYER_MAP; norms are not quantizable), so this single carve-out
        # leaves exactly the 116 ViT Linears (attn qkv/proj, mlp fc1/fc2, mergers) going to fp8.
        excludes.append("*visual.pos_embed")
        layer_config = dict(layer_config or {})
        layer_config["*visual*"] = "ptpc_fp8"
        print("--vit-fp8: quantizing vision-tower Linears to fp8 (ptpc: per-channel W + dynamic "
              "per-token act); dropped '*visual*' from the BF16 exclude, kept '*visual.pos_embed' "
              "excluded (nn.Embedding). SERVE with VLLM_Q3VL_FUSE_VIT_GELU_FC1=0.")
    quant_config = template.get_config(
        scheme=args.scheme, layer_config=layer_config, exclude_layers=excludes
    )
    # Inject the block-scale FP8 QLayerConfig for each expert glob (quark has no preset string).
    if _block_act and _lc_for_globs:
        blk = build_blockscale_fp8_qlayer_config(args.fp8_last_n_act)
        if quant_config.layer_quant_config is None:
            quant_config.layer_quant_config = {}
        for glob in _lc_for_globs:
            quant_config.layer_quant_config[glob] = blk
        print(
            f"--fp8-last-n-act {args.fp8_last_n_act}: injected per-group block-scale FP8 "
            f"QLayerConfig for {len(_lc_for_globs)} expert glob(s) "
            f"(weight+act per-group; group_size={'128' if args.fp8_last_n_act == 'block128' else '32'})."
        )

    # --weight-only: keep the MXFP4 *base* weights quantized but leave its activations at 16-bit
    # (W4A4 -> W4A16). Only the GLOBAL config's input quant is nulled; per-layer FP8 overrides
    # (--fp8-last-n-layers) keep their activation quant, because vLLM has no FP8-weight-only
    # scheme -- the FP8 experts must stay W8A8. Net effect with --fp8-last-n-layers: W4A16
    # everywhere (BF16 activations) EXCEPT the last-N experts, which are W8A8.
    if args.weight_only:
        quant_config.global_quant_config.input_tensors = None

    # --rotation: prepend the offline-R1 Hadamard as a PRE-quant optimization so quark rotates
    # the weights first, then observes quant scales in the rotated (near-Gaussian) basis. Folded
    # into weights -> the exported checkpoint is still plain W4A4 (no runtime op). The deepstack
    # mergers are handled separately below (they bypass the norm-fold quark keys off).
    if args.rotation:
        # quark's RotationProcessor.r1() reads model.config.hidden_size for the R1 size, but Q3VL's
        # top-level Qwen3VLMoeConfig keeps hidden_size under text_config -> mirror it up (the residual
        # dim IS the text hidden). rotate_offline_residual_extras derives its R1 size independently
        # (from the merger out_features), so both use the same 4096 -> identical R1.
        if getattr(model.config, "hidden_size", None) in (None, 0):
            model.config.hidden_size = model.config.text_config.hidden_size
            print(f"--rotation: set model.config.hidden_size = {model.config.hidden_size}")
        existing_algos = list(quant_config.algo_config or [])
        quant_config.algo_config = [build_q3vl_rotation_config(), *existing_algos]
        print("--rotation: offline-R1 Hadamard rotation enabled (Q3VL-MoE mapping).")

    # --smoothquant: pre-quant SmoothQuant (per-channel activation-outlier migration into weights).
    # Folded into weights -> exported checkpoint serves as-is on native W4A4. Needs the text-only
    # calib forward built above. quark writes the algo into quantization_config -> stripped for serving.
    if args.smoothquant or args.auto_smoothquant:
        n_exp = int(getattr(model.config.text_config, "num_experts", 128))
        existing_algos = list(quant_config.algo_config or [])
        if args.auto_smoothquant:
            quant_config.algo_config = [build_q3vl_autosmoothquant_config(n_experts=n_exp), *existing_algos]
            print(f"--auto-smoothquant: AutoSmoothQuant (per-layer alpha auto-search, MSE, {n_exp} experts) enabled (Q3VL-MoE mapping).")
        else:
            quant_config.algo_config = [build_q3vl_smoothquant_config(n_experts=n_exp, alpha=args.sq_alpha), *existing_algos]
            print(f"--smoothquant: SmoothQuant (alpha={args.sq_alpha}, {n_exp} experts) enabled (Q3VL-MoE mapping).")

    quantizer = ModelQuantizer(quant_config, multi_device=not cpu_mode)
    quantized_model = quantizer.quantize_model(model, calib_dataloader)

    # Freeze converts the in-place fake-quant observers to real quantized parameters;
    # quark's PTQ example requires this before export_safetensors or the packed weights
    # are not materialized.
    quantized_model = quantizer.freeze(quantized_model)

    # Remove the dead fused expert tensors quark's unfuse leaves behind (else they export as
    # redundant BF16 next to the quantized per-expert weights).
    _drop_fused_moe_params(quantized_model)

    # --rotation post-step: rotate the residual-touching modules quark's norm-fold mapping can't
    # reach -- the 3 deepstack mergers (inject mid-stack) and the router gates (raw-weight routers).
    if args.rotation:
        rotate_offline_residual_extras(quantized_model)

    # Enforce the invariant that the excluded vision tower stayed in BF16.
    if args.vit_fp8:
        assert_vision_tower_fp8(quantized_model)
    else:
        assert_vision_tower_unquantized(quantized_model)

    # Export the Hugging Face safetensors layout consumed by vLLM-ROCm.
    export_safetensors(model=quantized_model, output_dir=save_dir)
    processor.save_pretrained(save_dir)
    copy_missing_processor_files(args.model_id, save_dir, args.hf_token)
    add_router_gate_excludes(save_dir)
    if args.attention_bf16:
        add_attention_excludes_vllm_naming(save_dir)
    remap_layer_quant_config_to_vllm(save_dir)
    if args.rotation or args.smoothquant or args.auto_smoothquant:
        strip_rotation_algo_config(save_dir)  # quark writes rotation/smooth/auto-smooth algo into quant_config -> vLLM can't parse it
    print(f"Saved MXFP4 (Quark) checkpoint to: {save_dir}")

    if args.push_to_hub:
        api = HfApi(token=args.hf_token)
        api.create_repo(
            repo_id=args.hf_repo_id, private=True, exist_ok=True, repo_type="model"
        )
        api.upload_folder(
            folder_path=save_dir, repo_id=args.hf_repo_id, repo_type="model"
        )
        print(f"Uploaded to private HF repo: {args.hf_repo_id}")


if __name__ == "__main__":
    main()
