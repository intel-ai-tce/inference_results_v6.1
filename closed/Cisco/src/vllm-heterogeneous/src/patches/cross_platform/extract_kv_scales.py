
"""
Extract KV cache scales (k_scale, v_scale) from a model checkpoint.

Reads all safetensors files and extracts per-layer k_scale and v_scale
values. Supports both compressed-tensors format (model.layers.N.self_attn.k_scale)
and Quark format (model.layers.N.self_attn.k_proj.output_scale).

Usage:
    # Extract fp8_dynamic scales to JSON (for override on MI350X):
    python3 extract_kv_scales.py /model/llama2-70b-chat-hf/fp8_dynamic \
        -o /model/llama2-70b-chat-hf/fp8_kv_scales.json

    # Compare scales between two models:
    python3 extract_kv_scales.py /model/llama2-70b-chat-hf/fp8_dynamic \
        --compare /model/llama2-70b-chat-hf/fp4_quantized_gptq
"""

import os
import sys
import json
import glob
import argparse


def extract_scales(model_dir):
    """Extract per-layer k_scale/v_scale from safetensors checkpoint."""
    from safetensors import safe_open

    scales = {}
    
    src_prio = {}
    sf_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not sf_files:
        print(f"ERROR: No .safetensors files found in {model_dir}")
        sys.exit(1)

    for sf_path in sf_files:
        f = safe_open(sf_path, framework="pt")
        for key in f.keys():
            layer_idx = None
            scale_type = None
            
            
            
            
            
            
            prio = 0

            
            if ".self_attn.k_scale" in key and ".k_proj" not in key:
                scale_type, prio = "k_scale", 2
            elif ".self_attn.v_scale" in key and ".v_proj" not in key:
                scale_type, prio = "v_scale", 2
            
            elif ".self_attn.k_proj.output_scale" in key:
                scale_type, prio = "k_scale", 2
            elif ".self_attn.v_proj.output_scale" in key:
                scale_type, prio = "v_scale", 2
            
            
            elif ".self_attn.k_proj.weight_scale" in key:
                scale_type, prio = "k_scale", 1
            elif ".self_attn.v_proj.weight_scale" in key:
                scale_type, prio = "v_scale", 1

            if scale_type is None:
                continue

            parts = key.split(".")
            try:
                layer_idx = int(parts[parts.index("layers") + 1])
            except (ValueError, IndexError):
                continue

            scales.setdefault(layer_idx, {})
            cur_prio = src_prio.get((layer_idx, scale_type), 0)
            
            
            if scale_type in scales[layer_idx] and prio < cur_prio:
                continue
            tensor = f.get_tensor(key).float().abs()
            
            
            
            if (scale_type in scales[layer_idx] and prio == cur_prio
                    and tensor.numel() != 1):
                continue
            
            
            scales[layer_idx][scale_type] = float(tensor.max().item())
            src_prio[(layer_idx, scale_type)] = prio

    return scales


def main():
    parser = argparse.ArgumentParser(
        description="Extract KV cache scales from a model checkpoint"
    )
    parser.add_argument("model_dir", help="Path to model checkpoint directory")
    parser.add_argument(
        "-o", "--output", help="Output JSON file path"
    )
    parser.add_argument(
        "--compare",
        help="Path to second model to compare scales against",
    )
    args = parser.parse_args()

    print(f"Extracting KV scales from: {args.model_dir}")
    scales = extract_scales(args.model_dir)

    if not scales:
        print("  WARNING: No k_scale/v_scale found in checkpoint!")
        print("  The model may not have pre-computed KV cache scales.")
        sys.exit(1)

    print(f"  Found {len(scales)} layers\n")

    n = len(scales)
    show = list(sorted(scales.keys()))
    show_set = set(show[:3]) | set(show[-2:]) if n > 5 else set(show)
    for i in sorted(scales):
        if i in show_set:
            ks = scales[i].get("k_scale", "N/A")
            vs = scales[i].get("v_scale", "N/A")
            ks_str = f"{ks:.6f}" if isinstance(ks, float) else str(ks)
            vs_str = f"{vs:.6f}" if isinstance(vs, float) else str(vs)
            print(f"  layer {i:3d}: k_scale={ks_str}  v_scale={vs_str}")
        elif i == min(show_set | {i for i in show if i > 2}, default=i):
            print(f"  ... ({n - 5} more layers) ...")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(scales, f, indent=2, sort_keys=True)
        print(f"\n  Saved to {args.output}")
        print(f"  Use with: export VLLM_KV_SCALE_OVERRIDE={args.output}")

    if args.compare:
        print(f"\n{'=' * 60}")
        print(f"Comparing with: {args.compare}")
        other = extract_scales(args.compare)
        if not other:
            print("  WARNING: No scales found in comparison model!")
            return

        print(f"  Found {len(other)} layers\n")

        all_layers = sorted(set(scales) | set(other))
        max_k_diff = 0.0
        max_v_diff = 0.0
        mismatches = 0

        for i in all_layers:
            s1 = scales.get(i, {})
            s2 = other.get(i, {})
            k1 = s1.get("k_scale", 0.0)
            k2 = s2.get("k_scale", 0.0)
            v1 = s1.get("v_scale", 0.0)
            v2 = s2.get("v_scale", 0.0)
            k_pct = abs(k1 - k2) / max(abs(k1), 1e-12) * 100
            v_pct = abs(v1 - v2) / max(abs(v1), 1e-12) * 100
            max_k_diff = max(max_k_diff, k_pct)
            max_v_diff = max(max_v_diff, v_pct)
            if k_pct > 1.0 or v_pct > 1.0:
                mismatches += 1

            if i < 3 or i >= len(all_layers) - 2 or k_pct > 20 or v_pct > 20:
                print(
                    f"  layer {i:3d}: k_scale {k1:.6f} vs {k2:.6f} "
                    f"({k_pct:5.1f}%)  v_scale {v1:.6f} vs {v2:.6f} "
                    f"({v_pct:5.1f}%)"
                )
            elif i == 3:
                print(f"  ...")

        print(f"\n  Max k_scale diff: {max_k_diff:.1f}%")
        print(f"  Max v_scale diff: {max_v_diff:.1f}%")
        print(f"  Layers with >1% diff: {mismatches}/{len(all_layers)}")

        if mismatches > 0:
            print(
                f"\n  SCALES DIFFER. To use {os.path.basename(args.compare)} as decoder"
                f"\n  with {os.path.basename(args.model_dir)} as prefiller, you need the"
                f"\n  scale override. Run:"
                f"\n    python3 extract_kv_scales.py {args.model_dir} -o /path/to/scales.json"
                f"\n    export VLLM_KV_SCALE_OVERRIDE=/path/to/scales.json"
            )
        else:
            print("\n  Scales match! No override needed.")


if __name__ == "__main__":
    main()
