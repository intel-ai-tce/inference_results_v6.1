#!/usr/bin/env python3
"""Idempotent patcher: add an ``mlperf_cnn`` calibration dataset to the Quark LLM
PTQ example so we can reproduce AMD's
``Llama-3.1-8B-Instruct-MXFP4-W4A4-MLCAL-C1000-GPTQ`` using the *local* MLPerf
CNN/DailyMail calibration file (this Quark build ships only pileval/wikitext/
cnn_dailymail, not the MLPerf set the model card used).

``mlperf_cnn`` loads a JSON list of ``{"instruction", "input"}`` records (the
MLPerf calibration format), substitutes the article into the instruction's
``{input}`` placeholder, chat-templates each prompt (the model card used
"chat-templated prompts"), and feeds them through the existing tensor calib path.

Calibration file path: env ``MLPERF_CALIB_JSON`` (default
``/data/cnn_dailymail_calibration.json``).

Patches (both in-image, so must be applied inside the container):
  1) quark/torch/utils/llm/data_preparation.py  -> add branch + dispatcher entry
  2) Quark/examples/.../llm_ptq/quantize_quark.py -> add "mlperf_cnn" arg choice

Safe to run repeatedly (idempotent via markers).
"""
import ast
import os
import shutil
import sys

MARKER = "QUARK-MLPERF-CNN"


def patch_data_prep() -> int:
    import quark
    path = os.path.join(os.path.dirname(quark.__file__), "torch", "utils", "llm", "data_preparation.py")
    src = open(path).read()
    if MARKER in src:
        print("[patch] data_preparation.py already patched -> no-op.")
        return 0

    # 1) add the mlperf_cnn branch inside get_calib_dataloader_to_tensor (the
    #    block that fills `text_data`), right before its `else: raise`.
    anchor = (
        '    elif dataset_name == "wikitext":\n'
        '        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")\n'
        '        text_data = dataset["text"][:num_calib_data]\n'
        "    else:\n"
        "        raise NotImplementedError\n"
    )
    if anchor not in src:
        sys.stderr.write("[patch] ERROR: to_tensor anchor not found in data_preparation.py\n")
        return 2
    branch = (
        '    elif dataset_name == "wikitext":\n'
        '        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")\n'
        '        text_data = dataset["text"][:num_calib_data]\n'
        f'    elif dataset_name == "mlperf_cnn":  # {MARKER}: local MLPerf CNN/DailyMail calib\n'
        "        import json as _json_mc, os as _os_mc\n"
        '        _p_mc = _os_mc.environ.get("MLPERF_CALIB_JSON", "/data/cnn_dailymail_calibration.json")\n'
        '        with open(_p_mc) as _f_mc:\n'
        "            _raw_mc = _json_mc.load(_f_mc)\n"
        "        text_data = []\n"
        "        for _ex in _raw_mc[:num_calib_data]:\n"
        "            if isinstance(_ex, dict):\n"
        '                _instr = _ex.get("instruction", "")\n'
        '                _inp = _ex.get("input", _ex.get("article", ""))\n'
        "            else:\n"
        '                _instr, _inp = "", str(_ex)\n'
        '            _prompt = _instr.replace("{input}", _inp) if "{input}" in _instr else ((_instr + "\\n\\n" + _inp).strip())\n'
        "            try:\n"
        '                _prompt = tokenizer.apply_chat_template([{"role": "user", "content": _prompt}], tokenize=False, add_generation_prompt=True)\n'
        "            except Exception:\n"
        "                pass\n"
        "            text_data.append(_prompt)\n"
        "    else:\n"
        "        raise NotImplementedError\n"
    )
    src = src.replace(anchor, branch, 1)

    # 2) route mlperf_cnn through the tensor loader in the dispatcher.
    disp_anchor = '    if dataset_name in ["pileval", "cnn_dailymail", "wikitext"]:\n'
    disp_new = f'    if dataset_name in ["pileval", "cnn_dailymail", "wikitext", "mlperf_cnn"]:  # {MARKER}\n'
    if disp_anchor not in src:
        sys.stderr.write("[patch] ERROR: dispatcher anchor not found in data_preparation.py\n")
        return 2
    src = src.replace(disp_anchor, disp_new, 1)

    ast.parse(src)
    shutil.copy(path, path + ".bak_mlperfcnn")
    open(path, "w").write(src)
    print("[patch] data_preparation.py patched ->", path)
    return 0


def patch_quant_script() -> int:
    path = "/lab-mlperf-inference/Quark/examples/torch/language_modeling/llm_ptq/quantize_quark.py"
    if not os.path.isfile(path):
        sys.stderr.write(f"[patch] WARNING: {path} not found; skipping arg-choice patch.\n")
        return 0
    src = open(path).read()
    if MARKER in src:
        print("[patch] quantize_quark.py already patched -> no-op.")
        return 0
    anchor = (
        "        choices=[\n"
        '            "pileval",\n'
        '            "wikitext",\n'
        '            "cnn_dailymail",\n'
    )
    if anchor not in src:
        sys.stderr.write("[patch] ERROR: --dataset choices anchor not found in quantize_quark.py\n")
        return 2
    new = anchor + f'            "mlperf_cnn",  # {MARKER}\n'
    src = src.replace(anchor, new, 1)
    ast.parse(src)
    shutil.copy(path, path + ".bak_mlperfcnn")
    open(path, "w").write(src)
    print("[patch] quantize_quark.py patched ->", path)
    return 0


def patch_eval_import() -> int:
    """Make the top-level ``eval_model`` import optional. The installed lm_eval
    (0.4.12) dropped ``load_yaml_config`` that quark.contrib.llm_eval imports, so
    the module-level import crashes even runs that pass --skip_evaluation (where
    eval_model is never called)."""
    path = "/lab-mlperf-inference/Quark/examples/torch/language_modeling/llm_ptq/quantize_quark.py"
    if not os.path.isfile(path):
        return 0
    src = open(path).read()
    if "QUARK-EVAL-OPTIONAL" in src:
        print("[patch] eval import already guarded -> no-op.")
        return 0
    anchor = "from quark.contrib.llm_eval import eval_model\n"
    if anchor not in src:
        sys.stderr.write("[patch] WARNING: eval_model import line not found; skipping guard.\n")
        return 0
    guard = (
        "try:  # QUARK-EVAL-OPTIONAL: lm_eval API drift; only needed without --skip_evaluation\n"
        "    from quark.contrib.llm_eval import eval_model\n"
        "except Exception as _e_eval:  # pragma: no cover\n"
        "    eval_model = None\n"
        "    import warnings as _w_eval\n"
        '    _w_eval.warn(f"eval_model unavailable ({_e_eval}); OK when --skip_evaluation is set.")\n'
    )
    src = src.replace(anchor, guard, 1)
    ast.parse(src)
    open(path, "w").write(src)
    print("[patch] eval import guarded ->", path)
    return 0


def main() -> int:
    rc = patch_data_prep()
    if rc not in (0,):
        return rc
    rc = patch_quant_script()
    if rc not in (0,):
        return rc
    return patch_eval_import()


if __name__ == "__main__":
    raise SystemExit(main())
