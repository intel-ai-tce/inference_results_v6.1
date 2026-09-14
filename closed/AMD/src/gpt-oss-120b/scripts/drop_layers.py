import os
import json
import shutil
from safetensors.torch import safe_open, save_file
import argparse


# USAGE of this script is as follows:
#
# python drop_layers.py --m /model/ -i 60 -f 100
#
# This picks the model from /model/ and writes a pruned model at /model/pruned_60_100/
# and drops layers from 60 to 100 


def drop_llama_layers(
    input_dir: str,
    output_dir: str,
    drop_layers: list[int],
    layer_prefix: str = "model.layers."
):
    """
    Create a truncated LLaMA safetensors checkpoint by dropping specified layers and updating all supporting files.

    Args:
        input_dir (str): HF-style model dir with config.json and one or more .safetensors shards.
        output_dir (str): Directory to save the truncated model.
        drop_layers (list[int]): List of layer indices to remove.
        layer_prefix (str): Prefix for transformer block keys (default: "model.layers.").
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Copy tokenizer and auxiliary files
    for fname in os.listdir(input_dir):
        if fname.startswith("tokenizer") or (fname.endswith("json") and fname != "config.json" and not fname.endswith(".safetensors.index.json")):
            shutil.copy(os.path.join(input_dir, fname), os.path.join(output_dir, fname))

    # 2. Load and update config.json
    config_path = os.path.join(input_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Gather all layer indices
    all_layers = set()
    shard_files = [f for f in os.listdir(input_dir) if f.endswith(".safetensors")]
    for shard in shard_files:
        with safe_open(os.path.join(input_dir, shard), framework="pt") as f:
            for key in f.keys():
                if key.startswith(layer_prefix):
                    parts = key.split('.')
                    idx = int(parts[2])
                    all_layers.add(idx)
    all_layers = sorted(all_layers)

    # Determine kept layers and new indexing
    kept = [i for i in all_layers if i not in drop_layers]
    new_index = {old: new for new, old in enumerate(kept)}

    # Update config
    config["num_hidden_layers"] = len(kept)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # 3. Process each .safetensors shard
    for shard in shard_files:
        in_path = os.path.join(input_dir, shard)
        out_path = os.path.join(output_dir, shard)
        new_tensors = {}
        with safe_open(in_path, framework="pt") as f:
            meta = f.metadata()
            for key in f.keys():
                if key.startswith(layer_prefix):
                    parts = key.split('.')
                    old_idx = int(parts[2])
                    if old_idx in drop_layers:
                        continue
                    parts[2] = str(new_index[old_idx])
                    new_key = '.'.join(parts)
                else:
                    new_key = key
                new_tensors[new_key] = f.get_tensor(key)
        save_file(new_tensors, out_path, metadata=meta)
        print(f"Processed shard {shard}: dropped layers {drop_layers}")

    # 4. Regenerate index file
    index = {
        "metadata": {},
        "weight_map": {}
    }
    for fname in sorted(os.listdir(output_dir)):
        if not fname.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(output_dir, fname), framework="pt") as f:
            for key in f.keys():
                index["weight_map"][key] = fname
    total_size = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir)
        if f.endswith(".safetensors")
    )
    index["metadata"]["total_size"] = total_size
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    print(f"Pruned model saved to {output_dir} with layers {kept} and updated index.")


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--initial", type=int, required=True)
parser.add_argument("-f", "--final", type=int, required=True)
parser.add_argument("-m", "--model", type=str, required=True)
args = parser.parse_args()

input_dir=args.model
output_dir=args.model + "/pruned_"+str(args.initial)+"_"+str(args.final)+"/"

# print(output_dir)

drop_llama_layers(input_dir=args.model, output_dir=output_dir, drop_layers=[i for i in range(args.initial,args.final+1)])


# Load the JSON file
with open(input_dir + '/tokenizer_config.json', 'r') as f:
    data = json.load(f)
# 1. Edit "model_max_length"
data['model_max_length'] = 131072
# 2. Remove "pad_token" if it exists
data.pop('pad_token', None)
# 3. Remove "padding_side" if it exists
data.pop('padding_side', None)
# 4. Change tokenizer_class
if data.get('tokenizer_class') == 'PreTrainedTokenizer':
    data['tokenizer_class'] = 'PreTrainedTokenizerFast'
# Save the modified JSON back
with open(output_dir + '/tokenizer_config.json', 'w') as f:
    json.dump(data, f, indent=2)