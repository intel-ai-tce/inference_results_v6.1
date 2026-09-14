import torch
import argparse
from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoProcessor
from pathlib import Path
import shutil

from auto_round import AutoRound


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/model/whisper-large-v3", required=False)
    parser.add_argument("--dataset_dir", required=False)
    parser.add_argument("--bits", default=4, help="number of bits for weight quantization")
    parser.add_argument("--group_size", default=-1, help="group_size")
    parser.add_argument("--sym", action='store_true', help="symmetric quant")
    parser.add_argument("--act_bits", default=8, help="number of bits for activation quantization")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    print(f"args: {args}")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_path, dtype=torch.float32, low_cpu_mem_usage=False, use_safetensors=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    processor = AutoProcessor.from_pretrained(args.model_path)

    output_dir = args.model_path + f"_calibrated-xpu"
    
    # quantize the model
    # '''
    autoround = AutoRound(model, tokenizer, processor,
                            bits=args.bits, group_size=int(args.group_size), sym=args.sym, # act_bits=args.act_bits,
                            iters=0, 
                            layer_config={
                                "out_proj": {
                                    "bits": 16, 
                                }
                            },
                        )
    autoround.quantize()
    print(f"Saving model to... {output_dir}")
    autoround.save_quantized(output_dir, format="auto_awq", inplace=False)
    # Copy preprocessor_config.json from original model (AutoRound doesn't generate this)
    
    preprocessor_file = Path(args.model_path) / "preprocessor_config.json"
    if preprocessor_file.exists():
        shutil.copy2(preprocessor_file, Path(output_dir) / "preprocessor_config.json")
        print(f" Copied preprocessor_config.json from {args.model_path}")
    else:
        print(f"preprocessor_config.json not found in {args.model_path}")

if __name__ == '__main__':
    main()
