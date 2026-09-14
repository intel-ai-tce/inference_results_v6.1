import argparse
import numpy as np
import torch
import os


DTYPE_MAP = {'fp16': torch.float16,
            'fp8': torch.float8_e5m2
            }


def convert_and_save(node_feat_path, dtype='fp16', num_nodes=269346174, in_memory=False):
    if not os.path.exists(node_feat_path):
        print(f"{node_feat_path} not found")
        return

    torch_dtype = DTYPE_MAP[dtype]

    dirname = os.path.dirname(node_feat_path)
    output_path = os.path.join(dirname, f"node_feat_{dtype}.pt")
    if in_memory:
        node_features = torch.from_numpy(np.load(node_feat_path, mmap_mode='r')).to(torch_dtype)
    else:
        node_features = torch.from_numpy(np.memmap(node_feat_path, dtype='float32', mode='r', shape=(num_nodes,1024))).to(torch_dtype)

    torch.save(node_features, output_path)
    print(f"Node features saved at {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--node-feature-path', type=str, help="Path to node features")
    parser.add_argument('--dtype', type=str, default='fp16', help="Precision to save feats")
    parser.add_argument('--num-nodes', type=int, help="Number of nodes")
    parser.add_argument('--in-memory', action='store_true', help="load feature in memory to convert")

    args = parser.parse_args()

    convert_and_save(args.node_feature_path,
                    args.dtype.lower(),
                    args.num_nodes,
                    args.in_memory
                    )

if __name__=="__main__":
    main()

