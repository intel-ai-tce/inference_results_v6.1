from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(next(iter(self.encodings.values())))
        

def get_mlperf_data(data_path: str,
                   tokenizer: AutoTokenizer = None,
                   batch_size: int = 1,
                   num_calib_data: int = 128,
                   seqlen: int = 2048,
                   device: str = 'cpu') -> DataLoader[torch.Tensor]:

    print("mlperf calibration data path: ", data_path)

    import pandas as pd
    if data_path.endswith(".parquet"):
        mlperf_df = pd.read_parquet(data_path)
    else:
        mlperf_df = pd.read_pickle(data_path)

    if 'input' in mlperf_df:
        input_token_col = 'input'
    elif 'text_input' in mlperf_df:
        input_token_col = 'text_input'
    else:
        raise RuntimeError(f"Input tokens not found in processed data")

    system_prompt_instruction = mlperf_df[input_token_col].tolist()[:num_calib_data]

    batch_encoded = tokenizer.batch_encode_plus(
        system_prompt_instruction,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=seqlen,
    )
    if device:
        batch_encoded = batch_encoded.to(device)

    tokenized_dataset = CustomDataset({"input_ids": batch_encoded["input_ids"]})

    calib_dataloader: DataLoader[List[Dict[str, torch.Tensor]]] = DataLoader(tokenized_dataset, batch_size=batch_size,
                                                                             shuffle=False, drop_last=True)  # type: ignore

    return calib_dataloader
