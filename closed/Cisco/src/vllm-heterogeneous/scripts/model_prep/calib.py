from typing import Any

import torch
from torch.utils.data import DataLoader


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: dict[str, Any]):
        self.encodings = encodings

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {}
        for key, val in self.encodings.items():
            sample = val[idx]
            if torch.is_tensor(sample):
                item[key] = sample.detach().clone()
            else:
                item[key] = torch.tensor(sample)
        return item

    def __len__(self) -> int:
        return len(next(iter(self.encodings.values())))


def _truncate(ids: list[int], seqlen: int) -> list[int]:
    return ids[:seqlen]


def _collate_token_rows(batch: list[dict[str, torch.Tensor]], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].numel() for item in batch)
    rows = []
    masks = []
    for item in batch:
        ids = item["input_ids"]
        attention_mask = item.get("attention_mask")
        if ids.numel() < max_len:
            pad_len = max_len - ids.numel()
            pad = torch.full(
                (pad_len,),
                pad_token_id,
                dtype=ids.dtype,
                device=ids.device,
            )
            ids = torch.cat([ids, pad])
            if attention_mask is not None:
                mask_pad = torch.zeros(
                    (pad_len,),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([attention_mask, mask_pad])
        rows.append(ids)
        if attention_mask is not None:
            masks.append(attention_mask)
    result = {"input_ids": torch.stack(rows, dim=0)}
    if masks:
        result["attention_mask"] = torch.stack(masks, dim=0)
    return result


def get_mlperf_data(
    data_path: str,
    tokenizer=None,
    batch_size: int = 1,
    num_calib_data: int = 128,
    seqlen: int = 2048,
    device: str = "cpu",
) -> DataLoader:
    import pickle

    print("mlperf calibration data path:", data_path)
    with open(data_path, "rb") as fh:
        mlperf_df = pickle.load(fh)

    if "tok_input" in mlperf_df:
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id or 0
        import os

        bucket_multiple = int(os.environ.get("DEEPSEEK_CALIB_BUCKET_MULTIPLE", "128") or "0")
        rows = mlperf_df["tok_input"].tolist()[:num_calib_data]
        input_ids = []
        attention_masks = []
        truncated = 0
        raw_lengths = []
        padded_lengths = []
        for row in rows:
            ids = list(row)
            if len(ids) > seqlen:
                truncated += 1
            ids = _truncate(ids, seqlen)
            raw_len = len(ids)
            bucket_len = raw_len
            if bucket_multiple > 0 and bucket_len < seqlen:
                bucket_len = min(seqlen, ((bucket_len + bucket_multiple - 1) // bucket_multiple) * bucket_multiple)
            attention_mask = [1] * raw_len
            if bucket_len > raw_len:
                pad_len = bucket_len - raw_len
                ids = ids + [pad_token_id] * pad_len
                attention_mask = attention_mask + [0] * pad_len
            raw_lengths.append(raw_len)
            padded_lengths.append(len(ids))
            tensor = torch.tensor(ids, dtype=torch.long)
            mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
            if device:
                tensor = tensor.to(device)
                mask_tensor = mask_tensor.to(device)
            input_ids.append(tensor)
            attention_masks.append(mask_tensor)
        print(
            "mlperf tok_input calibration rows="
            f"{len(input_ids)} min_len={min(raw_lengths)} max_len={max(raw_lengths)} "
            f"avg_len={sum(raw_lengths) / len(raw_lengths):.1f} "
            f"padded_avg_len={sum(padded_lengths) / len(padded_lengths):.1f} "
            f"bucket_multiple={bucket_multiple} truncated={truncated}"
        )
        tokenized_dataset = CustomDataset({
            "input_ids": input_ids,
            "attention_mask": attention_masks,
        })
        return DataLoader(
            tokenized_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=lambda batch: _collate_token_rows(batch, pad_token_id),
        )
    else:
        if "input" in mlperf_df:
            input_col = "input"
        elif "templated_text_input" in mlperf_df:
            input_col = "templated_text_input"
        elif "text_input" in mlperf_df:
            input_col = "text_input"
        else:
            raise RuntimeError(f"Input prompts not found in calibration data: {data_path}")
        prompts = mlperf_df[input_col].tolist()[:num_calib_data]
        batch_encoded = tokenizer.batch_encode_plus(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=seqlen,
        )
        if device:
            batch_encoded = batch_encoded.to(device)
        tokenized_dataset = CustomDataset({
            "input_ids": batch_encoded["input_ids"],
            "attention_mask": batch_encoded["attention_mask"],
        })

    return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
