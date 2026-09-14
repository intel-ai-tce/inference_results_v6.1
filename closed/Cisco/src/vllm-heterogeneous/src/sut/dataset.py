import os
import logging

import numpy as np

log = logging.getLogger(__name__)


class Dataset:
    def __init__(self, total_sample_count=24576, perf_count_override=None,
                 dataset_path=None, prompt_source=None, tokenizer_path=None,
                 add_special_tokens=True):
        self.dataset_path = dataset_path
        self.prompt_source = prompt_source
        self.tokenizer_path = tokenizer_path
        self.add_special_tokens = add_special_tokens
        self._load()
        self.total_sample_count = min(len(self.input_ids), total_sample_count)
        self.perf_count = perf_count_override or self.total_sample_count

    def _load(self):
        if os.path.isdir(self.dataset_path):
            self._load_npy_dir(self.dataset_path)
        elif os.path.isfile(self.dataset_path):
            self._load_tabular(self.dataset_path)
        else:
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

    @staticmethod
    def _find_npy(dirpath, prefix):
        """Find an npy file matching the canonical or prefixed name."""
        exact = os.path.join(dirpath, f"{prefix}.npy")
        if os.path.isfile(exact):
            return exact
        for fname in sorted(os.listdir(dirpath)):
            if fname.endswith(".npy") and (
                fname.startswith(prefix) or fname.endswith(f"_{prefix}.npy")
            ):
                return os.path.join(dirpath, fname)
        return None

    def _load_npy_dir(self, dirpath):
        """Load from a directory containing input_ids_padded*.npy + input_lens*.npy."""
        ids_path = self._find_npy(dirpath, "input_ids_padded")
        lens_path = self._find_npy(dirpath, "input_lens")

        if not ids_path:
            raise FileNotFoundError(
                f"No input_ids_padded*.npy found in {dirpath}")
        if not lens_path:
            raise FileNotFoundError(
                f"No input_lens*.npy found in {dirpath}")

        padded = np.load(ids_path)
        lengths = np.load(lens_path)

        self.input_ids = [
            padded[i, :lengths[i]].tolist() for i in range(len(lengths))
        ]
        self.text_prompts = None
        self.stop_ids = []

        log.info("Loaded %d samples from %s (npy, max_len=%d)",
                 len(self.input_ids), dirpath, int(lengths.max()))

    def _load_tabular(self, filepath):
        """Load from a pickle or parquet file."""
        import pandas as pd
        if filepath.endswith(".parquet"):
            data = pd.read_parquet(filepath)
        elif filepath.endswith(".json"):
            data = pd.read_json(filepath)
        else:
            data = pd.read_pickle(filepath)

        if self.prompt_source in ("text_input", "templated_text_input"):
            col = self.prompt_source
            if col not in data:
                raise ValueError(f"Prompt source column not found: {col}")
            if not self.tokenizer_path:
                raise ValueError(
                    f"tokenizer_path is required for prompt_source={col}")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path, trust_remote_code=True)
            self.input_ids = [
                tokenizer.encode(
                    str(x), add_special_tokens=self.add_special_tokens)
                for x in data[col]
            ]
            self.text_prompts = [str(x) for x in data[col]]
            log.info("Loaded %d samples from %s (%s tokenized locally)",
                     len(self.input_ids), filepath, col)
        else:
            if self.prompt_source:
                col = self.prompt_source
                if col not in data:
                    raise ValueError(f"Prompt source column not found: {col}")
            elif "tok_input" in data:
                col = "tok_input"
            elif "input_tokens" in data:
                col = "input_tokens"
            else:
                raise ValueError("No input token column found in dataset")

            data[col] = data[col].apply(
                lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
            self.input_ids = list(data[col])
            self.text_prompts = None
            log.info("Loaded %d samples from %s (%s)",
                     len(self.input_ids), filepath, col)

        self.stop_ids = []
        if "tok_stop_sequence" in data.columns:
            self.stop_ids = list(data["tok_stop_sequence"])

    def LoadSamplesToRam(self, sample_list):
        pass

    def UnloadSamplesFromRam(self, sample_list):
        pass
