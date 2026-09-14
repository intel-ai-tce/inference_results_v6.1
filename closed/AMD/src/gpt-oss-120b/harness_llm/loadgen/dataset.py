import os
import numpy as np

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__file__)

import random


class Dataset:
    def __init__(self,
        total_sample_count=24576,
        perf_count_override=None,
        dataset_path=None,
        debug_toolkit=None
    ):
        self.dataset_path = dataset_path
        self.debug_toolkit = debug_toolkit
        self.load_processed_dataset()
        self.total_sample_count = min(len(self.input_ids), total_sample_count)
        self.perf_count = perf_count_override or self.total_sample_count

    def load_processed_dataset(self):
        if not os.path.isfile(self.dataset_path):
            log.warning("Processed pickle file {} not found. Please check that the path is correct".format(self.dataset_path))

        log.info("Loading dataset...")
        import pandas as pd
        if self.dataset_path.endswith(".parquet"):
            processed_data = pd.read_parquet(self.dataset_path)
        else:
            processed_data = pd.read_pickle(self.dataset_path)

        input_tokens = []
        if 'tok_input' in processed_data:
            input_token_col = 'tok_input'
        elif 'input_tokens' in processed_data:
            input_token_col = 'input_tokens'
        else:
            log.error(f"Input tokens not found in processed data")

        # Convert values with type np.ndarray to list
        # May want to move this to prepare_dataset.sh?
        processed_data[input_token_col] = processed_data[input_token_col].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

        if self.debug_toolkit is not None and self.debug_toolkit.debug_dataset_sampling_size > 0:
            input_tokens = self.select_subset_of_dataset(processed_data[input_token_col])
        else:
            input_tokens = processed_data[input_token_col]

        self.input_ids = []
        self.attention_masks = []
        self.query_types = []

        for ids in input_tokens:
            self.input_ids.append(ids)

        self.stop_ids = []
        if 'tok_stop_sequence' in processed_data.columns:
            stop_tokens = processed_data['tok_stop_sequence']
            for ids in stop_tokens:
                self.stop_ids.append(ids)

        log.info("Finished loading dataset.")

    def select_subset_of_dataset(self, input_tokens):
        input_lengths = input_tokens.apply(len)
        num_bins = self.debug_toolkit.debug_dataset_sampling_num_bins
        sample_size = self.debug_toolkit.debug_dataset_sampling_size
        np.random.seed(self.debug_toolkit.debug_dataset_sampling_random_seed)

        self.debug_toolkit.save_sampling_histogram_image(
            data=input_lengths,
            title="Original Input Length Distribution")

        sampled_data = []
        sampled_input_lengths = []
        hist, bin_edges = np.histogram(input_lengths, bins=num_bins)
        for i in range(num_bins):  
            bin_indices = ((input_lengths >= bin_edges[i]) & (input_lengths < bin_edges[i + 1])).index  
            num_samples_bin = int(hist[i] / sum(hist) * sample_size)

            if len(bin_indices) > 0:  
                sampled_indices = np.random.choice(bin_indices, num_samples_bin, replace=False)  
                sampled_data.extend(input_tokens[sampled_indices])
                sampled_input_lengths.extend(input_tokens[sampled_indices].apply(len))

        self.debug_toolkit.save_sampling_histogram_image(
            data=sampled_input_lengths,
            title="Sampled Input Length Distribution")

        return sampled_data

    def postProcess(self, out_tokens, input_seq_lens=None, query_id_list=None, sample_index_list=None):
        pass

    def LoadSamplesToRam(self, sample_list):
        pass

    def UnloadSamplesFromRam(self, sample_list):
        pass

    def __del__(self):
        pass
