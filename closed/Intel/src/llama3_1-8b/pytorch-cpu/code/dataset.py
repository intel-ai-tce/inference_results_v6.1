import torch
import numpy as np
import logging
import json
import copy
import os
import pickle

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("DATASET")

from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_MODEL_LEN=int(os.environ.get('MAX_MODEL_LEN', 4096))
MAX_SAMPLES=13368 # maximum samples available in the dataset

PROMPT = ("Summarize the following news article in 128 tokens. Please output the summary only, without any other text.\n\nArticle:\n{input}\n\nSummary:")

class Dataset(object):
    """ Dataset class for cnn-dailymail """

    def __init__(self, dataset_path=None, model_name="Llama-3.1-8B-Instruct", total_sample_count=MAX_SAMPLES):
        self.dataset_path = dataset_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
            model_max_length=MAX_MODEL_LEN,
            padding_side="right",
            use_fast=True,)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.total_sample_count = total_sample_count
        self.loadDataset()

    def loadDataset(self):
        if not os.path.isfile(self.dataset_path):
            log.warn(
                "Processed pickle file {} not found. Please check that the path is correct".format(
                    self.dataset_path
                )
            )

        log.info("Loading dataset...")
        import pandas as pd

        self.processed_data = pd.read_json(self.dataset_path)

        self.input = self.processed_data.input.tolist()[:self.total_sample_count]
        self.input_ids = self.processed_data.tok_input.tolist()[:self.total_sample_count]
        self.input_lens = [len(x) for x in self.input_ids]
        self.targets = self.processed_data.output.tolist()[:self.total_sample_count]
        del self.processed_data
        log.info("Finished loading dataset.")

    def getInputLengths(self):
        return self.input_lens

    def postProcess(self, out_tokens):
        """ Postprocesses output prediction """
        #TODO: Create response object in postProcess(?)
        output_seq = out_tokens

        return [np.asarray(out, dtype=np.int32) for out in output_seq]

    def __getitem__(self, index):
        """ Returns sample at 'index' """
        if index >= len(self.input_ids) or index < 0:
            raise IndexError(f"Index {index} out of bounds for dataset of size {len(self.input_ids)}")
        input_ids = self.input_ids[index]
        input_len = self.input_lens[index]
        source = self.input[index]

        return input_ids, input_len, source, self.targets[index]

    def __len__(self):
        return len(self.dataset)

    @torch.no_grad()
    def getSamples(self, sample_index_list): # Assumes batch size 1
        """ Returns samples given 'sample_index_list' """
        if len(sample_index_list)==1:
            return self[sample_index_list[0]]

        input_ids_list = []
        input_len_list = []
        for index in sample_index_list:

            input_ids, input_len, _, _ = self[index]
            input_ids_list.append(input_ids)
            input_len_list.append(input_len)

        return input_ids_list, input_len_list, _, _