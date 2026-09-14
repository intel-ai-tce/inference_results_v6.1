import os
import time
import sys

import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from harness_llm.backends.common.utils import ProgressReporter


DEBUG_MODEL_OUTPUT_PATH = 'model_output.txt'
DEBUG_SAMPLES_LATENCY_OUTPUT_PATH = 'samples_latency_data.txt'


class DebugToolkit:

    def __init__(self, harness_config: dict, llm_config: dict):
        self.samples_latency_data = {}
        self.harness_config = harness_config

        self.model_output_path = self.harness_config.get('debug_model_output_path', DEBUG_MODEL_OUTPUT_PATH)
        self.latency_output_path = self.harness_config.get('debug_latency_output_path', DEBUG_SAMPLES_LATENCY_OUTPUT_PATH)
        self.debug_print_finished = self.harness_config.get('debug_print_finished', False)
        self.debug_dataset_sampling_size = self.harness_config.get('debug_dataset_sampling_size', 0)
        self.debug_dataset_sampling_num_bins = self.harness_config.get('debug_dataset_sampling_num_bins', 10)
        self.debug_dataset_sampling_random_seed = self.harness_config.get('debug_dataset_sampling_random_seed', 42)
        if self.harness_config['debug_dump_model_output']:
            open(self.model_output_path, 'w').close()
        self.tokenizer = AutoTokenizer.from_pretrained(self._get_model_path(llm_config))

        if self.harness_config['debug_record_sample_latencies']:
            open(self.latency_output_path, 'w').close()


    def dump(self, text_token_ids):
        with open(self.model_output_path, 'a') as file:  
            for token_ids in text_token_ids:
                text = self.tokenizer.decode(token_ids)
                file.write(f'{text}\n\n')


    def record_sample_latencies(self, sample_id, output_token_ids, input_token_count=None):
        sample_data = self.samples_latency_data.get(sample_id)
        if sample_data is None:
            self.samples_latency_data[sample_id] = SampleData(sample_id)
            if input_token_count:
                self.samples_latency_data[sample_id].input_token_count = input_token_count
        else:
            if sample_data.first_token_time is None:
                sample_data.first_token_time = time.perf_counter_ns()
                assert output_token_ids != None, "output_token_ids should not be None when receiving the first token"

            if output_token_ids is None:
                sample_data.last_token_time = time.perf_counter_ns()

                with open(self.latency_output_path, 'a') as file:
                    ttft = sample_data.first_token_time - sample_data.sent_time
                    tpot = (sample_data.last_token_time - sample_data.first_token_time) / sample_data.output_token_count
                    file.write(f"{sample_data.id=} {ttft=:.0f} {tpot=:.0f} isl={sample_data.input_token_count} osl={sample_data.output_token_count}\n")
            else:
                sample_data.output_token_count += len(output_token_ids)

            self.samples_latency_data[sample_id] = sample_data

    def _get_model_path(self, config: dict):
        return config.get('model', config.get('model_path', None))

    def print_server_progress(self, n_finished, n_finished_first, servers, instance_count):
        tm = ProgressReporter.format_timestamp()

        msg = (
            "\r"
            + tm
            + "Processed prompts: "
            + str(n_finished)
            + " first tokens: "
            + str(n_finished_first)
            + " "
            + " | ".join((str(d)+":"+str(servers[d].sent)+"/"+str(servers[d].finished)+" q:"+str(servers[d].sent-servers[d].finished) for d in range(instance_count)))
            + " "
        )

        sys.stdout.write(msg)
        sys.stdout.flush()

    def _create_histogram_image(self, data_plot, filename, title="Histogram", bins=10):
        plt.figure(figsize=(10, 6))
        plt.hist(data_plot, bins=bins, color='skyblue', edgecolor='black')
        plt.title(title)
        plt.xlabel("Input Length")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)

        mean_val = np.mean(data_plot)
        median_val = np.median(data_plot)
        std_val = np.std(data_plot)

        stats_text = f"Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd Dev: {std_val:.2f}"
        plt.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                    ha='right', va='top')

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

        print(f"Histogram saved to {filename}")

    def save_sampling_histogram_image(self, data, title):
        output_dir = 'histogram_outputs'
        os.makedirs(output_dir, exist_ok=True)
        file_name = title.lower().replace(" ", "_")
        timestr = time.strftime("%Y%m%d-%H%M%S")
        histogram_file = os.path.join(output_dir, f"{file_name}_{timestr}.png")

        self._create_histogram_image(
            data_plot=data,
            filename=histogram_file,
            title=title,
            bins=self.debug_dataset_sampling_num_bins
        )


class SampleData:
    def __init__(self, id):
        self.id = id
        self.sent_time = time.perf_counter_ns()
        self.first_token_time = None
        self.last_token_time = None
        self.input_token_count = None
        self.output_token_count = 0
