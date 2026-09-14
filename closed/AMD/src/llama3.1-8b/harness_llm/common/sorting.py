import logging
import random
import heapq

from harness_llm.common.rpd_trace_utils import rpd_trace_range_non_timed
from time import time_ns


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__file__)


def validate_sorting_strategy(sorting):
    sorting_strategies = ("ascending", "descending", "lexicographic", "modulo_desc", "modulo_asc", "batchpacking", "ignore", "buckets", "greedy_pack")
    strategy = sorting.get("strategy", "ignore")
    if strategy not in sorting_strategies:
        raise ValueError(f"No such sorting strategy '{strategy}")
    if strategy == "buckets":
        if not hasattr(sorting, "buckets"):
            raise ValueError("Sorting strategy 'buckets' requires 'buckets' attribute in sorting metadata")
        if len(sorting["buckets"]) == 0:
            raise ValueError("Sorting strategy 'buckets' requires at least one bucket")
        if abs(sum(sorting["buckets"]) - 100) > 1e-6:
            raise ValueError(f"Sum of buckets {sum(sorting['buckets'])} must be close to 100")


class SortingStrategy:
    """Sorting strategy"""

    def __init__(
        self,
        data_object,
        max_num_batched_tokens,
        enable_optimizations = False
    ):
        self.data_object = data_object
        self.max_num_batched_tokens = max_num_batched_tokens
        self.num_partitions = 0
        self.enable_optimizations = enable_optimizations

    @rpd_trace_range_non_timed("SortingStrategy")
    def make_ranges(self, query_samples):
        query_chunk_size = (
            len(query_samples) + self.num_partitions - 1
        ) // self.num_partitions
        ranges = []
        for i in range(self.num_partitions):
            start = i * query_chunk_size
            end = start + query_chunk_size
            if end > len(query_samples):
                end = None
            ranges.append((start, end))
        return ranges
    
    @rpd_trace_range_non_timed("SortingStrategy")
    def make_bucket_ranges(self, query_samples, buckets):
        bucket_ranges = []
        total_samples = len(query_samples)
        current_start = 0

        for i, bucket_percentage in enumerate(buckets):
            bucket_size = int((bucket_percentage / 100.0) * total_samples)
            end = current_start + bucket_size
            if i == len(buckets) - 1 or end > total_samples:
                end = None
            
            bucket_ranges.append((current_start, end))
            if end is not None:
                current_start = end
            else:
                break
        log.info(f"Bucket ranges: {bucket_ranges}")
        return bucket_ranges


    @rpd_trace_range_non_timed("SortingStrategy")
    def sort_by_length(self, query_samples, weight=1):
        reord_start = time_ns()
        ranges = self.make_ranges(query_samples)
        evened_out_samples = self.even_out_token_count(
            query_samples, ranges[0][1] - ranges[0][0]
        )
        reordered_samples = []
        for start, stop in ranges:
            chunk = evened_out_samples[start:stop]
            chunk.sort(
                key=lambda sample: weight
                * len(self.data_object.input_ids[sample.index])
            )
            
            if self.enable_optimizations:
                # optimization: long-short interleaving here
                # for each chunk, long-short interleaving, j+2, j+3, k-2, k-3 (four different combinations)
                # add condition check to avoid index out of range problem

                l = len(chunk)
                j, k = 1, l-1
                while j+2 > k-2 and j < l-2 and k > 2:
                    tmp = chunk[j]
                    chunk[j] = chunk[k]
                    chunk[k] = tmp
                    j = j + 2
                    k = k-2
            
            reordered_samples.extend(chunk)
        reord_dur = (time_ns() - reord_start) / 1_000_000
        log.info(f"Reorder took: {reord_dur} ms")

        return ranges, reordered_samples


    @rpd_trace_range_non_timed("SortingStrategy")
    def batch_packing(self, query_samples, batch):
        reord_start = time_ns()
        ranges = self.make_ranges(query_samples)
        evened_out_samples = self.even_out_token_count(
            query_samples, ranges[0][1] - ranges[0][0]
        )
        reordered_samples = []
        for start, stop in ranges:
            chunk = evened_out_samples[start:stop]
            # Sort by input_ids length in descending
            chunk.sort(key=lambda sample: len(self.data_object.input_ids[sample.index]), reverse=True)
            # bacth size should be larger than the longest input
            if batch > len(self.data_object.input_ids[chunk[0].index]):
                while len(chunk) > 0:
                    capacity = batch
                    i = 0
                    # Packing by batch size with greedy algorithm
                    while i < len(chunk):
                        if capacity > len(self.data_object.input_ids[chunk[i].index]):
                            capacity = capacity - len(self.data_object.input_ids[chunk[0].index])
                            # looks like there is a typo/bug in the above line here for chunk index
                            #capacity = capacity - len(self.data_object.input_ids[chunk[i].index])
                            reordered_samples.append(chunk[i])
                            chunk.pop(i)
                        else:
                            i = i + 1
        reord_dur = (time_ns() - reord_start) / 1_000_000
        log.info(f"Batch packing took: {reord_dur} ms")

        return ranges, reordered_samples


    @rpd_trace_range_non_timed("SortingStrategy")
    def sort_lexicog(self, query_samples):
        reord_start = time_ns()
        ranges = self.make_ranges(query_samples)
        evened_out_samples = self.even_out_token_count(
            query_samples, ranges[0][1] - ranges[0][0]
        )
        reordered_samples = []
        for start, stop in ranges:
            chunk = evened_out_samples[start:stop]
            chunk.sort(key=lambda sample: self.data_object.input_ids[sample.index])
            reordered_samples.extend(chunk)
        reord_dur = (time_ns() - reord_start) / 1_000_000
        log.info(f"Reorder took: {reord_dur} ms")

        return ranges, reordered_samples


    @rpd_trace_range_non_timed("SortingStrategy")
    def sort_modulo(self, query_samples, weight=1):
        reord_start = time_ns()

        query_samples.sort(
            key=lambda sample: weight
                * len(self.data_object.input_ids[sample.index])
        )

        parts = [[] for _ in range(self.num_partitions)]
        for index, value in enumerate(query_samples):
            part_index = index % self.num_partitions
            parts[part_index].append(value)
                
        # example, random.shuffle(query_samples)
        # added random shuffle to see the effect
        #for i in range(len(parts)):
            #random.shuffle(parts[i])

        if self.enable_optimizations:
            # for each part, long-short interleaving
            for i in range(len(parts)):
                l = len(parts[i])
                j, k = 1, l-1
                while j+2 > k-2 and j < l-2 and k > 2:
                    tmp = parts[i][j]
                    parts[i][j] = parts[i][k]
                    parts[i][k] = tmp
                    j = j + 2
                    k = k-2
        
        reordered_samples = []
        for i in range(len(parts)):
            reordered_samples.extend(parts[i])

        ranges = self.make_ranges(query_samples)

        reord_dur = (time_ns() - reord_start) / 1_000_000
        log.info(f"Reorder took: {reord_dur} ms")

        return ranges, reordered_samples


    @rpd_trace_range_non_timed("SortingStrategy")
    def greedy_pack(self, query_samples, sorting=None):
        """Token-balanced Offline distribution across data-parallel instances.

        Assigns each query, in the exact order LoadGen issued it, to the instance
        with the smallest accumulated NORMALIZED token load (load / capacity),
        i.e. online least-loaded-bin / LPT. This uses token count -- not sample
        count -- as the balancing criterion so no single GPU straggles and gates
        Offline throughput.

        Capacity awareness: an optional ``sorting.capacities`` list (one entry per
        partition, in device order) lets a slower GPU receive proportionally less
        work so all instances finish together. Entries are relative per-GPU
        throughput in any consistent unit (e.g. the samples/s printed by the
        engine's ``[LB]`` log). When absent, all capacities are 1.0, which reduces
        to pure equal token balancing.

        Unlike the *_sort strategies this does NOT reorder the stream by length:
        samples keep their arrival order within each instance, only the choice of
        which instance handles each sample is load-driven. It is therefore a pure
        distribution step, not a sort of the processing stream.

        Returns variable-size contiguous ranges that map 1:1 onto the returned
        reordered sample list consumed by OfflineBaseSUT.issue_queries.
        """
        reord_start = time_ns()
        input_ids = self.data_object.input_ids
        n = self.num_partitions

        capacities = sorting.get("capacities", None) if sorting is not None else None
        if capacities is not None:
            capacities = [float(c) for c in capacities]
            if len(capacities) != n:
                raise ValueError(
                    f"sorting.capacities has {len(capacities)} entries but there "
                    f"are {n} partitions"
                )
            if any(c <= 0 for c in capacities):
                raise ValueError("sorting.capacities entries must all be > 0")
        else:
            capacities = [1.0] * n

        bins = [[] for _ in range(n)]
        loads = [0] * n
        # Min-heap keyed on (normalized_load, partition_index) where normalized
        # load = accumulated_tokens / capacity ~ predicted finish time. The second
        # element is a deterministic tie-breaker so equal loads route to the
        # lowest partition index first.
        heap = [(0.0, i) for i in range(n)]
        heapq.heapify(heap)
        for sample in query_samples:
            _, part = heapq.heappop(heap)
            bins[part].append(sample)
            loads[part] += len(input_ids[sample.index])
            heapq.heappush(heap, (loads[part] / capacities[part], part))

        reordered_samples = []
        ranges = []
        start = 0
        for b in bins:
            end = start + len(b)
            ranges.append((start, end))
            reordered_samples.extend(b)
            start = end

        reord_dur = (time_ns() - reord_start) / 1_000_000
        log.info(
            f"Greedy token-balanced pack took: {reord_dur} ms; "
            f"capacities={capacities}; partition sizes={[len(b) for b in bins]}; "
            f"partition tokens={loads}"
        )
        return ranges, reordered_samples

    @rpd_trace_range_non_timed("SortingStrategy")
    def even_out_token_count(self, query_samples, query_chunk_size):
        full_buckets = []
        buckets = [[] for _ in range(self.num_partitions)]
        bucket_sizes = [0 for _ in range(self.num_partitions)]

        # test it out
        random.shuffle(query_samples)
        random.shuffle(query_samples)
        #random.shuffle(query_samples[:4000])
        #random.shuffle(query_samples[4001:])
        """ 
        if self.enable_optimizations:
            # optimization: random shuffle the samples before assigning them to buckets
            #random.shuffle(query_samples)

            splits = 25 #24, 25, 50, 20, 332, 15, 10, 13, 14, 16, 17, 18, 19, 26, 250, 500 
            span = len(query_samples)//splits
            i = 0
            while (i+2 <= splits):
                query_samples[i*span:(i+1)*span].sort(
                    key=lambda sample: -1
                        * len(self.data_object.input_ids[sample.index])
                )
                query_samples[(i+1)*span:(i+2)*span].sort(
                    key=lambda sample: 1
                        * len(self.data_object.input_ids[sample.index])
                )
                i = i + 2
            # random shuffle for the remaining ones
            random.shuffle(query_samples[(i+2)*span:])
            """    
        for sample in query_samples:
            smallest_bucket = bucket_sizes.index(min(bucket_sizes))
            buckets[smallest_bucket].append(sample)
            bucket_sizes[smallest_bucket] += len(
                self.data_object.input_ids[sample.index]
            )
            if len(buckets[smallest_bucket]) == query_chunk_size and len(buckets) > 1:
                full_buckets.append(buckets[smallest_bucket])
                del buckets[smallest_bucket]
                del bucket_sizes[smallest_bucket]
        reordered_samples = []
        for bucket in full_buckets + buckets:
            reordered_samples.extend(bucket)
        return reordered_samples


    @rpd_trace_range_non_timed("SortingStrategy")
    def sort_samples(self, query_samples, sorting, num_partitions):
        self.num_partitions = num_partitions
        sorting_strategy = sorting.get('strategy', 'ignore')
        if sorting_strategy != "ignore":
            log.info(f"Sorting samples in {sorting_strategy} order")
        if sorting_strategy == "ascending":
            return self.sort_by_length(query_samples, weight=1)
        elif sorting_strategy == "descending":
            return self.sort_by_length(query_samples, weight=-1)
        elif sorting_strategy == "batchpacking" and self.max_num_batched_tokens > 0:
            return self.batch_packing(query_samples, batch=self.max_num_batched_tokens)
        elif sorting_strategy == "lexicographic":
            return self.sort_lexicog(query_samples)
        elif sorting_strategy == "modulo_desc":
            return self.sort_modulo(query_samples, weight=-1)
        elif sorting_strategy == "modulo_asc":
            return self.sort_modulo(query_samples, weight=1)
        elif sorting_strategy == "greedy_pack":
            return self.greedy_pack(query_samples, sorting)
        elif sorting_strategy == "buckets":
            if len(sorting['buckets']) != self.num_partitions:
                raise ValueError(f"Number of buckets {len(sorting['buckets'])} does not match number of partitions {self.num_partitions}")
            return (self.make_bucket_ranges(query_samples, sorting['buckets']), query_samples)
        else:
            return (self.make_ranges(query_samples), query_samples)
