import multiprocessing as mp
import threading
import queue
import logging
import time
import array
import numpy as np
import mlperf_loadgen as lg

from consumer import Consumer
from item import InputItem

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("RGAT-SUT")

class SUT():

    def __init__(self,
                checkpoint_path,
                dataset_path,
                fan_out=[-1,-1,-1],
                batch_size=1,
                num_proc=4,
                workers_per_proc=1,
                cpus_per_proc=30,
                start_core=1,
                warmup = False,
                performance_sample_count=1024,
                total_sample_count=24576,
                use_tpp=False,
                use_bf16=False,
                use_qint8_gemm=False,
                use_fused_sampler=True,
                accuracy=False
                ):

        self.checkpoint_path = checkpoint_path
        self.dataset_path = dataset_path
        self.fan_out = fan_out
        self.batch_size = batch_size
        self.num_proc = num_proc
        self.workers_per_proc = workers_per_proc
        self.cpus_per_proc = cpus_per_proc
        self.initial_core = start_core
        self.warmup = warmup
        
        self.procs = [None] * self.num_proc
        self.total_workers = self.num_proc # * self.workers_per_proc

        self.performance_sample_count = performance_sample_count
        self.total_sample_count = total_sample_count
        
        self.use_tpp = use_tpp
        self.use_bf16 = use_bf16
        self.use_qint8_gemm = use_qint8_gemm
        self.use_fused_sampler = use_fused_sampler

        self.accuracy = accuracy

        self.lock = mp.Lock()
        self.init_counter = mp.Value("i", 0)
        self.input_queue = mp.JoinableQueue()
        self.output_queue = mp.Queue()

        self.cv = mp.Condition(lock=self.lock)
        self.mp_context = mp.get_context('spawn')
        self.manager = self.mp_context.Manager()
        self.pred_labels = self.manager.list([])

    def start(self):
        """ Creates and Starts the processes and threads"""

        # Create processes
        self.createProcesses()
        
        # Start processes
        log.info("Starting processes")
        for proc in self.procs:
            if proc is None:
                raise RuntimeError("Process is None")
            proc.start()
        
        # Wait for all consumers to be ready (including if they're warming up)
        with self.cv:
            self.cv.wait_for(lambda : self.init_counter.value==self.total_workers)

        # Start Loadgen response thread
        self.response_thread = threading.Thread(target=self.responseLoadgen)
        self.response_thread.start()

    def stop(self):
        """ Stops processes and threads and exit """

        for _ in range(self.total_workers * self.workers_per_proc):
            self.input_queue.put(None)

        for proc in self.procs:
            proc.join()

        self.output_queue.put(None)

        predictions = []
        labels = []
        if self.accuracy:
            for preds, labs in self.pred_labels:
                predictions.append(preds)
                labels.append(labs)

            preds = np.concatenate(predictions)
            labs = np.concatenate(labels)
            from sklearn import metrics
            acc = metrics.accuracy_score(labs, preds)
            print(f"Accuracy: {acc}")


    def createProcesses(self):
        """ Create 'mp' instances or processes"""

        #start_core = self.initial_core
        for proc_idx in range(self.num_proc):
            self.procs[proc_idx] = Consumer(self.checkpoint_path, self.dataset_path, self.input_queue, self.output_queue, self.lock, self.cv, self.init_counter, proc_idx, self.initial_core[proc_idx * self.workers_per_proc: (proc_idx + 1) * self.workers_per_proc], self.cpus_per_proc, self.workers_per_proc, warmup=self.warmup, fan_out=self.fan_out, batch_size = self.batch_size, use_tpp=self.use_tpp, use_bf16=self.use_bf16, use_qint8_gemm=self.use_qint8_gemm, pred_labels = self.pred_labels, accuracy=self.accuracy, use_fused_sampler=self.use_fused_sampler)

            # start_core += self.cpus_per_proc

    def get_sut(self):
        lgSUT = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return lgSUT
    
    def get_qsl(self):

        lgQSL = lg.ConstructQSL(self.total_sample_count,
                                self.performance_sample_count,
                                self.procs[0].data_obj.load_samples_to_ram,
                                self.procs[0].data_obj.unload_samples_from_ram)
        return lgQSL

    def issue_queries(self, sample_query_list):
        
        num_queries = len(sample_query_list)
        # sample_query_list.sort(key=lambda x : x.index)
        if num_queries==1:
            self.input_queue.put(sample_query_list)
            return

        j = 0
        bs = self.batch_size
        while j < len(sample_query_list):
            qbatch = sample_query_list[j:min(j+bs, num_queries)]
            qidx = [qitem.index for qitem in qbatch]
            resp_ids = [qitem.id for qitem in qbatch]
            qitem = InputItem(resp_ids, qidx)
            # log.info(f"putting batch {len(qbatch)} of {len(sample_query_list)}")
            self.input_queue.put(qitem)
            j += bs

    def flush_queries(self):
        pass

    def responseLoadgen(self):
        while True:
            next_task = self.output_queue.get()

            if next_task is None:
                log.info('Exiting response thread')
                break

            query_id_list = next_task.query_id_list
            processed_result = next_task.result
            # array_type_code = next_task.array_type_code
            # batch_size = len(query_id_list)

            for id, out in zip(query_id_list, processed_result):
                response_array = array.array("B", out.tobytes())
                bi = response_array.buffer_info()
                responses = [lg.QuerySampleResponse(id, bi[0], bi[1]*response_array.itemsize)]
                lg.QuerySamplesComplete(responses)

    def __del__(self):
        pass
