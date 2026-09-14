import multiprocessing as mp
import logging
import time
import os
import thread_binder # type: ignore
import collections
import secrets
import torch
from dataset import IGBHeteroDGLDataset
import dgl
from tpp_pytorch_extension.gnn.common_inference import gnn_utils
import tpp_pytorch_extension as ppx
import numpy as np
from backend import Backend

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Consumer")

class Consumer(mp.Process):
    def __init__(self, model_checkpoint_path, dataset_path, input_queue=None, out_queue=None, lock=None, cond_var=None, init_counter=None, proc_idx=None, start_core_idx=0, cpus_per_proc=30, workers_per_proc=1, warmup=False, fan_out=[-1,-1,-1], batch_size=1, use_tpp=False, use_bf16=False, use_qint8_gemm=False, pred_labels=None, accuracy=False, use_fused_sampler=True):

        mp.Process.__init__(self)
        self.model_checkpoint_path = model_checkpoint_path
        self.dataset_path = dataset_path
        self.task_queue = input_queue
        self.out_queue = out_queue
        self.lock = lock
        self.cond_var = cond_var
        self.init_counter = init_counter
        self.proc_idx = proc_idx

        self.start_core_idx = start_core_idx 
        self.num_cores = cpus_per_proc
        self.workers_per_proc = workers_per_proc
        self.cores_per_worker = self.num_cores // self.workers_per_proc
        self.warmup = warmup
        self.fan_out = fan_out
        self.batch_size = batch_size
        self.use_tpp = use_tpp
        self.use_bf16 = use_bf16
        self.use_qint8_gemm = use_qint8_gemm
        self.pred_labels = pred_labels
        self.accuracy = accuracy
        self.use_fused_sampler = use_fused_sampler

        # self.end_core_idx = start_core_idx + self.num_cores - 1
        self.affinity = [] # list(range(self.start_core_idx[0], self.num_cores + self.start_core_idx[0]))
        for i in self.start_core_idx:
            self.affinity.extend(list(range(i, self.cores_per_worker + i)))

        self.data_obj = IGBHeteroDGLDataset(self.dataset_path, 'full', use_label_2K=True, fanout=self.fan_out)

        self.workers = []
        self.latencies = collections.defaultdict(list)

        self.profile = False

    def load_dataset(self):
        self.data_obj.create_graph()
        self.nfeat = self.data_obj.graph.ndata['feat']
        self.labels = self.data_obj.graph.ndata['label']['paper']
        self.in_feats = self.data_obj.in_feats
    
    def doWarmup(self):
        log.info(f"Starting warmup")
        
        for i in range(0, 10):
            index = [secrets.randbelow(788379) for i in range(self.batch_size)] 
            tic = time.time()
            blocks = self.data_obj.get_batch(index)
            mid = time.time()
            
            input_nodes = blocks[0].srcdata[dgl.NID]
            
            '''
            batch_inputs = self.data_obj.load_subtensor_dict(
                self.nfeat, input_nodes
            )
            '''
            output = self.model([blocks, self.nfeat, input_nodes])

        log.info("Process {} Warmup Completed".format(self.proc_idx))
        with self.cond_var:
            self.init_counter.value += 1
            self.cond_var.notify()

    def handleTasks(self, i, task_queue, result_queue, start_core, num_cores):
        cores = [j for j in range(start_core, start_core + num_cores)]
        thread_binder.set_thread_affinity(cores)
        log.info(f"Process {self.proc_idx} binded to cores {cores}")

        with torch.autograd.profiler.profile(
            enabled=self.profile, record_shapes=False
            ) as prof:
            if self.use_tpp and self.profile:
                ppx.reset_debug_timers()
            while True:
                try:
                    next_task = task_queue.get()
                    if next_task is None:
                        log.info("Exiting worker thread : {}".format(i))
                        break

                    query_id_list = next_task.query_id_list
                    sample_index_list = next_task.sample_index_list
                    #log.info("Fetching batch")
                    inputs = self.data_obj.get_batch(sample_index_list)
                    input_nodes = inputs[0].srcdata[dgl.NID]

                    '''
                    batch_inputs = self.data_obj.load_subtensor_dict(
                        self.nfeat, input_nodes
                    )
                    '''
                    # startTime = time.time()
                    result = self.model([inputs, self.nfeat, input_nodes], query_id_list) 
                    # log.info(f"Process {self.proc_idx} worker {i} latency: {time.time() - startTime}")
                    #TODO: Remove this before submission
                    if self.accuracy:
                        labels = inputs[-1].dstdata['label']['paper'].cpu().numpy()
                        #log.info(f"Fetched ground truth labels of size {labels.shape}")
                        with self.cond_var:
                        #    log.info(f"{result.result[0]} vs {labels[0]}")
                            self.pred_labels.append((result.result, labels))
                            self.cond_var.notify()
                    result_queue.put(result)
                    task_queue.task_done()

                except Exception as ex:
                    # Error occured
                    log.error(ex)
                    break

        if prof:
            ppx.print_debug_timers(0)

    def run(self):
        os.sched_setaffinity(self.pid, self.affinity)
        log.info(f"Process {self.proc_idx} affinity {self.affinity}")
        self.load_dataset()
        self.data_obj.create_node_sampler(self.fan_out, use_fused=self.use_fused_sampler)
        
        self.model = Backend(self.model_checkpoint_path, use_tpp=self.use_tpp, use_bf16=self.use_bf16, use_qint8_gemm=self.use_qint8_gemm)

        # Load model
        self.model.load_model()

        # Do Warmup
        if self.warmup:
            self.doWarmup()

        else:
            with self.cond_var:
                self.init_counter.value += 1
                self.cond_var.notify()
        
        for i in range(self.workers_per_proc):
            log.info(f"Starting worker {i} on {self.start_core_idx[i]}, number of cores {self.cores_per_worker}")
            worker = mp.Process(target=self.handleTasks, args=(i, self.task_queue, self.out_queue, self.start_core_idx[i], self.cores_per_worker))
            self.workers.append(worker)

        for w in self.workers:
            w.start()

        for w in self.workers:
            w.join()

        log.info("{} : Exiting consumer process".format(os.getpid()))
