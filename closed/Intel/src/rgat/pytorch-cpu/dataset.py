import argparse
import time
import numpy as np
import torch
import os.path as osp

import torch
import dgl
from dgl.data import DGLDataset
from tpp_pytorch_extension.gnn.common_inference import gnn_utils
from tpp_pytorch_extension._C._qtype import create_qtensor_int8sym

import warnings
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("GNN-Dataset")


class IGBHeteroDGLDataset(DGLDataset):
    def __init__(self,
            path,
            dataset_size='tiny',
            use_label_2K=False,
            in_memory=True,
            fanout=[15, 10, 5]):
        
        self.dir = path
        self.dataset_size = dataset_size
        self.use_label_2K = use_label_2K
        self.fan_out = fanout
        self.in_memory = in_memory
        self.ntypes = ['author', 'institute', 'fos', 'conference', 'journal', 'paper']
        self.etypes = None
        self.edge_dict = {}
        self.paper_nodes_num = {'tiny':100000, 'small':1000000, 'medium':10000000, 'large':100000000, 'full':269346174}
        self.author_nodes_num = {'tiny':357041, 'small':1926066, 'medium':15544654, 'large':116959896, 'full':277220883}

    def create_graph(self):
        log.info("Constructing graph")

        path = osp.join(self.dir, self.dataset_size, 'struct.graph')
        try:
            self.graph = dgl.data.utils.load_graphs(path)[0][0]
        except:
            print(f'Could not load graph from {path}')

        self.graph.predict = 'paper'
        
        print(self.graph)

        log.info("Loading features")

        label_file = 'node_label_19.npy' if not self.use_label_2K else 'node_label_2K.npy'
        
        paper_feat_path = osp.join(self.dir, self.dataset_size, 'processed', 'paper', 'node_feat_int8.pt')
        paper_scf_path = osp.join(self.dir, self.dataset_size, 'processed', 'paper', 'node_feat_scf.pt')
        paper_lbl_path = osp.join(self.dir, self.dataset_size, 'processed', 'paper', label_file)

        paper_node_features = torch.load(paper_feat_path, mmap=self.in_memory, weights_only=True)
        paper_feat_scf = torch.load(paper_scf_path, mmap=self.in_memory, weights_only=True)
        paper_node_labels = torch.from_numpy(np.fromfile(paper_lbl_path, dtype=np.float32)).to(torch.long)
        self.in_feats = paper_node_features.shape[1]
        
        author_feat_path = osp.join(self.dir, self.dataset_size, 'processed', 'author', 'node_feat_int8.pt')
        author_scf_path = osp.join(self.dir, self.dataset_size, 'processed', 'author', 'node_feat_scf.pt')
        author_node_features = torch.load(author_feat_path, mmap=self.in_memory, weights_only=True)
        author_feat_scf = torch.load(author_scf_path, mmap=self.in_memory, weights_only=True)
        
        institute_feat_path = osp.join(self.dir, self.dataset_size, 'processed','institute', 'node_feat_int8.pt')
        institute_scf_path = osp.join(self.dir, self.dataset_size, 'processed', 'institute', 'node_feat_scf.pt')
        institute_node_features = torch.load(institute_feat_path, mmap=self.in_memory, weights_only=True)
        institute_feat_scf = torch.load(institute_scf_path, mmap=self.in_memory, weights_only=True)
        
        fos_feat_path = osp.join(self.dir, self.dataset_size, 'processed', 'fos', 'node_feat_int8.pt')
        fos_scf_path = osp.join(self.dir, self.dataset_size, 'processed', 'fos', 'node_feat_scf.pt')
        fos_node_features = torch.load(fos_feat_path, mmap=self.in_memory, weights_only=True)
        fos_feat_scf = torch.load(fos_scf_path, mmap=self.in_memory, weights_only=True)
        
        conference_feat_path = osp.join(self.dir, self.dataset_size, 'processed', 'conference', 'node_feat_int8.pt')
        conference_scf_path = osp.join(self.dir, self.dataset_size, 'processed', 'conference', 'node_feat_scf.pt')
        conference_node_features = torch.load(conference_feat_path, mmap=self.in_memory, weights_only=True)
        conference_feat_scf = torch.load(conference_scf_path, mmap=self.in_memory, weights_only=True)
        
        journal_feat_path = osp.join(self.dir, self.dataset_size, 'processed', 'journal', 'node_feat_int8.pt')
        journal_scf_path = osp.join(self.dir, self.dataset_size, 'processed', 'journal', 'node_feat_scf.pt')
        journal_node_features = torch.load(journal_feat_path, mmap=self.in_memory, weights_only=True)
        journal_feat_scf = torch.load(journal_scf_path, mmap=self.in_memory, weights_only=True)
        
        num_paper_nodes = self.paper_nodes_num[self.dataset_size]
        graph_paper_nodes = self.graph.num_nodes('paper')
        
        block_size = 128
        
        if graph_paper_nodes < num_paper_nodes:
            log.info("graph_paper < num_paper")
            self.graph.nodes['paper'].data['feat'] = create_qtensor_int8sym(paper_node_features[0:graph_paper_nodes,:], paper_feat_scf[0:graph_paper_nodes], block_size, 1, False)
            
            self.graph.num_paper_nodes = graph_paper_nodes
            self.graph.nodes['paper'].data['label'] = paper_node_labels[0:graph_paper_nodes]
        else:
            self.graph.nodes['paper'].data['feat'] = create_qtensor_int8sym(paper_node_features, paper_feat_scf, block_size, 1, False)
            
            self.graph.num_paper_nodes = paper_node_features.shape[0]
            self.graph.nodes['paper'].data['label'] = paper_node_labels[0:graph_paper_nodes]
        
        self.graph.nodes['author'].data['feat'] = create_qtensor_int8sym(author_node_features, author_feat_scf, block_size, 1, False)
        self.graph.num_author_nodes = author_node_features.shape[0]
      
        self.graph.nodes['fos'].data['feat'] = create_qtensor_int8sym(fos_node_features, fos_feat_scf, block_size, 1, False)
        self.graph.num_fos_nodes = fos_node_features.shape[0]

        self.graph.nodes['institute'].data['feat'] = create_qtensor_int8sym(institute_node_features, institute_feat_scf, block_size, 1, False)
        self.graph.num_institute_nodes = institute_node_features.shape[0]

        self.graph.nodes['conference'].data['feat'] = create_qtensor_int8sym(conference_node_features, conference_feat_scf, block_size, 1, False)
        
        self.graph.num_journal_nodes = journal_node_features.shape[0]
        self.graph.nodes['journal'].data['feat'] = create_qtensor_int8sym(journal_node_features, journal_feat_scf, block_size, 1, False)
        
        self.val_idx = torch.load(osp.join(self.dir, self.dataset_size, 'processed', "val_idx.pt"), weights_only=True)
        print(f"val id list size {self.val_idx.size()}")

    def create_node_sampler(self, fan_out=[15,10,5], use_fused=True):
        self.sampler = dgl.dataloading.NeighborSampler(fan_out,
            #prefetch_node_feats={k: ['feat'] for k in self.graph.ntypes},
            #prefetch_labels={self.graph.predict: ['label']},
            fused=use_fused
            )
        print(f"Is used fused: {use_fused}")

    def __getitem__(self):
        return self.graph

    def __len__(self):
        return 1

    def get_batch(self, batch_id_list):        
        node_ids = [self.val_idx[i] for i in batch_id_list]
        _, _, block = self.sampler.sample_blocks(self.graph, {'paper': torch.tensor(node_ids, device='cpu')})

        # for i in range(len(self.fan_out)):
        #     for ntype in self.graph.ntypes:
        #         block[i].apply_nodes(lambda nodes: {'feat': nodes.data['feat'].to(torch.bfloat16)}, ntype=ntype) #.to(dtype)}, ntype=ntype)
    
        return block #(block, block[0].srcdata['feat'])

    def load_subtensor_dict(self, nfeat, input_nodes):
        """
        Extracts features and labels for a set of nodes.
        """
        batch_inputs={}
        ntypes = nfeat.keys()
        for ntype in ntypes:
            batch_inputs[ntype] = gnn_utils.gather_features(nfeat[ntype], input_nodes[ntype])

        return batch_inputs

    def load_samples_to_ram(self, sample_index_list):
        log.info("Loading samples to ram")
        tic = time.time()
        self.iterator = None
        '''
        self.iterator = dgl.dataloading.DataLoader(self.graph,
                    {self.graph.predict: sample_index_list}, 
                    self.subgraph_sampler,
                    batch_size=1,
                    shuffle=False
                    )
        '''
        toc = time.time()
        log.info(f"Loading {len(sample_index_list)} samples to ram took {round(toc-tic,3)} sec")

    def get_sample_iterator(self, index_list, batch_size=1):
        return self.iterator

    def collate_fn(self, batch):
        pass

    def unload_samples_from_ram(self, sample_index_list):
        del self.iterator
