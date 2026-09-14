import torch
import torch.nn.functional as F
import logging
from item import OutputItem

from rgat import RGAT, opt_impl, block

import tpp_pytorch_extension as ppx
from tpp_pytorch_extension._C._qtype import remap_and_quantize_qint8 # type: ignore

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Backend")

class Backend(object):
    def __init__(self, checkpoint_path, use_tpp=False, use_bf16=False, use_qint8_gemm=False):

        self.checkpoint_path = checkpoint_path
        self.use_tpp = use_tpp
        self.use_bf16 = use_bf16
        self.use_qint8_gemm = use_qint8_gemm

        self.etypes = ['affiliated_to',
                'rev_written_by',
                'rev_venue',
                'rev_topic',
                'rev_affiliated_to',
                'rev_published',
                'cites',
                'published',
                'topic',
                'venue',
                'written_by',
                ]

        self.in_feats = 1024
        self.hidden_ch = 512
        self.n_classes = 2983
        self.num_layers = 3
        self.num_heads = 4


    def load_state_dict(self):
        ckpt = torch.load(self.checkpoint_path, map_location=torch.device('cpu'), weights_only=True)
        ckpt = ckpt['model_state_dict']
        for key in (list(ckpt.keys())):
            newkey = key.replace('module.', '')
            ckpt[newkey] = ckpt.pop(key)
        return ckpt

    def load_model(self):
        log.info("Loading model")
        self.model = RGAT(self.etypes, self.use_tpp, self.use_qint8_gemm, self.use_bf16).to('cpu')
        self.model.eval()

        self.model.load_state_dict(self.load_state_dict())
        if self.use_tpp:
            block(self.model.model)

        # import intel_extension_for_pytorch as ipex
        # self.model = ipex.optimize(self.model)
        
        if self.use_qint8_gemm:
            for l in range(len(self.model.model.layers)):
                mkeys = self.model.model.layers[l].mods.keys()
                for k in mkeys:
                    self.model.model.layers[l].mods[k].fc.weight = \
                        torch.nn.Parameter(remap_and_quantize_qint8(self.model.model.layers[l].mods[k].fc.weight), requires_grad=False)
        else:
            model.to(torch.bfloat16)

        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
    
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_mb = (param_size + buffer_size) / 1024**2
        print('model size: {:.3f}MB'.format(size_mb))

    def __call__(self, inputs, query_id_list=[]):
        with torch.no_grad():
            outputs = self.model.forward_gather(inputs[0], inputs[1], inputs[2]).argmax(1).cpu().to(torch.float32).numpy()
            return OutputItem(query_id_list, outputs)
