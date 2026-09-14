# Bandwidth Requirement

Minimum network (ingress) bandwidth formulas per
[inference_rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#b1-ingress-bandwidth):

    DeepSeek-R1:   throughput * 3136  * dtype_size
    GPT-OSS-120B:  throughput * 15330 * dtype_size

Hosts use PCIe Gen5 (≈4 GB/s per lane, 64 GB/s per x16 link). Assuming a NIC on a PCIe x16
link to DRAM and the 4-bit input data types used in this submission, the maximum supported
QPS is far above the QPS achieved here:

| Benchmark    | Precision | Max supported QPS (64 GB/s ÷ (bytes/sample)) |
| ------------ | --------- | -------------------------------------------- |
| DeepSeek-R1  | 4 bit     | 64 GB / (3136 * 1 B)  = 10,204,082           |
| GPT-OSS-120B | 4 bit     | 64 GB / (15330 * 1 B) = 2,087,410            |

Note on the distributed SUT: at 512-GPU scale Crusoe runs a ZMQ distributed SUT (1 dedicated
head + 64× 8-GPU nodes). Only tokenized input/output **token streams** cross node boundaries
over 8× 400 Gbit/s RoCE per node (3200 Gbps/node aggregate); there is no cross-node RCCL /
collective traffic on the critical path. Inter-node bandwidth is therefore not a binding
constraint relative to the per-host ingress requirement above.
