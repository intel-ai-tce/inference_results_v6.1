# Bandwidth Requirement

Formulas for minimum network bandwidth are as [follows](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#b1-ingress-bandwidth): 

DeepSeek-R1: $throughput * 3136 * dtype size$

GPT-OSS-120B: $throughput * 15330 * dtype size$

Llama2-70b: $throughput * 1024 * dtype size$

Llama3.1-405b: $throughput * 20000 * dtype size$

Modern servers use PCIe gen5, which provides throughput of 4 gigabyte/second per lane and 64 GB/s for x16 connection. Assuming that
a NIC is connected via a PCIe x16 to the DRAM and data types used in our submission, the maximum supported throughput is well above what was found in our submission

| Benchmark |   Precision |       QPS    |
| --------- |-------------|--------------|
| DeepSeek-R1| 4 bit| 64 GB/(3136 * 1 Byte) = 10204082 |
| GPT-OSS-120B| 4 bit| 64 GB/(15330 * 1 Byte) = 2087410 |
| Llama2-70b| 4 bit| 64 GB/(1024 * 1 Byte) = 31250000 |
| Llama3.1-405b| 4 bit| 64 GB/(20000 * 1 Byte) = 1600000 |
