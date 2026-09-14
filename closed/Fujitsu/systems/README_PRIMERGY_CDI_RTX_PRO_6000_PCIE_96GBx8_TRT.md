# Informations of Nameplate power for PRIMERGY CDI

Currently, PRIMERGY CDI is primarily marketed in Japan, and sales in the Asia-Pacific region have only recently commenced. Due to the limited availability of English documentation for other regions at this time, we regret that we can only provide the product catalogs in Japanese.

Please note that PRIMERGY CDI is a composable system designed to allow for flexible resource allocation; therefore, a single server is configured using multiple interconnected units. The power information disclosed in PRIMERGY_CDI_RTX_PRO_6000_PCIE_96GBx8_TRT_power.yaml represents the specific components involved in the computation, namely the compute nodes and the PCIe Box.

Regarding the compute node, the PRIMERGY RX2530 M7 is available with various PSU options (500W, 900W, 1600W, 2200W, and 2400W). For this specific configuration, we have utilized two 1,600W PSUs.

Please find the specifications for the components below:


## PRIMERGY CDI Overview
https://www.fsastech.com/ja-jp/products/primergy/solution/cdi/pdf/cdi_catalog.pdf

## Compute Node
Model name: PRIMERGY RX2530 M7
https://www.fsastech.com/ja-jp/products/primergy/assets-i/pdf/c202404/rx2530m7_catalog.pdf
Description: 1U server node equipped with CPUs.
Power Configuration: 1,600W PSU × 2
Maximum Power Consumption: 2,608.6W

## PCIe Box for CDI (PCIe x10, 600W, PCIe Gen5)
Model Name: PY-PCD1P6
Overview: 10-slot expansion box for housing PCIe devices such as GPUs and SSDs
Power Configuration: 3,000W PSU x 4 (3+1 redundancy)
Maximum Power Consumption: 7,700W



