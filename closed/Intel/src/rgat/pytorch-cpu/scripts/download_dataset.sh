#!/bin/bash
# Redirect all output to 'output.log'
exec > /logs/data_download_output.log 2>&1
#download and prepare data

ntypes=('author' 'paper' 'fos' 'institute' 'conference' 'journal')
dataset_size="full"
CHECKSUMS=()
url=""
for ntype in "${ntypes[@]}"; do
    echo "Processing: $ntype"
    # Determine the URL based on ntype
    if [[ "$ntype" == "author" ]]; then
        url="https://igb-public-awsopen.s3.amazonaws.com/IGBH/processed/author/node_feat.npy"
        echo "/author/node_feat.npy This file size is 1TB , please expect longer downloading and validation time" 
         CHECKSUMS=(
            /data/IGBH/$dataset_size/processed/$ntype/node_feat.npy,2ec2512b554088381c04ec013e893c8d
          )
    elif [[ "$ntype" == "fos" ]]; then
        url="https://igb-public-awsopen.s3.amazonaws.com/IGBH/processed/fos/node_feat.npy"
         CHECKSUMS=(
            /data/IGBH/$dataset_size/processed/$ntype/node_feat.npy,3ef3df19e2475c387fec10bac82773df
          )
    elif [[ "$ntype" == "institute" ]]; then
        url="https://igb-public-awsopen.s3.amazonaws.com/IGBH/processed/institute/node_feat.npy"
         CHECKSUMS=(
            /data/IGBH/$dataset_size/processed/$ntype/node_feat.npy,12eaeced22d17b4e97d4b4742331c819
          )
    elif [[ "$ntype" == "conference" ]]; then
        url="https://igb-public-awsopen.s3.amazonaws.com/IGBH/processed/conference/node_feat.npy"
        CHECKSUMS=(
            /data/IGBH/$dataset_size/processed/$ntype/node_feat.npy,898ff529b8cf972261fedd50df6377f8
          )
    elif [[ "$ntype" == "journal" ]]; then
        url="https://igb-public-awsopen.s3.amazonaws.com/IGBH/processed/journal/node_feat.npy"
         CHECKSUMS=(
            /data/IGBH/$dataset_size/processed/$ntype/node_feat.npy,49d51b554b3004f10bee19d1c7f9b416
          )
    elif [[ "$ntype" == "paper" ]]; then
        url="https://igb-public-awsopen.s3.amazonaws.com/IGBH/processed/paper/node_feat.npy"
        echo "/paper/node_feat.npy This file size is 1TB , please expect longer downloading and validation time" 
         CHECKSUMS=(
            /data/IGBH/$dataset_size/processed/$ntype/node_feat.npy,71058b9ac8011bafa1c5467504452d13
          )
    fi
    pushd /data
    echo "Downloading: $ntype"
    wget -c -P /data/IGBH/$dataset_size/processed/$ntype $url
    popd
    echo "Beginning validation of downloaded file..."
    cd /workspace
    for ITEM in "${CHECKSUMS[@]}"; do
        FILENAME=$(echo ${ITEM} | cut -d',' -f1)
        CHECKSUM=$(echo ${ITEM} | cut -d',' -f2)
        bash run_validation.sh ${FILENAME} ${CHECKSUM}
    done
    pushd scripts/dataset
    echo "Begin coverting $ntype/node_feat.npy"
    bash cvt.sh /data/IGBH $dataset_size int8 128 $ntype
    echo "End coverting $ntype/node_feat.npy"
    popd
    # delete the proceesed original data to minimize space utilization
    rm /data/IGBH/$dataset_size/processed/$ntype/node_feat.npy
    echo "Deleted $ntype/node_feat.npy to optimize disk space" 
done
#struct.graph
edge_index=('paper__cites__paper' 'paper__written_by__author' 'author__affiliated_to__institute' 'paper__topic__fos' 'paper__published__journal' 'paper__venue__conference')
for index in "${edge_index[@]}"; do
    echo "Downloading edge_index : $index"
    url="https://igb-public-awsopen.s3.amazonaws.com/IGBH/processed/$index/edge_index.npy"
    wget -c -P /data/IGBH/$dataset_size/processed/$index $url
done
CHECKSUMS=(
            /data/IGBH/$dataset_size/processed/paper__cites__paper/edge_index.npy,f4897f53636c04a9c66f6063ec635c16
            /data/IGBH/$dataset_size/processed/paper__written_by__author/edge_index.npy,df39fe44bbcec93a640400e6d81ffcb5
            /data/IGBH/$dataset_size/processed/author__affiliated_to__institute/edge_index.npy,e35dba208f81e0987207f78787c75711
            /data/IGBH/$dataset_size/processed/paper__topic__fos/edge_index.npy,427fb350a248ee6eaa8c21cde942fda4
            /data/IGBH/$dataset_size/processed/paper__published__journal/edge_index.npy,38505e83bde8e5cf94ae0a85afa60e13
            /data/IGBH/$dataset_size/processed/paper__venue__conference/edge_index.npy,541b8d43cd93579305cfb71961e10a7d
          )
 echo "Begining validation of dowloaded file..."
 cd /workspace
 for ITEM in "${CHECKSUMS[@]}"; do
     FILENAME=$(echo ${ITEM} | cut -d',' -f1)
     CHECKSUM=$(echo ${ITEM} | cut -d',' -f2)
     bash run_validation.sh ${FILENAME} ${CHECKSUM}
 done
echo " running split_seed"
python split_seeds.py  --path /data/IGBH --dataset_size full
wget -c -P /data/IGBH/$dataset_size/processed/paper https://igb-public-awsopen.s3.amazonaws.com/IGBH/processed/paper/node_label_2K.npy
pushd scripts/store_graph
echo " generating store_graph"
bash store_graph.sh /data/ full
echo " store_graph is generated"
popd

