#!/bin/bash

#https://github.com/IllinoisGraphBenchmark/IGB-Datasets/blob/main/igb/download_igbh600m.sh
data_dir=${1:-PWD}
WORKDIR=${PWD}

echo "IGBH600M download starting"

if [ ! -d ${data_dir}]; then
	mkdir -p ${data_dir}
fi

cd ${data_dir}
if [ ! -d ${data_dir}/igbh/full/processed ]; then
	mkdir -p ${data_dir}/igbh/full/processed
fi

cd igbh/full/processed

# paper
mkdir paper
cd paper
wget -c https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/paper/node_feat.npy
wget -c https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/paper/node_label_19.npy
wget -c https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/paper/node_label_2K.npy
wget -c https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/paper/paper_id_index_mapping.npy

NODE_FEAT_PATH=${PWD}/node_feat.npy
COMPRESSED_FEAT=${PWD}/node_feat_fp16.pt

python ${WORKDIR}/compress_node_features.py --node-feature-path ${NODE_FEAT_PATH} --dtype fp16 --num-nodes 269346174

if [ -f ${COMPRESSED_FEAT} ]; then
        rm ${NODE_FEAT_PATH}
fi

cd ..

# paper__cites__paper
mkdir paper__cites__paper
cd paper__cites__paper
wget -c https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/paper__cites__paper/edge_index.npy

cd ..

# author
mkdir author
cd author
wget -c https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/author/author_id_index_mapping.npy
wget -c https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/author/node_feat.npy

NODE_FEAT_PATH=${PWD}/node_feat.npy
COMPRESSED_FEAT=${PWD}/node_feat_fp16.pt

python ${WORKDIR}/compress_node_features.py --node-feature-path ${NODE_FEAT_PATH} --dtype fp16 --num-nodes 277220883

if [ -f ${COMPRESSED_FEAT} ]; then
        rm ${NODE_FEAT_PATH}
fi

cd ..


# conference
mkdir conference
cd conference
wget -c https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/conference/conference_id_index_mapping.npy
wget -c https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/conference/node_feat.npy

NODE_FEAT_PATH=${PWD}/node_feat.npy
COMPRESSED_FEAT=${PWD}/node_feat_fp16.pt
python ${WORKDIR}/compress_node_features.py --node-feature-path ${NODE_FEAT_PATH} --dtype fp16 --in-memory

if [ -f ${COMPRESSED_FEAT} ]; then
        rm ${NODE_FEAT_PATH}
fi

cd ..

# institute
mkdir institute
cd institute
wget -c https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/institute/institute_id_index_mapping.npy
wget -c https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/institute/node_feat.npy

NODE_FEAT_PATH=${PWD}/node_feat.npy
COMPRESSED_FEAT=${PWD}/node_feat_fp16.pt
python ${WORKDIR}/compress_node_features.py --node-feature-path ${NODE_FEAT_PATH} --dtype fp16 --in-memory

if [ -f ${COMPRESSED_FEAT} ]; then
        rm ${NODE_FEAT_PATH}
fi

cd ..

# journal
mkdir journal
cd journal
wget -c https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/journal/journal_id_index_mapping.npy
wget -c https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/journal/node_feat.npy

NODE_FEAT_PATH=${PWD}/node_feat.npy
COMPRESSED_FEAT=${PWD}/node_feat_fp16.pt
python ${WORKDIR}/compress_node_features.py --node-feature-path ${NODE_FEAT_PATH} --dtype fp16 --in-memory

if [ -f ${COMPRESSED_FEAT} ]; then
        rm ${NODE_FEAT_PATH}
fi

cd ..

# fos
mkdir fos
cd fos
wget -c https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/fos/fos_id_index_mapping.npy
wget -c https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/fos/node_feat.npy

NODE_FEAT_PATH=${PWD}/node_feat.npy
COMPRESSED_FEAT=${PWD}/node_feat_fp16.pt
python ${WORKDIR}/compress_node_features.py --node-feature-path ${NODE_FEAT_PATH} --dtype fp16 --in-memory

if [ -f ${COMPRESSED_FEAT} ]; then
        rm ${NODE_FEAT_PATH}
fi

cd ..

# author__affiliated_to__institute
mkdir author__affiliated_to__institute
cd author__affiliated_to__institute
wget -c https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/author__affiliated_to__institute/edge_index.npy
cd ..

# paper__published__journal
mkdir paper__published__journal
cd paper__published__journal
wget -c https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/paper__published__journal/edge_index.npy
cd ..

# paper__topic__fos
mkdir paper__topic__fos
cd paper__topic__fos
wget -c https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/paper__topic__fos/edge_index.npy
cd ..

# paper__venue__conference
mkdir paper__venue__conference
cd paper__venue__conference
wget -c https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/paper__venue__conference/edge_index.npy
cd ..

# paper__written_by__author
mkdir paper__written_by__author
cd paper__written_by__author
wget -c https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/paper__written_by__author/edge_index.npy
cd ..

echo "IGBH-IGBH download complete"
