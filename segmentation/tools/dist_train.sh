#!/usr/bin/env bash
CONFIG=$1
GPUS=$2
# PORT=7956
# PORT=7957
PORT=7966

OMP_NUM_THREADS=1 /home/kubernetes/dependency/cxk/paddle_develop/anaconda3-pytorch1.7-python3.7-cuda11_Semantic/bin/python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/train.py $CONFIG --launcher pytorch ${@:3} \
