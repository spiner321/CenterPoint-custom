#/bin/bash

NGPUS=$1
PY_ARGS=${@:2}

python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py ${PY_ARGS}