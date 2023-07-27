#/bin/bash

NGPUS=$1
# PY_ARGS=${@:2}

# set random port
while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

CUDA_VISIBLE_DEVICES=2,3 \
python -m torch.distributed.launch --nproc_per_node=${NGPUS} --rdzv_endpoint=localhost:${PORT} train.py \
    --config /data/kimgh/CenterPoint-custom/CenterPoint-static/configs/nia/radar/nia_centerpoint_voxelnet_01voxel_radar_train.py \
    --work_dir /data/kimgh/cp_test \
    --validate