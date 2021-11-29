# #!/usr/bin/env zsh
PYTHON=${PYTHON:-"python"}

CFG=$1
GPUS=$2
OUTPUT_DIR=$3
PY_ARGS=${@:5}
PORT=${PORT:-29500}

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
      tools/train.py $CFG --work_dir $OUTPUT_DIR --seed 0 --launcher pytorch ${PY_ARGS}

python tools/to_d2_backbone.py --input $OUTPUT_DIR/latest.pth --output $OUTPUT_DIR/d2_backbone/latest.pth

python tools/d2_train.py --config-file benchmarks/detection/configs/coco_R_50_FPN_1x_infomin.yaml --num-gpu 8 \
       OUTPUT_DIR $OUTPUT_DIR/d2_det/mask_rcnn_full_1x \
       MODEL.WEIGHTS $OUTPUT_DIR/d2_backbone/latest.pth