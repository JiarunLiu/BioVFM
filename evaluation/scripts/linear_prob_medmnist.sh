# !/bin/bash

# ==============================================================================
# Linear probe on MedMNIST
# 
# Useage:
#     bash scripts/linear_prob_medmnist.sh <PRETRAIN_MODEL_DIR> <MODEL_FLAG> <GPU_ID>
# 
# ==============================================================================


PRETRAIN_MODEL_DIR=$1
MODEL_FLAG=$2
GPU_ID=$3


# full list
DATASETS=(
    "pathmnist"
    "chestmnist"
    "dermamnist"
    "octmnist"
    "pneumoniamnist"
    "retinamnist"
    "breastmnist"
    "bloodmnist"
    "tissuemnist"
    "organamnist"
    "organcmnist"
    "organsmnist"
)


for DATA_FLAG in ${DATASETS[@]}; do
    echo "Training ${DATA_FLAG} with ${MODEL_FLAG} and image size ${IMG_SIZE}..."

    CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
        --data_flag ${DATA_FLAG} \
        --model_flag ${MODEL_FLAG} \
        --img_size 224 \
        --num_devices 1 \
        --lr 1e-2 \
        --global_pool \
        --linear_prob \
        --wandb_tags "linear_prob,global_pool" \
        --pretrain_model "${PRETRAIN_MODEL_DIR}/checkpoint-${PRETRAIN_EPOCH}.pth" 

done