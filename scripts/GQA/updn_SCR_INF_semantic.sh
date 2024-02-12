set -e

# set arguments
dataset=$1
GPU_NUM=$2
source scripts/common.sh ${dataset}

SAVE_DIR=./saved_models_${dataset}
DATA_DIR=./data/${dataset}

for seed in 7; do
    expt=${dataset}_updn_SCR_INF_semantic_${seed}
    mkdir -p ${SAVE_DIR}/${expt}

    # train model
    CUDA_VISIBLE_DEVICES=${GPU_NUM} python -u main.py \
    --reproducePaper \
    --hint_type hat \
    --hint_path INF_semantic \
    --feature_name detectron_glove_wAttr \
    --visfis_all \
    --use_bbox \
    --impt_threshold ${impt_threshold} \
    --seed ${seed} \
    --data_dir ${DATA_DIR} \
    --dataset ${dataset} \
    --split train \
    --split_test dev \
    --batch_size 256 \
    --max_epochs 50 \
    --use_scr_loss \
    --scr_hint_loss_weight 0.01 \
    --scr_compare_loss_weight 10 \
    --model_importance gradcam \
    --FI_predicted_class false \
    --infusion \
    --checkpoint_path ${SAVE_DIR}/${expt} \
    >> ${SAVE_DIR}/${expt}/verbose_log.txt

    # calc all metrics
    source scripts/calc_metrics/val_updn_fpvg.sh ${dataset} ${expt} ${GPU_NUM}
done