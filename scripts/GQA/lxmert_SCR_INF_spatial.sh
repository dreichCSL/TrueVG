set -e

# set arguments
dataset=$1
GPU_NUM=$2
source scripts/common.sh ${dataset}

SAVE_DIR=./saved_models_${dataset}
DATA_DIR=./data/${dataset}

for seed in 7; do
    expt=${dataset}_lxmert_SCR_INF_spatial_${seed}
    mkdir -p ${SAVE_DIR}/${expt}

    # train model
    CUDA_VISIBLE_DEVICES=${GPU_NUM} python -u main.py \
    --reproducePaper \
    --hint_type hat \
    --hint_path INF_spatial \
    --feature_name detectron_glove_wAttr \
    --visfis_all \
    --use_bbox \
    --impt_threshold ${impt_threshold} \
    --seed ${seed} \
    --data_dir ${DATA_DIR} \
    --dataset ${dataset} \
    --split train \
    --split_test dev \
    --lxmert_small_model \
    --lxmert_hid_dim 128 \
    --model_type lxmert \
    --batch_size 64 \
    --max_epochs 35 \
    --grad_clip 5 \
    --learning_rate 5e-5 \
    --use_scr_loss \
    --scr_hint_loss_weight 1e-6 \
    --scr_compare_loss_weight 1e-4 \
    --model_importance gradcam \
    --FI_predicted_class false \
    --infusion \
    --checkpoint_path ${SAVE_DIR}/${expt} \
    >> ${SAVE_DIR}/${expt}/verbose_log.txt

    # calc all metrics
    source scripts/calc_metrics/val_lxmert_fpvg.sh ${dataset} ${expt} ${GPU_NUM}
done 
