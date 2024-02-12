set -e

# set arguments
dataset=$1
GPU_NUM=$2
source scripts/common.sh ${dataset}

SAVE_DIR=./saved_models_${dataset}
DATA_DIR=./data/${dataset}

for seed in 7; do
    expt=${dataset}_lxmert_visFIS_INF_semantic_${seed}
    mkdir -p ${SAVE_DIR}/${expt}

    # visfis - Align-Cos + Invariance-FI + suff + uncertainty
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
    --lxmert_small_model \
    --lxmert_hid_dim 128 \
    --model_type lxmert \
    --batch_size 64 \
    --max_epochs 35 \
    --grad_clip 5 \
    --learning_rate 5e-5 \
    --OBJ11 \
    --aug_type suff-uncertainty \
    --use_zero_loss \
    --use_direct_alignment \
    --model_importance gradcam \
    --FI_predicted_class false \
    --infusion \
    --checkpoint_path ${SAVE_DIR}/${expt} \
    >> ${SAVE_DIR}/${expt}/verbose_log.txt

    # calc all metrics
    source scripts/calc_metrics/val_lxmert_fpvg.sh ${dataset} ${expt} ${GPU_NUM}
done 
