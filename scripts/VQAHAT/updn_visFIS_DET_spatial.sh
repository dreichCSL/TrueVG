set -e

# set arguments
dataset=$1
GPU_NUM=$2
source scripts/common.sh ${dataset}

SAVE_DIR=./saved_models_${dataset}
DATA_DIR=./data/${dataset}

for seed in 7 77 777 7777 77777; do
    expt=${dataset}_updn_visFIS_DET_spatial_${seed}
    mkdir -p ${SAVE_DIR}/${expt}

    # train model
    CUDA_VISIBLE_DEVICES=${GPU_NUM} python -u main.py \
    --hint_type hat \
    --hint_path DET_spatial \
    --feature_vqahat_vf_original_symbolic_features \
    --feature_name detectron_glove_wAttr_coco \
    --num_obj_max_manual_setting 36 \
    --visfis_hatcp \
    --impt_threshold ${impt_threshold} \
    --seed ${seed} \
    --data_dir ${DATA_DIR} \
    --dataset ${dataset} \
    --split train \
    --split_test dev \
    --batch_size 64 \
    --max_epochs 50 \
    --OBJ11 \
    --aug_type suff-uncertainty \
    --use_zero_loss \
    --use_direct_alignment \
    --FI_predicted_class false \
    --model_importance gradcam \
    --checkpoint_path ${SAVE_DIR}/${expt} \
    >> ${SAVE_DIR}/${expt}/verbose_log.txt

    # calc all metrics
    source scripts/calc_metrics/val_updn_fpvg_vqahat.sh ${dataset} ${expt} ${GPU_NUM}
done