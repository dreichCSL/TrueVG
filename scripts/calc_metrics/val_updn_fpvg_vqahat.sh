set -e

# set arguments
dataset=$1
checkpoint=$2
GPU_NUM=$3
source scripts/common.sh ${dataset}

SAVE_DIR=./saved_models_${dataset}
DATA_DIR=./data/${dataset}

for split_test in testid testood
do
  # CONTENT tests for semantic matching in FPVG eval
  for filter_objects_variable in original relevant_iou_50pct_VQAHAT irrelevant_neg_overlap_25pct_VQAHAT relevant_iou_50pct_VQAHAT_CONTENT irrelevant_neg_overlap_25pct_VQAHAT_CONTENT

  do
    if [ $filter_objects_variable == "original" ]
      then filter_mode="none"
    else filter_mode="select"
    fi

      CUDA_VISIBLE_DEVICES=${GPU_NUM} python -u main.py \
      --ACC_only \
      --feature_name detectron_glove_wAttr_coco \
      --feature_vqahat_vf_original_symbolic_features \
      --num_obj_max_manual_setting 36 \
      --visfis_hatcp \
      --do_not_discard_items_without_hints \
      --impt_threshold ${impt_threshold} \
      --calc_dp_level_metrics \
      --batch_size 64 \
      --seed 7 \
      --data_dir ${DATA_DIR} \
      --dataset ${dataset} \
      --split_test ${split_test} \
      --filter_objects=${filter_objects_variable} \
      --filter_mode=${filter_mode} \
      --checkpoint_path ${SAVE_DIR}/${checkpoint} \
      --load_checkpoint_path ${SAVE_DIR}/${checkpoint}/model-best.pth

  done
done

# get/print accuracies and FPVG (spatial matching)
python scripts/evaluation/FPVG_function.py --dataset ${dataset} --modelname ${checkpoint}
# FPVG (semantic matching); uses CONTENT test results
python scripts/evaluation/FPVG_function.py --dataset ${dataset} --modelname ${checkpoint} --content_based
