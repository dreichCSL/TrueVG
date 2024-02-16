set -e
source scripts/common.sh

python ./preprocessing/preprocessing_truevg_vqa.py --data_dir ./data/hatcp
