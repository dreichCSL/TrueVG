# set path
PROJ_DIR=PATH_TO_CODE/TrueVG/
cd ${PROJ_DIR}
export PYTHONPATH=${PROJ_DIR}

# set threshold
dataset=$1
case "$dataset" in
    #case 1
    "hatcp") impt_threshold=0.55 FI_metrics=KOI ;;
    #case 2
    "gqacp") impt_threshold=0.3 split_postfix=-100k FI_metrics=LOO ;;
esac
