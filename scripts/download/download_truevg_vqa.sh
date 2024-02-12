set -e
# IMPORTANT NOTE: set correct path to main code directory in common.sh!
source scripts/common.sh

mkdir -p data
cd data

## setup vqa-hat
# get shared files from zenodo
wget https://zenodo.org/records/10357278/files/hatcp_data.zip
unzip hatcp_data.zip
mv hatcp_truevg_release hatcp
rm hatcp_data.zip
cd hatcp

# visual features
wget https://zenodo.org/records/10357278/files/VQAHAT_train36_WE_600D.h5
wget https://zenodo.org/records/10357278/files/VQAHAT_val36_WE_600D.h5
# copy two files from the repo instead of zenodo (wrong files in zenodo)
cp ../../files/VQAHAT_train36_imgid2img.pkl .
cp ../../files/VQAHAT_val36_imgid2img.pkl .

# still in data/hatcp/, download certain files needed in preprocessing
# vqa-hat spatial heat maps from https://abhishekdas.com/vqa-hat/; combined: ~750MB
mkdir VQAHAT
cd ./VQAHAT
wget http://s3.amazonaws.com/vqa-hat/vqahat_train.zip
unzip vqahat_train.zip
rm vqahat_train.zip
wget http://s3.amazonaws.com/vqa-hat/vqahat_val.zip
unzip vqahat_val.zip
rm vqahat_val.zip
cd ..

# ms-coco image annotations from https://cocodataset.org/
mkdir COCO
cd ./COCO
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
rm annotations_trainval2017.zip
cd ..

# vqa questions from https://visualqa.org/
mkdir VQAv1
cd ./VQAv1
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Train_mscoco.zip
unzip Questions_Train_mscoco.zip
rm Questions_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Val_mscoco.zip
unzip Questions_Val_mscoco.zip
rm Questions_Val_mscoco.zip
cd ..