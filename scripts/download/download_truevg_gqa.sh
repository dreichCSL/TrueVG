set -e
# IMPORTANT NOTE: set correct path to main code directory in common.sh!
source scripts/common.sh

mkdir -p data
cd data

## setup vqa-hat
# get shared files from zenodo
wget https://zenodo.org/records/10357278/files/gqacp_data.zip
unzip gqacp_data.zip
mv gqacp_truevg_release gqacp
rm gqacp_data.zip
cd gqacp

# visual features, download (and rename)
wget https://zenodo.org/records/10357278/files/output_gqa_detectron_GloveEmbeddings_wAttr_600D.h5 -O output_gqa_detectron_objects.h5
wget https://zenodo.org/records/10357278/files/output_gqa_detectron_GloveEmbeddings_wAttr_600D.json -O output_gqa_detectron_objects_info.json

# still in data/gqacp/, download files needed in preprocessing
# gqa balanced split scene graphs
mkdir GQA
cd ./GQA
wget https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip
unzip sceneGraphs.zip
rm sceneGraphs.zip
#wget https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip
#unzip questions1.2.zip
#rm questions1.2.zip
cd ..
