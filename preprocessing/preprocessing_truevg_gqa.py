import pickle, json
import os
from pathlib import Path

import argparse
import preprocessing.truevg.prep_infusion as prep_infusion
import preprocessing.truevg.prep_evaluation as prep_evaluation
import preprocessing.truevg.prep_hints as prep_hints

def prepare_data(data_dir):

    print("Loading scene graphs and questions.")
    sg = json.load(open(os.path.join(data_dir, 'GQA', 'train_sceneGraphs.json'), 'r'))
    sg.update(json.load(open(os.path.join(data_dir, 'GQA', 'val_sceneGraphs.json'), 'r')))
    # qa = json.load(open(os.path.join(data_dir, 'GQA', 'train_balanced_questions.json'), 'r'))
    # qa.update(json.load(open(os.path.join(data_dir, 'GQA', 'val_balanced_questions.json'), 'r')))

    qa = json.load(open(os.path.join(data_dir, 'questions', 'gqacp_train_questions.json'), 'r'))
    qa.update(json.load(open(os.path.join(data_dir, 'questions', 'gqacp_dev_questions.json'), 'r')))
    qa.update(json.load(open(os.path.join(data_dir, 'questions', 'gqacp_testid_questions.json'), 'r')))
    qa.update(json.load(open(os.path.join(data_dir, 'questions', 'gqacp_testood_questions.json'), 'r')))


    print("Preparing infusion database.")
    relevant_objects_per_imgid_qid_objid = prep_infusion.get_relevant_object_indices_h5_mod(sg, qa, data_dir)
    print("Writing to disk (intermediate result).")
    pickle.dump(relevant_objects_per_imgid_qid_objid, open(os.path.join(data_dir, 'true_grounding', 'relevant_objects_per_imgid_qid_objid.pkl'), 'wb'))
    infusion_database = prep_infusion.find_infusion_objects(sg, qa, relevant_objects_per_imgid_qid_objid, data_dir)
    print("Writing to disk.")
    pickle.dump(infusion_database, open(os.path.join(data_dir, 'true_grounding', 'infusion_database_gqa.pkl'), 'wb'))
    # infusion_database = pickle.load(open(os.path.join(data_dir, 'true_grounding', 'infusion_database_gqa.pkl'), 'rb'))


    print("Preparing FPVG rel file (spatial matching)")
    rel_obj = prep_evaluation.get_relevant_object_indices_spatial_matching(sg, qa, data_dir, threshold=0.5, matching_method='iou')
    print("Writing to disk.")
    pickle.dump(rel_obj, open(os.path.join(data_dir, 'true_grounding', 'GQA_detectron_trainval_relevant_objects_path_iou_50pct.pkl'), 'wb'))
    # rel_obj = pickle.load(open(os.path.join(data_dir, 'true_grounding', 'GQA_detectron_trainval_relevant_objects_path_iou_50pct.pkl'), 'rb'))


    print("Preparing FPVG irrel file (spatial matching)")
    irrel_obj = prep_evaluation.get_relevant_object_indices_spatial_matching(sg, qa, data_dir, threshold=0.25, matching_method='neg_overlap')
    print("Writing to disk.")
    pickle.dump(irrel_obj, open(os.path.join(data_dir, 'true_grounding', 'GQA_detectron_trainval_irrelevant_objects_path_neg_overlap_25pct.pkl'), 'wb'))


    print("Determining True VG questions.")
    truevg_qids = prep_evaluation.get_truevg_qids(rel_obj, irrel_obj, infusion_database, fpvg_mode=True)
    print("Writing to disk.")
    pickle.dump(truevg_qids, open(os.path.join(data_dir, 'true_grounding', 'GQA_balanced_trainval_detectron_true_grounding_qids.pkl'), 'wb'))


    print("Determining FI scores, DET spatial.")
    FIscores_DET_spatial = prep_hints.get_FIscores_DET_spatial_matching(sg, qa, data_dir)
    print("Writing to disk.")
    Path(os.path.join(data_dir, 'hints', 'DET_spatial')).mkdir(parents=True, exist_ok=True)
    pickle.dump(FIscores_DET_spatial, open(os.path.join(data_dir, 'hints', 'DET_spatial', 'FIscores.pkl'), 'wb'))

    print("Determining FI scores, INF semantic.")
    FIscores_INF_semantic = prep_hints.get_FIscores_INF_semantic_matching(qa, rel_obj, infusion_database, data_dir)
    print("Writing to disk.")
    Path(os.path.join(data_dir, 'hints', 'INF_semantic')).mkdir(parents=True, exist_ok=True)
    pickle.dump(FIscores_INF_semantic, open(os.path.join(data_dir, 'hints', 'INF_semantic', 'FIscores.pkl'), 'wb'))

    print("Determining FI scores, INF spatial.")
    FIscores_INF_spatial = prep_hints.get_FIscores_INF_spatial_matching(qa, FIscores_DET_spatial, infusion_database, data_dir)
    print("Writing to disk.")
    Path(os.path.join(data_dir, 'hints', 'INF_spatial')).mkdir(parents=True, exist_ok=True)
    pickle.dump(FIscores_INF_spatial, open(os.path.join(data_dir, 'hints', 'INF_spatial', 'FIscores.pkl'), 'wb'))


    print("Preparing FPVG rel and irrel file (semantic matching)")
    rel_obj_semantic, irrel_obj_semantic = prep_evaluation.get_relevant_object_indices_semantic_matching(FIscores_DET_spatial, infusion_database)
    print("Writing to disk.")
    pickle.dump(rel_obj_semantic, open(os.path.join(data_dir, 'true_grounding', 'GQA_detectron_trainval_relevant_objects_path_iou_50pct_CONTENT.pkl'), 'wb'))
    pickle.dump(irrel_obj_semantic, open(os.path.join(data_dir, 'true_grounding', 'GQA_detectron_trainval_irrelevant_objects_path_neg_overlap_25pct_CONTENT.pkl'), 'wb'))


    print("Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    # parser.add_argument('--dataset', type=str, choices=['gqacp', 'hatcp'])
    # parser.add_argument('--modelname', type=str)
    # parser.add_argument('--content_based', action='store_true')
    # parser.add_argument('--features', type=str, default=None, choices=[None, 'bottomupOriginal_coco_refHINT_cocoAnno',
    #                                                                    'bottomupOriginal_coco_directHINT'])
    args = parser.parse_args()

    # prepare necessary files for training and evaluation with GQA
    # the following files are expected to exist:
    required_files = [
        os.path.join(args.data_dir, 'GQA', 'train_sceneGraphs.json'),
        os.path.join(args.data_dir, 'GQA', 'val_sceneGraphs.json'),
        os.path.join(args.data_dir, 'questions', 'gqacp_train_questions.json'),
        os.path.join(args.data_dir, 'questions', 'gqacp_dev_questions.json'),
        os.path.join(args.data_dir, 'questions', 'gqacp_testid_questions.json'),
        os.path.join(args.data_dir, 'questions', 'gqacp_testood_questions.json'),
        os.path.join(args.data_dir, 'output_gqa_detectron_objects_info.json'),
        os.path.join(args.data_dir, 'output_gqa_detectron_objects.h5'),
        os.path.join(args.data_dir, 'preprocessing', 'objectname_classes.pkl'),
        os.path.join(args.data_dir, 'preprocessing', 'attribute_categories.pkl')
    ]
    # make sure that all files exist
    missing_files = 0
    for f in required_files:
        if not os.path.isfile(f):
            print("Required file is missing: ", f)
            missing_files += 1
    assert missing_files == 0, "Number of required files missing: {}".format(missing_files)

    # assert sum(required_files) == 4, "Some required files are missing."

    prepare_data(args.data_dir)