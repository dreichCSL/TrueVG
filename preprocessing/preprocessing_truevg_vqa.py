import pickle, json
import os
from pathlib import Path

import argparse
import preprocessing.truevg.prep_infusion as prep_infusion
import preprocessing.truevg.prep_evaluation as prep_evaluation
import preprocessing.truevg.prep_hints as prep_hints

def prepare_data(data_dir):

    print("Preparing infusion database.")
    Path(os.path.join(data_dir, 'true_grounding')).mkdir(parents=True, exist_ok=True)
    ref_object_scores = prep_infusion.get_heatmap_scores_for_ref_objects(data_dir)
    print("Writing to disk (intermediate result).")
    pickle.dump(ref_object_scores, open(os.path.join(data_dir, 'true_grounding', 'ref_object_scores.pkl'), 'wb'))
    # ref_object_scores = pickle.load(open(os.path.join(data_dir, 'true_grounding', 'ref_object_scores.pkl'), 'rb'))


    print("Preparing FPVG rel file (spatial matching)")
    rel_obj = prep_evaluation.get_relevant_object_indices_spatial_matching_bottomup_cocoref(ref_object_scores, data_dir, threshold=0.5, matching_method='iou')
    print("Writing to disk.")
    pickle.dump(rel_obj, open(os.path.join(data_dir, 'true_grounding', 'VQAHAT_bottomup_trainval_relevant_objects_path_iou_50pct.pkl'), 'wb'))
    print("Preparing FPVG irrel file (spatial matching)")
    irrel_obj = prep_evaluation.get_relevant_object_indices_spatial_matching_bottomup_cocoref(ref_object_scores, data_dir, threshold=0.25, matching_method='neg_overlap')
    print("Writing to disk.")
    pickle.dump(irrel_obj, open(os.path.join(data_dir, 'true_grounding', 'VQAHAT_bottomup_trainval_irrelevant_objects_path_neg_overlap_25pct.pkl'), 'wb'))
    # rel_obj = pickle.load(open(os.path.join(data_dir, 'true_grounding', 'VQAHAT_bottomup_trainval_relevant_objects_path_iou_50pct.pkl'), 'rb'))
    # irrel_obj = pickle.load(open(os.path.join(data_dir, 'true_grounding', 'VQAHAT_bottomup_trainval_irrelevant_objects_path_neg_overlap_25pct.pkl'), 'rb'))


    infusion_database = prep_infusion.find_infusion_objects_vqahat_coco(ref_object_scores, rel_obj, data_dir)
    # relevant_objects_per_imgid_qid_objid = prep_infusion.get_relevant_object_indices_h5_mod(sg, qa, data_dir)
    # print("Writing to disk (intermediate result).")
    # pickle.dump(relevant_objects_per_imgid_qid_objid, open(os.path.join(data_dir, 'true_grounding', 'relevant_objects_per_imgid_qid_objid.pkl'), 'wb'))
    # infusion_database = prep_infusion.find_infusion_objects(sg, qa, relevant_objects_per_imgid_qid_objid, data_dir)
    print("Writing to disk.")
    pickle.dump(infusion_database, open(os.path.join(data_dir, 'true_grounding', 'infusion_database_vqahat.pkl'), 'wb'))
    # infusion_database = pickle.load(open(os.path.join(data_dir, 'true_grounding', 'infusion_database_vqahat.pkl'), 'rb'))



    print("Determining FI scores, DET spatial.")
    FIscores_DET_spatial = prep_hints.get_FIscores_DET_spatial_matching_bottomup_cocoref(ref_object_scores, data_dir)
    print("Writing to disk.")
    Path(os.path.join(data_dir, 'hints', 'DET_spatial')).mkdir(parents=True, exist_ok=True)
    pickle.dump(FIscores_DET_spatial, open(os.path.join(data_dir, 'hints', 'DET_spatial', 'FIscores.pkl'), 'wb'))
    # FIscores_DET_spatial = pickle.load(open(os.path.join(data_dir, 'hints', 'DET_spatial', 'FIscores.pkl'), 'rb'))

    print("Determining FI scores, INF semantic.")
    FIscores_INF_semantic = prep_hints.get_FIscores_INF_semantic_matching_bottomup_cocoref(rel_obj, infusion_database)
    # # FIscores_INF_semantic = prep_hints.get_FIscores_INF_semantic_matching(qa, rel_obj, infusion_database, data_dir)
    print("Writing to disk.")
    Path(os.path.join(data_dir, 'hints', 'INF_semantic')).mkdir(parents=True, exist_ok=True)
    pickle.dump(FIscores_INF_semantic, open(os.path.join(data_dir, 'hints', 'INF_semantic', 'FIscores.pkl'), 'wb'))


    print("Preparing FPVG rel and irrel file (semantic matching)")
    rel_obj_semantic, irrel_obj_semantic = prep_evaluation.get_relevant_object_indices_semantic_matching(FIscores_DET_spatial, infusion_database, num_objects=36)
    print("Writing to disk.")
    pickle.dump(rel_obj_semantic, open(os.path.join(data_dir, 'true_grounding', 'VQAHAT_bottomup_trainval_relevant_objects_path_iou_50pct_CONTENT.pkl'), 'wb'))
    pickle.dump(irrel_obj_semantic, open(os.path.join(data_dir, 'true_grounding', 'VQAHAT_bottomup_trainval_irrelevant_objects_path_neg_overlap_25pct_CONTENT.pkl'), 'wb'))


    print("Determining True VG questions.")
    truevg_qids = prep_evaluation.get_truevg_qids(rel_obj, irrel_obj, infusion_database)
    print("Writing to disk.")
    pickle.dump(truevg_qids, open(os.path.join(data_dir, 'true_grounding', 'VQAHAT_bottomup_trainval_true_grounding_qids.pkl'), 'wb'))


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
        os.path.join(args.data_dir, 'questions', 'hatcp_train_questions.json'),
        os.path.join(args.data_dir, 'questions', 'hatcp_dev_questions.json'),
        os.path.join(args.data_dir, 'questions', 'hatcp_testid_questions.json'),
        os.path.join(args.data_dir, 'questions', 'hatcp_testood_questions.json'),
        os.path.join(args.data_dir, 'COCO', 'annotations', 'instances_train2017.json'),
        os.path.join(args.data_dir, 'COCO', 'annotations', 'instances_val2017.json'),
        os.path.join(args.data_dir, 'VQAv1', 'OpenEnded_mscoco_train2014_questions.json'),
        os.path.join(args.data_dir, 'VQAv1', 'OpenEnded_mscoco_val2014_questions.json'),
        os.path.join(args.data_dir, 'preprocessing', 'VQAHAT_img_db.pkl'),
        os.path.join(args.data_dir, 'preprocessing', "object_classes_bottomup.txt"),
        os.path.join(args.data_dir, 'preprocessing', "attribute_classes_bottomup.txt")

        # os.path.isfile(os.path.join(args.data_dir, 'output_gqa_detectron_objects_info.json')),
        # os.path.isfile(os.path.join(args.data_dir, 'output_gqa_detectron_objects.h5')),
        # os.path.isfile(os.path.join(args.data_dir, 'preprocessing', 'objectname_classes.pkl')),
        # os.path.isfile(os.path.join(args.data_dir, 'preprocessing', 'attribute_categories.pkl'))
    ]
    # make sure that all files exist
    missing_files = 0
    for f in required_files:
        if not os.path.isfile(f):
            print("Required file is missing: ", f)
            missing_files += 1
    assert missing_files == 0, "Number of required files missing: {}".format(missing_files)
    # assert sum([1 for f in required_files if os.path.isfile(f)]) == len(required_files), "Some required files are missing."

    prepare_data(args.data_dir)