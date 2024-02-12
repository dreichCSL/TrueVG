import pickle, json
import os, sys
import argparse

import calculate_FPVG_coco as fpvg_coco
import calculate_FPVG_gqa as fpvg_gqa


def get_results(dataset='gqacp', modelname=[], content_based=False, features=None):

    if not isinstance(modelname, list):
        modelnames = modelname.split(',')
    else:
        modelnames = modelname
    data_dir = os.path.join('data', dataset)
    model_dir = os.getcwd()

    if dataset == 'hatcp':
        # data_dir ='data/hatcp_full'
        target = pickle.load(open(os.path.join(data_dir, 'processed/train_target.pkl'), 'rb'))
        target.extend(pickle.load(open(os.path.join(data_dir, 'processed/val_target.pkl'), 'rb')))
        target_dict = {i['question_id']:i for i in target}

        ans_dict = pickle.load(open(os.path.join(data_dir, 'processed/trainval_label2ans.pkl'), 'rb'))

        # preparing Q/A data
        # vqahat_annos = {}
        # for subset in ["train", "dev", "test-id", "test-ood", "testid", "testood"]:
        #     try:
        #         tmp_file = json.load(open(os.path.join(data_dir, f'questions/{subset}_annotations.json'), 'rb'))
        #         vqahat_annos.update({str(q['question_id']):q for q in tmp_file['annotations']})
        #     except FileNotFoundError:
        #         # this is just a precaution to continue processing if files have not been renamed (test-id -> testid)
        #         continue

        qa_vqahat_id = json.load(open(os.path.join(data_dir, 'questions/hatcp_testid_questions.json'), 'r'))
        qa_vqahat_ood = json.load(open(os.path.join(data_dir, 'questions/hatcp_testood_questions.json'), 'r'))

        # tmp_file = pickle.load(open(os.path.join(data_dir, 'true_grounding/VQA-HAT_bottomupOriginal_36objs_true_grounding_qids_refHINT_cocoAnno.pkl'), 'rb'))
        tmp_file = pickle.load(open(os.path.join(data_dir, 'true_grounding/VQAHAT_bottomup_trainval_true_grounding_qids.pkl'), 'rb'))

        good_qids_bottomup_vqahat = {i:1 for i in tmp_file}



        for qid in qa_vqahat_id:
            qa_vqahat_id[qid].update({'score_set': target_dict[int(qid)]['scores']})
            qa_vqahat_id[qid].update({'answer_set': [ans_dict[label] for label in target_dict[int(qid)]['labels']]})

        # qa_vqahat_id_tvg = {i: qa_vqahat_id[i] for i in qa_vqahat_id if i in good_qids_bottomup_vqahat}
        # only questions of type "other"
        # qa_vqahat_id_other = {i:qa_vqahat_id[i] for i in qa_vqahat_id if vqahat_annos[i]['answer_type'] == 'other'}
        qa_vqahat_id_other = {i:qa_vqahat_id[i] for i in qa_vqahat_id if qa_vqahat_id[i]['answer_type'] == 'other'}

        # True VG tests: only questions with complete relevant content/location
        qa_vqahat_id_other_tvg = {i:qa_vqahat_id_other[i] for i in qa_vqahat_id_other if i in good_qids_bottomup_vqahat}


        for qid in qa_vqahat_ood:
            qa_vqahat_ood[qid].update({'score_set': target_dict[int(qid)]['scores']})
            qa_vqahat_ood[qid].update({'answer_set': [ans_dict[label] for label in target_dict[int(qid)]['labels']]})

        # qa_vqahat_ood_tvg = {i: qa_vqahat_ood[i] for i in qa_vqahat_ood if i in good_qids_bottomup_vqahat}
        # only questions of type "other"
        # qa_vqahat_ood_other = {i:qa_vqahat_ood[i] for i in qa_vqahat_ood if vqahat_annos[i]['answer_type'] == 'other'}
        qa_vqahat_ood_other = {i:qa_vqahat_ood[i] for i in qa_vqahat_ood if qa_vqahat_ood[i]['answer_type'] == 'other'}

        # True VG tests: only questions with complete relevant content/location
        qa_vqahat_ood_other_tvg = {i:qa_vqahat_ood_other[i] for i in qa_vqahat_ood_other if i in good_qids_bottomup_vqahat}


        # for easier access
        qa_vqahat = {}
        # qa_vqahat['id'] = {'all': qa_vqahat_id, 'all_tvg': qa_vqahat_id_tvg, 'other': qa_vqahat_id_other, 'tvg_other': qa_vqahat_id_other_tvg}
        # qa_vqahat['ood'] = {'all': qa_vqahat_ood, 'all_tvg': qa_vqahat_ood_tvg, 'other': qa_vqahat_ood_other, 'tvg_other': qa_vqahat_ood_other_tvg}
        qa_vqahat['id'] = {'other': qa_vqahat_id_other, 'tvg_other': qa_vqahat_id_other_tvg}
        qa_vqahat['ood'] = {'other': qa_vqahat_ood_other, 'tvg_other': qa_vqahat_ood_other_tvg}


        # results = {}
        for modelname in modelnames:
            print("\n------------------ Model: {} -------------------".format(modelname))
            # results[modelname] = {}
            for testtype in ["id", "ood"]:
                # results[modelname][testtype] = {}
                for q_set in list(qa_vqahat[testtype]):
                # for q_set in ['all', 'other', 'tvg_other']:
                    if features is None or features == 'bottomupOriginal_coco_refHINT_cocoAnno':
                        features = 'bottomupOriginal_coco_refHINT_cocoAnno'
                        # file_extension = 'refHINT_cocoAnno'
                    # elif features == 'bottomupOriginal_coco_directHINT':
                    #     file_extension = 'directHINT_ORIGINAL'
                    if content_based is True:
                        # # only get FPVG for TVG subsets and ignore other sets
                        # if q_set != 'tvg_other': continue
                        print("CONTENT-based FPVG. Test type: {} \t QA-set: {}".format(testtype, q_set))
                        tmp = fpvg_coco.calculate_FPVG(os.path.join(model_dir,
                                                                    f'saved_models_hatcp/{modelname}/test{testtype}_none_original_none_gt_metrics.pkl'),
                                                       os.path.join(model_dir,
                                                                    f'saved_models_hatcp/{modelname}/test{testtype}_select_relevant_iou_50pct_VQAHAT_CONTENT_none_gt_metrics.pkl'),
                                                       os.path.join(model_dir,
                                                                    f'saved_models_hatcp/{modelname}/test{testtype}_select_irrelevant_neg_overlap_25pct_VQAHAT_CONTENT_none_gt_metrics.pkl'),
                                                       # os.path.join(model_dir,
                                                       #              f'saved_models_hatcp/{modelname}/test{testtype}_select_relevant_iou_50pct_VQAHAT_{file_extension}_CONTENT_none_gt_metrics.pkl'),
                                                       # os.path.join(model_dir,
                                                       #              f'saved_models_hatcp/{modelname}/test{testtype}_select_irrelevant_neg_overlap_25pct_VQAHAT_{file_extension}_CONTENT_none_gt_metrics.pkl'),
                                                       data_dir=data_dir,
                                                       framework='visfis_vqahat',
                                                       features="bottomup",
                                                       qa_input=qa_vqahat[testtype][q_set],
                                                       obj_number=36, label2ans=ans_dict,
                                                       target_dict=target_dict,
                                                       return_details=True, verbose=0,
                                                       content_based_rel=True)
                    else:
                        print("LOCATION-based FPVG. Test type: {} \t QA-set: {}".format(testtype, q_set))
                        tmp = fpvg_coco.calculate_FPVG(os.path.join(model_dir, f'saved_models_hatcp/{modelname}/test{testtype}_none_original_none_gt_metrics.pkl'),
                                                       os.path.join(model_dir,
                                                                    f'saved_models_hatcp/{modelname}/test{testtype}_select_relevant_iou_50pct_VQAHAT_none_gt_metrics.pkl'),
                                                       os.path.join(model_dir,
                                                                    f'saved_models_hatcp/{modelname}/test{testtype}_select_irrelevant_neg_overlap_25pct_VQAHAT_none_gt_metrics.pkl'),
                                                       data_dir=data_dir,
                                                       framework='visfis_vqahat', features="bottomup",
                                                       qa_input=qa_vqahat[testtype][q_set],
                                                       obj_number=36, label2ans=ans_dict, target_dict=target_dict,
                                                       return_details=True, verbose=0)
                    print(tmp[-1])

    elif dataset == 'gqacp':
        ans_dict = pickle.load(open(os.path.join(data_dir, 'processed/trainval_label2ans.pkl'), 'rb'))

        qa_gqacp_id = json.load(open(os.path.join(data_dir, 'questions/gqacp_testid_questions.json'), 'r'))
        qa_gqacp_ood = json.load(open(os.path.join(data_dir, 'questions/gqacp_testood_questions.json'), 'r'))
        tmp_file = pickle.load(open(os.path.join(data_dir, 'true_grounding/GQA_balanced_trainval_detectron_true_grounding_qids.pkl'), 'rb'))
        good_qids_gqa = {i:1 for i in tmp_file}

        qa_gqacp_id_tvg = {i:qa_gqacp_id[i] for i in qa_gqacp_id if i in good_qids_gqa}
        qa_gqacp_ood_tvg = {i:qa_gqacp_ood[i] for i in qa_gqacp_ood if i in good_qids_gqa}

        qa_gqacp = {}
        qa_gqacp['ood'] = {'all': qa_gqacp_ood, 'all_tvg': qa_gqacp_ood_tvg}
        qa_gqacp['id'] = {'all': qa_gqacp_id, 'all_tvg': qa_gqacp_id_tvg}

        for modelname in modelnames:
            print("\n----------- Model: {} ------------".format(modelname))
            for testtype in ["id", "ood"]:
                for q_set in list(qa_gqacp[testtype]):
                    if content_based is True:
                        print("CONTENT-based FPVG. Test type: {} \t QA-set: {}".format(testtype, q_set))
                        tmp = fpvg_gqa.calculate_FPVG(os.path.join(model_dir,f'saved_models_gqacp/{modelname}/test{testtype}_none_original_none_gt_metrics.pkl'),
                                                      os.path.join(model_dir,f'saved_models_gqacp/{modelname}/test{testtype}_select_relevant_iou_50pct_trainval_CONTENT_none_gt_metrics.pkl'),
                                                      os.path.join(model_dir,f'saved_models_gqacp/{modelname}/test{testtype}_select_irrelevant_neg_overlap_25pct_trainval_CONTENT_none_gt_metrics.pkl'),
                                                      data_dir=data_dir,
                                                      framework='visfis', features='detectron',
                                                      content_based_rel=True, label2ans=ans_dict,
                                                      qa_input=qa_gqacp[testtype][q_set], return_details=True,
                                                      verbose=0)
                    else:
                        print("LOCATION-based FPVG. Test type: {} \t QA-set: {}".format(testtype, q_set))
                        tmp = fpvg_gqa.calculate_FPVG(os.path.join(model_dir, f'saved_models_gqacp/{modelname}/test{testtype}_none_original_none_gt_metrics.pkl'),
                                                      os.path.join(model_dir, f'saved_models_gqacp/{modelname}/test{testtype}_select_relevant_iou_50pct_trainval_none_gt_metrics.pkl'),
                                                      os.path.join(model_dir, f'saved_models_gqacp/{modelname}/test{testtype}_select_irrelevant_neg_overlap_25pct_trainval_none_gt_metrics.pkl'),
                                                      data_dir=data_dir,
                                                      framework='visfis', features='detectron', label2ans=ans_dict,
                                                      qa_input=qa_gqacp[testtype][q_set], return_details=True, verbose=0)
                    print(tmp[-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['gqacp', 'hatcp'])
    parser.add_argument('--modelname', type=str)
    parser.add_argument('--content_based', action='store_true')

    args = parser.parse_args()

    # assumes call from main code directory, which is enforced by scripts/common.sh e.g. when launching a training script
    get_results(dataset=args.dataset, modelname=args.modelname, content_based=args.content_based)































