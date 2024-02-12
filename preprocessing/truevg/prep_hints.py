import os
import copy
import numpy as np
import json, pickle
import h5py

from preprocessing.truevg.prep_evaluation import get_iou, get_overlap

######################################
## GQA           #####################
######################################

def get_FIscores_DET_spatial_matching(sg, qa, data_dir, min_iou=0.5):
    # FI-scores determined by IoU with question-relevant reference bboxes;
    # info is given in GQA's scene-graph and GQA's qa file

    # get final object bbox for each question per img
    relevant_objects_per_img_and_question = {}

    # get reference/annotated relevant object ids per question (and image)
    for q_id in list(qa):
        img_id = qa[q_id]['imageId']
        object_list = list(qa[q_id]['annotations']['answer'].values())
        object_list.extend(list(qa[q_id]['annotations']['fullAnswer'].values()))
        object_list.extend(list(qa[q_id]['annotations']['question'].values()))
        object_list = list(set(object_list))

        if len(object_list):
            dict_entry = relevant_objects_per_img_and_question.get(img_id, {})
            dict_entry.update({q_id: object_list})
            relevant_objects_per_img_and_question[img_id] = dict_entry

    # now find overlap with detectron output objects
    img_id_list = list(relevant_objects_per_img_and_question.keys())

    img_rep_gqa_json = json.load(open(os.path.join(data_dir, 'output_gqa_detectron_objects_info.json'), 'r'))
    img_rep_gqa_feats = h5py.File(os.path.join(data_dir, 'output_gqa_detectron_objects.h5'), 'r')

    FIscores_per_qid = {}

    print("Processing a total of {} images. This may take a while.".format(len(img_id_list)))
    for c,img_id in enumerate(img_id_list):
        if c % 5000 == 0:
            print("Processed images: {}.".format(c))

        # feature files processing
        try:
            img_index_h5 = img_rep_gqa_json[str(img_id)]['index']
            in_bboxes = img_rep_gqa_feats['bboxes'][img_index_h5]
        except:
            print("File for image {} not found, skipping.".format(img_id))
            continue

        # first get all ref boxes in a dict
        q_for_img = relevant_objects_per_img_and_question[img_id]
        q_id_list = list(q_for_img.keys())

        for q_id in q_id_list:

            # then find the iou match among ref boxes (min 0.5). if not found, skip
            best_obj_ious = []
            for box_idx, obj in enumerate(in_bboxes):
                x1, y1, x2, y2 = list(in_bboxes[box_idx])  # (4,) x,y,x2,y2
                hyp_box = {'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)}
                best_obj_ious.append(0.0)
                if np.sum([x1, y1, x2, y2]) == 0: continue
                for obj_id in q_for_img[q_id]:
                    ref_o = sg[str(img_id)]['objects'][obj_id]
                    ref_box = {'x1': ref_o['x'], 'y1': ref_o['y'], 'x2': ref_o['x'] + ref_o['w'], 'y2': ref_o['y'] + ref_o['h']}
                    iou = get_iou(ref_box, hyp_box)
                    best_obj_ious[-1] = max(best_obj_ious[-1], iou)

            hint_entry = np.array(best_obj_ious)
            # only consider hints for qid if meaningful overlap with some relevant ref object exists
            if len(np.where(hint_entry > min_iou)[0]) > 0:
                FIscores_per_qid[q_id] = np.array(best_obj_ious)

    return FIscores_per_qid


def get_FIscores_INF_semantic_matching(qa, rel_obj, infusion_database, data_dir, max_objects=100):
    # infusion database used to determine post-Infusion visual feature representation
    # consider how many objects are actually in the image so we don't give score to 0-paddings
    img_rep_gqa_json = json.load(open(os.path.join(data_dir, 'output_gqa_detectron_objects_info.json'), 'r'))

    rel_obj_dict = {qid:rel_obj[imgid][qid] for imgid in rel_obj for qid in rel_obj[imgid] if len(rel_obj[imgid][qid])}

    # create the hat values across 100 objects
    FIscores_per_qid = {}
    # only use questions that already have meaningful (>0.5 IOU) spatial matching, for fair comparability with same qids across all hint types
    for qid in list(set(qa.keys()) & set(rel_obj_dict.keys())):
        img_id = qa[qid]['imageId']
        num_objects = min(img_rep_gqa_json[img_id]['objectsNum'], max_objects)
        FIscores_per_qid[qid] = np.zeros((max_objects)) + 0.01  # avoid plain zero
        FIscores_per_qid[qid][num_objects:] = 0  # zero for padded objects
        # overwrite zeros in ascending order of iou matching score
        relevant_objects = []
        newly_added_objects = []
        infusion_boxes = []
        try:
            infusion_boxes = [b for objid in infusion_database[img_id][qid] for b in infusion_database[img_id][qid][objid]]
            infusion_box_indices_dict = {_[0]: 1 for _ in infusion_boxes}
        except KeyError:
            pass
        for infusion_box_idx, infusion_box_info in infusion_boxes:
            if infusion_box_idx >= max_objects:
                infusion_box_idx = 100  # we use this number as an indicator in model training; max_objects > 100 not supported
            # adding new box
            if infusion_box_idx == 100:
                if num_objects < max_objects:  # simply append the annotated object that's missing from the detections
                    relevant_objects.append(num_objects)
                    newly_added_objects.append(num_objects)
                    num_objects += 1  # increase count because we're adding an object
                else:  # we need to overwrite some box to accomodate the annotated object here
                    for i in reversed(range(max_objects)):
                        if infusion_box_indices_dict.get(i, 0):  # if this object is already needed, skip
                            continue
                        else:
                            infusion_box_indices_dict[i] = 1  # add to used objects
                            relevant_objects.append(i)
                            newly_added_objects.append(i)
                            break
            # existing box is used
            else:
                relevant_objects.append(infusion_box_idx)

        if len(relevant_objects):
            FIscores_per_qid[qid][relevant_objects] = 0.99

        # remove this entry if no annotations or iou matches found for objects in this img rep
        if (np.sum(FIscores_per_qid[qid]==0.01) == num_objects):
            del(FIscores_per_qid[qid])

    return FIscores_per_qid


def get_FIscores_INF_spatial_matching(qa, FIscores_spatial, infusion_database, data_dir, max_objects=100):
    # difference to DET spatial matching:
    # set feature importance to 0.99 (same value as in INF-semantic) where a new visual object is added:
    # 0.99 represents full bbox overlap;
    # otherwise use the same FI scores as DET spatial (no other visual objects have changed bboxes from Infusion)

    # consider how many objects are actually in the image so we don't give score to 0-paddings
    img_rep_gqa_json = json.load(open(os.path.join(data_dir, 'output_gqa_detectron_objects_info.json'), 'r'))

    # create the hat values across 100 objects
    FIscores_per_qid = {}
    # only use questions that already have meaningful (>0.5 IOU) spatial matching, for fair comparability with same qids across all hint types
    for qid in FIscores_spatial:
        img_id = qa[qid]['imageId']
        num_objects = min(img_rep_gqa_json[img_id]['objectsNum'], max_objects)
        FIscores_per_qid[qid] = copy.deepcopy(FIscores_spatial[qid])  # avoid plain zero
        # overwrite zeros in ascending order of iou matching score
        relevant_objects = []
        newly_added_objects = []
        infusion_boxes = []
        try:
            infusion_boxes = [b for objid in infusion_database[img_id][qid] for b in infusion_database[img_id][qid][objid]]
            infusion_box_indices_dict = {_[0]: 1 for _ in infusion_boxes}
        except KeyError:
            pass
        for infusion_box_idx, infusion_box_info in infusion_boxes:
            if infusion_box_idx >= max_objects:
                infusion_box_idx = 100  # we use this number as an indicator in model training; max_objects > 100 not supported
            # adding new box
            if infusion_box_idx == 100:
                if num_objects < max_objects:  # simply append the annotated object that's missing from the detections
                    relevant_objects.append(num_objects)
                    newly_added_objects.append(num_objects)
                    num_objects += 1  # increase count because we're adding an object
                else:  # we need to overwrite some box to accomodate the annotated object here
                    for i in reversed(range(max_objects)):
                        if infusion_box_indices_dict.get(i, 0):  # if this object is already needed, skip
                            continue
                        else:
                            infusion_box_indices_dict[i] = 1  # add to used objects
                            relevant_objects.append(i)
                            newly_added_objects.append(i)
                            break

        if len(newly_added_objects):
            FIscores_per_qid[qid][newly_added_objects] = 0.99

    return FIscores_per_qid

######################################
## VQA-HAT       #####################
######################################

def get_FIscores_DET_spatial_matching_bottomup_cocoref(ref_object_scores, data_dir):

    # get final object bbox for each question per img
    relevant_objects_per_img_and_question = {}

    # get reference/annotated relevant object ids per question (and image)
    for q_id_img_id in list(ref_object_scores):
        object_list = []
        q_id, img_id = q_id_img_id.split('_')
        for obj in ref_object_scores[q_id_img_id]:
            # append relevant bbox if heat map FI score is > 0.55 for this object (threshold from visfis)
            if obj[2] > 0.55:
                object_list.append(obj[0])
        if len(object_list):
            dict_entry = relevant_objects_per_img_and_question.get(img_id, {})
            dict_entry.update({q_id: object_list})  # list of bboxes
            relevant_objects_per_img_and_question[img_id] = dict_entry
    img_id_list = list(relevant_objects_per_img_and_question.keys())

    # contains bottomup detection info: bboxes, class, attribute
    img_db = pickle.load(open(os.path.join(data_dir, 'preprocessing', 'VQAHAT_img_db.pkl'), 'rb'))

    FIscores_per_qid = {}
    print("Processing a total of {} images. This may take a while.".format(len(img_id_list)))
    for c,img_id in enumerate(img_id_list):
        if c % 5000 == 0:
            print("Processed images: {}.".format(c))

        # feature files processing
        try:
            in_bboxes = img_db[int(img_id)]['bboxes']
        except:
            print("File for image {} not found, skipping.".format(img_id))
            continue

        # first get all ref boxes in a dict
        q_for_img = relevant_objects_per_img_and_question[img_id]
        q_id_list = list(q_for_img.keys())

        for q_id in q_id_list:
            # then find the iou match among ref boxes (min 0.5). if not found, skip
            best_obj_ious = []
            for box_idx, obj in enumerate(in_bboxes):
                x1, y1, x2, y2 = list(in_bboxes[box_idx][:4])  # (4,) x,y,x2,y2
                hyp_box = {'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)}
                best_obj_ious.append(0.0)
                if np.sum([x1, y1, x2, y2]) == 0: continue
                for ref_o in q_for_img[q_id]:
                    ref_box = {'x1': int(ref_o[0]), 'y1': int(ref_o[1]), 'x2': int(ref_o[2]), 'y2': int(ref_o[3])}
                    overlap = get_iou(ref_box, hyp_box)
                    best_obj_ious[-1] = max(best_obj_ious[-1], overlap)
            FIscores_per_qid[int(q_id)] = np.array(best_obj_ious)

    return FIscores_per_qid


def get_FIscores_INF_semantic_matching_bottomup_cocoref(rel_obj, infusion_database):
    max_objects = 36
    rel_obj_dict = {qid: rel_obj[imgid][qid] for imgid in rel_obj for qid in rel_obj[imgid] if len(rel_obj[imgid][qid])}
    rel_dicts = {0.99: rel_obj_dict}

    FIscores_per_qid = {}
    for img_id in infusion_database:
        for qid in infusion_database[img_id]:
            FIscores_per_qid[int(qid)] = np.zeros((max_objects)) + 0.01  # avoid plain zero
            # overwrite zeros in ascending order of iou matching score
            for iou_match in list(rel_dicts.keys()):
                relevant_objects = []
                infusion_boxes = []
                try:
                    infusion_boxes = [b for objid in infusion_database[img_id][qid] for b in infusion_database[img_id][qid][objid]]
                    infusion_box_indices_dict = {_[0]: 1 for _ in infusion_boxes}
                except KeyError:
                    pass
                for infusion_box_idx, infusion_box_info in infusion_boxes:
                    if infusion_box_idx >= max_objects:
                        infusion_box_idx = 100
                    # adding new box
                    if infusion_box_idx == 100:
                        for i in reversed(range(max_objects)):
                            if infusion_box_indices_dict.get(i, 0):  # if this object is already needed, skip
                                continue
                            else:
                                infusion_box_indices_dict[i] = 1  # add to used objects
                                relevant_objects.append(i)
                                break
                    # existing box is used
                    else:
                        relevant_objects.append(infusion_box_idx)
                if len(relevant_objects):
                    FIscores_per_qid[int(qid)][relevant_objects] = iou_match
            # remove this entry if no annotations or iou matches found for objects in this img rep
            if np.sum(FIscores_per_qid[int(qid)]==0.01) == max_objects:
                del(FIscores_per_qid[int(qid)])

    return FIscores_per_qid