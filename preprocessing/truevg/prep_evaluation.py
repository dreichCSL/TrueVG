import numpy as np
import json, pickle
import os
import h5py


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Parameters
    ----------
    bb1 : dict, bounding box of object
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict, bounding box of area box
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y2) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    Returns
    -------
    float
        in [0, 1]
    """

    if not (bb1['x1'] < bb1['x2']) \
            or not (bb1['y1'] < bb1['y2']) \
            or not (bb2['x1'] < bb2['x2']) \
            or not (bb2['y1'] < bb2['y2']):
        return 0.0
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = int((x_right - x_left)) * int((y_bottom - y_top))
    # compute the area of both AABBs
    bb1_area = int((bb1['x2'] - bb1['x1'])) * int((bb1['y2'] - bb1['y1']))
    bb2_area = int((bb2['x2'] - bb2['x1'])) * int((bb2['y2'] - bb2['y1']))
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas minus one time the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def get_overlap(bb1, bb2):
    """
    Calculate the overlap percentage of bb1 with bb2 (how much area of bb1 does bb2 cover?).
    Parameters
    ----------
    bb1 : dict, bounding box of object
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict, bounding box of area box
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y2) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    Returns
    -------
    float
        in [0, 1]
    """

    if not (bb1['x1'] < bb1['x2']) \
            or not (bb1['y1'] < bb1['y2']) \
            or not (bb2['x1'] < bb2['x2']) \
            or not (bb2['y1'] < bb2['y2']):
        return 0.0
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = int((x_right - x_left)) * int((y_bottom - y_top))
    # compute the area of both AABBs
    bb1_area = int((bb1['x2'] - bb1['x1'])) * int((bb1['y2'] - bb1['y1']))
    ## change: we want to know the area of the object's bbox that is covered by the absolute position area box
    overlap_rate = intersection_area / float(bb1_area)
    assert overlap_rate >= 0.0
    assert overlap_rate <= 1.0
    return overlap_rate



def get_truevg_qids(rel_obj, irrel_obj, infusion_database, fpvg_mode=True):
    # fpvg_mode=False is only used for investigating certain statistics
    # this function creates the list of truevg samples

    if fpvg_mode is True:
        # only those qids are used for FPVG that have some objects to run classification with
        rel_qids = [i for imgid in rel_obj for i in rel_obj[imgid] if len(rel_obj[imgid][i])>0]
        irrel_qids = [i for imgid in irrel_obj for i in irrel_obj[imgid] if len(irrel_obj[imgid][i])>0]
        all_qids = list(set(rel_qids) & set(irrel_qids))
    else:
        # fpvg_mode=False is only used for investigating certain statistics
        all_qids = [j for i in infusion_database for j in infusion_database[i]]

    # find those qids that have all needed (location) boxes with correct content
    infused_qids = []
    for imgid in infusion_database:
        for qid in infusion_database[imgid]:
            for objid in infusion_database[imgid][qid]:
                for box in infusion_database[imgid][qid][objid]:
                    # if infusion happens (an existing object is altered or a new object is added), then add this qid
                    if (box[1][0]!=11) or (box[1][0]==11 and box[0]==100):
                        infused_qids.append(qid)
    # exclude all qids that would require infusion (=they're missing visual content and are NOT truevg)
    truevg_qids = list(set(all_qids) - set(infused_qids))

    return truevg_qids


######################################
## GQA           #####################
######################################

def get_relevant_object_indices_semantic_matching(hints_spatial, infusion_database, num_objects=100, vqahat=False):
    # this function determines rel/irrel detected objects with SEMANTIC matching for FPVG evaluation

    qid2imgid = {qid:imgid for imgid in infusion_database for qid in infusion_database[imgid]}
    all_rel = {}
    all_irrel = {}
    for qid in hints_spatial:
        qid = str(qid)
        imgid = qid2imgid.get(qid, None)
        if imgid is not None:
            rel_objects = []
            irrel_filter = []
            for objid in infusion_database[imgid][qid]:
                for obj in infusion_database[imgid][qid][objid]:
                    if obj[0] < 100:
                        if obj[1][0] == 11:  # all content correct
                            rel_objects.append(obj[0])
                            irrel_filter.append(obj[0])
                        else:
                            irrel_filter.append(obj[0])

            if vqahat is True:
                imgid = int(imgid)
            if len(rel_objects) > 0:
                tmp_entry = all_rel.get(str(imgid), {})
                tmp_entry[qid] = sorted(rel_objects)
                all_rel[str(imgid)] = tmp_entry

            tmp_entry = all_irrel.get(str(imgid), {})
            # all objects are irrel, that aren't partial semantic/location overlaps
            tmp_entry[qid] = [i for i in range(num_objects) if i not in irrel_filter]
            all_irrel[str(imgid)] = tmp_entry

    return all_rel, all_irrel


# copied and modified from official FPVG code
def get_relevant_object_indices_spatial_matching(sg, qa, data_dir, threshold=0.5, matching_method='iou', feature_source="detectron"):
    # this function determines rel/irrel detected objects with SPATIAL matching for FPVG evaluation

    # matching_method: overlap or iou

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

    if feature_source == "detectron":
        img_rep_gqa_json = json.load(open(os.path.join(data_dir, 'output_gqa_detectron_objects_info.json'), 'r'))
        img_rep_gqa_feats = h5py.File(os.path.join(data_dir, 'output_gqa_detectron_objects.h5'), 'r')
    elif feature_source == "ref":
        img_rep_gqa_json = json.load(open(os.path.join(data_dir, 'output_gqa_ref_objects_info.json'), 'r'))
        img_rep_gqa_feats = h5py.File(os.path.join(data_dir, 'output_gqa_ref_objects.h5'), 'r')

    print("Processing a total of {} images. This may take a while.".format(len(img_id_list)))
    for c,img_id in enumerate(img_id_list):
        if c % 5000 == 0:
            print("Processed images: {}.".format(c))

        # feature files processing
        try:
            img_index_h5 = img_rep_gqa_json[str(img_id)]['index']
            in_bboxes = img_rep_gqa_feats['bboxes'][img_index_h5]
            num_bboxes = img_rep_gqa_json[str(img_id)]['objectsNum']
        except:
            print("File for image {} not found, skipping.".format(img_id))
            continue

        # first get all ref boxes in a dict
        q_for_img = relevant_objects_per_img_and_question[img_id]
        q_id_list = list(q_for_img.keys())
        ref_box_match = {}
        for q_id in q_id_list:
            for obj_id in q_for_img[q_id]:
                ref_box_match[obj_id] = None

        # then find the iou match among ref boxes (min 0.5). if not found, skip
        for obj_id in list(ref_box_match.keys()):
            best_obj_idx = []
            ref_o = sg[str(img_id)]['objects'][obj_id]
            ref_box = {'x1': ref_o['x'], 'y1': ref_o['y'], 'x2': ref_o['x'] + ref_o['w'], 'y2': ref_o['y'] + ref_o['h']}
            for box_idx, obj in enumerate(in_bboxes):
                x1, y1, x2, y2 = list(in_bboxes[box_idx])  # (4,) x,y,x2,y2
                # if boxes have no coordinates, they are empty: skip
                if np.sum([x1, y1, x2, y2]) == 0: continue
                hyp_box = {'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)}
                if matching_method == 'iou':
                    overlap = get_iou(ref_box, hyp_box)
                elif matching_method == 'neg_overlap':
                    overlap = get_overlap(hyp_box, ref_box)  # how much of hyp box is relevant

                if overlap > threshold:
                    best_obj_idx.append(box_idx)
            # getting matching box index for obj_id in img_id
            if best_obj_idx is not None:
                ref_box_match[obj_id] = best_obj_idx

        # now store the found matching indices of detected objects in a dict for all questions in this image
        for q_id in list(q_for_img.keys()):
            dict_entry = q_for_img[q_id]
            try:
                box_idx_list = []
                for i in dict_entry:
                    box_idx_list.extend(ref_box_match[i])
                if matching_method == 'neg_overlap':
                    # only keep those that did not have any matches with any of the ref_boxes
                    box_idx_list = set([_ for _ in range(num_bboxes)]) - set(box_idx_list)
                q_for_img[q_id] = sorted(list(set(box_idx_list)))
            except:
                # answer object was not found with sufficient iou/overlap match among detected objects
                q_for_img[q_id] = []
        relevant_objects_per_img_and_question[img_id] = q_for_img

    return relevant_objects_per_img_and_question


######################################
## VQA-HAT       #####################
######################################

def get_relevant_object_indices_spatial_matching_bottomup_cocoref(ref_object_scores, data_dir,
                                                                  threshold=0.5, matching_method='iou'):
    # this function gets FPVG files with spatial matching
    # based on FPVG original code

    # matching_method: neg_overlap or iou

    # to get object bbox for each question per img
    relevant_objects_per_img_and_question = {}

    # get reference/annotated relevant object ids per question (and image)
    for q_id_img_id in list(ref_object_scores):
        object_list = []
        q_id, img_id = q_id_img_id.split('_')
        for obj in ref_object_scores[q_id_img_id]:
            if obj[2] > 0.55:  # threshold from visfis paper used for VQAHAT determination of rel objects
                object_list.append(obj[0])
        if len(object_list):
            dict_entry = relevant_objects_per_img_and_question.get(img_id, {})
            dict_entry.update({q_id: object_list})  # list of bboxes
            relevant_objects_per_img_and_question[img_id] = dict_entry

    img_id_list = list(relevant_objects_per_img_and_question.keys())
    # img_db contains bottomup-detected bbox info and detected class/attribute
    img_db = pickle.load(open(os.path.join(data_dir, 'preprocessing', 'VQAHAT_img_db.pkl'), 'rb'))

    # bottomup number
    num_bboxes = 36

    print("Processing a total of {} images. This may take a while.".format(len(img_id_list)))
    for c,img_id in enumerate(img_id_list):
        if c % 5000 == 0:
            print("Processed images: {}.".format(c))

        # bottomup detection
        try:
            in_bboxes = img_db[int(img_id)]['bboxes']
        except:
            print("File for image {} not found, skipping.".format(img_id))
            continue

        # first get all ref boxes in a dict
        q_for_img = relevant_objects_per_img_and_question[img_id]
        q_id_list = list(q_for_img.keys())
        ref_box_match = {}
        for q_id in q_id_list:
            for obj_id in q_for_img[q_id]:
                ref_box_match[str(obj_id)] = None  # bbox list of 4 coordinates to string for dict compatibility; reverse further down

        # then find the iou match among ref boxes (min 0.5). if not found, skip
        for obj_id in list(ref_box_match.keys()):
            best_obj_idx = []
            ref_o = [int(i) for i in obj_id.strip('][').split(', ')]  # reverts string of int list to regular int list
            ref_box = {'x1': ref_o[0], 'y1': ref_o[1], 'x2': ref_o[2], 'y2': ref_o[3]}
            for box_idx, obj in enumerate(in_bboxes):
                x1, y1, x2, y2 = list(in_bboxes[box_idx])  # (4,) x,y,x2,y2
                # if boxes have no coordinates, they are empty: skip
                if np.sum([x1, y1, x2, y2]) == 0: continue
                hyp_box = {'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)}
                if matching_method == 'iou':
                    overlap = get_iou(ref_box, hyp_box)
                elif matching_method == 'neg_overlap':
                    overlap = get_overlap(hyp_box, ref_box)  # how much of hyp box is relevant
                if overlap > threshold:
                    best_obj_idx.append(box_idx)
            # getting matching box index for obj_id in img_id
            if best_obj_idx is not None:
                ref_box_match[obj_id] = best_obj_idx

        # now store the found matching indices of detected objects in a dict for all questions in this image
        for q_id in list(q_for_img.keys()):
            dict_entry = q_for_img[q_id]
            try:
                box_idx_list = []
                for i in dict_entry:
                    box_idx_list.extend(ref_box_match[str(i)])
                if matching_method == 'neg_overlap':
                    # only keep those that did not have any matches with any of the ref_boxes
                    box_idx_list = set([_ for _ in range(num_bboxes)]) - set(box_idx_list)
                q_for_img[q_id] = sorted(list(set(box_idx_list)))
            except:
                # answer object was not found with sufficient iou/overlap match among detected objects
                q_for_img[q_id] = []
        relevant_objects_per_img_and_question[img_id] = q_for_img

    return relevant_objects_per_img_and_question
