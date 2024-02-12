import numpy as np
import pickle, json
import os
import h5py
import copy
from PIL import Image

from preprocessing.truevg.prep_evaluation import get_iou

######################################
## GQA           #####################
######################################

def find_infusion_objects(sg, qa, relevant_objects_per_img_and_question, data_dir):
    # this function creates the infusion database

    reference_gqa_json = json.load(open(os.path.join(data_dir, 'output_gqa_detectron_objects_info.json')))
    print("Loading image DB with object detection results.")
    img_db = h5py.File(os.path.join(data_dir, 'preprocessing', 'img_db.h5'), 'r', driver='core')

    tmp_dict = pickle.load(open(os.path.join(data_dir, 'preprocessing', 'objectname_classes.pkl'), 'rb'))
    objs_dict_tmp = sorted(list(tmp_dict.keys()))
    obj_dict = {c:i for c,i in enumerate(objs_dict_tmp)}

    cat_attr_dict = pickle.load(open(os.path.join(data_dir, 'preprocessing', 'attribute_categories.pkl'), 'rb'))
    cat_attr_dict = {i[0]: sorted(i[1])for i in cat_attr_dict.items()}  # sorted attr lists
    cat_attr_dict_reversed = {i:cat for cat in cat_attr_dict for i in cat_attr_dict[cat]}
    attr_cat_list = sorted(cat_attr_dict)

    out_embeddings = {}
    content_relevant_objects_per_img_and_question = {}

    print("Processing a total of {} images. This may take a while.".format(len(sg)))
    for idx, img_id in enumerate(sorted(list(sg.keys()))):
        if idx % 5000 == 0:
            print("Processed images: {}.".format(idx))

        try:
            img_index_h5 = reference_gqa_json[img_id]['index']
            num_objects = reference_gqa_json[img_id]['objectsNum']
        except:
            print("File for image {} not found, skipping.".format(img_id))
            continue

        out_embeddings[img_id] = [{} for _ in range(num_objects)]
        for obj_idx in range(num_objects):
            out_embeddings[img_id][obj_idx]['name'] = obj_dict[int(img_db['name'][img_index_h5][obj_idx])]
            attribute_names = img_db['attributes'][img_index_h5][obj_idx].tolist()
            # turn attribute classes to names
            if attribute_names[0] == -1:
                # exception handling for five cases in total where no attributes were determined
                out_embeddings[img_id][obj_idx]['attributes'] = []
            else:
                out_embeddings[img_id][obj_idx]['attributes'] = \
                    [cat_attr_dict[cat_name][attr_class_id] for (cat_name,attr_class_id) in zip(attr_cat_list,attribute_names)]

        # check which objects match in content
        if relevant_objects_per_img_and_question.get(img_id, None) is None:
            # if this happens, we need to check if the image has no annotations or if our detections produced no matches
            qids_to_process = []
            for q_id in qa:
                if qa[q_id]['imageId'] == img_id:
                    qids_to_process.append(q_id)
        else:
            qids_to_process = list(relevant_objects_per_img_and_question[img_id].keys())

        for qid in qids_to_process:
            ref_relevant_object_ids = list(set([i for i in qa[qid]['annotations']['answer'].values()] +
                                               [i for i in qa[qid]['annotations']['fullAnswer'].values()] +
                                               [i for i in qa[qid]['annotations']['question'].values()]))
            for obj_id in ref_relevant_object_ids:
                try:
                    relevant_boxes = list(set([rel_box for rel_box in relevant_objects_per_img_and_question[img_id][qid][obj_id] if rel_box < 100]))
                except KeyError:
                    relevant_boxes = []

                matching_content = []
                relevant_boxes_match_degrees = {}
                for box_idx in relevant_boxes:
                    relevant_boxes_match_degrees[box_idx] = [0, {'name': "", 'attributes': []}]
                    if sg[img_id]['objects'][obj_id]['name'] == out_embeddings[img_id][box_idx]['name']:
                        relevant_boxes_match_degrees[box_idx][0] = 10  # -> 11 for full match of ID and all attributes
                    else:
                        relevant_boxes_match_degrees[box_idx][1]['name'] = sg[img_id]['objects'][obj_id]['name']
                    ref_attributes = sg[img_id]['objects'][obj_id]['attributes']
                    if len(ref_attributes) == 0:  # automatically correct if no attributes annotated
                        relevant_boxes_match_degrees[box_idx][0] += 1

                    tmp_dict = {_: 1 for _ in out_embeddings[img_id][box_idx]['attributes']}
                    for attribute_name in ref_attributes:
                        if tmp_dict.get(attribute_name, 0):
                            relevant_boxes_match_degrees[box_idx][0] += (1 /len(ref_attributes))
                        else:
                            # get attribute name that we want to replace in the detected/recognized attributes
                            try:
                                to_be_replaced_attr_name = out_embeddings[img_id][box_idx]['attributes'][attr_cat_list.index(cat_attr_dict_reversed[attribute_name])]
                            except IndexError:
                                print(attribute_name)
                                break

                            relevant_boxes_match_degrees[box_idx][1]['attributes'] = \
                                relevant_boxes_match_degrees[box_idx][1]['attributes'] + [(to_be_replaced_attr_name, attribute_name)]

                    if relevant_boxes_match_degrees[box_idx][0] == 11:  # full match, retain this one
                        del relevant_boxes_match_degrees[box_idx]  # remove from consideration for infusion
                        matching_content.append((box_idx, [11, {}]))  # keep as relevant box

                # if there's no fully-matched boxes (content and location)
                if not len(matching_content):
                    infusion_item = None
                    # check for location-matched boxes with wrong recognitions
                    if len(relevant_boxes_match_degrees):
                        # check if at least one full match is available
                        infusion_candidates = sorted(relevant_boxes_match_degrees.items(), key=lambda x: [1][0], reverse=True)  # last item has the highest match, so infuse that
                        # create list of already used boxes so we don't accidentally reuse/infuse one that's used for a different obj_id either as infusion or matching box
                        try:
                            already_used_boxes = [b for o_id in content_relevant_objects_per_img_and_question[img_id][qid] for b in content_relevant_objects_per_img_and_question[img_id][qid][o_id]]
                        except KeyError:
                            already_used_boxes = []
                        for candidate in infusion_candidates:
                            if candidate[0] in already_used_boxes: continue  # skip already used boxes
                            infusion_item = candidate
                            break  # stop when first=best candidate was selected

                    # if all relevant boxes are already "taken" or if no location-matched box exists, create a new one with correct annotation
                    if infusion_item is None:
                        # if no match at all insert a new object for this obj_id (new obj identified as 100)
                        infusion_item = (100, [11, {'name': sg[img_id]['objects'][obj_id]['name'],
                                                    'attributes': sg[img_id]['objects'][obj_id]['attributes'],
                                                    'coordinates': [sg[img_id]['objects'][obj_id]['x'],
                                                                    sg[img_id]['objects'][obj_id]['y'],
                                                                    sg[img_id]['objects'][obj_id]['x'] + sg[img_id]['objects'][obj_id]['w'],
                                                                    sg[img_id]['objects'][obj_id]['y'] + sg[img_id]['objects'][obj_id]['h']]
                                                    }])
                    matching_content.append(infusion_item)

                content_relevant_objects_per_img_and_question[img_id] = content_relevant_objects_per_img_and_question.get(img_id, {})
                content_relevant_objects_per_img_and_question[img_id][qid] = content_relevant_objects_per_img_and_question[img_id].get(qid, {})
                content_relevant_objects_per_img_and_question[img_id][qid][obj_id] = content_relevant_objects_per_img_and_question[img_id][qid].get(obj_id, [])
                content_relevant_objects_per_img_and_question[img_id][qid][obj_id] = matching_content

            # make sure that we get as many objects as are listed as relevant in annotations
            if len(ref_relevant_object_ids):
                assert len(ref_relevant_object_ids) == len(content_relevant_objects_per_img_and_question[img_id][qid])

    return content_relevant_objects_per_img_and_question


def get_relevant_object_indices_h5_mod(sg, qa, data_dir, threshold=0.5, feature_source="detectron"):
    # function is based on FPVG original released code to create rel/irrel files;
    # difference here is that we additionally need exact info about which annotated objs match which detected obj
    # (this is not needed in FPVG);
    # this function creates information file that's used in the creation of the infusion database;

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
        img_rep_gqa_feats = h5py.File(os.path.join(data_dir, 'output_gqa_ref_objects_info.h5'), 'r')

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
                overlap = get_iou(ref_box, hyp_box)
                if overlap > threshold:
                    best_obj_idx.append(box_idx)
            # getting matching box index for obj_id in img_id
            if best_obj_idx is not None:
                ref_box_match[obj_id] = best_obj_idx

        # now store the found matching indices of detected objects in a dict for all questions in this image
        for q_id in list(q_for_img.keys()):
            question_obj_ids = {}
            dict_entry = q_for_img[q_id]  # the object ids
            try:
                for i in dict_entry:
                    question_obj_ids[i] = ref_box_match[i]
                q_for_img[q_id] = question_obj_ids
            except:
                # answer object was not found with sufficient iou/overlap match among detected objects
                q_for_img[q_id] = {}
        relevant_objects_per_img_and_question[img_id] = q_for_img

    return relevant_objects_per_img_and_question

######################################
## VQA-HAT       #####################
######################################

def find_infusion_objects_vqahat_coco(ref_object_scores, relevant_objects_per_img_and_question, data_dir):
    # build infusion_database for vqahat

    # bottomup detection processing
    num_bboxes = 36
    # img_db contains bottomup detection info: bbox, class, attribute
    img_db = pickle.load(open(os.path.join(data_dir, 'preprocessing', 'VQAHAT_img_db.pkl'), 'rb'))
    # vocab
    object_name_file = os.path.join(data_dir, 'preprocessing', "object_classes_bottomup.txt")
    attribute_name_file = os.path.join(data_dir, 'preprocessing', "attribute_classes_bottomup.txt")
    with open(object_name_file, 'r') as f:
        objs = f.read()
    # obj dicts
    # split: just take the first entry if multiple synonyms
    # start at count 1: class 0 will then be UNKTOKEN
    obj_idx2name = {o_idx: o_name.split(',')[0] for o_idx, o_name in enumerate(objs.split('\n'), 1)}
    obj_idx2name[0] = "UNKTOKEN"
    # process attributes
    with open(attribute_name_file, 'r') as f:
        attrs = f.read()
    # attr dicts
    # split: just take the first entry if multiple synonyms
    attr_idx2name = {a_idx: a_name.split(',')[0] for a_idx, a_name in enumerate(attrs.split('\n'), 1)}
    attr_idx2name[0] = "UNKTOKEN"

    # mapping unmatched coco names to gqa names
    coco_names_dict = {'sports ball': "ball", 'baseball glove': "glove", 'tennis racket': "racket",
                       'potted plant': "plant", 'tv': "television", 'mouse': "computer mouse",
                       'remote': "remote control", 'hair drier': "hair dryer"}

    img_ids_list = list(relevant_objects_per_img_and_question)
    out_embeddings = {}
    content_relevant_objects_per_img_and_question = {}

    print("Processing a total of {} images. This may take a while.".format(len(img_ids_list)))
    for idx, img_id in enumerate(img_ids_list):
        if idx % 5000 == 0:
            print("Processed images: {}.".format(idx))

        # feature file processing
        try:
            out_embeddings[img_id] = [{} for _ in range(num_bboxes)]
            boxes = []
            obj_names = []
            for obj_idx in range(num_bboxes):
                obj_class = int(img_db[int(img_id)]['obj_classes'][obj_idx])
                obj_name = obj_idx2name[obj_class]
                attr_class = int(img_db[int(img_id)]['attr_classes'][obj_idx])
                attr_name = attr_idx2name[attr_class]
                out_embeddings[img_id][obj_idx]['name'] = obj_name
                out_embeddings[img_id][obj_idx]['attributes'] = attr_name
                out_embeddings[img_id][obj_idx]['box'] = img_db[int(img_id)]['bboxes'][obj_idx]

        except KeyError:
            print("File for image {} not found, skipping.".format(img_id))
            continue

        qids_to_process = list(relevant_objects_per_img_and_question[img_id.lstrip('0')].keys())
        for qid in qids_to_process:
            # # this list will contain all relevant bboxes for this qid, with coordinates and object name
            ref_relevant_object_ids = []
            for ref_obj_tuple in ref_object_scores[qid+"_"+img_id.lstrip('0')]:
                if ref_obj_tuple[2] > 0.55:  # threshold for relevance from visfis and relevant_objects file
                    ref_relevant_object_ids.append(ref_obj_tuple[:2])  # bbox and object name

            for obj_id_idx, obj_id_tmp in enumerate(ref_relevant_object_ids):
                obj_id = copy.deepcopy(obj_id_tmp)
                ref_box_list = obj_id[0]
                obj_id[1] = coco_names_dict.get(obj_id[1], obj_id[1])
                try:
                    relevant_boxes = []
                    relevant_boxes_tmp = list(set([rel_box for rel_box in relevant_objects_per_img_and_question[img_id.lstrip('0')][qid] if rel_box < num_bboxes]))
                    # identify which rel objects match which ref object
                    for rel_box in relevant_boxes_tmp:
                        if np.sum(ref_box_list) == 0: continue
                        ref_box = {'x1': int(ref_box_list[0]), 'y1': int(ref_box_list[1]), 'x2': int(ref_box_list[2]), 'y2': int(ref_box_list[3])}
                        x1, y1, x2, y2 = out_embeddings[img_id][rel_box]['box'].tolist()
                        hyp_box = {'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)}
                        overlap = get_iou(ref_box, hyp_box)
                        if overlap > 0.5:
                            relevant_boxes.append(rel_box)
                            # this means, ALL matching objects are determined to match this annotated object.
                            # no exclusion of detected objects for the next ref object is taking place here
                except KeyError:
                    relevant_boxes = []

                matching_content = []
                relevant_boxes_match_degrees = {}
                for box_idx in relevant_boxes:
                    relevant_boxes_match_degrees[box_idx] = [0, {'name': "", 'attributes': []}]
                    if obj_id[1] == out_embeddings[img_id][box_idx]['name']:
                        relevant_boxes_match_degrees[box_idx][0] = 10  # -> 11 for full match of ID and all attributes
                    else:
                        relevant_boxes_match_degrees[box_idx][1]['name'] = obj_id[1]
                    ref_attributes = []
                    if len(ref_attributes) == 0:  # automatically correct if no attributes annotated
                        relevant_boxes_match_degrees[box_idx][0] += 1
                    if relevant_boxes_match_degrees[box_idx][0] == 11:  # full match, retain this one
                        del relevant_boxes_match_degrees[box_idx]  # remove from consideration for infusion
                        matching_content.append((box_idx, [11, {}]))  # keep as relevant box

                # if there's no fully-matched boxes (content and location)
                if not len(matching_content):
                    infusion_item = None
                    # check for location-matched boxes with wrong recognitions
                    if len(relevant_boxes_match_degrees):
                        # check if at least one full match is available
                        infusion_candidates = sorted(relevant_boxes_match_degrees.items(), key=lambda x: [1][0], reverse=True)  # last item has the highest match, so infuse that
                        # create list of already used boxes so we don't accidentally reuse/infuse one that's used for a different obj_id either as infusion or matching box
                        try:
                            already_used_boxes = [b for o_id in content_relevant_objects_per_img_and_question[img_id][qid] for b in content_relevant_objects_per_img_and_question[img_id][qid][o_id]]
                        except KeyError:
                            already_used_boxes = []
                        for candidate in infusion_candidates:
                            if candidate[0] in already_used_boxes: continue  # skip already used boxes
                            infusion_item = candidate
                            break  # stop when first=best candidate was selected

                    # if all relevant boxes are already "taken" or if no location-matched box exists, create a new one with correct annotation (note: ?need to correctly set attribute as combo of 39 words! random select?)
                    if infusion_item is None:
                        # if no match at all insert a new object for this obj_id (new obj identified as 100)
                        infusion_item = (100, [11, {'name': obj_id[1],
                                                    'attributes': [],
                                                    'coordinates': obj_id[0]
                                                    }])
                    matching_content.append(infusion_item)

                content_relevant_objects_per_img_and_question[img_id] = content_relevant_objects_per_img_and_question.get(img_id, {})
                content_relevant_objects_per_img_and_question[img_id][qid] = content_relevant_objects_per_img_and_question[img_id].get(qid, {})
                content_relevant_objects_per_img_and_question[img_id][qid][obj_id_idx] = content_relevant_objects_per_img_and_question[img_id][qid].get(obj_id_idx, [])
                content_relevant_objects_per_img_and_question[img_id][qid][obj_id_idx] = matching_content

            # make sure that we get as many boxes as annotated
            if len(ref_relevant_object_ids):
                assert len(ref_relevant_object_ids) == len(content_relevant_objects_per_img_and_question[img_id][qid])

    return content_relevant_objects_per_img_and_question


def get_heatmap_scores_for_ref_objects(data_dir):
    # VQAHAT: this function creates an intermediate file as "annotation" file for bbox relevance

    # load and prepare data
    questions = json.load(open(os.path.join(data_dir, 'questions', 'hatcp_train_questions.json'), 'r'))
    questions.update(json.load(open(os.path.join(data_dir, 'questions', 'hatcp_dev_questions.json'), 'r')))
    questions.update(json.load(open(os.path.join(data_dir, 'questions', 'hatcp_testid_questions.json'), 'r')))
    questions.update(json.load(open(os.path.join(data_dir, 'questions', 'hatcp_testood_questions.json'), 'r')))

    coco_imgs_anno_train = json.load(open(os.path.join(data_dir, 'COCO', 'annotations', 'instances_train2017.json'), 'r'))
    coco_imgs_anno_val = json.load(open(os.path.join(data_dir, 'COCO', 'annotations', 'instances_val2017.json'), 'r'))
    coco_imgs = {img['id']: {'height': img['height'], 'width': img['width']}
                 for img in coco_imgs_anno_train['images'] + coco_imgs_anno_val['images']}
    coco_categories = {i['id']: i['name'] for i in coco_imgs_anno_train['categories']}
    coco_imgs_anno_dict = {}
    for i in coco_imgs_anno_train['annotations'] + coco_imgs_anno_val['annotations']:
        coco_imgs_anno_dict[i['image_id']] = coco_imgs_anno_dict.get(i['image_id'], []) + [[i['bbox'], coco_categories[i['category_id']]]]

    v1_q = json.load(open(os.path.join(data_dir, 'VQAv1', 'OpenEnded_mscoco_val2014_questions.json'), 'r'))
    v1_q['questions'].extend(json.load(open(os.path.join(data_dir, 'VQAv1', 'OpenEnded_mscoco_train2014_questions.json'), 'r'))['questions'])
    v1_q_dict = {}
    for i in v1_q['questions']:
        v1_q_dict[i['image_id']] = v1_q_dict.get(i['image_id'], []) + [i]

    # helper function for loading and pre-processing heat maps
    def pil_loader(hint_img_qid, img_w_h_tuple):
        path = os.path.join(data_dir, 'VQAHAT')
        if os.path.exists(os.path.join(path, 'vqahat_train', str(hint_img_qid) + '_1.png')):
            path = os.path.join(path, 'vqahat_train', str(hint_img_qid) + '_1.png')
        elif os.path.exists(os.path.join(path, 'vqahat_val', str(hint_img_qid) + '_1.png')):
            path = os.path.join(path, 'vqahat_val', str(hint_img_qid) + '_1.png')
        else:
            print("HAT heat map file doesn't exist. Exit.")
            return 0

        with open(path, 'rb') as f:
            with Image.open(f) as img:
                img_original = img.convert('RGB')
                # resize to match HINT maps
                img_resized = img_original.resize(img_w_h_tuple)
                img_out = np.array(img_resized)[:,:,0]  # only one channel
                img_out = np.swapaxes(img_out, 0,1)  # swap height and width to get regular order of x,y (ie. width, height)
                return img_out

    hint_annotations_out = {}
    print("Processing a total of {} questions. This may take a while.".format(len(questions)))
    for counter, qid in enumerate(list(questions)):
        if counter % 5000 == 0:
            print("Processed questions: {}.".format(counter))

        q = questions[qid]
        hint_img_qid = None
        hint_qid = None
        img_id = None

        original_questions = v1_q_dict[q['imageId']]
        for q_original in original_questions:
            # need to confirm the question, only possible by actually matching question text!
            if q_original['question'] == q['question']:
                hint_img_qid = q_original['question_id']
                img_id = q_original['image_id']
                hint_qid = qid
                # break after first hit, not necessary to look further
                break
        if hint_qid is not None:
            img_w, img_h = coco_imgs[int(img_id)]['width'], coco_imgs[int(img_id)]['height']
            img_matrix = pil_loader(hint_img_qid, (img_w, img_h))
        else:
            print("Nothing found for ", q)

        # now get bboxes for calculating score
        try:
            bboxes = coco_imgs_anno_dict[img_id]
        except KeyError:
            bboxes = []

        for bbox_info in bboxes:
            # calculate FI scores for coco reference objects based on spatial matching with heat maps
            bbox = np.array(bbox_info[0], dtype=int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]
            avg_rel = np.average(img_matrix[x1:x2, y1:y2])
            # happens when annotated bbox is out of bounds, ie a wrong annotation
            if np.isnan(avg_rel):
                hint_score = 0.0
                continue
            # now irrel avg score
            img_inv = img_matrix.copy()
            img_inv[x1:x2, y1:y2] = 0
            irrel_sum = np.sum(img_inv)
            # total number of elements minus area of bbox
            irrel_elements = np.ndarray.flatten(img_inv).shape[0] - ((x2-x1) * (y2-y1))
            avg_irrel = irrel_sum / irrel_elements
            hint_score = avg_rel / (avg_rel + avg_irrel)
            if np.isnan(hint_score): hint_score = 0.0
            # output
            hint_annotations_out[str(hint_qid) + "_" + str(img_id)] = hint_annotations_out.get(str(hint_qid) + "_" + str(img_id), []) + [[[x1, y1, x2, y2], bbox_info[1], hint_score]]

    # return FI scores for each ref object in coco, for each question individually
    return hint_annotations_out


