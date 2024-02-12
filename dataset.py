import os, io
import json
import pickle as cPickle
import numpy as np
import utils
import h5py
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pickle
from torch.utils.data.sampler import Sampler

import re
import pdb
import copy


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s').replace('-', ' ').replace('.', '').replace('"', '').replace('n\'t', ' not').replace('$', ' dollar ')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                if '-' in w:
                    print(w)
                tokens.append(self.add_word(w))
        else:
            for w in words:
                if w in self.word2idx:
                    tokens.append(self.word2idx[w])
                else:
                    tokens.append(len(self.word2idx))
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)




class SelfCriticalDataset(Dataset):

    def __init__(self, split,
                 hint_type,
                 dictionary,
                 opt,
                 discard_items_without_hints=False):
        super(SelfCriticalDataset, self).__init__()
        self.split = split
        self.hint_type = hint_type
        self.dictionary = dictionary  # questions' dictionary
        self.opt = opt
        self.data_dir = opt.data_dir
        self.discard_items_without_hints = discard_items_without_hints
        self.infusion_spatial_hints = {}

        if self.opt.infusion:
            if self.opt.dataset == 'hatcp':  #self.opt.feature_vqahat_vf_original_symbolic_features:
                if self.opt.reproducePaper:
                    # use provided inf_db
                    self.infusion_database = pickle.load(open(os.path.join(self.data_dir, 'true_grounding', 'infusion_database_vqahat_reproducePaper.pkl'), 'rb'))
                else:
                    # use self-generated inf_db
                    self.infusion_database = pickle.load(open(os.path.join(self.data_dir, 'true_grounding', 'infusion_database_vqahat.pkl'), 'rb'))
                self.object_embeddings = pickle.load(open(os.path.join(self.data_dir, 'true_grounding', 'VQAHAT_word_embedding_dict.pkl'),'rb'))
                self.attribute_embeddings = self.object_embeddings
            elif self.opt.dataset == 'gqacp':
                if self.opt.reproducePaper:
                    # use provided inf_db
                    self.infusion_database = pickle.load(open(os.path.join(self.data_dir, 'true_grounding', 'infusion_database_gqa_reproducePaper.pkl'), 'rb'))
                else:
                    # use self-generated inf_db
                    self.infusion_database = pickle.load(open(os.path.join(self.data_dir, 'true_grounding', 'infusion_database_gqa.pkl'), 'rb'))
                self.object_embeddings = pickle.load(open(os.path.join(self.data_dir, 'true_grounding', 'GQA_obj_class_embedding_dict.pkl'), 'rb'))
                self.attribute_embeddings = pickle.load(open(os.path.join(self.data_dir, 'true_grounding', 'GQA_attr_class_embedding_dict.pkl'), 'rb'))

        if hint_type is None and self.discard_items_without_hints:
            raise Exception("Cannot discard items without hints because hint_type is not specified")

        ## load data 
        # load hint 
        if self.hint_type is not None:
            if self.opt.dataset in ['xaicp', 'gqacp', 'hatcp']:
                if self.opt.visfis or self.opt.visfis_all or self.opt.visfis_hatcp:
                    hint_fname = f'hints/{self.opt.hint_path}/FIscores.pkl'
                    self.hint = cPickle.load(open(os.path.join(self.data_dir, hint_fname), 'rb'))
                    print("Hints used:", len(self.hint))
                else:
                    hint_fname = f'hints/{self.opt.hint_path}/{self.split}_{self.hint_type}.pkl'
                    self.hint = cPickle.load(open(os.path.join(self.data_dir, hint_fname), 'rb'))

                print(f"loaded hints from {hint_fname}")


            if self.opt.num_obj_max_manual_setting < 100:
                # reduce hints to number of objects used as input
                for qid in self.hint:
                    self.hint[qid] = self.hint[qid][:self.opt.num_obj_max_manual_setting]

        # support controlled hint exp
        if self.opt.random_suff or self.opt.random_unc or self.opt.random_inv_FI or self.opt.random_align:
            hint_fname = f'hints/{self.opt.dataset}_hints_random.pkl'
            self.hint_random = cPickle.load(open(os.path.join(self.data_dir, hint_fname), 'rb'))
            print("loaded random hint")

        # get qid_to_target
        self.qid_to_target = self.get_qid_to_target()
        print('loaded qid_to_targets')
        # get questions
        self.questions = self.get_questions() 
        # get annotations
        self.annotations = self.get_annotations()
        print(f"loaded questions/annotations")

        # get ans2label / label2ans
        ans2label_path = os.path.join(self.data_dir, 'processed', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(self.data_dir, 'processed', 'trainval_label2ans.pkl')

        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        print('loaded ans2label and label2ans')
        self.num_ans_candidates = len(self.ans2label)
        print(f'num_ans_candidates is {self.num_ans_candidates}')


        self.filterObjects = {}
        # GQACP: accuracy and FPVG spatial / semantic matching
        if self.opt.filter_objects in ["relevant_iou_50pct_trainval", "irrelevant_neg_overlap_25pct_trainval",
                                           "relevant_iou_50pct_trainval_CONTENT", "irrelevant_neg_overlap_25pct_trainval_CONTENT"]:
            # filterobject options
            self.filterObjects = {
                'irrelevant_neg_overlap_25pct_trainval': pickle.load(
                    open(os.path.join(self.opt.data_dir, 'true_grounding',
                                      'GQA_detectron_trainval_irrelevant_objects_path_neg_overlap_25pct.pkl'),'rb')),
                'relevant_iou_50pct_trainval': pickle.load(
                    open(os.path.join(self.opt.data_dir, 'true_grounding',
                                      'GQA_detectron_trainval_relevant_objects_path_iou_50pct.pkl'),'rb')),
                'relevant_iou_50pct_trainval_CONTENT': pickle.load(
                    open(os.path.join(self.opt.data_dir, 'true_grounding',
                                      'GQA_detectron_trainval_relevant_objects_path_iou_50pct_CONTENT.pkl'),'rb')),
                'irrelevant_neg_overlap_25pct_trainval_CONTENT': pickle.load(
                    open(os.path.join(self.opt.data_dir, 'true_grounding',
                                      'GQA_detectron_trainval_irrelevant_objects_path_neg_overlap_25pct_CONTENT.pkl'),'rb'))
            }

        # VQAHAT
        elif self.opt.filter_objects in ["irrelevant_neg_overlap_25pct_VQAHAT", "relevant_iou_50pct_VQAHAT",
                                         "irrelevant_neg_overlap_25pct_VQAHAT_CONTENT", "relevant_iou_50pct_VQAHAT_CONTENT"]:

            self.filterObjects = {
                'irrelevant_neg_overlap_25pct_VQAHAT': pickle.load(
                    open(os.path.join(self.opt.data_dir, 'true_grounding',
                                      'VQAHAT_bottomup_trainval_irrelevant_objects_path_neg_overlap_25pct.pkl'),
                                      'rb')),
                'relevant_iou_50pct_VQAHAT': pickle.load(
                    open(os.path.join(self.opt.data_dir, 'true_grounding',
                                      'VQAHAT_bottomup_trainval_relevant_objects_path_iou_50pct.pkl'),
                                      'rb')),
                'irrelevant_neg_overlap_25pct_VQAHAT_CONTENT': pickle.load(
                    open(os.path.join(self.opt.data_dir, 'true_grounding',
                                      'VQAHAT_bottomup_trainval_irrelevant_objects_path_neg_overlap_25pct_CONTENT.pkl'), 'rb')),
                'relevant_iou_50pct_VQAHAT_CONTENT': pickle.load(
                    open(os.path.join(self.opt.data_dir, 'true_grounding',
                                      'VQAHAT_bottomup_trainval_relevant_objects_path_iou_50pct_CONTENT.pkl'), 'rb'))
            }
            tmpdict = {}
            for imgid in self.filterObjects[self.opt.filter_objects]:
                tmpdict[int(imgid)] = {}
                for qid in self.filterObjects[self.opt.filter_objects][imgid]:
                    tmpdict[int(imgid)][int(qid)] = self.filterObjects[self.opt.filter_objects][imgid][qid]
            self.filterObjects[self.opt.filter_objects] = tmpdict
        elif self.opt.filter_objects == 'original':
            pass
        else:
            raise ValueError("Trying to filter objects, but the specified file entry ({}) does not exist in dict.".format(self.opt.filter_objects))


        # load features and spatials from hdf5 file
        self.image_id2ix = {}
        self.hf = {}
        self.features = {}
        self.spatials = {}
        self.get_features() 
        
        # calc v feature length
        # if self.opt.model_type == 'lxmert':
        #     assert self.opt.spatial_type == 'simple'
        # self.full_v_dim = self.len_pure_visual + \
        #                 utils.get_spatial_length(self.opt.spatial_type, self.opt.spatial_length) + \
        #                 utils.get_oracle_length(self.opt.oracle_type, self.opt.oracle_embed_size)
        self.full_v_dim = self.len_pure_visual
        # me: we set full_v_dim with bbox only, ie either 600+4 or 1024+6

        # init nn.Embedding
        if self.opt.oracle_type == 'wordvec':
            self.oracle_embed = nn.Embedding(2, self.opt.oracle_embed_size, padding_idx=-1)  # Binary
        else:
            self.oracle_embed = None

        self.init_vqx()
        self.tokenize()
        self.tensorize()
        
        print(f"split {self.split} len {self.datalen}")
        # clean up
        del self.questions, self.annotations
        del self.dictionary, self.ans2label, self.label2ans, self.qid_to_target
        try:
            del self.hint
        except AttributeError:
            pass
    
    
    def get_qid_to_target(self):
        if self.opt.dataset in ["vqacp2", "hatcp", "gqacp", 'vinvl', 'bottomup']:
            train_target = cPickle.load(open(os.path.join(self.data_dir, 'processed', f'train_target.pkl'), 'rb'))
            val_target = cPickle.load(open(os.path.join(self.data_dir, 'processed', f'val_target.pkl'), 'rb'))
            target = train_target + val_target

        qid_to_target = {}
        for t in target:
            question_id = t['question_id']
            assert question_id not in qid_to_target
            qid_to_target[question_id] = t
        return qid_to_target

    def get_questions(self):
        if self.opt.dataset == 'vqacp2':
            if self.split == "train":
                f = os.path.join(self.data_dir, f'vqacp_v2_train_questions.json')
            else:
                f = os.path.join(self.data_dir, f'vqacp_v2_test_questions.json')
            
            return json.load(open(f))
        elif self.opt.dataset == 'vqa2':
            year = '2015' if self.split == 'test' else '2014'
            f = os.path.join(self.data_dir, f'v2_OpenEnded_mscoco_{self.split}{year}_questions.json')
            return json.load(open(f))['questions']
        elif self.opt.dataset in ['xaicp','gqacp', 'hatcp', 'vinvl', 'bottomup']:

            split_name = self.split
            if split_name == 'val':
                f = os.path.join(self.data_dir, f'{split_name}_balanced_questions.json')
            elif self.opt.visfis_all:
                f = os.path.join(self.data_dir, 'questions', f'gqacp_{split_name}_questions.json')
            elif self.opt.visfis_hatcp:
                f = os.path.join(self.data_dir, 'questions', f'hatcp_{split_name}_questions.json')

            if self.opt.visfis_hatcp:
                output = json.load(open(f))
                output = {int(k): v for k, v in output.items()}

            else:
                output = json.load(open(f))


            return output


        else:
            raise ValueError(f'cannot get questions for {self.opt.dataset}')
        
    def get_annotations(self):
        if self.opt.dataset == 'vqacp2':
            if self.split == "train":
                f = os.path.join(self.data_dir, f'vqacp_v2_{self.split}_annotations.json')
            else:
                f = os.path.join(self.data_dir, f'vqacp_v2_test_annotations.json')
            return json.load(open(f))
        elif self.opt.dataset == 'vqa2':
            year = '2015' if self.split == 'test' else '2014'
            f = os.path.join(self.data_dir, f'v2_mscoco_{self.split}{year}_annotations.json')
            return json.load(open(f))['annotations']
        elif self.opt.dataset in ['xaicp','gqacp', 'hatcp', 'vinvl', 'bottomup']:

            split_name = self.split
            if split_name == 'val':
                f = os.path.join(self.data_dir, f'{split_name}_balanced_questions.json')
            elif self.opt.visfis_all:
                f = os.path.join(self.data_dir, 'questions', f'gqacp_{split_name}_questions.json')
            elif self.opt.visfis_hatcp:
                f = os.path.join(self.data_dir, 'questions', f'hatcp_{split_name}_questions.json')

            if self.opt.visfis_hatcp:
                output = json.load(open(f))
                output = {int(k): v for k, v in output.items()}
            else:
                output = json.load(open(f))

            return output

        else:
            raise ValueError(f'cannot get annotations for {self.opt.dataset}')
    
    def get_features(self):
        if self.opt.dataset in ['xaicp']:  #, 'gqacp']: # shared train/val features -> xai
            print(f'loading hdf5 for combined train/val')
            # read image_id2ix
            _path = os.path.join(self.data_dir, f'{self.opt.dataset}_imgid2img.pkl')
            self.image_id2ix = cPickle.load(open(_path, 'rb'))
            # read hdf5
            h5_path = os.path.join(self.data_dir, f'{self.opt.dataset}.hdf5')
            self.hf = h5py.File(h5_path, 'r')
            self.features = self.hf.get('image_features')
            self.spatials = self.hf.get('spatial_features')
            # get para
            self.len_pure_visual = self.features.shape[2]
            self.num_objects = self.features.shape[1]

        elif self.opt.dataset in ['gqacp'] and self.opt.bottomup: # shared train/val features -> xai
            print(f'loading hdf5 for combined train/val')
            # read image_id2ix
            _path = os.path.join(self.data_dir, f'{self.opt.dataset}_imgid2img.pkl')
            self.image_id2ix = cPickle.load(open(_path, 'rb'))
            # read hdf5
            h5_path = os.path.join(self.data_dir, f'{self.opt.dataset}.hdf5')
            self.hf = h5py.File(h5_path, 'r')
            self.features = self.hf.get('image_features')
            # self.spatials = self.hf.get('spatial_features')
            # get para
            self.len_pure_visual = self.features.shape[2]
            self.num_objects = self.features.shape[1]

        elif self.opt.dataset == "gqacp":
            print(f'loading hdf5 for combined train/val')

            if self.opt.feature_name == 'detectron_glove_wAttr':
                tmp_file = json.load(open(os.path.join(self.opt.data_dir, 'output_gqa_detectron_objects_info.json'), 'r'))
                h5_path = os.path.join(self.opt.data_dir, 'output_gqa_detectron_objects.h5')
            elif self.opt.feature_name == 'oracle_glove_wAttr':
                tmp_file = json.load(open(os.path.join(self.opt.data_dir, 'output_gqa_ref_objects_info.json'), 'r'))
                h5_path = os.path.join(self.opt.data_dir, 'output_gqa_ref_objects.h5')

            self.image_id2ix = {qid: tmp_file[qid]['index'] for qid in tmp_file}

            self.hf = h5py.File(h5_path, 'r')
            self.features = self.hf.get('features')
            self.features = self.features[:, :self.opt.num_obj_max_manual_setting, :]
            #self.features_bboxes[split] = self.hf[split].get('bboxes')
            if self.opt.use_bbox is True:
                if self.opt.feature_name == 'detectron_glove_wAttr_coco':
                    self.image_widthHeight = {int(qid): (tmp_file[qid]['width'], tmp_file[qid]['height']) for qid in tmp_file}
                else:
                    self.image_widthHeight = {qid: (tmp_file[qid]['width'], tmp_file[qid]['height']) for qid in tmp_file}
                self.features_bboxes = self.hf.get('bboxes')
                self.features_bboxes = self.features_bboxes[:, :self.opt.num_obj_max_manual_setting, :]
                self.len_pure_visual = self.features.shape[2] + 4
            else:
                self.len_pure_visual = self.features.shape[2]
            self.num_objects = self.features.shape[1]

        elif self.opt.dataset == "hatcp" and self.opt.feature_vqahat_vf_original_symbolic_features:  # cp -> need to load both train and val
            print(f'loading hdf5 for {self.split} split')
            # read image_id2ix
            self.image_id2ix = {}
            self.image_id2ix["train"] = cPickle.load(open(os.path.join(self.data_dir,'VQAHAT_train36_imgid2img.pkl'), 'rb'))
            self.image_id2ix["val"] = cPickle.load(open(os.path.join(self.data_dir,'VQAHAT_val36_imgid2img.pkl'), 'rb'))
            # read hdf5
            self.hf = {}
            self.features = {}

            h5_path = os.path.join(self.data_dir, 'VQAHAT_train36_WE_600D.h5')
            self.hf["train"] = h5py.File(h5_path, 'r')
            self.features["train"] = self.hf["train"].get('image_features')


            h5_path = os.path.join(self.data_dir, 'VQAHAT_val36_WE_600D.h5')
            self.hf["val"] = h5py.File(h5_path, 'r')
            self.features["val"] = self.hf["val"].get('image_features')


            self.len_pure_visual = self.features["train"].shape[2]
            self.num_objects = self.features["train"].shape[1]

        else:
            raise ValueError("unsupported dataset in get_features()")

    def init_vqx(self):
        
        print("initializing vqx...")
        count = 0
        self.entries = {}
        print("number of questions ", len(self.questions))
        # iter through questions
        for index, question_id in tqdm(enumerate(self.questions)):
            image_id = self.questions[question_id]['imageId']
            answer_ori = self.annotations[question_id]['answer']

            if self.discard_items_without_hints and question_id not in self.hint.keys():
                # ignore discarded item
                continue
            elif self.hint_type is not None and question_id in self.hint.keys():
                hint = self.hint[question_id]
                hint_flag = 1
            else:
                hint = np.zeros((self.num_objects))
                hint_flag = 0
            
            # add hint as oracle to v_feature
            hint_scores = torch.from_numpy(hint)
            hint_scores = hint_scores.float().unsqueeze(1)
            
            # support controlled hint exp
            # NOTE THIS WONT WORK WITH MY CHANGES
            if self.opt.random_suff or self.opt.random_unc or self.opt.random_inv_FI or self.opt.random_align:
                hint_random_scores = self.hint_random[question_id]
                hint_random_scores = torch.from_numpy(hint_random_scores)
                hint_random_scores = hint_random_scores.float().unsqueeze(1)
                hint_scores = (hint_scores, hint_random_scores)
            
            if self.opt.dataset in ["vqacp2", "hatcp"] or (self.opt.feature_vqahat_vf_original is True or self.opt.feature_vqahat_vf_original_symbolic_features is True): # two splits
                if image_id in self.image_id2ix['train']:
                    cur_split = 'train'
                else:
                    cur_split = 'val'

            # new_entry = {'image': self.image_id2ix[cur_split][image_id] if self.opt.dataset in ["vqacp2", "hatcp"] or (self.opt.feature_vqahat_vf_original is True or self.opt.feature_vqahat_vf_original_symbolic_features is True) else self.image_id2ix[image_id],
            #              'image_id': image_id,
            #              'question_id': question_id,
            #              'question': self.questions[question_id]['question'],
            #              'answer': self.qid_to_target[question_id],
            #              'hint': hint_scores,
            #              'hint_flag': hint_flag,
            #             'answer_ori': answer_ori}
            new_entry = {'image': self.image_id2ix[cur_split][image_id] if self.opt.dataset in ["vqacp2", "hatcp"] or (self.opt.feature_vqahat_vf_original is True or self.opt.feature_vqahat_vf_original_symbolic_features is True) else self.image_id2ix[image_id],
                         'image_id': image_id,
                         'question_id': question_id,
                         'question': self.questions[question_id]['question'],
                         'answer': self.qid_to_target[question_id],
                         'hint': hint,
                         'hint_flag': hint_flag,
                        'answer_ori': answer_ori}

            self.entries[count] = new_entry
            count += 1
        self.datalen = count
        print(f"split {self.split} init_vqx count {count}")
        return count

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for e_id in range(len(self.entries)):
            entry = self.entries[e_id]
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        for e_id in range(len(self.entries)):
            entry = self.entries[e_id]
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if labels is None:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None
            elif len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    def padding_img_feats(self, img_feat, img_feat_pad_size, mode='original'):
        if img_feat.shape[0] > img_feat_pad_size:
            img_feat = img_feat[:img_feat_pad_size]


        img_feat = np.pad(
            img_feat,
            ((0, img_feat_pad_size - img_feat.shape[0]), (0, 0)),
            mode='constant',
            constant_values=0
        )
        return img_feat

    def __getitem__(self, index):
        entry = self.entries[index]
        imgid = entry['image_id']

        qid = entry['question_id']

        hint_score = copy.deepcopy(entry['hint'])
        hint_flag = entry['hint_flag']


        # get features/spatials
        if self.opt.dataset in ["vqacp2"]: # two splits
            if imgid in self.image_id2ix['train']:
                cur_split = 'train'
            else:
                cur_split = 'val'
            image_ix = self.image_id2ix[cur_split][imgid]
            frcn_feat = torch.from_numpy(np.array(self.features[cur_split][image_ix]))
            spatials = torch.from_numpy(np.array(self.spatials[cur_split][image_ix]))
        else: # one split
            if self.opt.dataset == 'hatcp' and (self.opt.feature_vqahat_vf_original or self.opt.feature_vqahat_vf_original_symbolic_features):
                if imgid in self.image_id2ix['train']:
                    cur_split = 'train'
                else:
                    cur_split = 'val'
                image_ix = self.image_id2ix[cur_split][imgid]
                frcn_feat = np.array(self.features[cur_split][image_ix])
            else:  # THIS IS THE STANDARD CASE FOR DETECTRON
                image_ix = self.image_id2ix[imgid]

                frcn_feat = np.array(self.features[image_ix])[:self.opt.num_obj_max_manual_setting]
                if self.opt.use_bbox is True:
                    # bbox_feat = np.array(self.features_bboxes[image_ix])
                    bbox_feat = np.array(self.features_bboxes[image_ix])[:self.opt.num_obj_max_manual_setting]
                    # normalize to 0-1 range
                    bbox_feat[:, 0] /= self.image_widthHeight[imgid][0]
                    bbox_feat[:, 1] /= self.image_widthHeight[imgid][1]
                    bbox_feat[:, 2] /= self.image_widthHeight[imgid][0]
                    bbox_feat[:, 3] /= self.image_widthHeight[imgid][1]


        if self.opt.infusion and self.split == 'train':  # don't use infusion in eval setting
            infusion_boxes = []
            try:
                if self.opt.feature_vqahat_vf_original_symbolic_features:
                    # this one for coco annotations
                    if self.opt.reproducePaper:
                        # to reproduce paper results with infusion (uses provided inf_db and does not re-sort obj-ids)
                        infusion_boxes = [b for objid in self.infusion_database[str(imgid)][str(qid)] for b in
                                          self.infusion_database[str(imgid)][str(qid)][objid]]
                    else:
                        # this is the corrected / deterministic case for testing with differently sorted inf_db dicts
                        # sort obj-ids, as their order affects the position of newly added objects
                        infusion_boxes = [b for objid in sorted(self.infusion_database[str(imgid)][str(qid)]) for b in
                                          self.infusion_database[str(imgid)][str(qid)][objid]]
                # elif self.opt.visfis_hatcp:
                #     infusion_boxes = [b for objid in self.infusion_database[str(imgid).zfill(12)][str(qid)] for b in self.infusion_database[str(imgid).zfill(12)][str(qid)][objid]]
                else:
                    if self.opt.reproducePaper:
                        # to reproduce paper results with infusion (uses provided inf_db and does not re-sort obj-ids)
                        infusion_boxes = [b for objid in self.infusion_database[imgid][qid] for b in
                                          self.infusion_database[imgid][qid][objid]]
                    else:
                        # this is the corrected / deterministic case for testing with differently sorted inf_db dicts
                        # sort obj-ids, as their order affects the position of newly added objects
                        infusion_boxes = [b for objid in sorted(self.infusion_database[imgid][qid]) for b in
                                          self.infusion_database[imgid][qid][objid]]

                infusion_box_indices_dict = {_[0]:1 for _ in infusion_boxes}
            except KeyError:
                pass

            for infusion_box_idx, infusion_box_info in infusion_boxes:
                # add new box
                if infusion_box_idx >= self.opt.num_obj_max_manual_setting:
                    infusion_box_idx = 100
                if infusion_box_idx == 100:
                    num_obj = np.sum(np.any(frcn_feat, axis=1))
                    if num_obj < self.opt.num_obj_max_manual_setting:  # simply append the annotated object that's missing from the detections
                        frcn_feat[num_obj][:300] = self.object_embeddings[infusion_box_info[1]['name']]
                        if self.opt.feature_vqahat_vf_original_symbolic_features is True:
                            att_to_add = infusion_box_info[1]['attributes']
                            if isinstance(att_to_add, list):
                                att_to_add = "UNKTOKEN"
                            frcn_feat[num_obj][300:600] = self.attribute_embeddings[att_to_add]
                        else:
                            for att_to_add in infusion_box_info[1]['attributes']:
                                frcn_feat[num_obj][300:600] += (self.attribute_embeddings[att_to_add] / len(infusion_box_info[1]['attributes']))
                        if self.opt.use_bbox:
                            bbox_feat[num_obj, 0] = infusion_box_info[1]['coordinates'][0] / self.image_widthHeight[imgid][0]
                            bbox_feat[num_obj, 1] = infusion_box_info[1]['coordinates'][1] / self.image_widthHeight[imgid][1]
                            bbox_feat[num_obj, 2] = infusion_box_info[1]['coordinates'][2] / self.image_widthHeight[imgid][0]
                            bbox_feat[num_obj, 3] = infusion_box_info[1]['coordinates'][3] / self.image_widthHeight[imgid][1]
                    else:  # we need to overwrite some box to accomodate the annotated object here
                        for i in reversed(range(self.opt.num_obj_max_manual_setting)):
                            if infusion_box_indices_dict.get(i, 0):  # if this object is already needed, skip
                                continue
                            else:
                                infusion_box_indices_dict[i] = 1  # add to used objects
                                # for vqahat only reset name, theres no attribute information
                                if self.opt.visfis_hatcp is True:
                                    frcn_feat[i][:300] = 0
                                else:
                                    frcn_feat[i][:] = 0
                                frcn_feat[i][:300] = self.object_embeddings[infusion_box_info[1]['name']]
                                if self.opt.feature_vqahat_vf_original_symbolic_features is True:
                                    att_to_add = infusion_box_info[1]['attributes']
                                    if isinstance(att_to_add, list):
                                        att_to_add = "UNKTOKEN"
                                    frcn_feat[i][300:600] = self.attribute_embeddings[att_to_add]
                                else:
                                    for att_to_add in infusion_box_info[1]['attributes']:
                                        frcn_feat[i][300:600] += (self.attribute_embeddings[att_to_add] / len(infusion_box_info[1]['attributes']))
                                if self.opt.use_bbox:
                                    bbox_feat[i, 0] = infusion_box_info[1]['coordinates'][0] / self.image_widthHeight[imgid][0]
                                    bbox_feat[i, 1] = infusion_box_info[1]['coordinates'][1] / self.image_widthHeight[imgid][1]
                                    bbox_feat[i, 2] = infusion_box_info[1]['coordinates'][2] / self.image_widthHeight[imgid][0]
                                    bbox_feat[i, 3] = infusion_box_info[1]['coordinates'][3] / self.image_widthHeight[imgid][1]
                                break

                else:
                    if infusion_box_info[0] == 11:
                        continue  # nothing to do here, existing box is perfect
                    elif infusion_box_info[0] >= 10:  # only attributes to change
                        current_attr = frcn_feat[infusion_box_idx][300:600]
                        if self.opt.feature_vqahat_vf_original_symbolic_features is True:
                            att_to_add = infusion_box_info[1]['attributes']
                            if isinstance(att_to_add, list):
                                att_to_add = "UNKTOKEN"
                            current_attr = self.attribute_embeddings[att_to_add]
                        else:
                            for att_to_change in infusion_box_info[1]['attributes']:
                                current_attr -= (self.attribute_embeddings[att_to_change[0]] / 39)
                                current_attr += (self.attribute_embeddings[att_to_change[1]] / 39)
                        frcn_feat[infusion_box_idx][300:600] = current_attr
                    elif infusion_box_info[0] >= 1:  # only name to change
                        frcn_feat[infusion_box_idx][:300] = self.object_embeddings[infusion_box_info[1]['name']]
                    else:  # both name and attributes to change
                        frcn_feat[infusion_box_idx][:300] = self.object_embeddings[infusion_box_info[1]['name']]
                        current_attr = frcn_feat[infusion_box_idx][300:600]
                        if self.opt.feature_vqahat_vf_original_symbolic_features is True:
                            att_to_add = infusion_box_info[1]['attributes']
                            if isinstance(att_to_add, list):
                                att_to_add = "UNKTOKEN"
                            current_attr = self.attribute_embeddings[att_to_add]
                        else:
                            for att_to_change in infusion_box_info[1]['attributes']:
                                current_attr -= (self.attribute_embeddings[att_to_change[0]] / 39)
                                current_attr += (self.attribute_embeddings[att_to_change[1]] / 39)
                        frcn_feat[infusion_box_idx][300:600] = current_attr


        if self.opt.filter_objects != "original":
            if self.opt.filter_mode == "select":
                # now modify if filterobjects exist
                try:
                    qid_filter = qid
                    filter_objects = self.opt.filter_objects.split(',')
                    # if this is a double filter
                    if len(filter_objects) > 1:
                        image_filter = {}
                        image_filter[qid_filter] = list(set(self.filterObjects[filter_objects[0]].get(imgid, {})[qid_filter]) & set(self.filterObjects[filter_objects[1]].get(imgid, {})[qid_filter]))
                    else:
                        image_filter = self.filterObjects[self.opt.filter_objects].get(imgid, {})
                    # if no filterObjects for this question given, use full image
                    if len([vip_box for vip_box in image_filter[qid_filter] if vip_box < self.opt.num_obj_max_manual_setting]) != 0:
                        image_filter_instance = [vip_box for vip_box in image_filter[qid_filter] if vip_box < self.opt.num_obj_max_manual_setting]
                        # create filter of all objects BUT the filterObjects (=reducing to that set instead of selecting it)
                        object_filter = sorted(list(set([i for i in range(len(frcn_feat))]) - set(image_filter_instance)))
                        # remove all listed objects (question_filter is a list of object indeces)
                        frcn_feat = np.delete(frcn_feat, object_filter, axis=0)
                        if self.opt.use_bbox is True:
                            bbox_feat = np.delete(bbox_feat, object_filter, axis=0)
                except KeyError:
                    # no filter objects found for this question
                    pass

            if self.opt.use_bbox is True:
                frcn_feat = self.padding_img_feats(frcn_feat, img_feat_pad_size=self.num_objects, mode=self.opt.filter_objects)
                bbox_feat = self.padding_img_feats(bbox_feat,img_feat_pad_size=self.num_objects, mode=self.opt.filter_objects)
                frcn_feat = torch.from_numpy(np.concatenate([frcn_feat, bbox_feat], axis=1))
            else:
                frcn_feat = torch.from_numpy(self.padding_img_feats(frcn_feat,img_feat_pad_size=self.num_objects,mode=self.opt.filter_objects))
            hint_score = torch.from_numpy(hint_score)
            hint_score = hint_score.float().unsqueeze(1)

        else:
            if self.opt.use_bbox is True:
                frcn_feat = torch.from_numpy(np.concatenate([frcn_feat, bbox_feat], axis=1))
            else:
                frcn_feat = torch.from_numpy(frcn_feat)
            hint_score = torch.from_numpy(hint_score)
            hint_score = hint_score.float().unsqueeze(1)


        curr_v_feature = frcn_feat
        
        question_ori = entry['question']
        answer_ori = entry['answer_ori']
        
        question = entry['q_token']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        
        
        if labels is not None:
            target.scatter_(0, labels, scores)

        return curr_v_feature, question, target, hint_score, qid, imgid, hint_flag, question_ori, answer_ori
    
    def __len__(self):
        return self.datalen

