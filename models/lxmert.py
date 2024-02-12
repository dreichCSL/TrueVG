import torch
import torch.nn as nn
import numpy as np
import os

import transformers

class LxmertVisualAnswerHead(nn.Module):
    # NOTE: ONLY FOR GQA-pretrained small LXMERT models; 1843 classes:
    # def __init__(self, hid_dim=768, num_labels=28):
    def __init__(self, hid_dim=768, num_labels=1843):
        super().__init__()
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            nn.GELU(),
            nn.LayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_labels),
        )

    def forward(self, hidden_states):
        return self.logit_fc(hidden_states)

class lxmert(nn.Module):
    def __init__(self, opt):
        super().__init__()
        num_labels = opt.num_ans_candidates
        hid_dim = opt.lxmert_hid_dim

        if opt.lxmert_small_model:
            assert hid_dim == 128
            # for small lxmert model (trained only on visfis_all):
            lxmert_config = transformers.LxmertConfig(visual_feat_dim=600, visual_attr_loss=False,
                                                      intermediate_size=512, num_attention_heads=4, hidden_size=128)
            if opt.visfis_all:
                if opt.feature_name[:6] == 'oracle':
                    # if oracle features in finetuning, use oracle pretrained lxmert model
                    self.lxmert_encoder = transformers.LxmertModel.from_pretrained(
                        os.path.join(opt.data_dir, 'lxmert_pretrained_models', "lxmertPretrain_ORA/pytorch_model.bin"),
                        config=lxmert_config)
                elif opt.infusion:
                    # if infusion features in finetuning, use infusion pretrained lxmert model
                    self.lxmert_encoder = transformers.LxmertModel.from_pretrained(
                        os.path.join(opt.data_dir, 'lxmert_pretrained_models', "lxmertPretrain_INF/pytorch_model.bin"),
                        config=lxmert_config)
                else:
                    self.lxmert_encoder = transformers.LxmertModel.from_pretrained(
                        os.path.join(opt.data_dir, 'lxmert_pretrained_models', "lxmertPretrain_DET/pytorch_model.bin"),
                        config=lxmert_config)

        else:
            print("ME: ONLY SMALL LXMERT FOR GQA/VISFIS IS SUPPORTED IN THIS VERSION")
            exit(1)


        self.answer_head = LxmertVisualAnswerHead(hid_dim, num_labels)

    def forward(self,
                input_ids,
                attention_mask,
                visual_feats,
                visual_pos,
                token_type_ids,
                return_dict,
                output_attentions):
        output = self.lxmert_encoder(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    visual_feats=visual_feats,
                                    visual_pos=visual_pos,
                                    token_type_ids=token_type_ids,
                                    return_dict=return_dict,
                                    output_attentions=output_attentions)
        result = self.answer_head(output['pooled_output'])
        return {'question_answering_score': result}