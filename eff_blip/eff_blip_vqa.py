from eff_blip.eff_modeling_blip import BertConfig, BertModel, BertLMHeadModel
from eff_blip.eff_blip_caption import create_vit, init_tokenizer, load_checkpoint

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer
import numpy as np

class BLIP_VQA(nn.Module):
    def __init__(self,
                 config,
                 med_config='eff_blip_configs/med_config.json',
                 image_size=480,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.text_len = config['text_len']

        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()

        decoder_config = BertConfig.from_json_file(med_config)

        decoder_config.encoder_width = vision_width
        decoder_config.text_len = config['text_len']
        decoder_config.vision_len = (config['image_size'] // 16) * (config['image_size'] // 16) + 1

        # Text Decoder Laplacian Adapter
        decoder_config.add_text_laplacian_adapter = config['add_text_laplacian_adapter']
        decoder_config.learnable_p = config['learnable_p']
        decoder_config.laplacian_adapter_p_self = config['laplacian_adapter_p_self']
        decoder_config.laplacian_adapter_p_cross = config['laplacian_adapter_p_cross']
        decoder_config.laplacian_adapter_mu = config['laplacian_adapter_mu']
        decoder_config.text_laplacian_adapter_reduction_factor = config['text_laplacian_adapter_reduction_factor']
        decoder_config.image_sample_len = config['image_sample_len']

        # Text Decoder FFN Adapter
        decoder_config.add_text_ffn_adapter = config['add_text_ffn_adapter']
        decoder_config.text_ffn_adapter_reduction_factor = config['text_ffn_adapter_reduction_factor']

        self.text_decoder = BertLMHeadModel(config=decoder_config)

    def forward(self, image, question, answer=None, n=None, weights=None):

        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)

        '''
        n: number of answers for each question
        weights: weight for each answer
        '''
        text_list = []
        text_ques_seq = []

        image_embeds_list = []
        image_atts_list = []

        flag = 0
        for b, n in enumerate(n):
            image_embeds_list += [image_embeds[b]] * n
            image_atts_list += [image_atts[b]] * n
            for i in range(n):
                text_list.append(question[b] + answer[flag])
                text_ques_seq.append(torch.arange(len(self.tokenizer(question[b]).input_ids) - 1))
                flag += 1

        image_embeds = torch.stack(image_embeds_list, 0)
        image_atts = torch.stack(image_atts_list, 0)

        text_ques_seq = torch.nn.utils.rnn.pad_sequence(text_ques_seq).T.to(image.device)

        text = self.tokenizer(text_list, padding='longest', truncation=True, max_length=self.text_len, return_tensors="pt").to(image.device)
        text.input_ids[:, 0] = self.tokenizer.bos_token_id
        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)
        decoder_targets.scatter_(1, text_ques_seq, -100)

        decoder_output = self.text_decoder(text.input_ids,
                                           attention_mask=text.attention_mask,
                                           encoder_hidden_states=image_embeds,
                                           encoder_attention_mask=image_atts,
                                           labels=decoder_targets,
                                           return_dict=True,
                                           reduction='none',
                                           )

        loss = weights * decoder_output.loss
        loss = loss.sum() / image.size(0)

        return loss

    def generate(self, image, question, sample=False, num_beams=3, max_length=10, min_length=1, top_p=0.9,
                 repetition_penalty=1.0):
        image_embeds = self.visual_encoder(image)

        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}

        text_ques_seq = []

        for b in range(image.size(0)):
            text_ques_seq.append(torch.arange(len(self.tokenizer(question[b]).input_ids) - 1))

        input_ids = self.tokenizer(question, padding='longest', truncation=True, max_length=self.text_len,  return_tensors="pt").input_ids.to(image.device)
        input_ids[:, 0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]

        if sample:
            # nucleus sampling
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                 max_length=max_length,
                                                 min_length=min_length,
                                                 do_sample=True,
                                                 top_p=top_p,
                                                 num_return_sequences=1,
                                                 eos_token_id=self.tokenizer.sep_token_id,
                                                 pad_token_id=self.tokenizer.pad_token_id,
                                                 repetition_penalty=1.1,
                                                 **model_kwargs)
        else:
            # beam search
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                 max_length=max_length,
                                                 min_length=min_length,
                                                 num_beams=num_beams,
                                                 eos_token_id=self.tokenizer.sep_token_id,
                                                 pad_token_id=self.tokenizer.pad_token_id,
                                                 repetition_penalty=repetition_penalty,
                                                 **model_kwargs)

        captions = []
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption)
        return captions

    def rank_answer(self, image, question, answer_list, answer_candidates, k):

        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        text_ques_seq = []
        start_x = []
        start_y = []

        for b in range(image.size(0)):
            ques_len = len(self.tokenizer(question[b]).input_ids) - 2
            text_ques_seq.append(torch.arange(ques_len))
            start_x.append(b)
            start_y.append(ques_len)

        input_ids = self.tokenizer(question, padding='longest', truncation=True, max_length=self.text_len, return_tensors="pt").input_ids.to(image.device)
        input_ids[:, 0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]

        answer_ids = answer_candidates.input_ids

        start_output = self.text_decoder(input_ids,
                                         encoder_hidden_states=image_embeds,
                                         encoder_attention_mask=image_atts,
                                         return_dict=True,
                                         reduction='none')
        logits = start_output.logits[start_x, start_y]  # first token's logit

        # topk_probs: top-k probability
        # topk_ids: [num_question, k]
        answer_first_token = answer_ids[:, 1]
        prob_first_token = F.softmax(logits, dim=1).index_select(dim=1, index=answer_first_token)
        topk_probs, topk_ids = prob_first_token.topk(k, dim=1)

        # answer input: [num_question*k, answer_len]
        text_list = []
        text_ques_seq = []

        for b, topk_id in enumerate(topk_ids):
            answer_ids_list = answer_ids.index_select(dim=0, index=topk_id)
            for ids in answer_ids_list:
                text_list.append(question[b] + self.tokenizer.decode(ids, skip_special_tokens=True))
                text_ques_seq.append(torch.arange(len(self.tokenizer(question[b]).input_ids) - 1))

        text_ques_seq = torch.nn.utils.rnn.pad_sequence(text_ques_seq).T.to(image.device)

        text = self.tokenizer(text_list, padding='longest', truncation=True, max_length=self.text_len, return_tensors="pt").to(image.device)
        text.input_ids[:, 0] = self.tokenizer.bos_token_id
        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)
        decoder_targets.scatter_(1, text_ques_seq, -100)

        # repeat encoder's output for top-k answers
        image_embeds = tile(image_embeds, 0, k)
        image_atts = tile(image_atts, 0, k)

        output = self.text_decoder(text.input_ids,
                                   attention_mask=text.attention_mask,
                                   encoder_hidden_states=image_embeds,
                                   encoder_attention_mask=image_atts,
                                   labels=decoder_targets,
                                   return_dict=True,
                                   reduction='none')

        log_probs_sum = -output.loss
        log_probs_sum = log_probs_sum.view(image.size(0), k)

        max_topk_ids = log_probs_sum.argmax(dim=1)
        max_ids = topk_ids[max_topk_ids >= 0, max_topk_ids]

        return max_ids

def blip_vqa(pretrained='', **kwargs):
    model = BLIP_VQA(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
    #         assert(len(msg.missing_keys)==0)
    return model


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))

