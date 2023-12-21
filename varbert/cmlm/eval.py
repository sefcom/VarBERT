import os
import sys
import json
import pprint
import logging
import argparse
import jsonlines
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm, trange

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    CamembertConfig,
    CamembertForMaskedLM,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

import torch.nn as nn
from   torch.nn import CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN, gelu


from transformers import RobertaConfig
from transformers.models.bert.modeling_bert import BertEmbeddings, BertModel, BertPreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaLMHead

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    "roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    "roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
    "distilroberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-pytorch_model.bin",
    "roberta-base-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-openai-detector-pytorch_model.bin",
    "roberta-large-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-openai-detector-pytorch_model.bin",
}

l = logging.getLogger('model_main')
vocab_size = 50001

class RobertaLMHead2(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size*8)
    
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#         self.layer_norm = nn.LayerNorm(8*config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, vocab_size, bias=False)
#         self.decoder = nn.Linear(8*config.hidden_size, vocab_size, bias=False)
        
        self.bias = nn.Parameter(torch.zeros(vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x


class RobertaForMaskedLMv2(RobertaForMaskedLM):

    def __init__(self, config):
        super().__init__(config)
        self.lm_head2 = RobertaLMHead2(config)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        masked_lm_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         print("inputs:",input_ids,"labels:",labels)
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head2(sequence_output)
        output_pred_scores = torch.topk(prediction_scores,k=20,dim=-1)
        outputs = (output_pred_scores,)  # Add hidden states and attention if they are here

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs
#         output = (prediction_scores,) + outputs[2:]
#         return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output


class RobertaForCMLM(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def forward(self,input_ids=None,attention_mask=None,token_type_ids=None,
        position_ids=None,head_mask=None,inputs_embeds=None,masked_lm_labels=None):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)
        output_pred_scores = torch.topk(prediction_scores,k=20,dim=-1)
        outputs = (output_pred_scores,)  # Add hidden states and attention if they are here

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    "openai-gpt": (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForMaskedLMv2, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    "camembert": (CamembertConfig, CamembertForMaskedLM, CamembertTokenizer),
}


class CMLDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size=512,limit=None):
        assert os.path.isfile(file_path)
        logger.info("Creating features from dataset file at %s", file_path)

        self.examples=[]
        with jsonlines.open(file_path, 'r') as f:
            for ix,line in tqdm(enumerate(f),desc="Reading Jsonlines",ascii=True):
                if limit is not None and ix>limit:
                    continue
                if (None in line["inputids"]) or (None in line["labels"]):
                    print("LineNum:",ix,line)
                    continue
                else:
                    self.examples.append(line)
            
        if limit is not None:
            self.examples = self.examples[0:limit]
        self.block_size = int(block_size)
        self.tokenizer = tokenizer
        self.truncated={}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        block_size =  self.block_size
        tokenizer = self.tokenizer
        
        item = self.examples[i]
        
        input_ids = item["inputids"]
        labels = item["labels"]
        assert len(input_ids) == len(labels)
        
        if len(input_ids) > block_size-2:
            self.truncated[i]=1
        
        if len(input_ids) >= block_size-2:
            input_ids = input_ids[0:block_size-2]
            labels = labels[0:block_size-2]
        elif len(input_ids) < block_size-2:
            input_ids = input_ids+[tokenizer.pad_token_id]*(self.block_size-2-len(input_ids))
            labels = labels + [tokenizer.pad_token_id]*(self.block_size-2-len(labels))
        
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
        labels = tokenizer.build_inputs_with_special_tokens(labels)
        
        assert len(input_ids) == len(labels)
        assert len(input_ids) == block_size
        try:
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            labels    = torch.tensor(labels, dtype=torch.long)
            mask_idxs = (input_ids==tokenizer.mask_token_id).bool()
            labels[~mask_idxs]=-100
        except:
            l.error(f"Unexpected error at index {i}: {sys.exc_info()[0]}")
            raise
        
        return input_ids , labels
    
parser = argparse.ArgumentParser()
parser.add_argument(
        "--model_name",
        type=str,
        help="The model checkpoint for weights initialization.",
    )
parser.add_argument(
        "--data_file",
        type=str,
        help="Input Data File to Score",
    )
parser.add_argument(
        "--meta_file",
        type=str,
        help="Input Meta File to Score",
    )

parser.add_argument(
        "--prefix",
        default="test",
        type=str,
        help="prefix to separate the output files",
    )

parser.add_argument(
        "--pred_path",
        default="outputs",
        type=str,
        help="path where the predictions will be stored",
    )

parser.add_argument(
        "--batch_size",
        default=20,
        type=int,
        help="Eval Batch Size",
    )
args = parser.parse_args()

device = torch.device("cuda")
n_gpu = torch.cuda.device_count()
config_class, model_class, tokenizer_class = MODEL_CLASSES["roberta"]

config = config_class.from_pretrained(args.model_name)
tokenizer = tokenizer_class.from_pretrained(args.model_name)
model = model_class.from_pretrained(
            args.model_name,
            from_tf=bool(".ckpt" in args.model_name),
            config=config,
        )

model.to(device)
tiny_dataset = CMLDataset(tokenizer,file_path=args.data_file,block_size=1024)
eval_sampler = SequentialSampler(tiny_dataset)
eval_dataloader = DataLoader(tiny_dataset, sampler=eval_sampler, batch_size=args.batch_size)

model.eval()
eval_loss = 0.0
nb_eval_steps = 0

matched={1:0,3:0,5:0,10:0}
totalmasked={1:0,3:0,5:0,10:0}

pred_list={
    1 : [],
    3 : [],
    5 : [],
    10: []
}
gold_list=[]

for batch in tqdm(eval_dataloader, desc="Evaluating"):
    inputs, labels = batch[0], batch[1]
    only_masked = inputs==tokenizer.mask_token_id
    masked_gold = labels[only_masked]
    
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    gold_list.append(masked_gold.tolist())

    with torch.no_grad():
        outputs = model(inputs, masked_lm_labels=labels)
        lm_loss = outputs[0]
        inference = outputs[1].indices.cpu()
        eval_loss += lm_loss.mean().item()
        
        # TopK Calculation
        masked_predict = inference[only_masked]
        for k in [1,3,5,10]:
            topked = masked_predict[:,0:k]
            pred_list[k].append(topked.tolist())
            for i,cur_gold_tok in enumerate(masked_gold):
                totalmasked[k]+=1
                cur_predict_scores = topked[i]
                if cur_gold_tok in cur_predict_scores:
                    matched[k]+=1
        
    nb_eval_steps += 1

eval_loss = eval_loss / nb_eval_steps
perplexity = torch.exp(torch.tensor(eval_loss))
print(f"Perplexity: ", perplexity)
for i in [1,3,5,10]:
    print("TopK:", i, matched[i]/totalmasked[i])
    
print("Truncated:", len(tiny_dataset.truncated))

os.makedirs(args.pred_path, exist_ok=True)
json.dump(pred_list,open(os.path.join(args.pred_path, args.prefix+"_pred_list.json"),"w"))
json.dump(gold_list,open(os.path.join(args.pred_path, args.prefix+"_gold_list.json"),"w"))