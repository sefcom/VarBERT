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
    RobertaTokenizerFast,
    get_linear_schedule_with_warmup,
)

import torch.nn as nn
from   torch.nn import CrossEntropyLoss, MSELoss
from transformers.activations import ACT2FN, gelu

from transformers import RobertaConfig
from transformers.models.bert.modeling_bert import BertEmbeddings, BertModel, BertPreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaLMHead

l = logging.getLogger('model_main')
var_size = 2

class RobertaLMHead2(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.out_vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.out_vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)
        return x

    
class RobertaLMHead3(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, var_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(var_size))

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
        self.lm_head3 = RobertaLMHead3(config)
        self.out_vocab_size = config.out_vocab_size
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
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
        prediction_scores_origin = self.lm_head3(sequence_output)  #32,1024,3
#         prediction_scores_origin_out = prediction_scores_origin[:,:,:var_size-1] # Prediction between two classes 0,1 ignoring -100
        output_pred_scores = torch.topk(prediction_scores,k=20,dim=-1)
        output_pred_scores_type = torch.topk(prediction_scores_origin,k=1,dim=-1)

        outputs = (output_pred_scores, output_pred_scores_type)  # Add hidden states and attention if they are here
        
        var_labels = labels[:,:,0]
        decomp_labels = labels[:,:,1]

        masked_lm_loss = None
        masked_lm_loss_origin = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss_fct_type = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.out_vocab_size), var_labels.view(-1))
            masked_lm_loss_origin = loss_fct_type(prediction_scores_origin.view(-1, var_size), decomp_labels.view(-1))
            masked_loss = masked_lm_loss + masked_lm_loss_origin
            outputs = (masked_lm_loss,) + outputs

        return outputs

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    "openai-gpt": (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForMaskedLMv2, RobertaTokenizerFast),
    "distilbert": (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    "camembert": (CamembertConfig, CamembertForMaskedLM, CamembertTokenizer),
}

fid_list = []

def var_origin_match(out, idx_to_word):
    ''' 
    Getting the results for variable origin prediction at the variable level 
    '''
    
    gold_origin = []
    pred_origin = []
    joint_result = defaultdict(list)
    varlevel_results = defaultdict(float)
    
    for each in tqdm(out):
        fid = each["fid"]
        goldorigin = each["gold_origin"]
        predorigin = each["pred_origin"]
        varposall = each["varposall"]
        assert len(goldorigin) == len(predorigin)
        assert len(goldorigin) == len(varposall)
        
        # ---- checkout the positions of each variable ----
        pos = defaultdict(list)
        for i, var in enumerate(varposall):
            pos[var].append(i)

        # ---- get single origin predictions for each occurrence of a variable -----
        for each_var in pos:
            single_gold_origin = set([goldorigin[each_pos] for each_pos in pos[each_var]])
            assert len(single_gold_origin) == 1
            single_gold_origin = list(single_gold_origin)[0]            
            var_origin_count = defaultdict(int)
            for each_pos in pos[each_var]:
                var_origin_count[predorigin[each_pos]] += 1
            single_pred_origin = 'decompiler' if var_origin_count['dwarf'] < var_origin_count['decompiler'] else 'dwarf'

            joint_result[fid].append((each_var, single_gold_origin, single_pred_origin))
            gold_origin.append(single_gold_origin)
            pred_origin.append(single_pred_origin)
    
    # ---- Getting origin statistics ------
    varlevel_results['ACCURACY'] = round(accuracy_score(gold_origin, pred_origin),4)
    varlevel_results['F1'] = round(f1_score(gold_origin, pred_origin, average='macro'),4)
    varlevel_results['P'] = round(precision_score(gold_origin, pred_origin, average='macro'),4)
    varlevel_results['R'] = round(recall_score(gold_origin, pred_origin, average='macro'),4)
    
    # ---- Dump Origin predictions variable level ----
    json.dump(joint_result, open(os.path.join(args.model_name,args.prefix+"_origin_predictions.json"),'w'))
    return varlevel_results


def var_match(out, idx_to_word):
    
    joint_result = defaultdict(list)
    joint_result_word = defaultdict(list)
    total_dwarf = 0
    total_matched = 0
    total_matched_nooov = 0
    total_nooov = 0
    total_oov = 0
    total_matched_origin = 0
    total_vars = 0
    
    for each in tqdm(out):
        total_vars = each["num_vars"]
        fid = each["fid"]
        pos = each["varpos"]
        gold = each["gold"]
        pred = each["pred"]
        goldorigin = each["gold_origin"]
        predorigin = each["pred_origin"]
        
        var_level_pred = []
            
        # Only for dwarf variables 
        for each_var in pos:
            if each_var == "-100":
                continue
            single_gold = set([gold[each_pos] for each_pos in pos[each_var]])
            assert len(single_gold) == 1 #There should be only 1 gold variable for each gold var position
            single_gold = list(single_gold)[0]
            
            # Majority voting with conflict resolved by prob
            all_pred_each_pos = {}
            for each_pos in pos[each_var]:
                if pred[each_pos] not in all_pred_each_pos:
                    all_pred_each_pos[pred[each_pos]] = {'count':1, 
                                                         # 'prob':prob[each_pos],
                                                        }
                else:
                    all_pred_each_pos[pred[each_pos]]['count']+=1
                    # all_pred_each_pos[pred[each_pos]]['prob']+=prob[each_pos]
            multi_pred = []
            for each in all_pred_each_pos:
                multi_pred.append((each, all_pred_each_pos[each]['count']))
            
            sorted_multi_pred = sorted(multi_pred, key=lambda x:x[1], reverse=True)
            single_pred = sorted_multi_pred[0][0]
            gold_word = idx_to_word.get(str(single_gold),"UNK")
            pred_word = idx_to_word.get(str(single_pred),"UNK")
            joint_result_word[fid].append((each_var, gold_word, pred_word))
            total_dwarf += 1
            if gold_word == "UNK": total_oov += 1
            else: total_nooov += 1
            if gold_word == pred_word:
                total_matched += 1
            if gold_word == pred_word and gold_word!="UNK":
                total_matched_nooov += 1
    ## VARLEVEL Resutls
    varlevel_results = {'TOTAL_VARS': total_dwarf,
                        'TOTAL_OOV':total_oov,
                       'TOTAL_NO_OOV':total_nooov,
                       'MATCHED':total_matched,
                       'MATCHED_NO_OOV':total_matched_nooov,
                       'MATCHED_TOTAL_PCT':round(total_matched/total_dwarf,6), ## all matches including OOV matches
                       'MATCHED_NO_OOV_TOTAL_PCT':round(total_matched_nooov/total_dwarf,6), ## all matches excluding OOV matches
                       'MATCHED_EXCLUDE_OOV_PCT' : round(total_matched_nooov/total_nooov,6), ## only No-oov matches out of total_nooov matches
                       }
    
    json.dump(joint_result_word, open(os.path.join(args.model_name,args.prefix+"_name_predictions.json"),'w'))
            
    return varlevel_results


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
        labels_type = item['labels_type']
        fid = item['fid']
        varmap_position = item['varmap_position']
        gold_texts = item['gold_texts']
        
        fid_list.append([fid, 
                         len([each for each in input_ids if each == tokenizer.mask_token_id]), 
                         len(input_ids),
                         varmap_position,
                         gold_texts,
                        ])
        assert len(input_ids) == len(labels)
        assert len(input_ids) == len(labels_type)
        
        if len(input_ids) > block_size-2:
            self.truncated[i]=1
        
        if len(input_ids) >= block_size-2:
            input_ids = input_ids[0:block_size-2]
            labels = labels[0:block_size-2]
            labels_type = labels_type[0:block_size-2]
        elif len(input_ids) < block_size-2:
            input_ids = input_ids+[tokenizer.pad_token_id]*(self.block_size-2-len(input_ids))
            labels = labels + [tokenizer.pad_token_id]*(self.block_size-2-len(labels))
            labels_type = labels_type + [tokenizer.pad_token_id]*(self.block_size-2-len(labels_type))
        
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
        labels = tokenizer.build_inputs_with_special_tokens(labels)
        labels_type = [-100] + labels_type + [-100]
        
        assert len(input_ids) == len(labels)
        assert len(input_ids) == len(labels_type)
        assert len(input_ids) == block_size
        try:
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            labels    = torch.tensor(labels, dtype=torch.long)
            labels_type    = torch.tensor(labels_type, dtype=torch.long)
            mask_idxs = (input_ids==tokenizer.mask_token_id).bool()
            labels[~mask_idxs]=-100
            labels = labels.reshape(labels.size()[0],1)
            labels_type = labels_type.reshape(labels_type.size()[0],1)
            labels = torch.cat((labels, labels_type), -1)
        except:
            l.error(f"Unexpected error at index {i}: {sys.exc_info()[0]}")
            raise
        
        return input_ids , labels
    
parser = argparse.ArgumentParser()
parser.add_argument(
        "--model_name",
        default = "model",
        type=str,
        help="The model checkpoint for weights initialization.",
    )
parser.add_argument(
        "--tokenizer_name",
        default = "tokenizer",
        type=str,
        help="The tokenizer",
    )
parser.add_argument(
        "--data_file",
        default="cmlm_input_ida_ft_test_latest.json",
        type=str,
        help="Input Data File to Score",
    )

parser.add_argument(
        "--prefix",
        default="test",
        type=str,
        help="prefix to separate the output files",
    )
parser.add_argument(
        "--batch_size",
        default=5,
        type=int,
        help="Eval Batch Size",
    )

parser.add_argument(
        "--pred_path",
        default="outputs",
        type=str,
        help="path where the predictions will be stored",
    )

parser.add_argument(
        "--out_vocab_map",
        default="out_vocab_file",
        type=str,
        help="path where the mapping of idx_to_word is present",
    )

parser.add_argument(
    "--block_size",
    default=1024,
    type=int,
    help="Optional input sequence length after tokenization."
    "The training dataset will be truncated in block of this size for training."
    "Default to the model max input length for single sentence inputs (take into account special tokens).",
)
args = parser.parse_args()


device = torch.device("cuda")
n_gpu = torch.cuda.device_count()
config_class, model_class, tokenizer_class = MODEL_CLASSES["roberta"]
config = config_class.from_pretrained(args.model_name)

if args.tokenizer_name:
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)
elif args.model_name:
    tokenizer = tokenizer_class.from_pretrained(args.model_name)

    
idx_to_word = json.load(open(args.out_vocab_map))
config.out_vocab_size = len(idx_to_word)+1
args.out_vocab_size = len(idx_to_word)+1

model = model_class.from_pretrained(
            args.model_name,
            from_tf=bool(".ckpt" in args.model_name),
            config=config,
        )

model.to(device)
tiny_dataset = CMLDataset(tokenizer,file_path=args.data_file,block_size=args.block_size)
eval_sampler = SequentialSampler(tiny_dataset)
eval_dataloader = DataLoader(tiny_dataset, sampler=eval_sampler, batch_size=args.batch_size, shuffle = False)

model.eval()


eval_loss = 0.0
nb_eval_steps = 0

matched={1:0,3:0,5:0,10:0}
matched_not_va={1:0,3:0,5:0,10:0}
matched_dwarf={1:0,3:0,5:0,10:0}
matched_dwarf_nonsingle={1:0,3:0,5:0,10:0}
totaldecomp = {1:0,3:0,5:0,10:0}
totaldwarf={1:0,3:0,5:0,10:0}
totaldwarf_nonsingle={1:0,3:0,5:0,10:0}
matched_va={1:0,3:0,5:0,10:0}
matched_oov={1:0,3:0,5:0,10:0}
totalmasked={1:0,3:0,5:0,10:0}
total_oov={1:0,3:0,5:0,10:0}

pred_list={
    1 : [],
    3 : [],
    5 : [],
    10: []
}
gold_list=[]
gold_list_type=[]
pred_list_type=[]
result_metrics = {"VARNAME":{ "TOP1":0,
                              "TOP3":0,
                              "TOP5":0,
                              "TOP10":0,
                             "TOTAL_MASKED":{'1':0, '3':0, '5':0, '10':0},
                             "TOTAL_MATCHED":{'1':0, '3':0, '5':0, '10':0},
                             "TOTAL_DWARF":{'1':0, '3':0, '5':0, '10':0},
                             "TOTAL_OOV":{'1':0, '3':0, '5':0, '10':0},
                             "DWARF_MATCHED_PCT":{'1':0, '3':0, '5':0, '10':0},
                             "DWARF_MATCHED_OOV_PCT":{'1':0, '3':0, '5':0, '10':0},
                             "DWARF_NO_MATCHED_OOV_PCT":{'1':0, '3':0, '5':0, '10':0},
                             "TOTAL_MATCHED_OOV_PCT":{'1':0, '3':0, '5':0, '10':0},
                            },
                  "VARLEVEL_NAME":{},
                  "VARLEVEL_ORIGIN":{},
                  "VARORIGIN":{"ACCURACY":0,
                             "F1":0,
                             "P":0,
                             "R":0,
                             "classification report":[]},
                  "MISC":{"Truncated":0,
                         "perplexity":0,
                          "loss":0
                         }
                 }

for batch in tqdm(eval_dataloader, desc="Evaluating"):
    inputs, labels = batch[0], batch[1]
    only_masked = inputs==tokenizer.mask_token_id
    masked_gold = labels[only_masked]
    
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    masked_gold_name = masked_gold[:,0]
    masked_gold_type = masked_gold[:,1]
    gold_list.append(masked_gold_name.tolist())
    gold_list_type.append(masked_gold_type.to('cpu').tolist())

    with torch.no_grad():
        lm_loss,inference,inference_type = model(inputs, labels=labels)
        eval_loss += lm_loss.mean().item()
        
        # TopK Calculation
        masked_predict = inference.indices.cpu()[only_masked]
        masked_predict2 = inference_type.indices[only_masked].squeeze(1).to('cpu').numpy()

        ### ------ For varname -------
        for k in [1,3,5,10]:
            
            topked = masked_predict[:,0:k]
            pred_list[k].append(topked.tolist())
            for i,goldtok in enumerate(masked_gold_name):
                totalmasked[k]+=1
                if goldtok.item() == -100: # For IDA/GHIDRA variable's gold labels have been marked with -100 so that model does not learn
                    totaldecomp[k] +=1 
                    continue
                totaldwarf[k] += 1
                if goldtok.item() == args.out_vocab_size -1: total_oov[k]+=1 #only for dwarf variables
                        
                # calculate prediction scores
                each_prediction = topked[i]
                if goldtok in each_prediction:
                    matched[k]+=1
                    matched_dwarf[k]+=1
                    if goldtok.item() == args.out_vocab_size -1: matched_oov[k]+=1
        
        ### ------ For VARORIGIN -------
        pred_list_type.append(masked_predict2.tolist())
        
    nb_eval_steps += 1

eval_loss = eval_loss / nb_eval_steps
perplexity = torch.exp(torch.tensor(eval_loss))
result_metrics['MISC']["perplexity"] = perplexity.item()
result_metrics['MISC']["loss"] = round(eval_loss,2)


# print('\n\n ------------ METRICS FOR VARNAME ------------')
for i in [1,3,5,10]:
    # print("TopK:",i,matched[i]/totalmasked[i])
    result_metrics['VARNAME']["TOTAL_MASKED"][str(i)] = totalmasked[i]
    result_metrics['VARNAME']['TOTAL_DWARF'][str(i)] = totaldwarf[i]
    result_metrics['VARNAME']['TOTAL_OOV'][str(i)] = total_oov[i]
    result_metrics['VARNAME']["TOTAL_MATCHED"][str(i)] = matched[i]
    result_metrics['VARNAME']["TOP"+str(i)] = round(matched[i]/totalmasked[i],6) #not considering the matches of non-dwarf variables but dino has both
    result_metrics['VARNAME']["DWARF_MATCHED_PCT"][str(i)] = round(matched_dwarf[i]/totaldwarf[i],6)
    result_metrics['VARNAME']["DWARF_MATCHED_OOV_PCT"][str(i)] = round(matched_oov[i]/totaldwarf[i],6)
    result_metrics['VARNAME']["DWARF_NO_MATCHED_OOV_PCT"][str(i)] = round((matched_dwarf[i] - matched_oov[i])/totaldwarf[i],6)
    result_metrics['VARNAME']["TOTAL_MATCHED_OOV_PCT"][str(i)] = round(matched_oov[i]/totalmasked[i],6)
result_metrics['MISC']['Truncated'] = len(tiny_dataset.truncated)


# print('\n\n ------------ METRICS FOR VARORIGIN ------------')
gold_list_type_values = [e for each in gold_list_type for e in each]
pred_list_type_values = [e for each in pred_list_type for e in each]

# print("Number of non 01 values in gold_list_type_values:",sum([1 for each in gold_list_type_values if each != 1 and each !=0]))
# print("Number of non 01 values in pred_list_type_values:",sum([1 for each in pred_list_type_values if  each != 1 and each !=0]))

print("VARNAME STATS")
# print("totalmasked:",totalmasked)
# print("matched:",matched)
# print("totaldwarf:",totaldwarf)
# print("totaldwarf_nonsingle:",totaldwarf_nonsingle)
# print("matched_oov:",matched_oov)
# print("total_oov:",total_oov[1])
# print("dwarf:",matched_dwarf)
# print("dwarf_nonsingle:",matched_dwarf_nonsingle)
print("DWARF VAR % in TOTAL:", round(totaldwarf[1]*100/totalmasked[1],2))
print("DECOMPILER VAR % in TOTAL:", round(totaldecomp[1]*100/totalmasked[1],2))


print('\nclassification report')
# print(gold_list_type)
# print("PRED:",pred_list_type)
print(classification_report(gold_list_type_values, pred_list_type_values))

result_metrics['VARORIGIN']['ACCURACY'] = round(accuracy_score(gold_list_type_values,pred_list_type_values),4)
result_metrics['VARORIGIN']['F1'] = round(f1_score(gold_list_type_values, pred_list_type_values, average='macro'),4)
result_metrics['VARORIGIN']['P'] = round(precision_score(gold_list_type_values, pred_list_type_values, average='macro'),4)
result_metrics['VARORIGIN']['R'] = round(recall_score(gold_list_type_values, pred_list_type_values, average='macro'),4)

print("Model saved at:", args.model_name)
l.debug("Model saved at:", args.model_name)

flat_dg, flat_dp, flat_g, flat_p = [],[],[], []

for i in range(len(gold_list_type)):
    flat_dg += gold_list_type[i]
    flat_g += gold_list[i]
    flat_dp += pred_list_type[i]
    for e in pred_list[1][i]:
        flat_p.append(e[0])
# print("All Variables :", len(flat_dg), len(flat_dp), len(flat_g), len(flat_p))
    

start_idx = 0
out = []
out_name = defaultdict(list)
out_type = defaultdict(list)
# total_examples = 0
for each in tqdm(fid_list):
    _id = each[0]
    num_vars = each[1]
    length = each[2]
    varmap_position = each[3]
    g = [idx_to_word[str(e)] if str(e) in idx_to_word else e for e in flat_g[start_idx:start_idx+num_vars]]
    p = [idx_to_word[str(e)] if str(e) in idx_to_word else e for e in flat_p[start_idx:start_idx+num_vars]]
    go = ['dwarf' if e==0 else 'decompiler' for e in flat_dg[start_idx:start_idx+num_vars]]
    po = ['dwarf' if e==0 else 'decompiler' for e in flat_dp[start_idx:start_idx+num_vars]]
    varmap_position_all = each[4]
    out_type['fid'].append(_id)
    out_type['num_vars'].append(num_vars)
    out_type['length'].append(length)
    out_type['gold_name'].append(g)
    out_type['pred_name'].append(p)
    out_type['gold_type'].append(go)
    out_type['pred_type'].append(po)
    
    out.append([num_vars,
                _id, 
                varmap_position,
                flat_g[start_idx:start_idx+num_vars], 
                flat_p[start_idx:start_idx+num_vars],
               go,
               po,
               varmap_position_all,
               ])
    start_idx += num_vars
    
    
### MERGE THE splits (longer functions broken down into multiple samples)
final_output = {}
final_list = []
for each in tqdm(out):
    fid = each[1].split("_")[0]
    num_vars = each[0]
    varpos = each[2]
    g = each[3]
    p = each[4]
    go = each[5]
    po = each[6]
    varposall = each[7]
    if fid not in final_output: # first time the function is added to final output -> add all
        final_output[fid] = {'fid':fid, "num_vars":num_vars, "varpos":varpos, "gold":g, "pred":p,"gold_origin":go, "pred_origin":po, "varposall":varposall}
    else:
        final_output[fid]["num_vars"]+=num_vars
        final_output[fid]['gold'] += g
        final_output[fid]['pred'] += p
        final_output[fid]['gold_origin'] += go
        final_output[fid]['pred_origin'] += po

#convert to list
for each in final_output:
    final_list.append(final_output[each])
# print("final_list:",len(final_list))   
    
    
# get the variable level metrics
varlevel_results = var_match(final_list, idx_to_word)
varlevel_origin_results = var_origin_match(final_list, idx_to_word)

    
for each in varlevel_results:
    result_metrics['VARLEVEL_NAME'][each]=varlevel_results[each]
for each in varlevel_origin_results:
    result_metrics['VARLEVEL_ORIGIN'][each]=varlevel_origin_results[each]

print(" \n\n ------------ Prediction Results ------------")
print(f"# Variable Name: ")
for k, v in result_metrics['VARNAME']['DWARF_NO_MATCHED_OOV_PCT'].items():
    print(f"\t\tTop {k}: {v}")
print(f"# Variable Origin: ")
for k, v in result_metrics['VARORIGIN'].items():
    print(f"\t\t{k}: {v}")

# Saving the results in model folder
with open(os.path.join(args.model_name, args.prefix+"_results.json"),'w') as f:
    json.dump(result_metrics, f)
    
df = pd.DataFrame(out_type)
# df.to_csv(os.path.join(args.pred_path, args.prefix+"_instance_lvl.csv"), index=False)