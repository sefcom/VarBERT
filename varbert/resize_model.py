import json
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
import argparse
import os
import shutil
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers.activations import ACT2FN, gelu

vocab_size = 50001
class RobertaLMHead2(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, vocab_size, bias=False)
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
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        type_label=None
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
        
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, vocab_size), labels.view(-1))
            
        masked_loss = masked_lm_loss 

        output = (prediction_scores,) + outputs[2:]
        return ((masked_loss,) + output) if masked_loss is not None else output


def get_resized_model(oldmodel, newvocab_len):    
    
    ## Change the bias dimensions 
    def _get_resized_bias(old_bias, new_size):
        old_num_tokens = old_bias.data.size()[0]
        if old_num_tokens == new_size:
            return old_bias

        # Create new biases
        new_bias = nn.Parameter(torch.zeros(new_size))
        new_bias.to(old_bias.device)

        # Copy from the previous weights
        num_tokens_to_copy = min(old_num_tokens, new_size)
        new_bias.data[:num_tokens_to_copy] = old_bias.data[:num_tokens_to_copy]
        return new_bias
        
    ## Change the decoder dimensions 
    
    cls_layer_oldmodel = oldmodel.lm_head2.decoder
    oldvocab_len, old_embedding_dim = cls_layer_oldmodel.weight.size()
    print(f"Old Vocab Size: {oldvocab_len} \t Old Embedding Dim: {old_embedding_dim}")
    if oldvocab_len == newvocab_len:
        return oldmodel

    # Create new weights
    cls_layer_newmodel = nn.Linear(in_features=old_embedding_dim, out_features=newvocab_len)
    cls_layer_newmodel.to(cls_layer_oldmodel.weight.device)

    # initialize all weights (in particular added tokens)
    oldmodel._init_weights(cls_layer_newmodel)

    # Copy from the previous weights
    num_tokens_to_copy = min(oldvocab_len, newvocab_len)
    cls_layer_newmodel.weight.data[:num_tokens_to_copy, :] = cls_layer_oldmodel.weight.data[:num_tokens_to_copy, :]
    oldmodel.lm_head2.decoder = cls_layer_newmodel    
    # Change the bias
    old_bias = oldmodel.lm_head2.bias
    oldmodel.lm_head2.bias = _get_resized_bias(old_bias, newvocab_len)
    
    return oldmodel



def main(args):
   
    model = RobertaForMaskedLMv2.from_pretrained(args.old_model)
    vocab = json.load(open(args.vocab_path))
    print(f"New Vocab Size : {len(vocab)}")
    newmodel = get_resized_model(model, len(vocab)+1)
    print(f"Final Cls Layer of New model : {newmodel.lm_head2.decoder}")
    newmodel.save_pretrained(args.out_model_path)
    
    # Move the other files
    files = ['tokenizer_config.json','vocab.json','training_args.bin','special_tokens_map.json','merges.txt']
    for file in files:
        shutil.copyfile(os.path.join(args.old_model,file), os.path.join(args.out_model_path,file))
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--old_model', type=str, help='name of the train file')
    parser.add_argument('--vocab_path', type=str, help='path to the out vocab, the size of output layer you need')
    parser.add_argument('--out_model_path', type=str, help='path to the new modified model')
    args = parser.parse_args()
    
    main(args)