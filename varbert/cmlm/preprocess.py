# from pymongo import MongoClient
from tqdm import tqdm
from transformers import RobertaTokenizerFast
import jsonlines
import sys
import json
import logging
import traceback
import random
import os
import re
import argparse
from multiprocessing import Process, Manager, cpu_count, Pool
from itertools import repeat
from collections import defaultdict

l = logging.getLogger('model_main')

def read_input_files(filename):
    
    samples, sample_ids = [], set()
    with jsonlines.open(filename,'r') as f:
        for each in tqdm(f):
            samples.append(each)
            sample_ids.add(str(each['mongo_id']))
    return sample_ids, samples 


def prep_input_files(input_file, num_processes, tokenizer, word_to_idx, max_sample_chunk, input_file_ids, output_file_ids, preprocessed_outfile):
    
    #------------------------ INPUT FILES ------------------------
    output_data = Manager().list()
    pool = Pool(processes=num_processes) # Instantiate the pool here
    each_alloc = len(input_file) // (num_processes-1)
    input_data = [input_file[i*each_alloc:(i+1)*each_alloc] for i in range(0,num_processes)]
    x = [len(each) for each in input_data]
    print(f"Allocation samples for each worker: {len(input_data)}, {x}")

    pool.starmap(generate_id_files,zip(input_data,
                                        repeat(output_data),
                                       repeat(tokenizer), 
                                       repeat(word_to_idx), 
                                       repeat(max_sample_chunk)
                                     ))
    pool.close()
    pool.join()

    # Write to Output file
    with jsonlines.open(preprocessed_outfile,'w') as f:
        for each in tqdm(output_data):
            f.write(each)
    
    # check : #source ids == target_ids after parallel processing
    print(f"src_tgt_intersection:", len(input_file_ids - output_file_ids), len(output_file_ids-input_file_ids))
 
    for each in tqdm(output_data):
        output_file_ids.add(str(each['_id']).split("_")[0])
    print("src_tgt_intersection:",len(input_file_ids - output_file_ids), len(output_file_ids-input_file_ids))
    print(len(output_data))
    
    # validate : vocab_check
    vocab_check = defaultdict(int)
    total = 0
    for each in tqdm(output_data):
        variables = each['orig_vars']
        for var in variables:
            total += 1
            normvar = normalize(var)
            _, vocab_stat = get_var_token(normvar,word_to_idx)
            if "in_vocab" in vocab_stat:
                vocab_check['in_vocab']+=1
            if "not_in_vocab" in vocab_stat:
                vocab_check['not_in_vocab']+=1
            if "part_in_vocab" in vocab_stat:
                vocab_check['part_in_vocab']+=1
               
    print(vocab_check, round(vocab_check['in_vocab']*100/total,2), round(vocab_check['not_in_vocab']*100/total,2), round(vocab_check['part_in_vocab']*100/total,2))
     

def generate_id_files(data, output_data,  tokenizer, word_to_idx, n):

    for d in tqdm(data):
        try:
            ppw = preprocess_word_mask(d,tokenizer,word_to_idx)
            outrow = {"words":ppw[4],"mod_words":ppw[6],"inputids":ppw[0],"labels":ppw[1],"gold_texts":ppw[2],"gold_texts_id":ppw[3],"meta":[],"orig_vars":ppw[5], "_id":ppw[7]}
            # if input length is more than max possible 1024 then split and make more sample found by tracing _id
            if len(outrow['inputids']) > n:
                for i in range(0, len(outrow['inputids']), n):
                    sample = {"words": outrow['words'][i:i+n],
                                 "mod_words":outrow['mod_words'][i:i+n],
                                 "inputids":outrow['inputids'][i:i+n],
                                 "labels":outrow["labels"][i:i+n],
                                 "gold_texts":outrow["gold_texts"],
                                 "gold_texts_id":outrow["gold_texts_id"],
                                 "orig_vars":outrow["orig_vars"],
                                 "meta":outrow["meta"],
                                 "_id":str(outrow['_id'])+"_"+str((i)//n),}
                    output_data.append(sample)
            else:
                output_data.append(outrow)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            traceback.print_exception(*sys.exc_info())

def change_case(strt): 
    return ''.join(['_'+i.lower() if i.isupper() else i for i in strt]).lstrip('_')
def is_camel_case(s):
    return s != s.lower() and s != s.upper() and "_" not in s

def normalize(k):
    if is_camel_case(k): k=change_case(k)
    else: k=k.lower()
    return k

def get_var_token(norm_variable_word,word_to_idx):
    vocab_check = defaultdict(int)
    token  = word_to_idx.get(norm_variable_word,args.vocab_size)
    if token == args.vocab_size:
        vocab_check['not_in_vocab']+=1
        if "_" in norm_variable_word:
            word_splits=norm_variable_word.split("_")
            word_splits = [ee for ee in word_splits if ee ]
            for x in word_splits:
                ptoken=word_to_idx.get(x,args.vocab_size)
                if ptoken!=args.vocab_size:
                    token=ptoken
                    vocab_check['part_in_vocab']+=1
                    break
    else:
        vocab_check['in_vocab']+=1
    return [token], vocab_check


def canonicalize_code(code):
    code = re.sub('//.*?\\n|/\\*.*?\\*/', '\\n', code, flags=re.S)
    lines = [l.rstrip() for l in code.split('\\n')]
    code = '\\n'.join(lines)
#     code = re.sub('@@\\w+@@(\\w+)@@\\w+', '\\g<1>', code)
    return code

def preprocess_word_mask(text,tokenizer, word_to_idx):
    # vars_map = text['vars_map'] #needed for vartype detection
    # vars_map = dict([[ee[1],ee[0]] for ee in vars_map])
    _id = text['mongo_id']
    ftext = canonicalize_code(text['norm_func'])
    words = ftext.replace("\n"," ").split(" ")
    pwords =[]
    tpwords =[]
    owords =[]
    towords =[]
    pos=0
    masked_pos=[]
    var_words =[]
    var_toks = []
    mod_words = []
    orig_vars = []

    vocab=tokenizer.get_vocab()

    for word in words:
        if re.search(args.var_loc_pattern, word):
            idx = 0
            for each_var in list(re.finditer(args.var_loc_pattern,word)):
                s = each_var.start()
                e = each_var.end()
                prefix = word[idx:s]
                var = word[s:e]
                orig_var = var.split("@@")[-2]

                # Somethings attached before the variables
                if prefix:
                    toks = tokenizer.tokenize(prefix)
                    for t in toks:
                        mod_words.append(t)
                        tpwords.append(vocab[t])
                        towords.append(vocab[t])

                # Original variable handling 
                    # ---- IF dwarf : inputs:<mask> labels:tokenized,normalized var 
                    # ---- NOT dwarf: inputs: tokenized,orig_vars labels:tokenized,orig_vars

                norm_variable_word = normalize(orig_var)
                var_tokens, _ = get_var_token(norm_variable_word,word_to_idx)
                var_toks.append(var_tokens)
                var_words.append(norm_variable_word) #Gold_texts (gold labels)
                mod_words.append(orig_var)
                orig_vars.append(orig_var)

                tpwords.append(vocab["<mask>"])
                towords.append(var_tokens[0])

                idx = e

            # Postfix if any
            postfix = word[idx:]
            if postfix:
                toks = tokenizer.tokenize(postfix)
                for t in toks:
                    mod_words.append(t)
                    tpwords.append(vocab[t])
                    towords.append(vocab[t])

        else:
            # When there is no variables
            toks = tokenizer.tokenize(word)
            for t in toks:
                if t == "<mask>":
                    continue #skip adding the <mask> token if code has <mask> keyword
                mod_words.append(t)
                tpwords.append(vocab[t])
                towords.append(vocab[t])

    assert len(tpwords) == len(towords)
    assert len(var_toks) == len(var_words)

         # 0       1      2          3            4        5      6          7  
    return tpwords,towords,var_words, var_toks, words, orig_vars, mod_words, _id


def main(args):
 
    tokenizer = RobertaTokenizerFast.from_pretrained(args.tokenizer)
    word_to_idx = json.load(open(args.vocab_word_to_idx))
    idx_to_word = json.load(open(args.vocab_idx_to_word))
    max_sample_chunk = args.max_chunk_size-2
    args.vocab_size = len(word_to_idx) #OOV as UNK[assinged id=vocab_size]
    print("Vocab_size:",args.vocab_size)

    train, test = [], []
    src_train_ids, srctest_ids = set(), set()
    tgt_train_ids, tgttest_ids = set(), set()
 
    src_train_ids, train = read_input_files(args.train_file)
    srctest_ids, test = read_input_files(args.test_file)
    print(f"Data size Train: {len(train)} \t Test: {len(test)}")
    
    num_processes = min(args.workers, cpu_count())
    print("Running with #workers : ",num_processes)
    
    prep_input_files(train, num_processes, tokenizer, word_to_idx, max_sample_chunk, src_train_ids, tgt_train_ids, args.out_train_file )
    prep_input_files(test, num_processes, tokenizer, word_to_idx, max_sample_chunk, srctest_ids, tgttest_ids, args.out_test_file)

        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, help='name of the train file')
    parser.add_argument('--test_file', type=str, help='name of name of the test file')
    
    parser.add_argument('--tokenizer', type=str, help='path to the tokenizer')
    
    parser.add_argument('--vocab_word_to_idx', type=str, help='Output Vocab Word to index file')
    parser.add_argument('--vocab_idx_to_word', type=str, help='Output Vocab Index to Word file')
    
    parser.add_argument('--vocab_size', type=int, default=50001, help='size of output vocabulary')
    parser.add_argument('--var_loc_pattern', type=str, default="@@\w+@@\w+@@", help='pattern representing variable location')
    parser.add_argument('--max_chunk_size', type=int, default=1024, help='size of maximum chunk of input for the model')
    parser.add_argument('--workers', type=int, default=20, help='number of parallel workers you need')
    
    parser.add_argument('--out_train_file', type=str, help='name of the output train file')
    parser.add_argument('--out_test_file', type=str, help='name of name of the output test file')
    args = parser.parse_args()
    
    main(args)