from tqdm import tqdm
from transformers import RobertaTokenizerFast
import jsonlines
import sys
import json
import traceback
import random
import os
import re
import argparse
from collections import defaultdict
from multiprocessing import Process, Manager, cpu_count, Pool
from itertools import repeat


def read_input_files(filename):
    
    samples, sample_ids = [], set()
    with jsonlines.open(filename,'r') as f:
        for each in tqdm(f):
            sample_ids.add(each['fid'])
            samples.append(each)
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
                output_file_ids.add(str(each['fid']).split("_")[0])
                f.write(each)

        # validate : # source ids == target_ids all ids are present after parallel processing
        print(len(input_file_ids), len(output_file_ids))
        print(f"src_tgt_intersection:", len(input_file_ids - output_file_ids), len(output_file_ids-input_file_ids))

        # validate : vocab_check
        vocab_check = defaultdict(int)
        total = 0
        for each in tqdm(output_data):
            variables = list(each['type_dict'].keys())
            for var in variables:
                total += 1
                _, vocab_stat = get_var_token(var, word_to_idx)
                if "in_vocab" in vocab_stat:
                    vocab_check['in_vocab']+=1
                if "not_in_vocab" in vocab_stat:
                    vocab_check['not_in_vocab']+=1
                if "part_in_vocab" in vocab_stat:
                    vocab_check['part_in_vocab']+=1

        print(vocab_check, round(vocab_check['in_vocab']*100/total,2), round(vocab_check['not_in_vocab']*100/total,2))

def get_var_token(norm_variable_word,word_to_idx):
    vocab_check = defaultdict(int)
    token  = word_to_idx.get(norm_variable_word,args.vocab_size)
    if token == args.vocab_size:
        vocab_check['not_in_vocab']+=1
    else:
        vocab_check['in_vocab']+=1
    return [token], vocab_check


def preprocess_word_mask(text,tokenizer, word_to_idx):
    type_dict = text['type_stripped_norm_vars']
    _id = text['_id'] if "_id" in text.keys() else text['fid']
    ftext = text['norm_func']
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
    labels_typ = []
    orig_vars = []
    varmap_position = defaultdict(list)
    mask_one_ida = False

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
                        labels_typ.append(-100)

                var_tokens, _ = get_var_token(orig_var,word_to_idx)
                var_toks.append(var_tokens)
                var_words.append(orig_var) # Gold_texts (gold labels)
                mod_words.append(orig_var)
                orig_vars.append(orig_var)

                # Second label generation variable Type (decomp vs dwarf)
                if orig_var not in type_dict or type_dict[orig_var] == args.decompiler: # if it is not present consider ida
                    labels_typ.append(1)
                    tpwords.append(vocab["<mask>"])
                    towords.append(-100)
                    varmap_position[-100].append(pos)
                elif type_dict[orig_var] == "dwarf":
                    labels_typ.append(0)
                    tpwords.append(vocab["<mask>"])
                    towords.append(var_tokens[0])
                    varmap_position[orig_var].append(pos)
                else:
                    print("ERROR: CHECK LABEL TYPE IN STRIPPED BIN DICTIONARY")
                    exit(0)
                pos += 1

                idx = e

            # Postfix if any
            postfix = word[idx:]
            if postfix:
                toks = tokenizer.tokenize(postfix)
                for t in toks:
                    mod_words.append(t)
                    tpwords.append(vocab[t])
                    towords.append(vocab[t])
                    labels_typ.append(-100)

        else:
            # When there are no variables
            toks = tokenizer.tokenize(word)
            for t in toks:
                if t == "<mask>":
                    continue # skip adding the <mask> token if code has <mask> keyword
                mod_words.append(t)
                tpwords.append(vocab[t])
                towords.append(vocab[t])
                labels_typ.append(-100)

    assert len(tpwords) == len(towords)
    assert len(labels_typ) == len(towords)
    assert len(var_toks) == len(var_words)

            # 0       1      2          3            4        5      6          7          8          9
    return tpwords,towords,var_words, var_toks, labels_typ, words, orig_vars, mod_words, type_dict, _id, varmap_position


def generate_id_files(data, output_data, tokenizer, word_to_idx, n):

    for d in tqdm(data):
        try:
            ppw = preprocess_word_mask(d,tokenizer,word_to_idx)
            outrow = {"words":ppw[5],"mod_words":ppw[7],"inputids":ppw[0],"labels":ppw[1],"gold_texts":ppw[2],"gold_texts_id":ppw[3],"labels_type":ppw[4],"meta":[],"orig_vars":ppw[6], "type_dict":ppw[8], "fid":ppw[9],'varmap_position':ppw[10]}
            # if input length is more than max possible 1024 then split and make more sample found by tracing _id
            if len(outrow['inputids']) > n:
                for i in range(0, len(outrow['inputids']), n):
                    sample = {"words": outrow['words'][i:i+n],
                                 "mod_words":outrow['mod_words'][i:i+n],
                                 "inputids":outrow['inputids'][i:i+n],
                                 "labels":outrow["labels"][i:i+n],
                                 "labels_type":outrow["labels_type"][i:i+n],
                                 "gold_texts":outrow["gold_texts"],
                                 "gold_texts_id":outrow["gold_texts_id"],
                                 "orig_vars":outrow["orig_vars"],
                                 "type_dict":outrow["type_dict"],
                                 "meta":outrow["meta"],
                                 "fid":outrow['fid']+"_"+str((i)//n),
                                 "varmap_position":outrow["varmap_position"],
                             }
                    output_data.append(sample)
            else:
                output_data.append(outrow)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            traceback.print_exception(*sys.exc_info())



def main(args):

    tokenizer = RobertaTokenizerFast.from_pretrained(args.tokenizer)
    word_to_idx = json.load(open(args.vocab_word_to_idx))
    idx_to_word = json.load(open(args.vocab_idx_to_word))
    max_sample_chunk = args.max_chunk_size-2
    args.vocab_size = len(word_to_idx) # OOV as UNK[assinged id=vocab_size]
    print(f"Vocab_size: {args.vocab_size}")

    train, test = [], []
    src_train_ids, srctest_ids = set(), set()
    tgt_train_ids, tgttest_ids = set(), set()
 
    src_train_ids, train = read_input_files(args.train_file)
    srctest_ids, test = read_input_files(args.test_file)
    print(f"Data size Train: {len(train)} \t Test: {len(test)}")

    num_processes = min(args.workers, cpu_count())
    print(f"Running with #workers : {num_processes}")

    prep_input_files(train, num_processes, tokenizer, word_to_idx, max_sample_chunk, src_train_ids, tgt_train_ids, args.out_train_file )
    prep_input_files(test, num_processes, tokenizer, word_to_idx, max_sample_chunk, srctest_ids, tgttest_ids, args.out_test_file)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, help='name of the train file')
    parser.add_argument('--test_file', type=str, help='name of name of the test file')
    parser.add_argument('--test_notintrain_file', type=str, help='name of the test not in train file')

    parser.add_argument('--tokenizer', type=str, help='path to the tokenizer')

    parser.add_argument('--vocab_word_to_idx', type=str, help='Output Vocab Word to index file')
    parser.add_argument('--vocab_idx_to_word', type=str, help='Output Vocab Index to Word file')

    parser.add_argument('--vocab_size', type=int, default=150001, help='size of output vocabulary')
    parser.add_argument('--var_loc_pattern', type=str, default="@@\w+@@\w+@@", help='pattern representing variable location')
    parser.add_argument('--decompiler', type=str, default="ida", help='decompiler for type prediction; ida or ghidra')
    parser.add_argument('--max_chunk_size', type=int, default=1024, help='size of maximum chunk of input for the model')
    parser.add_argument('--workers', type=int, default=30, help='number of parallel workers you need')

    parser.add_argument('--out_train_file', type=str, help='name of the output train file')
    parser.add_argument('--out_test_file', type=str, help='name of name of the output test file')
    # parser.add_argument('--out_test_notintrain_file', type=str, help='name of the output test not in train file')


    args = parser.parse_args()

    main(args)
