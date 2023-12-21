# Generate vocab from preprocessed sets with fid

import argparse
import os
import json
import jsonlines as jsonl
import re
from collections import defaultdict
from tqdm import tqdm

def load_jsonl_files(file_path):
    data = []
    with jsonl.open(file_path) as ofd:
        for each in tqdm(ofd, desc=f"Loading data from {file_path}"):
            data.append(each)
    return data

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def calculate_distribution(data, dataset_type):
    var_distrib = defaultdict(int)
    for each in tqdm(data):
        func = each['norm_func']
        pattern = "@@\w+@@\w+@@"
        if dataset_type == 'varcorpus':
            dwarf_norm_type = each['type_stripped_norm_vars']
        
        for each_var in list(re.finditer(pattern,func)):
            s = each_var.start()
            e = each_var.end()
            var = func[s:e]
            orig_var = var.split("@@")[-2]
            
            # Collect variables only dwarf
            if dataset_type == 'varcorpus':
                if orig_var in dwarf_norm_type:
                    var_distrib[orig_var]+=1
            elif dataset_type == 'hsc':
                var_distrib[orig_var]+=1

    sorted_var_distrib = sorted(var_distrib.items(), key = lambda x : x[1], reverse=True)
    return sorted_var_distrib


def build_vocab(data, vocab_size, existing_vocab=None):
    if existing_vocab:
        vocab_list = list(existing_vocab)
    else:
        vocab_list = []
    for idx, each in tqdm(enumerate(data)):
        if len(vocab_list) == args.vocab_size:
            print("limit reached:", args.vocab_size, "Missed:",len(data)-idx-1)
            break
        if each[0] in vocab_list:
            continue
        else:
            vocab_list.append(each[0])
    
    idx2word, word2idx = {}, {}
    for i,each in enumerate(vocab_list):
        idx2word[i] = each
        word2idx[each] = i

    return idx2word, word2idx

def save_json(data, output_path, filename):
    with open(os.path.join(output_path, filename), 'w') as w:
        w.write(json.dumps(data))


def main(args):
    # Load existing human vocabulary if provided
    if args.existing_vocab:
        with open(args.existing_vocab, 'r') as f:
            human_vocab = json.load(f)
    else:
        human_vocab = None

    # Load train and test data
    train_data = load_jsonl_files(args.train_file)
    test_data = load_jsonl_files(args.test_file)

    # TODO add check to 
    var_distrib_train =  calculate_distribution(train_data, args.dataset_type)
    var_distrib_test = calculate_distribution(test_data, args.dataset_type)
    
    # save only if needed
    # save_json(var_distrib_train, args.output_dir, 'var_distrib_train.json')
    # save_json(var_distrib_test, args.output_dir, 'var_distrib_test.json')

    print("Train data distribution", len(var_distrib_train))
    print("Test data distribution", len(var_distrib_test))

    existing_vocab_data = {}
    if args.existing_vocab:
        print("Human vocab size", len(human_vocab))
        existing_vocab_data = read_json(args.existing_vocab)
    
    # Build and save the vocabulary
    idx2word, word2idx = build_vocab(var_distrib_train, args.vocab_size, existing_vocab=existing_vocab_data)
    print("Vocabulary size", len(idx2word))
    save_json(idx2word, args.output_dir, 'idx_to_word.json')
    save_json(word2idx, args.output_dir, 'word_to_idx.json')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Dataset Vocabulary Generator")
    parser.add_argument("--dataset_type", type=str, choices=['hsc', 'varcorpus'], required=True, help="Create vocab for HSC (source code) or VarCorpus (decompiled code)")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training data file")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test data file")
    parser.add_argument("--existing_vocab", type=str, help="Path to the existing human vocabulary file")
    parser.add_argument("--vocab_size", type=int, default=50000, help="Limit for the vocabulary size")
    parser.add_argument("--output_dir", type=str, required=True, help="Path where the output vocabulary will be saved")

    args = parser.parse_args()
    main(args)