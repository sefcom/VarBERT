import argparse
import random
import numpy as np
import time
import os
import math
import json
import jsonlines
from tqdm import tqdm
from collections import defaultdict
import re
import pandas as pd

def update_function(data, pattern):
    new_data = []
    for body in tqdm(data):
        idx = 0
        new_body = ""
        for each_var in list(re.finditer(pattern, body)):
            s = each_var.start()
            e = each_var.end()
            prefix = body[idx:s]
            var = body[s:e]
            orig_var = var.split("@@")[-2]
            new_body += prefix + orig_var
            idx = e
        new_body += body[idx:]
        new_data.append(new_body)
    return new_data

def process_file(input_file, output_file, pattern):
    data = []
    with jsonlines.open(input_file) as f:
        for each in tqdm(f):
            data.append(each['norm_func'])

    new_data = update_function(data, pattern)

    mlm_data = []
    for idx, each in tqdm(enumerate(new_data)):
        if len(each) > 0 and not each.isspace():
            mlm_data.append({'text': each, 'source': 'human', '_id': idx})

    with jsonlines.open(output_file, mode='w') as f:
        for each in tqdm(mlm_data):
            f.write(each)

def main(args):
    process_file(args.train_file, args.output_train_file, "@@\w+@@\w+@@")
    process_file(args.test_file, args.output_test_file, "@@\w+@@\w+@@")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and save jsonl files for MLM')
    parser.add_argument('--train_file', type=str, required=True, help='Path to the input train JSONL file')
    parser.add_argument('--test_file', type=str, required=True, help='Path to the input test JSONL file')
    parser.add_argument('--output_train_file', type=str, required=True, help='Path to the output train JSONL file for MLM')
    parser.add_argument('--output_test_file', type=str, required=True, help='Path to the output test JSONL file for MLM')
    args = parser.parse_args()
    main(args)
