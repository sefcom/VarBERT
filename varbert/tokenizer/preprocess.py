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

def update_data(data, pattern):
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

def main(args):   
    data = []
    with jsonlines.open(args.input_file) as f:
        for each in tqdm(f):
            data.append(each['norm_func'])

    new_data = update_data(data, "@@\w+@@\w+@@")

    with jsonlines.open(args.output_file, mode='w') as writer:
        for item in new_data:
            writer.write({'func': item})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process JSONL files.')
    parser.add_argument('--input_file', type=str, help='Path to the input HSC jsonl file')
    parser.add_argument('--output_file', type=str, help='Path to the save HSC jsonl file for tokenization')
    args = parser.parse_args()
    main(args)
