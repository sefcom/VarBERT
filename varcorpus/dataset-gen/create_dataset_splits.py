import os
import subprocess
import json
import sys
from tqdm import tqdm
import concurrent.futures
from multiprocessing import Manager
from itertools import islice
from functools import partial
import time
import re
import hashlib
import numpy as np
import itertools
import random
import logging
from collections import OrderedDict, Counter
from preprocess_vars import normalize_variables

l = logging.getLogger('main')

manager = Manager()
md5_to_funcline = manager.dict()
funcline_to_data = manager.dict()
cfuncs_md5 = manager.dict()
cppfuncs_md5 = manager.dict()
dwarf_vars = manager.list()
decompiler_vars = manager.list()
new_funcs = manager.list()
    
def read_json(filename):
    with open(filename, 'r') as r:
        data = json.loads(r.read())
    if data:
        return data

def write_json(filename, data):    
    with open(filename, 'w') as w:
        w.write(json.dumps(data))

def write_list(filename, data, distribution=False, save_list=True):    
    if distribution:
        c = Counter(data)
        sorted_dict  = dict(sorted(c.items(), key=lambda item: item[1], reverse = True))
        write_json(filename + '_distribution.json', sorted_dict)

    if save_list: 
        with open(filename, 'w') as w:
            w.write('\n'.join(data))


def write_jsonlines(path, data):
    with open(path, 'a+') as w:
        w.write(json.dumps(data) + "\n")
    return True

# collect hash(func_body)
def hash_funcs(workdir, decomp, file_):
    global md5_to_funcline
    global funcline_to_data
    global cfuncs_md5
    global cppfuncs_md5
 
    var_regex = r"@@(var_\d+)@@(\w+)@@"
    json_data = read_json(os.path.join(workdir, "map", decomp, file_))
    if not json_data:
        return

    id_ = 0
    for name, data in islice(json_data.items(), 0, len(json_data)):
        func = data['func']     
        up_func = re.sub(var_regex, "\\2", func)
        # dedup func body
        func_body = '{'.join(up_func.split('{')[1:])   
        # for Ghidra
        # if 'anon_var' in func_body or 'anon_func' in func_body:
        #     continue
        func_body = func_body.replace('\t', '').replace('\n', '').replace(' ','')
        md5 = hashlib.md5(func_body.encode('utf-8')).hexdigest()
        id_ += 1
        data['id'] = id_
        data['func_name'] = name
        data['md5'] = md5
        md5_to_funcline[md5] = name
        funcline_to_data[name] = data
        if data['language'] == 'c':
            cfuncs_md5[md5] = name
        else:
            cppfuncs_md5[md5] = name

# de-dup functions before you split!
def func_dedup( workdir, decomp, start: int, batch_size: int, input):    
    end = start + batch_size
    for hash_, funcline  in islice(md5_to_funcline.items(), start, end):
        try:       
            binary_name = funcline.split('_')[0]
            ret = write_jsonlines(os.path.join(workdir, "dedup_func", decomp, f'{binary_name}.jsonl'), funcline_to_data[funcline])
            if not ret:
                l.error(f"error in dumping {hash_} | {funcline}")
        except Exception as e:
            l.error(f"Error in deduplicating functions{e}")

def find_c_and_cpp_files(dedup_dir, decomp):

    files = os.listdir(os.path.join(dedup_dir, decomp))
    random.seed(5)
    random.shuffle(files)

    c_name_and_lines = OrderedDict()
    cpp_name_and_lines = OrderedDict()
    # find c and cpp files
    c_funcs_per_binary, cpp_funcs_per_binary = [], []
    for f in files:
        with open(os.path.join(dedup_dir, decomp, f), 'r') as r:
            lines = r.readlines()
            if lines:
                if json.loads(lines[0])['language'] == 'c':
                    c_funcs_per_binary.append(len(lines))
                    c_name_and_lines[f] = len(lines)
                else:
                    cpp_funcs_per_binary.append(len(lines))
                    cpp_name_and_lines[f] = len(lines)

    l.info(f"c files: {len(c_funcs_per_binary)} \tcpp files: {len(cpp_funcs_per_binary)}")
    return c_funcs_per_binary, cpp_funcs_per_binary, c_name_and_lines, cpp_name_and_lines

# decide splits
def decide_per_binary_split(dedup_dir, c_funcs_per_binary, cpp_funcs_per_binary, c_name_and_lines, cpp_name_and_lines, workdir, decomp):
    l.debug("Deciding  binary splits...")
    def split_files(list_, name_and_lines):
        list_ = list(name_and_lines.items())
        random.shuffle(list_)
        train_, test_ = 0, 0
        # sum_ = sum(list_)  
        sum_ = sum([lines for _, lines in list_]) 
        # if number of total functions are fewer, update the limits
        t1 = round(sum_ * 0.709, 2)
        t2 = round(sum_ * 0.805, 2)
   
        l.debug(f"total functions!: {sum_} \tlower limit: {t1} \tupper limit: {t2}")
        got = []
        found = False
        for elem in range(0, len(list_)):
            # train_ += list_[elem]
            _, lines = list_[elem]
            train_ += lines
            got.append(list_[elem])
            if t1 <= train_ <= t2:
                l.debug(f"functions in train split!: {train_}")
                l.debug(f"found the split! pick files until: {elem}")
                break
        l.debug(f"functions in test split: {sum_ - train_}")
        # train_files = dict(islice(name_and_lines.items(), 0, elem+1))
        # test_files = dict(islice(name_and_lines.items(), elem+1, len(list_)))
        train_files = dict(list_[:elem+1])
        test_files = dict(list_[elem+1:])
        return train_files, test_files
    
    if len(c_funcs_per_binary):
        c_train_files, c_test_files = split_files(c_funcs_per_binary, c_name_and_lines)
        while True:            
            if len(c_test_files) == 0:
                l.debug("retrying for c...")
                c_train_files, c_test_files = split_files(c_funcs_per_binary, c_name_and_lines)
            else:
                break
    if len(cpp_funcs_per_binary):
        cpp_train_files, cpp_test_files = split_files(cpp_funcs_per_binary, cpp_name_and_lines)
        while True:
            if len(cpp_test_files) == 0:
                l.debug("retrying for cpp...")
                cpp_train_files, cpp_test_files = split_files(cpp_funcs_per_binary, cpp_name_and_lines)
            else:
                break
    
    if len(cpp_funcs_per_binary) and len(c_funcs_per_binary):
        train_files = {**c_train_files, **cpp_train_files}
        test_files = {**c_test_files, **cpp_test_files}
    elif len(cpp_funcs_per_binary):
        train_files = cpp_train_files
        test_files = cpp_test_files
    elif len(c_funcs_per_binary):
        train_files = c_train_files
        test_files = c_test_files

    with open(os.path.join(workdir, "splits", decomp, "train_files.json"), 'w') as w:
        w.write(json.dumps(train_files))
    
    with open(os.path.join(workdir, "splits", decomp, "test_files.json"), 'w') as w:
        w.write(json.dumps(test_files))
    return list(train_files.keys()), list(test_files.keys())

# decide splits
def decide_per_func_split(workdir, decomp):
    l.debug("Deciding  function splits...")
    def split_funcs(all_funcs):

        shuffled_funcnames = list(all_funcs.values())
        random.seed(5)
        random.shuffle(shuffled_funcnames)
  
        total = len(all_funcs)
        train_ =  int(total * 0.80)
        train_funcs = shuffled_funcnames[:train_]
        test_funcs = shuffled_funcnames[train_:]
        return train_funcs, test_funcs
    
    l.info(f"Total C funcs {len(cfuncs_md5)}")
    c_train_funcs, c_test_funcs = split_funcs(cfuncs_md5)
    l.info(f"Total CPP funcs {len(cppfuncs_md5)}")
    cpp_train_funcs, cpp_test_funcs = split_funcs(cppfuncs_md5)

    # total funcs now
    c_train_funcs.extend(cpp_train_funcs)
    c_test_funcs.extend(cpp_test_funcs)

    with open(os.path.join(workdir, "splits", decomp, "train_funcs.txt"), 'w') as w:
        w.write("\n".join(c_train_funcs))

    with open(os.path.join(workdir, "splits", decomp, "test_funcs.txt"), 'w') as w:
        w.write("\n".join(c_test_funcs))

    return c_train_funcs, c_test_funcs

# create binary splits
def create_per_binary_split(files, filename, workdir, decomp):

    with open(os.path.join(workdir, "splits", decomp,'per-binary', f'{filename}.jsonl'), 'w') as w:
        for file_ in files:            
            with open(os.path.join(workdir, "dedup_func", decomp, file_), 'r') as r:
                lines = r.readlines()
            for line in lines:
                w.write(json.dumps(json.loads(line)) + '\n')

# create func splits
def create_per_func_split(funcs, filename, workdir, decomp):

    with open(os.path.join(workdir, "splits", decomp,'per-func', f'{filename}.jsonl'), 'w') as w:        
        for func in funcs:
            if func not in funcline_to_data:
                print(func)
            w.write( json.dumps(funcline_to_data[func])+ "\n")

def get_variables(decomp, func_):
    
    global dwarf_vars
    global decompiler_vars
         
    var_regex = r"@@(var_\d+)@@(\w+)@@"
    data = json.loads(func_)
    vars_ = data['type_stripped_vars']
    for var, ty in vars_.items():
        if ty == "dwarf":
            dwarf_vars.append(var)            
        elif ty == decomp:            
            decompiler_vars.append(var)
        else:
            l.error(f"Error in getting variables:  {var} {ty}")
            
def substitute_norm(lookup, func):

    global new_funcs
    try:
        data = json.loads(func)
        func = data['func']
        tyvars_= data['type_stripped_vars']        
        # if the variable is removed in cleaning invalid vars
        if not tyvars_:
            return        
        vars_ = []
        for v in tyvars_:
            if tyvars_[v] == 'dwarf':
                vars_.append(v)
        # vars_map can be empty if there are no dwarf variable names in the func
        vars_map = []
        norm_func = func        
        for var in vars_:
            og_var = rf"(@@var_\d+@@)({var})(@@)"
            if var in lookup:
                if lookup[var] == '':
                    continue
                else:
                    norm_func = re.sub(og_var, rf'\1{lookup[var]}\3', norm_func)
                vars_map.append([var, lookup[var]])
            else:
                l.error(f"something fishy! {var} {data['id']}")

        up_data = data
        up_data['norm_func'] = norm_func
        up_data['vars_map'] = vars_map
        up_data['fid'] = str(data['id']) + "-" + data['func_name'].replace("_","-")

        # for preprocessing for vocab (vars_map and type_stripped_norm_vars same thing)
        norm_var_type = {}
        for pair in vars_map:
            norm_var = pair[1]
            var = pair[0]
            if norm_var in norm_var_type and up_data["type_stripped_vars"][var] != 'dwarf':
                norm_var_type[norm_var] = 'dwarf'
            else:
                norm_var_type[norm_var] = up_data["type_stripped_vars"][var]
        
        up_data['type_stripped_norm_vars'] = norm_var_type
        new_funcs.append(up_data)
    except Exception as e:
        l.error(f'Error in updating function in substitute_norm: {e}')

def write_norm(filename, funcs):
    with open(filename, 'w') as w:
        for func in funcs:
            w.write(json.dumps(func) + "\n")

def create_train_and_test_sets(workdir, decomp, WORKERS=4):

    total_steps = 4
    progress = tqdm(total=total_steps, desc="Initializing")

    # create dirs
    dirs = [f"dedup_func/{decomp}", "lookup",  f"splits/{decomp}/per-func",  f"splits/{decomp}/per-binary",  f"vars/{decomp}/per-func", f"vars/{decomp}/per-binary" ]
    for d in dirs:
        os.makedirs(os.path.join(workdir, d), exist_ok=True)
    files = os.listdir(os.path.join(workdir, 'map', decomp))
    
    progress.set_description("Deduplicating functions")
    # create hash of funcs for de-dup
    with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as executor:
        # executor.map(hash_funcs, files)
        partial_func = partial(hash_funcs, workdir, decomp)
        executor.map(partial_func, files)
    
    l.info(f"functions before de-dup: {len(funcline_to_data)} \tfunctions after de-dup: {len(md5_to_funcline)}")
    
    batch = len(md5_to_funcline) // (WORKERS - 1)
    if os.listdir(os.path.join(workdir, "dedup_func", decomp)):
        print("files in dedup dir!, remove them")
        l.error("files in dedup dir!, remove them")
        exit(1)

    l.debug("now saving dedup code!")
    with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as executor:
        cal_partial = partial(func_dedup,  workdir, decomp, batch_size=batch, input=input,)
        executor.map(cal_partial, [batch * i for i in range(WORKERS)])
    
    l.info(f'Binaries in dedup func: {len(os.listdir(os.path.join(workdir, "dedup_func", decomp)))}')
    progress.update(1)
    # find c and cpp files
    c_funcs_per_binary, cpp_funcs_per_binary, c_name_and_lines, cpp_name_and_lines = find_c_and_cpp_files(os.path.join(workdir, "dedup_func"), decomp)
    
    # # per-binary splits
    progress.set_description("Creating Binary Split")
    train_b, test_b = decide_per_binary_split(os.path.join(workdir, "dedup_func"), c_funcs_per_binary, cpp_funcs_per_binary, c_name_and_lines, cpp_name_and_lines, workdir, decomp)   
    l.debug(f"per binary numbers: Train: {len(train_b)} \tTest: {len(test_b)}")
    create_per_binary_split(train_b, "train", workdir, decomp)
    create_per_binary_split(test_b, "test", workdir, decomp)
    train_b.extend(test_b)
    all_b = train_b
    l.debug(f"total samples in binary split: {len(all_b)}")
    progress.update(1)
    
    # per-func splits    
    progress.set_description("Creating Function Split")
    train_fn, test_fn = decide_per_func_split(workdir, decomp)
    l.debug(f"per func numbers: Train: {len(train_fn)} \tTest: {len(test_fn)}")
    create_per_func_split(train_fn, "train", workdir, decomp)
    create_per_func_split(test_fn, "test", workdir, decomp)    
    train_fn.extend(test_fn)
    all_fn = train_fn
    l.debug(f"total samples in func split: {len(all_fn)}")
    progress.update(1)

    progress.set_description("Updating variables and saving splits")
    clean_up_variables(WORKERS, workdir, decomp)
    progress.update(1)
    progress.set_description("Train and Test splits created!")
    progress.close()
    
# TODO: Fix later
def clean_up_variables(WORKERS, workdir, decomp):
    # replace variables
    import glob
    files = glob.glob(f'{workdir}/splits/{decomp}/**/*', recursive=True)
    for f in files:
        new_funcs[:] = []
        if os.path.isfile(f) and f.endswith('jsonl') and 'all' not in f and 'final' not in f:
            tmp = f.split('/')
            ty = tmp[-2]
            file_ = tmp[-1]
            with open(f, 'r') as r:
                funclines = r.readlines()
            with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as executor: 
                partial_func = partial(get_variables, decomp)
                executor.map(partial_func, funclines)
            
            # create variable lists
            var_out_file = os.path.join(workdir, "vars", decomp,  f"{ty}", f"{file_[:-6]}_dwarf_vars")
            write_list(var_out_file, list(dwarf_vars), True, False)

            write_list(os.path.join(workdir, "vars", decomp,  f"{ty}", f"{file_[:-6]}_decomp_vars"), list(decompiler_vars), True, False) 
            
            clean_var_out = os.path.join(workdir, "vars", decomp,  f"{ty}")
            # subprocess.call(['python3.8', 'preprocess_vars.py', f'{var_out_file}_distribution.json', file_[:-6], clean_var_out, os.path.join(workdir, 'lookup')])
            normalize_variables(f'{var_out_file}_distribution.json', file_[:-6], clean_var_out, os.path.join(workdir, 'lookup'), WORKERS)
            import time; time.sleep(10)
            with open(os.path.join(workdir, 'lookup', 'universal_lookup.json'), 'r') as r:
                lookup = json.loads(r.read())

            with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as executor:
                partial_func = partial(substitute_norm, lookup)
                executor.map(partial_func, funclines)
            # how so many funcs?? # FIXME
            write_norm(os.path.join(workdir, "splits", decomp, ty, f"final_{file_[:-6]}.jsonl"), new_funcs)
            l.info(f"Functions in file {f}:  {str(len(new_funcs))}")
# create_train_and_test_sets('/tmp/varbert_tmpdir', 'ghidra')
