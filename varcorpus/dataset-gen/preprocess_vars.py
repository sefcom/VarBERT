import os
import re
import json
import logging
import concurrent.futures
from itertools import islice
from functools import partial
from multiprocessing import Manager

l = logging.getLogger('main')

manager = Manager()
vars_ =  manager.dict()
clean_vars = manager.dict()

def read_(filename):
    with open(filename, 'r') as r:
        data = json.loads(r.read())
    return data

def read_jsonl(filename):
    with open(filename, 'r') as r:
        data = r.readlines()
    return data

def count_(data):
    tot = 0
    for name, c in data.items():
        tot += int(c)
    return tot

def change_case(str_):

    # aP
    if len(str_) == 2:
        var = str_.lower()

    # _aP # __A 
    elif len(str_) == 3 and (str_[0] == '_' or str_[:2] == '__'):
        var = str_.lower()

    else:        
        s1 = re.sub('([A-Z]+)([A-Z][a-z]+)', r'\1_\2', str_)
        #s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', str_)
        var = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    return var


def clean(data, lookup, start,  batch_size: int):

    global vars_
    global clean_vars    
    end = start + batch_size

    # change case and lower
    for var in islice(data, start, end):
        if var in lookup:
            vars_[var] = lookup[var]
            up_var = lookup[var]        
        else:
            up_var = change_case(var)      
            # rm underscore
            up_var = up_var.strip('_')
            vars_[var] = up_var
        if up_var in clean_vars:
            clean_vars[up_var] += data[var]
        else:
            clean_vars[up_var] = data[var]

def normalize_variables(filename, ty, outdir, lookup_dir, WORKERS):
    existing_lookup = False
    if os.path.exists(os.path.join(lookup_dir, 'universal_lookup.json')):          
        with open(os.path.join(lookup_dir, 'universal_lookup.json'), 'r') as r:
            lookup = json.loads(r.read())
            existing_lookup = True
    else:
        lookup = {}
    l.debug(f"len of lookup: {len(lookup)}")
    data = read_(filename)
    l.debug(f"cleaning variables for: {filename} | count: {len(data)}")

    batch = len(data) // (WORKERS - 1)
    with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as executor:
        cal_partial = partial(clean,  data, lookup, batch_size=batch,)
        executor.map(cal_partial, [batch * i for i in range(WORKERS)])
    l.debug(f"norm vars for file {filename} before: {len(vars_)} after: {len(set(vars_.values()))}")
    sorted_clean_vars  = dict(sorted(clean_vars.items(), key=lambda item: item[1], reverse = True))

    with open(os.path.join(outdir, f"{ty}_clean.json"), 'w') as w:
        w.write(json.dumps(dict(sorted_clean_vars)))
             
    if existing_lookup:
        for v in vars_:
            if v not in lookup:
                lookup[v] = vars_[v]

        with open(os.path.join(lookup_dir, 'universal_lookup.json'), 'w') as w:
            w.write(json.dumps(lookup))
    else:
        with open(os.path.join(lookup_dir, 'universal_lookup.json'), 'w') as w:
            w.write(json.dumps(dict(vars_)))
    
    
