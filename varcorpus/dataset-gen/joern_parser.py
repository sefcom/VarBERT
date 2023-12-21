import re
import os
import json
import time
import logging
import subprocess
from collections import defaultdict, OrderedDict
from cpgqls_client import CPGQLSClient, import_code_query

l = logging.getLogger('main')

class JoernParser:
    def __init__(self, binary_name, binary_type, decompiler,
                 decompiled_code, workdir,  port, outpath ):
        self.binary_name = binary_name
        self.binary_type = binary_type
        self.decompiler = decompiler
        self.dc_inpath = decompiled_code
        self.port = port
        self.client = CPGQLSClient(f"localhost:{self.port}")  
        self.joern_workdir = os.path.join(workdir, 'tmp_joern')
        self.joern_data = defaultdict(dict)
        self.joern_outpath = outpath
        self.parse_joern()
    

    def edit_dc(self):
        try:
            regex_ul = r'(\d)(uLL)'
            regex_l = r'(\d)(LL)'
            regex_u = r'(\d)(u)'
            
            with open(f'{self.dc_inpath}', 'r') as r:
                data = r.read()   

            tmp_1 = re.sub(regex_ul, r'\g<1>', data)
            tmp_2 = re.sub(regex_l, r'\g<1>', tmp_1)
            final = re.sub(regex_u, r'\g<1>', tmp_2)
            
            with open(os.path.join(self.joern_workdir, f'{self.binary_name}.c'), 'w') as w:
                w.write(final)
        
        except Exception as e:
            l.error(f"Error in joern :: {self.decompiler} {self.binary_type} {self.binary_name} :: {e} ")


    def clean_up(self):
        self.client.execute(f'close("{self.binary_name}")')
        self.client.execute(f'delete("{self.binary_name}")')
        

    def split(self):
        try:
            l.debug(f"joern parsing! {self.decompiler} {self.binary_type} {self.binary_name}") 
            regex = r"(\((\"([\s\S]*?)\"))((, )(\"([\s\S]*?)\")((, )(\d*)(, )(\d*)))((, )(\"([\s\S]*?)\")((, )(\d*)(, )(\d*)))"
            r = re.compile(regex)		

            self.client.execute(import_code_query(self.joern_workdir, f'{self.binary_name}'))
            fetch_q = f'show(cpg.identifier.l.map(x => (x.location.filename, x.method.name, x.method.lineNumber.get, x.method.lineNumberEnd.get, x.name, x.lineNumber.get, x.columnNumber.get)).sortBy(_._7).sortBy(_._6).sortBy(_._1))'

            result = self.client.execute(fetch_q)
            res_stdout = result['stdout']
            
            if '***temporary file:' in res_stdout:
                tmp_file = res_stdout.split(':')[-1].strip()[:-3]

                with open(tmp_file, 'r') as tf_read:
                    res_stdout = tf_read.read()
                subprocess.check_output(['rm', '{}'.format(tmp_file)])
                                
            matches = r.finditer(res_stdout,re.MULTILINE)		
            raw_data = defaultdict(dict)		
            track_func = set()
            
            random = defaultdict(list)
            random_tmp = []
            temp = ''
            test = set()
            main_dict = defaultdict(dict)
            tmp_file_name = ''
            for m in matches:
                
                file_path = m.group(3)
                func_name = m.group(7)
                func_start = m.group(10)
                func_end = m.group(12)
                var_name = m.group(16)
                var_line = m.group(19)
                var_col = m.group(21)
        
                if tmp_file_name != self.binary_name:
                    tmp_file_name = self.binary_name
                    random = defaultdict(list)
                    raw_data = defaultdict(dict)
                    random_tmp = []
                        
                pkg_func = func_name + '_' + func_start 
                if pkg_func != temp:			
                    track_func = set()
                    random = defaultdict(list)
                    random_tmp = []
                
                if var_name not in random_tmp:
                    random_tmp.append(var_name)

                temp = pkg_func
                track_func.add(pkg_func)
                test.add(pkg_func)
                random[var_name].append(var_line)
                
                raw_data[pkg_func].update({"func_start":func_start, "func_end" : func_end})
                raw_data[pkg_func].update({"variable": dict(random)})
                raw_data[pkg_func].update({'tmp':random_tmp})

            with open(self.joern_outpath, 'w') as w:
                w.write(json.dumps(raw_data))
            self.joern_data = raw_data

        except Exception as e:
            l.error(f"Error in joern :: {self.decompiler} {self.binary_type} {self.binary_name} :: {e} ")
    
    def parse_joern(self):
        time.sleep(2)
        self.edit_dc()
        raw_data = self.split()
        self.clean_up()
    
        

