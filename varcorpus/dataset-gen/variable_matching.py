from binary import Binary
from utils import read_text, read_json, write_json
import os
import string
import re
import json
from collections import defaultdict, Counter, OrderedDict
import sys
import hashlib
import logging
from parse_decompiled_code import IDAParser, GhidraParser
from typing import Any, List, Dict

l = logging.getLogger('main')

class DataLoader:
    def __init__(self, binary_name, dwarf_info_path, decompiler, decompiled_code_strip_path, 
     decompiled_code_type_strip_path, joern_strip_data, joern_type_strip_data, data_map_dump, language, decompiled_code_type_strip_ln_fn_names) -> None:
        self.binary_name = binary_name
        self.decompiler = decompiler
        self.decompiled_code_strip_path = decompiled_code_strip_path
        self.decompiled_code_type_strip_path = decompiled_code_type_strip_path
        self.joern_strip_data = joern_strip_data
        self.joern_type_strip_data = joern_type_strip_data
        self.dwarf_vars, self.dwarf_funcs, self.linkage_name_to_func_name, self.decompiled_code_type_strip_ln_fn_names = self._update_values((dwarf_info_path), decompiled_code_type_strip_ln_fn_names)
        self.data_map_dump = data_map_dump
        self.language = language
        self.sample = {}
        self.run()
    
    def _update_values(self, path, decompiled_code_type_strip_ln_fn_names):
        if isinstance(path, str):
            tmp = read_json(path)
        tmp = path
        linkage_name_to_func_name = {}
        if 'vars_per_func' in tmp.keys():
            dwarf_vars = tmp['vars_per_func']
            dwarf_funcs = dwarf_vars.keys()
        if 'linkage_name_to_func_name' in tmp.keys():
            if tmp['linkage_name_to_func_name']:
                linkage_name_to_func_name = tmp['linkage_name_to_func_name']
            else:
                for funcname, _ in tmp['vars_per_func'].items():
                    linkage_name_to_func_name[funcname] = funcname
        # TODO: if setting corpus language to c and cpp then enable this
        # else: 
        #     # C does not have linkage names, also at the time of building data set, we did not have cpp implementation in, so hack it
        #     for funcname, _ in tmp['vars_per_func'].items():
        #         linkage_name_to_func_name[funcname] = funcname
        #     print("linkage", linkage_name_to_func_name)

        # cpp
        if os.path.exists(decompiled_code_type_strip_ln_fn_names):
            decompiled_code_type_strip_ln_fn_names = read_json(decompiled_code_type_strip_ln_fn_names)
        else:
            # c
            decompiled_code_type_strip_ln_fn_names = linkage_name_to_func_name
        return dwarf_vars, dwarf_funcs, linkage_name_to_func_name, decompiled_code_type_strip_ln_fn_names

    def run(self):
        if self.decompiler.lower() == 'ida':
            parser_strip = IDAParser(read_text(self.decompiled_code_strip_path), self.binary_name, None)
            parser_type_strip = IDAParser(read_text(self.decompiled_code_type_strip_path), self.binary_name, self.decompiled_code_type_strip_ln_fn_names)
        elif self.decompiler.lower() == 'ghidra':
            parser_strip = GhidraParser(read_text(self.decompiled_code_strip_path), self.binary_name)
            parser_type_strip = GhidraParser(read_text(self.decompiled_code_type_strip_path), self.binary_name)
            
        # load joern data        
        joern_data_strip = JoernDataLoader(self.joern_strip_data, parser_strip.func_name_to_addr, None)
        joern_data_type_strip = JoernDataLoader(self.joern_type_strip_data, parser_type_strip.func_name_to_addr, parser_type_strip.func_name_to_linkage_name)
        # filter out functions that are not common
        common_func_addr_decompiled_code = set(parser_strip.func_addr_to_name.keys()).intersection(parser_type_strip.func_addr_to_name.keys())
        common_func_addr_from_joern = set(joern_data_strip.joern_addr_to_name.keys()).intersection(joern_data_type_strip.joern_addr_to_name.keys())
        
        # joern names to addr
        # remove comments at the last because Joern's numbers are with comments        
        for addr in common_func_addr_from_joern:
            try:
                if addr not in common_func_addr_decompiled_code:
                    continue
                # remove decompiler/compiler generated functions - we get dwarf names from type-strip
                if addr not in parser_type_strip.func_addr_to_name:
                    continue
                l.debug(f'Trying for func addr {self.binary_name} :: {self.decompiler} {addr}')
                
                func_name_type_strip = parser_type_strip.func_addr_to_name[addr]
                func_name_strip = parser_strip.func_addr_to_name[addr]

                # if function is compiler-generated (artifical) we do not have collect DWARF info fot that function (we need only source functions)
                # For Ghidra: we use addr as funcname in dwarf info that is collected from binary
                dwarf_addr_compatible_addr = None
                if self.decompiler.lower() == 'ghidra':
                    dwarf_addr_compatible_addr = str(int(addr[1:-1], 16) - 0x00100000)
                    if dwarf_addr_compatible_addr not in self.dwarf_funcs:
                        l.debug(f'Func not in DWARF / Source! {self.binary_name} :: {self.decompiler} {dwarf_addr_compatible_addr}')
                        continue
                
                # For IDA func name collected from binary is w/o line number
                elif self.decompiler.lower() == 'ida':
                    # funcnames from cpp have '::' in type-strip dc and not in parsed dwarf data - get linkage names so we are good
                    funcname_from_type_strip = parser_type_strip.functions[addr]['func_name_no_line']

                    if funcname_from_type_strip not in self.dwarf_funcs and funcname_from_type_strip.strip().split('::')[-1] not in self.dwarf_funcs:
                        l.debug(f'Func not in DWARF / Source! {self.binary_name} :: {self.decompiler} {funcname_from_type_strip}')
                        continue

                dc_functions_strip = parser_strip.functions[addr]
                dc_functions_type_strip = parser_type_strip.functions[addr]

                dwarf_func_name = '_'.join(func_name_type_strip.split('_')[:-1])
                cfunc_strip = CFunc(self, dc_functions_strip, joern_data_strip, "strip", func_name_strip, dwarf_func_name, addr, None)
                cfunc_type_strip = CFunc(self, dc_functions_type_strip, joern_data_type_strip, "type-strip", func_name_type_strip,  dwarf_func_name, addr, self.linkage_name_to_func_name)

                # unequal number of joern vars! do not proceed
                if len(cfunc_strip.joern_vars)  != len(cfunc_type_strip.joern_vars):
                    l.debug(f'Return! unequal number of joern vars! {self.binary_name} :: {self.decompiler} {addr}')
                    continue

                # unequal number of local vars!
                if len(cfunc_strip.local_vars_dc) == len(cfunc_type_strip.local_vars_dc):
                    # identify missing var and update lines to current lines (I n don't need for type_strip? but doing it for uniformity)
                    l.debug(f'Identify missing vars! {self.binary_name} :: {self.decompiler} :: {addr}')
                    cfunc_strip.joern_vars = self.identify_missing_vars(cfunc_strip)
                    cfunc_type_strip.joern_vars = self.identify_missing_vars(cfunc_type_strip)
                    
                    l.debug(f'Removing invalid variable names! {self.binary_name} :: {self.decompiler} :: {addr}')
                    
                    # remove invalid variable names
                    valid_strip_2_type_strip_varnames, cfunc_strip.joern_vars, cfunc_type_strip.joern_vars = self.remove_invalid_variable_names(cfunc_strip, cfunc_type_strip, 
                                                                                        parser_strip, parser_type_strip)

                    l.debug(f'Start matching variable names! {self.binary_name} :: {self.decompiler} :: {addr}')
                    matchvar = MatchVariables(cfunc_strip, cfunc_type_strip, valid_strip_2_type_strip_varnames, self.dwarf_vars, self.decompiler, cfunc_type_strip.dwarf_func_name, self.binary_name, self.language, dwarf_addr_compatible_addr)

                    sample_name, sample_data = matchvar.dump_sample()
                    if sample_data:
                        l.debug(f'Dump sample! {self.binary_name} :: {self.decompiler} :: {addr}')
                        self.sample[sample_name] = sample_data
                    
            except Exception as e:
                l.error(f"ERRR {self.binary_name} :: {e} {addr}")
        l.info(f'Variable Matching complete for {self.binary_name} :: {self.decompiler} :: samples :: {len(self.sample)}')
        write_json(self.data_map_dump, dict(self.sample))

    def identify_missing_vars(self, cfunc):
        # 1. update file lines to func lines
        # 2. joern may miss some occurences of variables without declaration (i.e. variables from .data, .bss section. find them and update corresponding lines)
        joern_vars = cfunc.joern_vars
        tmp_dict = {}
        for var, var_lines in joern_vars.items():
            updated_lines = []
            updated_lines[:] = [int(number) - int(cfunc.joern_func_start) +1 for number in var_lines]  
            var = var.strip('~')
            find_all = rf'([^\d\w_-])({var})([^\d\w_-])'                  
            for i, line in enumerate(cfunc.func_lines, 1):                        
                matches = re.search(find_all, line)                 
                if matches:                          
                    if i not in updated_lines and not line.startswith('//'):                          
                        updated_lines.append(i) 

            updated_lines = list(set(updated_lines))   
                         
            tmp_dict[var] = updated_lines
        return tmp_dict
        

    def remove_invalid_variable_names(self, cfunc_strip, cfunc_type_strip, parser_strip, parser_type_strip):
        
        # https://www.hex-rays.com/products/ida/support/idadoc/1361.shtml
        if self.decompiler.lower() == 'ida':
            data_types = ['_BOOL1', '_BOOL2', '_BOOL4', '__int8', '__int16', '__int32', '__int64', '__int128', '_BYTE', '_WORD', '_DWORD', '_QWORD', '_OWORD', '_TBYTE', '_UNKNOWN', '__pure', '__noreturn', '__usercall', '__userpurge', '__spoils', '__hidden', '__return_ptr', '__struct_ptr', '__array_ptr', '__unused', '__cppobj', '__ptr32', '__ptr64', '__shifted', '__high']

        # anymore undefined?
        if  self.decompiler.lower() == 'ghidra':
            data_types = ['ulong', 'uint', 'ushort', 'ulonglong', 'bool', 'char', 'int', 'long', 'undefined', 'undefined1', 'undefined2', 'undefined4', 'undefined8', 'byte', 'FILE', 'size_t']

        invalid_vars = ['null', 'true', 'false', 'True', 'False', 'NULL', 'char', 'int']
        # also func names which may have been identified as var names

        zip_joern_vars = dict(zip(cfunc_strip.joern_vars, cfunc_type_strip.joern_vars))
        # strip: type-strip
        valid_strip_2_type_strip_varnames = {}
        valid_strip_vars = {}
        valid_type_strip_vars = {}

        for strip_var, type_strip_var in zip_joern_vars.items():
            try:
                strip_var_woc, type_strip_var_woc = strip_var, type_strip_var
                strip_var = strip_var.strip('~')
                type_strip_var = type_strip_var.strip('~')

                # remove func name 
                if strip_var in parser_strip.func_name_wo_line.keys() or type_strip_var in parser_type_strip.func_name_wo_line.keys():
                    continue
                # remove func name with :: 
                if strip_var.strip().split('::')[-1] in parser_strip.func_name_wo_line.keys() or type_strip_var.strip().split('::')[-1] in parser_type_strip.func_name_wo_line.keys():
                    continue
                # (rn: sub_2540_475)
                if strip_var in parser_strip.func_name_to_addr.keys() or type_strip_var in parser_type_strip.func_name_to_addr.keys():
                    continue
                # remove types
                if strip_var in data_types or type_strip_var in data_types:
                    continue
                # remove invalid name
                if strip_var in invalid_vars or type_strip_var in invalid_vars:
                    continue

                if '::' in strip_var:
                    strip_var_woc = strip_var.strip().split('::')[-1]
                if '::' in type_strip_var: 
                    type_strip_var_woc = type_strip_var.strip().split('::')[-1]
                        
                # create valid mapping
                valid_strip_2_type_strip_varnames[strip_var_woc] = type_strip_var_woc

                # update joern vars
                if strip_var in cfunc_strip.joern_vars:
                    valid_strip_vars[strip_var_woc] = cfunc_strip.joern_vars[strip_var]
                        
                elif f'~{strip_var}' in cfunc_strip.joern_vars:
                    # if strip_var_woc:
                    valid_strip_vars[strip_var_woc] = cfunc_strip.joern_vars[f'~{strip_var}']
                        
                if type_strip_var in cfunc_type_strip.joern_vars:
                    # if type_strip_var_woc:
                    valid_type_strip_vars[type_strip_var_woc] = cfunc_type_strip.joern_vars[type_strip_var]
                    # else:
                        # valid_type_strip_vars[type_strip_var] = cfunc_type_strip.joern_vars[type_strip_var]

                if f'~{type_strip_var}' in cfunc_type_strip.joern_vars:
                    # if type_strip_var_woc:
                    valid_type_strip_vars[type_strip_var_woc] = cfunc_type_strip.joern_vars[f'~{type_strip_var}']
                    # else:
                        # valid_type_strip_vars[type_strip_var] = cfunc_type_strip.joern_vars[f'~{type_strip_var}']
            except Exception as e:
                l.error(f'Error in removing invalid variable names! {self.binary_name} :: {self.decompiler} :: {e}')     
        return valid_strip_2_type_strip_varnames, valid_strip_vars, valid_type_strip_vars        


class JoernDataLoader:
    def __init__(self, joern_data, decompiled_code_name_to_addr, func_name_to_linkage_name) -> None:
        self.functions = {}
        self.joern_data = joern_data
        self.decompiled_code_name_to_addr = decompiled_code_name_to_addr
        self.joern_name_to_addr = {}
        self.joern_addr_to_name = {}
        self.joern_start_line_to_funcname = {}
        self._load(func_name_to_linkage_name)

    def _load(self, func_name_to_linkage_name):
        # key: func name, val: {var: lines}
        func_line_num_to_addr = {}
        for k, v in self.decompiled_code_name_to_addr.items():
            line_num = str(k.split('_')[-1])
            func_line_num_to_addr[line_num] = v
        try:
            counter = 0
            for func_name, v in self.joern_data.items():   
                try:
                    # check if func name in decompiled code funcs and get addr mapping 
                    # type-strip - replace demangled name with mangled name (cpp) or simply a func name (c)
                    # some func names from joern are different from parsed decompiled code (IDA: YaSkkServ::`anonymous namespace'::signal_dictionary_update_handler | Joern: signal_dictionary_update_handler )          
                    if func_name_to_linkage_name:
                        tmp_wo_line = func_name.split('_')
                        if len(tmp_wo_line) <=1 :
                            continue
                        name_wo_line, line_num = '_'.join(tmp_wo_line[:-1]), tmp_wo_line[-1]

                        if name_wo_line in func_name_to_linkage_name:
                            func_name = func_name_to_linkage_name[name_wo_line]
                            func_name = f'{func_name}_{line_num}'

                    if func_name in self.decompiled_code_name_to_addr:                        
                        addr = self.decompiled_code_name_to_addr[func_name]       
                    elif line_num in func_line_num_to_addr:
                        addr = func_line_num_to_addr[line_num]
                    else:
                        l.warning("joern and IDA func name did not match")

                    self.joern_addr_to_name[addr] = func_name
                    self.joern_name_to_addr[func_name] = addr
                    self.joern_start_line_to_funcname[v['func_start']] = func_name

                    joern_vars = v['variable']
                    start = v['func_start']
                    end = v['func_end']             
                    if start and end:
                        self.functions[func_name] = {'func_name': func_name,
                            'variables': joern_vars,
                            # 'var_lines': joern_var_lines,
                            'start': start,
                            'end': end}
                    counter += 1
                except Exception as e:
                    l.error(f'Error in loading joern data! {e}')
        except Exception as e:   
            l.error(f' error in getting joern vars! {e}')


class CFunc:

    def __init__(self, dcl, dc_func, joern_data, binary_type, func_name, dwarf_func_name, func_addr, linkage_name_to_func_name ) -> None:
                
        self.local_vars_dc = dc_func['local_vars']       
        
        self.func = dc_func['func']
        self.func_addr = func_addr
        self.func_prototype = None
        self.func_body = None
        self.func_lines = None
        
        self.func_name = func_name
        self.func_name_no_line = '_'.join(func_name.split('_')[:-1])
        self.line = func_name.split('_')[-1]
        self.binary_name = dcl.binary_name
        self.binary_type = binary_type
        
        self.decompiler = dcl.decompiler
        self.dwarf_func_name = dwarf_func_name
        self.dwarf_mangled_func_name = linkage_name_to_func_name

        # start
        if self.func_name in joern_data.functions:
            self.joern_func_start = joern_data.functions[self.func_name]['start']
        elif self.line in joern_data.joern_start_line_to_funcname:
            tmp_func_name = joern_data.joern_start_line_to_funcname[self.line]
            self.joern_func_start = joern_data.functions[tmp_func_name]['start']

        # variables
        if self.func_name in joern_data.functions:
            self.all_vars_joern = joern_data.functions[self.func_name]['variables']
        elif self.line in joern_data.joern_start_line_to_funcname:
            tmp_func_name = joern_data.joern_start_line_to_funcname[self.line]
            self.all_vars_joern = joern_data.functions[tmp_func_name]['variables']

        self.set_func_details()

    def __repr__(self) -> str:
        pass
    
    @property
    def joern_vars(self):
        return self.all_vars_joern
    
    @joern_vars.setter
    def joern_vars(self, new_value):
        self.all_vars_joern = new_value

    def set_func_details(self) -> None:
        if self.func:
            self.func_lines = self.func.split('\n')
            self.func_prototype = self.func.split('{')[0]
            self.func_body = '{'.join(self.func.split('{')[1:])


class MatchVariables:

    def __init__(self, strip: CFunc, type_strip: CFunc, valid_strip_2_type_strip_varnames, dwarf_vars, decompiler, dwarf_func_name, binary_name, language, dwarf_addr_compatible_addr=None) -> None:
        self.cfunc_strip = strip
        self.cfunc_type_strip = type_strip
        self.mapped_vars = valid_strip_2_type_strip_varnames
        self.dwarf_vars = dwarf_vars
        self.labelled_vars = {}
        self.modified_func = None
        self.decompiler = decompiler
        self.dwarf_func_name = dwarf_func_name
        self.md5_hash = None
        self.binary_name = binary_name
        self.language = language
        self.dwarf_addr_compatible_addr = dwarf_addr_compatible_addr
        self.match()
    
    def match(self):        
        if len(self.cfunc_strip.joern_vars) != len(self.cfunc_type_strip.joern_vars):
            return        
        if len(self.mapped_vars) == 0:
            return

        self.modified_func = self.update_func()
        # label DWARF and decompiler-gen
        self.labelled_vars = self.label_vars()    
        self.md5_hash = self.func_hash()
 
    def update_func(self):
        def rm_comments(func):
            cm_regex = r'// .*'
            cm_func = re.sub(cm_regex, ' ', func).strip()
            return cm_func

        # pre-process variables and replace them with "@@dwarf_var_name@@var_id@@"
        varname2token = {}
        for i, varname in enumerate(self.mapped_vars, 0):
            varname2token[varname] = f"@@var_{i}@@{self.mapped_vars[varname]}@@"
        new_func = self.cfunc_strip.func

        # if no line numbers available
        allowed_prefixes = [" ", "&", "(", "*", "++", "--", "!"]
        allowed_suffixes = [" ", ")", ",", ";", "[", "++", "--"]
        for varname, newname in varname2token.items():
            for p in allowed_prefixes:
                for s in allowed_suffixes:
                    new_func = new_func.replace(f"{p}{varname}{s}", f"{p}{newname}{s}")
        
        # no var is labelled as stderr. it is nether in global vars nor in vars for this function, so we do not add it
        if '@@' not in new_func:
            return None
        return rm_comments(new_func)
       

    def label_vars(self):
        labelled_vars = {}
        for strip_var, type_strip_var in self.mapped_vars.items():
            if self.decompiler == 'ida':
                check_value = self.cfunc_type_strip.dwarf_func_name   
            elif self.decompiler == 'ghidra':
                check_value = self.dwarf_addr_compatible_addr
            if type_strip_var in self.dwarf_vars[check_value] or type_strip_var in self.dwarf_vars['global_vars']:
                labelled_vars[type_strip_var] = 'dwarf'
            else:
                labelled_vars[type_strip_var] = self.decompiler
        return labelled_vars

    
    def func_hash(self) -> None:
        var_regex = r"@@(var_\d+)@@(\w+)@@"
        up_func = re.sub(var_regex, "\\2", self.modified_func)
        func_body = '{'.join(up_func.split('{')[1:])   
        md5 = hashlib.md5(func_body.encode('utf-8')).hexdigest()      
        return md5
    

    def dump_sample(self):
        if self.decompiler == 'ida':
            func_name_dwarf = str(self.cfunc_type_strip.dwarf_mangled_func_name[str(self.dwarf_func_name)])
        elif self.decompiler == 'ghidra':
            func_name_dwarf = str(self.cfunc_type_strip.dwarf_func_name)
        if self.modified_func:
            name = f'{self.binary_name}_{self.cfunc_strip.func_addr}'
            data = { 'func': self.modified_func,
                                'type_stripped_vars': dict(self.labelled_vars),
                                'stripped_vars': list(self.mapped_vars.keys()),
                                'mapped_vars': dict(self.mapped_vars),
                                'func_name_dwarf':func_name_dwarf,
                                'dwarf_mangled_func_name': str(self.dwarf_func_name),
                                'hash': self.md5_hash,
                                'language': self.language
                                }
        
            return name, data
        else:
            return None, None
