import re
import logging


l = logging.getLogger('main')
#
# Raw IDA decompiled code pre-processing
#
class IDAParser:
    def __init__(self, decompiled_code, binary_name, linkage_name_to_func_name) -> None:
        self.decompiled_code = decompiled_code
        self.functions = {}
        self.func_addr_to_name = {}
        self.func_name_to_addr = {}
        self.linkage_name_to_func_name = linkage_name_to_func_name
        self.func_name_to_linkage_name = self.create_dict_fn_to_ln()
        self.binary_name = binary_name
        self.func_name_wo_line = {}
        self.preprocess_ida_raw_code()

    def create_dict_fn_to_ln(self):

        # linkage name to func name (it is the signature abc(void), so split at '(')
        func_name_to_linkage_name = {}
        if self.linkage_name_to_func_name:
            for ln, fn in self.linkage_name_to_func_name.items():
                func_name_to_linkage_name[fn.split('(')[0]] = ln
        return func_name_to_linkage_name
    
    def preprocess_ida_raw_code(self):

        data = self.decompiled_code
        func_addr_to_line_count = {}
        functions, self.func_name_to_addr, self.func_addr_to_name, self.func_name_wo_line = self.split_ida_c_file_into_funcs(data)
        for addr, func in functions.items():
            try:
                func_sign = func.split('{')[0].strip()
                func_body = '{'.join(func.split('{')[1:])
        
                if not addr in self.func_addr_to_name:
                    continue
                func_name = self.func_addr_to_name[addr]

                # find local variables
                varlines_bodylines = func_body.strip("\n").split('\n\n')
                if len(varlines_bodylines) >= 2:
                    var_dec_lines = varlines_bodylines[0]
                    local_vars = self.find_local_vars(var_dec_lines)
                else:
                    local_vars = []  
                self.functions[addr] = {'func_name': func_name,
                                        'func_name_no_line': '_'.join(func_name.split('_')[:-1]), # refer to comments in split_ida_c_file_into_funcs
                                        'func': func,
                                        'func_prototype': func_sign,
                                        'func_body': func_body,
                                        'local_vars': local_vars,
                                        'addr': addr
                                        }

            except Exception as e:
                l.error(f'Error in {self.binary_name}:{func_name}:{addr} = {e}')
        l.info(f'Functions after IDA parsing {self.binary_name} :: {len(self.functions)}')

    def split_ida_c_file_into_funcs(self, data: str):

        line_count = 1
        chunks = data.split('//----- ')
        data_declarations = chunks[0].split('//-------------------------------------------------------------------------')[2]
        func_dict, func_name_to_addr, func_addr_to_name, func_name_with_linenum_linkage_name   = {}, {}, {}, {}
        func_name_wo_line = {}
        for chunk in chunks:        
            lines = chunk.split('\n')
            line_count = line_count
            if not lines:
                continue
            if '-----------------------------------' in lines[0]:
                name = ''
                func_addr = lines[0].strip('-').strip()            
                func = '\n'.join(lines[1:])
                func_dict[func_addr] = func  

                # func name and line number
                all_func_lines = func.splitlines()
                first_line = all_func_lines[0]
                sec_line = all_func_lines[1]
                if '(' in first_line:
                    name = first_line.split('(')[0].split(' ')[-1].strip('*') + '_' + str(line_count + 1)
                elif '//' in first_line and '(' in sec_line:
                    name = sec_line.split('(')[0].split(' ')[-1].strip('*') + '_' + str(line_count + 2)

                if name:
                    # for cpp we use linkage name/mangled names instead of original function names
                    # 1. dwarf info gives func name w/o class name but IDA has class_name::func_name. 2. to help with function overloading or same func name in different classes
                    # if there is no mangled name, original function is copied to dict so we have all the func names.
                    # so in case of C, we can use same dict. it is essentially func name

                    tmp_wo_line = name.split('_')
                    if len(tmp_wo_line) <=1 :
                        continue
                    name_wo_line, line_num = '_'.join(tmp_wo_line[:-1]), tmp_wo_line[-1]

                    # replace demangled name with mangled name (cpp) or simply a func name (c) 
                    if name_wo_line in self.func_name_to_linkage_name: # it is not func from source
                        # type-strip 
                        name = self.func_name_to_linkage_name[name_wo_line]   
                        name = f'{name}_{line_num}'
                    elif name_wo_line.strip().split('::')[-1] in self.func_name_to_linkage_name:
                        name = f'{name}_{line_num}'

                    # strip  
                    func_name_to_addr[name] = func_addr
                    func_addr_to_name[func_addr] = name
                    func_name_wo_line[name_wo_line] = func_addr

            line_count += len(lines) - 1         
        return func_dict, func_name_to_addr, func_addr_to_name, func_name_wo_line

    def find_local_vars(self, lines):
        # use regex
        local_vars = []
        regex = r"(\w+(\[\d+\]|\d{0,6}));"
        matches = re.finditer(regex, lines)
        if matches:
            for m in matches:
                tmpvar = m.group(1)
                if not tmpvar:
                    continue
                lv = tmpvar.split('[')[0]
                local_vars.append(lv)
        return local_vars
    

#
# Raw Ghidra decompiled code pre-processing
#

class GhidraParser:
    def __init__(self, decompiled_code_path, binary_name) -> None:
        self.decompiled_code_path = decompiled_code_path
        self.functions = {}
        self.func_addr_to_name = {}
        self.func_name_to_addr = {}
        self.linkage_name_to_func_name = None
        self.func_name_to_linkage_name = None
        self.binary_name = binary_name
        self.func_name_wo_line = {}
        self.preprocess_ghidra_raw_code()

    def preprocess_ghidra_raw_code(self):
        data = self.decompiled_code_path
        functions, self.func_name_to_addr, self.func_addr_to_name, self.func_name_wo_line = self.split_ghidra_c_file_into_funcs(data)
        if not functions:
            return
        for addr, func in functions.items():
            try:
                func_sign = func.split('{')[0].strip()
                func_body = '{'.join(func.split('{')[1:])
                if not addr in self.func_addr_to_name:
                    continue
                func_name = self.func_addr_to_name[addr]

                varlines_bodylines = func_body.strip("\n").split('\n\n')
                if len(varlines_bodylines) >= 2:
                    var_dec_lines = varlines_bodylines[0]
                    local_vars = self.find_local_vars(varlines_bodylines)
                else:
                    local_vars = []                

                self.functions[addr] = {'func_name': func_name,
                                        'func_name_no_line': '_'.join(func_name.split('_')[:-1]),
                                        'func': func,
                                        'func_prototype': func_sign,
                                        'func_body': func_body,
                                        'local_vars': local_vars,
                                        # 'arguments': func_args,
                                        'addr': addr
                                        }
            except Exception as e:
                l.error(f'Error in {self.binary_name}:{func_name}:{addr}  = {e}')

    def split_ghidra_c_file_into_funcs(self, data: str):
        
        chunks = data.split('//----- ')
      
        func_dict, func_name_to_addr, func_addr_to_name   = {}, {}, {}
        func_name_wo_line = {}

        line_count = 1
        for chunk in chunks[1:]:     
            line_count = line_count   
            lines = chunk.split('\n')
            if not lines:
                continue
            if '-----------------------------------' in lines[0]:
                func_addr = lines[0].strip('-').strip()            
                func = '\n'.join(lines[1:])
                func_dict[func_addr] = func

                # get func name and line number TODO: Ghidra had a func split in two lines - maybe use regex for it
                all_func_lines = func.split('\n\n')
                first_line = all_func_lines[0]
                if '(' in first_line:
                    t_name = first_line.split('(')[0]
                    if "\n" in t_name:
                        name = t_name.split('\n')[-1].split(' ')[-1].strip('*') + '_' + str(line_count + 1) 
                    else:
                        name = t_name.split(' ')[-1].strip('*') + '_' + str(line_count + 1)                  
                
                if name:
                    name_wo_line = '_'.join(name[:-1])
                    func_name_to_addr[name] = func_addr           
                    func_addr_to_name[func_addr] = name
                    func_name_wo_line[name_wo_line] = func_addr

            line_count += len(lines) - 1
        return func_dict, func_name_to_addr, func_addr_to_name, func_name_wo_line
    
    def find_local_vars(self, varlines_bodylines):
        
        all_vars = []
        try: 
            first_curly = ''
            sep = ''
            dec_end = 0
            for elem in range(len(varlines_bodylines)):
                if varlines_bodylines[elem] == '{' and first_curly == '':
                    first_curly = '{'
                    dec_start = elem + 1
                if first_curly == '{' and sep == '' and varlines_bodylines[elem] == '  ':
                    dec_end = elem - 1

            for index in range(dec_start, dec_end + 1):
                if index < len(varlines_bodylines):
                    f_sp = varlines_bodylines[index].split(' ')
                    if f_sp:
                        tmp_var = f_sp[-1][:-1].strip('*')

                        if tmp_var.strip().startswith('['):
                            up_var = f_sp[-2:-1][0].strip('*')
                            all_vars.append(up_var)
                        else:
                            all_vars.append(tmp_var)

        except Exception as e:
            l.error(f'Error in finding local vars {self.binary_name} :: {e}')

        return all_vars
