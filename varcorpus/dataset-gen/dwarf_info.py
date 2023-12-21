import logging
from elftools.elf.elffile import ELFFile
from elftools.dwarf.locationlists import (
    LocationEntry, LocationExpr, LocationParser)
from elftools.dwarf.descriptions import (
    describe_DWARF_expr, _import_extra, describe_attr_value,set_global_machine_arch)
from collections import defaultdict

l = logging.getLogger('main')

def is_elf(binary_path: str):

    with open(binary_path, 'rb') as rb:
        if rb.read(4) != b"\x7fELF":
            l.error(f"File is not an ELF format: {binary_path}")
            return None
        rb.seek(0)
        elffile = ELFFile(rb)
        if not elffile.has_dwarf_info():
            l.error(f"No DWARF info found in: {binary_path}")
            return None
        return elffile.get_dwarf_info()

class Dwarf:
    def __init__(self, binary_path, binary_name, decompiler) -> None:
        self.binary_path = binary_path
        self.binary_name = binary_name
        self.decompiler = decompiler
        self.dwarf_info = is_elf(self.binary_path)
        self.vars_in_each_func = defaultdict(list)
        self.spec_offset_var_names = defaultdict(set) if decompiler == 'ghidra' else None
        self.linkage_name_to_func_name = {}
        self._load()
    
    def _load(self):
        if not self.dwarf_info:
            l.error(f"Failed to load DWARF information for: {self.binary_path}")
            return
        if self.decompiler == 'ida':
            self.spec_offset_var_names = self.collect_spec_and_names()
        self.vars_in_each_func, self.linkage_name_to_func_name = self.read_dwarf()

    # for Ghidra only
    def collect_spec_and_names(self):
        # 1. get CU's offset 2. get spec offset, if any function resolves use that function name with vars
        spec_offset_var_names = defaultdict(set)
        for CU in self.dwarf_info.iter_CUs():
            try:
                current_cu_offset = CU.cu_offset
                top_DIE = CU.get_top_DIE()
                self.die_info_rec(top_DIE)
                current_spec_offset = 0
                
                for i in range(0, len(CU._dielist)):
                    if CU._dielist[i].tag == 'DW_TAG_subprogram':
                        
                        spec_offset, current_spec_offset  = 0, 0
                        spec = CU._dielist[i].attributes.get("DW_AT_specification", None)
                        
                        # specification value + CU offset = defination of function
                        # 15407 +  0xbd1 = 0x8800
                        # <8800>: Abbrev Number: 74 (DW_TAG_subprogram)
                        # <8801>   DW_AT_external  
                        
                        if spec:
                            spec_offset = spec.value
                            current_spec_offset = spec_offset + current_cu_offset      

                    if CU._dielist[i].tag == 'DW_TAG_formal_parameter' or CU._dielist[i].tag == 'DW_TAG_variable':
                        if self.is_artifical(CU._dielist[i]):
                            continue
                        name = self.get_name(CU._dielist[i])  
                        if not name:
                            continue
                        if current_spec_offset:
                            spec_offset_var_names[str(current_spec_offset)].add(name)
             
            except Exception as e:
                l.error(f"Error in collect_spec_and_names :: {self.binary_path} :: {e}")
        return spec_offset_var_names
    
    # for global vars - consider only variables those have locations. 
    # otherwise variables like stderr, optind, make it to global variables list and 
    # sometimes these can be marked as valid variables for a particular binary

    def read_dwarf(self):
        # for each binary
        if not self.dwarf_info:
            return defaultdict(list), {}
        vars_in_each_func = defaultdict(list)
        previous_func_offset = 0
        l.debug(f"Reading DWARF info from :: {self.binary_path}")
        for CU in self.dwarf_info.iter_CUs():
            try:
                current_func_name, ch, func_name = 'global_vars', '', ''
                func_offset = 0
                tmp_list, global_vars_list, local_vars_and_args_list = [], [], []

                # caches die info 
                top_DIE = CU.get_top_DIE()
                self.die_info_rec(top_DIE)

                # 1. global vars  2. vars and args in subprogram
                for i in range(0, len(CU._dielist)):
                    try:
                        if CU._dielist[i].tag == 'DW_TAG_variable':                        
                            if not CU._dielist[i].attributes.get('DW_AT_location'):
                                continue    
                            if self.is_artifical(CU._dielist[i]):
                                continue
                            var_name = self.get_name(CU._dielist[i]) 
                            if not var_name: 
                                continue
                            tmp_list.append(var_name)         

                        # shouldn't check location for parameters because we don't add their location in DWARF
                        if CU._dielist[i].tag == 'DW_TAG_formal_parameter':
                            param_name = self.get_name(CU._dielist[i])  
                            if not param_name:
                                continue
                            tmp_list.append(param_name)

                        # for last func and it's var
                        local_vars_and_args_list = tmp_list
                        if CU._dielist[i].tag == 'DW_TAG_subprogram':
                            if self.is_artifical(CU._dielist[i]):
                                continue                        
                            # IDA
                            if self.decompiler == 'ghidra':
                                # func addr is func name for now
                                low_pc = CU._dielist[i].attributes.get('DW_AT_low_pc', None)
                                addr = None
                                if low_pc is not None:
                                    addr = str(low_pc.value)

                                ranges = CU._dielist[i].attributes.get('DW_AT_ranges', None)
                                if ranges is not None:
                                    addr =  self.dwarf_info.range_lists().get_range_list_at_offset(ranges.value)[0].begin_offset

                                if not addr:
                                    continue
                                func_name = addr    
                            # Ghidra
                            if self.decompiler == 'ida':
                                func_name = self.get_name(CU._dielist[i]) 
                                if not func_name:
                                    continue

                                func_linkage_name = self.get_linkage_name(CU._dielist[i])
                                func_offset = CU._dielist[i].offset
                                # because func name from dwarf is without class name but IDA gives us funcname with classname
                                # so we match them using linkage_name
                                if func_linkage_name:
                                    self.linkage_name_to_func_name[func_linkage_name] = func_name
                                    func_name = func_linkage_name
                                else:
                                    # need it later for matching
                                    self.linkage_name_to_func_name[func_name] = func_name
                                    func_name = func_name
                             
                                # because DIE's are serialized and subprogram comes before vars and params
                                vars_from_specification_subprogram = []                        
                                if previous_func_offset in self.spec_offset_var_names:
                                    vars_from_specification_subprogram = self.spec_offset_var_names[str(previous_func_offset)]
                                previous_func_offset = str(func_offset)
                     
                            if current_func_name != func_name:
                                if current_func_name == 'global_vars':
                                    global_vars_list.extend(tmp_list)
                                    vars_in_each_func[current_func_name].extend(global_vars_list)
                                else:
                                    if self.decompiler == 'ida':
                                        if vars_from_specification_subprogram: 
                                            tmp_list.extend(vars_from_specification_subprogram)
                                    vars_in_each_func[current_func_name].extend(tmp_list)                 
                                    ch = current_func_name
                                current_func_name = func_name
                                tmp_list = []      
                    except Exception as e:
                        l.error(f"Error in reading DWARF {e}")  
                      
                if current_func_name != ch and func_name:
                    vars_in_each_func[func_name].extend(local_vars_and_args_list)
   
            except Exception as e:
                l.error(f"Error in read_dwarf :: {self.binary_name} :: {e}")  
            l.debug(f"Number of functions in {self.binary_name}: {str(len(vars_in_each_func))}")        
        return vars_in_each_func, self.linkage_name_to_func_name

    def get_name(self, die):        
        name = die.attributes.get('DW_AT_name', None)                   
        if name:
            return name.value.decode('ascii')
        
    def get_linkage_name(self, die):
        name = die.attributes.get('DW_AT_linkage_name', None)                   
        if name:
            return name.value.decode('ascii')
    
    # compiler generated variables or functions (destructors, ...)
    def is_artifical(self, die):
        return die.attributes.get('DW_AT_artificial', None)                  
       
    def die_info_rec(self, die, indent_level='    '):
        """ A recursive function for showing information about a DIE and its
            children.
        """ 
        child_indent = indent_level + '  '
        for child in die.iter_children():
            self.die_info_rec(child, child_indent)
    
    @classmethod
    def get_vars_in_each_func(cls, binary_path):
        dwarf = cls(binary_path)
        return dwarf.vars_in_each_func
    
    @classmethod
    def get_vars_for_func(cls, binary_path, func_name):
        dwarf = cls(binary_path)
        return dwarf.vars_in_each_func[func_name] + dwarf.vars_in_each_func['global_vars']