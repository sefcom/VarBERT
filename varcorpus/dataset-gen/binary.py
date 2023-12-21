import subprocess
from pathlib import Path
import hashlib
from elftools.elf.elffile import ELFFile
from elftools.dwarf.locationlists import (
    LocationEntry, LocationExpr, LocationParser)
from elftools.dwarf.descriptions import (
    describe_DWARF_expr, _import_extra, describe_attr_value,set_global_machine_arch)
from collections import defaultdict
from utils import write_json
import os
import sys
import shutil
import logging
from utils import subprocess_
from dwarf_info import Dwarf
from strip_types import type_strip_target_binary

l = logging.getLogger('main')

class Binary:
    def __init__(self, target_binary, b_name, path_manager, language, decompiler, load_data=True):
        self.target_binary = target_binary         # to binary_path
        self.binary_name = b_name
        self.hash = None
        self.dwarf_data = None
        self.strip_binary  = None
        self.type_strip_binary = None
        self.path_manager = path_manager
        self.language = language
        self.decompiler = decompiler
        self.dwarf_dict = None
    
        if load_data:
            self.dwarf_dict = self.load()
            self.strip_binary, self.type_strip_binary = self.modify_dwarf()
            
    # read things from binary!
    def load(self):
        try:
            l.debug(f'Reading DWARF info from :: {self.target_binary}')
            if not self.is_elf_has_dwarf():
                return None

            self.hash = self.md5_hash()
            self.dwarf_data = Dwarf(self.target_binary, self.binary_name, self.decompiler)
            dwarf_dict = self.dump_data()
            l.debug(f'Finished reading DWARF info from :: {self.target_binary}')
            return dwarf_dict
        except Exception as e:
            l.error(f"Error in reading  DWARF info from :: {self.target_binary} :: {e}")

    def md5_hash(self):
        return subprocess.check_output(['md5sum', self.target_binary]).decode('utf-8').strip().split(' ')[0]
                
    def is_elf_has_dwarf(self):
        try:
            with open(self.target_binary, 'rb') as f:
                elffile = ELFFile(f)
                if not elffile.has_dwarf_info():
                    return None                
                set_global_machine_arch(elffile.get_machine_arch())
                dwarf_info = elffile.get_dwarf_info()                
                return dwarf_info
        except Exception as e:
            l.error(f"Error in is_elf_has_dwarf :: {self.target_binary} :: {e}")

    def dump_data(self):
        dump = defaultdict(dict)
        dump['hash'] = self.hash
        dump['vars_per_func'] = self.dwarf_data.vars_in_each_func
        dump['linkage_name_to_func_name'] = self.dwarf_data.linkage_name_to_func_name
        dump['language'] = self.language
        write_json(os.path.join(self.path_manager.tmpdir, 'dwarf', self.binary_name), dict(dump))
        return dump

    def _strip_binary(self, target_binary):

        res = subprocess_(['strip', '--strip-all', target_binary])
        if isinstance(res, Exception):
            l.error(f"error in stripping binary! {res} :: {target_binary}")
            return None
        return target_binary

    def _type_strip_binary(self, in_binary, out_binary):

        type_strip_target_binary(in_binary, out_binary, self.decompiler)
        if not os.path.exists(out_binary):
            l.error(f"error in stripping binary! :: {in_binary}")
        return out_binary

    def modify_dwarf(self): 

        shutil.copy(self.target_binary, self.path_manager.strip_bin_dir)
        type_strip = self._type_strip_binary(os.path.join(self.path_manager.strip_bin_dir, self.binary_name), os.path.join(self.path_manager.type_strip_bin_dir, self.binary_name))
        strip = self._strip_binary(os.path.join(self.path_manager.strip_bin_dir, self.binary_name))        
        return strip, type_strip



