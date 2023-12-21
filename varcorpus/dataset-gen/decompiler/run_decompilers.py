import subprocess
import time
import os
import shutil
import re
from collections import defaultdict
import json
import sys
import tempfile
from decompiler.ida_dec import decompile_
# from logger import get_logger
from pathlib import Path
import logging
l = logging.getLogger('main')

class Decompiler:
    def __init__(self, decompiler, decompiler_path, decompiler_workdir, 
                 binary_name, binary_path, binary_type,
                 decompiled_binary_code_path, failed_path, type_strip_addrs, type_strip_mangled_names) -> None:
        self.decompiler = decompiler
        self.decompiler_path = decompiler_path
        self.decompiler_workdir = decompiler_workdir
        self.binary_name = binary_name
        self.binary_path = binary_path
        self.binary_type = binary_type
        self.decompiled_binary_code_path = decompiled_binary_code_path
        self.failed_path = failed_path
        self.type_strip_addrs= type_strip_addrs
        self.type_strip_mangled_names = type_strip_mangled_names
        self.decompile()

    def decompile(self):
        
        if self.decompiler == "ida":
            success, path = decompile_(self.decompiler_workdir, Path(self.decompiler_path), self.binary_name,
                Path(self.binary_path), self.binary_type, Path(self.decompiled_binary_code_path),
                    Path(self.failed_path), Path(self.type_strip_addrs), Path(self.type_strip_mangled_names))                        
            if not success:
                l.error(f"Decompilation failed for {self.binary_name} :: {self.binary_type} :: check logs at {path}")
                return None, None
            return success, path
        
        elif self.decompiler == "ghidra":
            try:
                current_script_dir = Path(__file__).resolve().parent
                subprocess.call(['{} {} tmp_project -scriptPath {} -postScript ghidra_dec.py {} {} {} -import {} -readOnly -log {}.log'.format(self.decompiler_path, self.decompiler_workdir, current_script_dir,
                                                                                                                                                                                self.decompiled_binary_code_path, self.failed_path,  
                                                                                                                                                                                self.decompiler_workdir, self.binary_path,  self.decompiler_workdir, 
                                                                                                                                                                                self.binary_name)], shell=True,
                                                                                                                                                                                stdout=subprocess.DEVNULL,
                                                                                                                                                                                stderr=subprocess.DEVNULL)

            except Exception as e:
                l.error(f"Decompilation failed for {self.binary_name} :: {self.binary_type} {e}")
                return None
            return True, self.decompiled_binary_code_path