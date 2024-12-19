import os
import json
import time
import psutil
import logging
import subprocess
from typing import List
from pathlib import Path
from concurrent.futures import process
from elftools.elf.elffile import ELFFile

# log = get_logger(sys._getframe().f_code.co_filename)

def read_text(file_name):
    with open(file_name, 'r') as r:
        data = r.read().strip()
    return data

def read_lines(file_name):
    with open(file_name, 'r') as r:
        data = r.read().split()
    return data

def read_json(file_name):
    with open(file_name, 'r') as r:
        data = json.loads(r.read())
    return data

def write_json(file_name, data):
    with open(f'{file_name}.json', 'w') as w:
        w.write(json.dumps(data))

def subprocess_(command, timeout=None, shell=False):
    try:
        subprocess.check_call(command, timeout=timeout, shell=shell)
    except Exception as e:
        l.error(f"error in binary! :: {command} :: {e}")
        return e       
   
def is_elf(binary_path: str):
    '''
    if elf return elffile obj
    '''
    with open(binary_path, 'rb') as rb:
        bytes = rb.read(4)
        if bytes == b"\x7fELF":
            rb.seek(0)
            elffile = ELFFile(rb)
            if elffile and elffile.has_dwarf_info():
                dwarf_info = elffile.get_dwarf_info()
                
                return elffile, dwarf_info

def create_dirs(dir_names, tmpdir, ty):
    """
    Create directories within the specified data directory.

    :param dir_names: List of directory names to create.
    :param tmpdir: Base data directory where directories will be created.
    :param ty: Subdirectory type, such as 'strip' or 'type-strip'. If None, only base directories are created.
    """
    # Path(os.path.join(tmpdir, 'dwarf')).mkdir(parents=True, exist_ok=True)
    for name in dir_names:
        if ty:
            target_path = os.path.join(tmpdir, name, ty)
        else:
            target_path = os.path.join(tmpdir, name)        
        Path(target_path).mkdir(parents=True, exist_ok=True)


def set_up_data_dir(tmpdir, workdir, decompiler):
    """
    Set up the data directory with required subdirectories for the given decompiler.

    :param tmpdir: The base data directory to set up.
    :param workdir: The working directory where some data directories will be created.
    :param decompiler: Name of the decompiler to create specific subdirectories.
    """

    base_dirs = ['binary', 'failed']
    dc_joern_dirs = ['dc', 'joern']
    map_dirs = [f"map/{decompiler}", f"dc/{decompiler}/type_strip-addrs", f"dc/{decompiler}/type_strip-names"]
    workdir_dirs = [f'{decompiler}_data', 'tmp_joern']

    # Create directories for 'strip' and 'type-strip'
    for ty in ['strip', 'type_strip']:
        create_dirs([f'{d}/{decompiler}' for d in dc_joern_dirs] + base_dirs, tmpdir, ty)

    # Create additional directories
    create_dirs(map_dirs, tmpdir, None)

    # Create workdir directories
    create_dirs(workdir_dirs, workdir, None)

    # copy binary
    create_dirs(['dwarf'], tmpdir, None)


### JOERN


l = logging.getLogger('main')

class JoernServer:
    def __init__(self, joern_path, port):
        self.joern_path = joern_path
        self.port = port
        self.process = None
        self.java_process = None

    def start(self):
        if self.is_server_running():
            l.error(f"Joern server already running on port {self.port}")
            self.stop()
            return
        
        # hack: joern can't find the shell script fuzzyc2cpg.sh (CPG generator via the shell in this version)
        current_dir = os.getcwd()
        try:
            os.chdir(self.joern_path)
            joern_cmd = [os.path.join(self.joern_path, 'joern'), '--server', '--server-port', str(self.port)]
            self.process = subprocess.Popen(joern_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            for _ in range(20):
                self.java_process =  self.is_server_running()
                if self.java_process:
                    l.debug(f"Joern server started on port {self.port}")
                    return
                time.sleep(4)
                l.debug(f"Retrying to start Joern server on port {self.port}")
        except Exception as e:
            l.error(f"Failed to start Joern server on port {self.port} :: {e}")
            return

        finally:
            os.chdir(current_dir)

    def stop(self):
        if self.java_process is not None:
            self.java_process.kill()
            time.sleep(3)
            if self.is_server_running():
                l.warning("Joern server did not terminate gracefully, forcing termination")
                self.java_process.kill()
  
    def is_server_running(self):
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] == 'java' and 'io.shiftleft.joern.console.AmmoniteBridge' in proc.info['cmdline'] and '--server-port' in proc.info['cmdline'] and str(self.port) in proc.info['cmdline']:
                        return psutil.Process(proc.info['pid'])
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
        except Exception as e:
            l.error(f"Error while checking if server is running: {e}")

        return False
    
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
    
    def exit(self):
        self.stop()

    def restart(self):
        self.server.stop()
        time.sleep(10)
        self.server.start()