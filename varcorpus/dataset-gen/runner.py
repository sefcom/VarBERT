import os
import time
import glob
import shutil
import logging
from tqdm import tqdm
from pathlib import Path
import concurrent.futures
from multiprocessing import Manager

from binary import Binary
from joern_parser import JoernParser
from variable_matching import DataLoader
from decompiler.run_decompilers import Decompiler
from utils import set_up_data_dir, JoernServer, read_json
from create_dataset_splits import create_train_and_test_sets

l = logging.getLogger('main')

class Runner:
    def __init__(self, decompiler, target_binaries, WORKERS, path_manager, PORT, language, DEBUG) -> None:
        self.decompiler = decompiler
        self.target_binaries =  target_binaries
        self.WORKERS = WORKERS
        self.path_manager = path_manager
        self.binary = None
        self.PORT = PORT
        self.language = language
        self.DEBUG = DEBUG
        self.collect_workdir = Manager().dict()

    def _setup_environment_for_ghidra(self):
        java_home = "/usr/lib/jvm/java-17-openjdk-amd64"
        os.environ["JAVA_HOME"] = java_home
        if os.environ.get('JAVA_HOME') != java_home:
            l.error(f"INCORRECT JAVA VERSION FOR GHIDRA: {os.environ.get('JAVA_HOME')}")

    def decompile_runner(self, binary_path: str):
        """
        Runner method to handle the decompilation of a binary.
        """
        try:
            decompile_results = {}
            binary_name = Path(binary_path).name
            l.info(f"Processing target binary: {binary_name}")
            workdir = f'{self.path_manager.tmpdir}/workdir/{binary_name}'
            l.debug(f"{binary_name} in workdir {workdir}")            
            set_up_data_dir(self.path_manager.tmpdir, workdir, self.decompiler)

            # collect dwarf info and create strip and type-strip binaries
            self.binary = Binary(binary_path, binary_name, self.path_manager, self.language, self.decompiler, True)
            if not (self.binary.strip_binary and self.binary.type_strip_binary):
                l.error(f"Strip/Type-strip binary not found for {self.binary.binary_name}")
                return
            l.debug(f"Strip/Type-strip binaries created for {self.binary.binary_name}")
            decomp_workdir = os.path.join(workdir, f"{self.decompiler}_data")
            decompiler_path = getattr(self.path_manager, f"{self.decompiler}_path")
            if self.decompiler == "ghidra":
                self._setup_environment_for_ghidra()            
            for binary_type in ["type_strip", "strip"]:             
                binary_path = getattr(self.binary, f"{binary_type}_binary")
                l.debug(f"Decompiling {self.binary.binary_name} :: {binary_type} with {self.decompiler} :: {binary_path}")

                dec = Decompiler(
                    decompiler=self.decompiler,
                    decompiler_path=Path(decompiler_path),
                    decompiler_workdir=decomp_workdir,
                    binary_name=self.binary.binary_name,
                    binary_path=binary_path,
                    binary_type=binary_type,
                    decompiled_binary_code_path=os.path.join(self.path_manager.dc_path, str(self.decompiler), binary_type, f'{binary_name}.c'),
                    failed_path=os.path.join(self.path_manager.failed_path_ida, binary_type),
                    type_strip_addrs=os.path.join(self.path_manager.dc_path, self.decompiler, f"type_strip-addrs", binary_name),
                    type_strip_mangled_names = os.path.join(self.path_manager.dc_path, self.decompiler, f"type_strip-names", binary_name)                    
                )
                dec_path = dec
                
                if not dec_path:  
                    l.error(f"Decompilation failed for {self.binary.binary_name} :: {binary_type} ")            
                l.info(f"Decompilation succesful for {self.binary.binary_name} :: {binary_type}!")
                decompile_results[binary_type] = dec_path

        except Exception as e:
            l.error(f"Error in decompiling {self.binary.binary_name} with {self.decompiler}: {e}")
            return

        self.collect_workdir[self.binary.binary_name] = workdir
        return self.binary.dwarf_dict, decompile_results
    
    def joern_runner(self, dwarf_data, binary_name, decompilation_results, PARSE):   

        workdir = self.collect_workdir[binary_name]
        joern_data_strip, joern_data_type_strip = '', ''
        try:  
            decompiled_code_strip_path = decompilation_results['strip'].decompiled_binary_code_path
            decompiled_code_type_strip_path = decompilation_results['type_strip'].decompiled_binary_code_path
            decompiled_code_type_strip_names = decompilation_results['type_strip'].type_strip_mangled_names
            dwarf_info_path = dwarf_data
            data_map_dump = os.path.join(self.path_manager.match_path, self.decompiler, binary_name)

            if not (os.path.exists(decompiled_code_strip_path) and os.path.exists(decompiled_code_type_strip_path)):
                l.error(f"Decompiled code not found for {binary_name} :: {self.decompiler}")
                return

            # paths for joern
            joern_strip_path = os.path.join(self.path_manager.joern_data_path, self.decompiler, 'strip', (binary_name + '.json'))
            joern_type_strip_path = os.path.join(self.path_manager.joern_data_path, self.decompiler, 'type_strip', (binary_name + '.json'))    
            
            if  PARSE:
                # JOERN
                l.info(f'Joern parsing for {binary_name} :: strip :: {self.decompiler} :: in {workdir}')
                joern_data_strip = JoernParser(binary_name=binary_name, binary_type='strip', decompiler=self.decompiler, 
                                                    decompiled_code=decompiled_code_strip_path, workdir=workdir, port=self.PORT, 
                                                    outpath=joern_strip_path).joern_data
                
                l.info(f'Joern parsing for {binary_name} :: type-strip :: {self.decompiler} :: in {workdir}')
                joern_data_type_strip = JoernParser(binary_name=binary_name, binary_type='type-strip', decompiler=self.decompiler, 
                                                    decompiled_code=decompiled_code_type_strip_path, workdir=workdir, port=self.PORT, 
                                                    outpath=joern_type_strip_path).joern_data
                

            else:   
                joern_data_strip = read_json(joern_strip_path)
                joern_data_type_strip = read_json(joern_type_strip_path)
            
            l.info(f'Joern parsing completed for {binary_name} :: strip :: {self.decompiler} :: {len(joern_data_strip)}')
            l.info(f'Joern parsing completed for {binary_name} :: type-strip :: {self.decompiler} :: {len(joern_data_type_strip)}')


            if not (joern_data_strip and joern_data_type_strip):
                l.info(f"oops! strip/ty-strip joern not found! {binary_name}")
                return

            l.info(f'Start mapping for {binary_name} :: {self.decompiler}')
            
            matchvariable = DataLoader(binary_name, dwarf_info_path, self.decompiler, decompiled_code_strip_path, 
                        decompiled_code_type_strip_path, joern_data_strip, joern_data_type_strip, data_map_dump, self.language, decompiled_code_type_strip_names)


        except Exception as e:
            l.info(f"Error! :: {binary_name} :: {self.decompiler} :: {e} ")


    def run(self, PARSE, splits):

        if PARSE:
            try:               
                joern_progress = tqdm(total=0, desc="Joern Processing")
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.WORKERS) as executor:
                    future_to_binary = {executor.submit(self.decompile_runner, binary): binary for binary in self.target_binaries}
                    # As each future completes, process it with joern_runner
                    # for future in concurrent.futures.as_completed(future_to_binary):
                    for future in tqdm(concurrent.futures.as_completed(future_to_binary), total=len(future_to_binary), desc="Decompiling Binaries"):
                        binary = future_to_binary[future]
                        binary_name = Path(binary).name
                        binary_info, decompilation_results = future.result()
                        if not decompilation_results:
                            l.error(f"No Decompilation results for {binary_name}")
                            return
                        l.info(f"Start Joern for :: {binary_name}")
                        try:
                            joern_progress.total += 1
                            # Setup JAVA_HOME for Joern
                            os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
                            java_env = os.environ.get('JAVA_HOME', None)
                            if java_env != '/usr/lib/jvm/java-11-openjdk-amd64':
                                l.error(f"INCORRECT JAVA VERSION JOERN: {java_env}")
                            with JoernServer(self.path_manager.joern_dir, self.PORT) as joern_server:
                                self.joern_runner(binary_info, binary_name, decompilation_results, PARSE)
                            joern_progress.update(1)
                        except Exception as e:
                            l.error(f"Error in starting Joern server for {binary_name} :: {e}")
                        finally:
                            if joern_server.is_server_running():
                                joern_server.stop()                                
                                l.debug(f"Stop Joern for :: {binary_name}")
                            time.sleep(3)
                
                # dedup functions -> create train and test sets
                joern_progress.close()
                if splits:
                    create_train_and_test_sets( f'{self.path_manager.tmpdir}', self.decompiler)

            except Exception as e:
                l.error(f"Error during parallel decompilation: {e}")

        else:
            create_train_and_test_sets( f'{self.path_manager.tmpdir}', self.decompiler)
            
        # if not debug, delete tmpdir and copy splits to data dir
        source_dir = f'{self.path_manager.tmpdir}/splits/'
        # Get all file paths
        file_paths = glob.glob(os.path.join(source_dir, '**', 'final*.jsonl'), recursive=True)

        for file_path in file_paths:
            # Check if the filename contains 'final'
            if 'final' in os.path.basename(file_path):
                relative_path = os.path.relpath(file_path, start=source_dir)
                new_destination_path = os.path.join(self.path_manager.data_dir, relative_path)
                # Create directories if they don't exist
                os.makedirs(os.path.dirname(new_destination_path), exist_ok=True)
                # Copy the file
                shutil.copy(file_path, new_destination_path)
                l.info(f"Copied {file_path} to {new_destination_path}")
        if self.path_manager.tmpdir and not self.DEBUG:
            shutil.rmtree(self.path_manager.tmpdir)
            l.info(f"Deleted {self.path_manager.tmpdir}")