import os
import tempfile

class PathManager:
    def __init__(self, args):
        # Base paths set by user input
        self.binaries_dir = args.binaries_dir
        self.data_dir = os.path.abspath(args.data_dir)
        self.joern_dir = os.path.abspath(args.joern_dir)
        if not args.tmpdir:
            self.tmpdir = tempfile.mkdtemp(prefix='varbert_tmpdir_', dir='/tmp')
        else:
            self.tmpdir = args.tmpdir
        print("self.tmpdir", self.tmpdir)
        os.makedirs(self.tmpdir, exist_ok=True)  
            
        self.ida_path = args.ida_path
        self.ghidra_path = args.ghidra_path
        self.corpus_language = args.corpus_language

        # Derived paths
        self.strip_bin_dir = os.path.join(self.tmpdir, 'binary/strip')
        self.type_strip_bin_dir = os.path.join(self.tmpdir, 'binary/type_strip')
        self.joern_data_path = os.path.join(self.tmpdir, 'joern')
        self.dc_path = os.path.join(self.tmpdir, 'dc')
        self.failed_path_ida = os.path.join(self.tmpdir, 'failed')
        self.dwarf_info_path = os.path.join(self.tmpdir, 'dwarf')
        self.match_path = os.path.join(self.tmpdir, 'map')
    
    