import os
import glob
import argparse
import logging.config
import concurrent.futures
from functools import partial
from multiprocessing import Manager

from pathmanager import PathManager
from runner import Runner


def main(args):
    if args.decompiler == 'ida' and not args.ida_path:
        parser.error("IDA path is required when decompiling with IDA.")
    elif args.decompiler == 'ghidra' and not args.ghidra_path:
        parser.error("Ghidra path is required when decompiling with Ghidra.")

    path_manager = PathManager(args)
    target_binaries = [
        os.path.abspath(path)
        for path in glob.glob(os.path.join(path_manager.binaries_dir, '**', '*'), recursive=True)
        if os.path.isfile(path)
    ]
    decompiler = args.decompiler
    if not args.corpus_language:
        pass
        #TODO:  detect

    splits = True if args.splits else False
    l.info(f"Decompiling {len(target_binaries)} binaries with {decompiler}")

    runner = Runner(
        decompiler=args.decompiler,
        target_binaries=target_binaries,
        WORKERS=args.WORKERS,
        path_manager=path_manager,
        PORT=8090,  
        language=args.corpus_language,
        DEBUG=args.DEBUG
    )

    runner.run(PARSE=True, splits=splits)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data set generation for VarBERT")

    parser.add_argument(
        "-b", "--binaries_dir",
        type=str,
        help="Path to binaries dir",
        required=True,
    )

    parser.add_argument(
        "-d", "--data_dir",
        type=str,
        help="Path to data dir",
        required=True,
    )

    parser.add_argument(
        "--tmpdir",
        type=str,
        help="Path to save intermediate files. Default is /tmp",
        required=False,
        default="/tmp/varbert_tmpdir"
    )

    parser.add_argument(
        "--decompiler",
        choices=['ida', 'ghidra'],
        type=str,
        help="choose decompiler IDA or Ghidra",
        required=True
    )

    parser.add_argument(
        "-ida", "--ida_path",
        type=str,
        help="Path to IDA",
        required=False,
    )

    parser.add_argument(
        "-ghidra", "--ghidra_path",
        type=str,
        help="Path to Ghidra",
        required=False,
    )

    parser.add_argument(
        "-joern", "--joern_dir",
        type=str,
        help="Path to Joern",
        required=False,
    )

    # TODO: if language not given, maybe detect it
    parser.add_argument(
        "-lang",  "--corpus_language",
        type=str,
        help="Corpus language",
        required=True,
    )


    parser.add_argument(
        "-w",  "--WORKERS",
        type=int,
        help="Number of workers",
        default=2,  
        required=False
    )

    parser.add_argument(
        "--DEBUG",
        help="Turn on debug logging mode", 
        action='store_true',
        required=False
    )
    parser.add_argument(
        "--splits",
        help="Create test and train split",
        action='store_true',
        required=False
    )
    args = parser.parse_args()
    from log import setup_logging
    setup_logging(args.tmpdir, args.DEBUG)
    l = logging.getLogger('main')
    main(args)

