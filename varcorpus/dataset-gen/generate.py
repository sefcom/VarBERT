import os
import glob
import argparse
import logging.config
import concurrent.futures
from functools import partial
from multiprocessing import Manager
import json

from pathmanager import PathManager
from runner import Runner


def main(args):
    if args.decompiler == 'ida' and not args.ida_path:
        parser.error("IDA path is required when decompiling with IDA.")
    elif args.decompiler == 'ghidra' and not args.ghidra_path:
        parser.error("Ghidra path is required when decompiling with Ghidra.")

    if not args.corpus_language and not args.language_map:
        parser.error("Either --corpus_language or --language_map is required.")
    if args.corpus_language and args.language_map:
        parser.error("Provide only one of --corpus_language or --language_map, not both.")
    if args.language_map and not os.path.isfile(args.language_map):
        parser.error(f"Language map file not found: {args.language_map}")
    if args.corpus_language and args.corpus_language not in {"c", "cpp"}:
        parser.error("Invalid --corpus_language. Allowed values are: c, cpp")

    language_map = None
    if args.language_map:
        try:
            with open(args.language_map, 'r') as f:
                language_map = json.load(f)
        except Exception as e:
            parser.error(f"Failed to read language map JSON: {e}")

    path_manager = PathManager(args)
    target_binaries = [
        os.path.abspath(path)
        for path in glob.glob(os.path.join(path_manager.binaries_dir, '**', '*'), recursive=True)
        if os.path.isfile(path)
    ]
    # Validate language_map if provided
    if language_map is not None:
        # keys are expected to be binary basenames; values 'c' or 'cpp'
        valid_langs = {'c', 'cpp'}
        invalid = {k: v for k, v in language_map.items() if v not in valid_langs}
        if invalid:
            parser.error(f"Invalid languages in language_map (allowed: c, cpp): {invalid}")
        missing = [os.path.basename(p) for p in target_binaries if os.path.basename(p) not in language_map]
        if missing:
            parser.error(f"language_map missing entries for binaries: {missing}")
    decompiler = args.decompiler
    if not args.corpus_language:
        pass
        #TODO:  detect

    splits = True if args.splits else False
    l.info(f"Decompiling {len(target_binaries)} binaries with {decompiler}")

    default_language = None if language_map is not None else args.corpus_language
    effective_language_map = language_map if language_map else {}

    runner = Runner(
        decompiler=args.decompiler,
        target_binaries=target_binaries,
        WORKERS=args.WORKERS,
        path_manager=path_manager,
        PORT=8090,  
        language=default_language,
        DEBUG=args.DEBUG,
        language_map=effective_language_map
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
        required=False,
    )

    parser.add_argument(
        "--language_map",
        type=str,
        help="Path to JSON mapping of binary name to language, e.g. {\"bin1\": \"c\", \"bin2\": \"cpp\"}",
        required=False,
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

