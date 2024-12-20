### Building VarCorpus

To build VarCorpus, we collected C and C++ packages from Gentoo and built them across four compiler optimizations (O0, O1, O2 and O3) with debugging information (-g enabled), using [Bintoo](https://github.com/sefcom/bintoo).

The script reads binaries from a directory, generates type-stripped and stripped binaries, decompiles and parses them using Joern to match variables, and saves deduplicated training and testing sets for both function and binary split in a data directory. This pipeline supports two decompilers, IDA and Ghidra; it can be extended to any decompiler that generates C-style decompilation output.

To generate the dataset, run the following command, replacing placeholders with correct paths and options:

```python
python3 generate.py \
    -b <path_to_binaries_directory> \
    -d <path_to_save_train_and_test_sets> \
    --decompiler <ida_or_ghidra> \
    -lang <C_or_CPP> \
    -ida <path_to_idat64> or -ghidra <path_to_analyzeHeadless> \
    -w <number_of_parallel_workers> \
    -joern <path_to_joern_directory> \
    --splits
```

The current implementation relies on a specific modified version of [Joern](https://github.com/joernio/joern). We made this modification to expedite the data set creation process. Please download the compatible Joern version from [Joern](https://www.dropbox.com/scl/fi/toh6087y5t5xyln47i5ih/modified_joern.tar.gz?rlkey=lfvjn1u7zvtp9a4cu8z8vgsof&dl=0) and save it. 

```
wget -O joern.tar.gz https://www.dropbox.com/scl/fi/toh6087y5t5xyln47i5ih/modified_joern.tar.gz?rlkey=lfvjn1u7zvtp9a4cu8z8vgsof&dl=0
tar xf joern.tar.gz
```

For `-joern`, provide the path to directory with joern executable from the downloaded version.
```
-joern <download_path>/joern/
```

### Use Docker

You can skip the setup and use our Dockerfile directly to build data set. 

Have your binaries directory and a output directory ready on your host machine. These will be mounted into the container so you can easily provide binaries and retrieve train and test sets.

####  For Ghidra:

```
docker build -t varbert -f ../../Dockerfile ..

docker run -it \
    -v $PWD/<binaries_dir>:/varbert_workdir/data/binaries \
    -v $PWD/<dir_to_save_sets>:/varbert_workdir/data/sets \
    varbert  \
    python3 /varbert_workdir/VarBERT/varcorpus/dataset-gen/generate.py \
    -b /varbert_workdir/data/binaries \
    -d /varbert_workdir/data/sets \
    --decompiler ghidra  \
    -lang <C_or_CPP> \
    -ghidra /varbert_workdir/ghidra_10.4_PUBLIC/support/analyzeHeadless  \
    -w <number_of_parallel_workers>  \
    -joern /varbert_workdir/joern \
    --splits
```

To enable debug mode add `--DEBUG` arg and mount a tmp directory from host to see intermediate files:

```
docker run -it \
    -v $PWD/<binaries_dir>:/varbert_workdir/data/binaries \
    -v $PWD/<dir_to_save_sets>:/varbert_workdir/data/sets \
    -v $PWD/<dir_to_save_tmpdir>:/tmp/varbert_tmpdir \
    varbert  \
    python3 /varbert_workdir/VarBERT/varcorpus/dataset-gen/generate.py \
    -b /varbert_workdir/data/binaries \
    -d /varbert_workdir/data/sets \
    --decompiler ghidra  \
    -lang <C_or_CPP> \
    -ghidra /varbert_workdir/ghidra_10.4_PUBLIC/support/analyzeHeadless  \
    -w <number_of_parallel_workers>  \
    --DEBUG  \
    -joern /varbert_workdir/joern \
    --splits
```

What this does:

-  `-v $PWD/<binaries_dir>:/varbert_workdir/data/binaries`: Mounts your local binaries directory into the container.
-  `-v $PWD/<dir_to_save_sets>:/varbert_workdir/data/sets`: Mounts the directory where you want to save the generated train/test sets.

Inside the container, your binaries are accessible at `/varbert_workdir/data/binaries`, resulting data sets will be saved to `/varbert_workdir/data/sets` and intermediate files are available at `/tmp/varbert_tmpdir`.



#### For IDA:

Please update Dockerfile to include your IDA and run. 

```
docker run -it \
    -v $PWD/<binaries_dir>:/varbert_workdir/data/binaries \
    -v $PWD/<dir_to_save_sets>:/varbert_workdir/data/sets \
    varbert  \
    python3 /varbert_workdir/VarBERT/varcorpus/dataset-gen/generate.py \
    -b /varbert_workdir/data/binaries \
    -d /varbert_workdir/data/sets \
    --decompiler ida  \
    -lang <C_or_CPP> \
    -ida <path_to_idat>  \
    -w <number_of_parallel_workers>  \
    -joern /varbert_workdir/joern \
    --splits
```


#### Notes:

- The train and test sets are split in an 80:20 ratio. If there aren't enough functions (or binaries) to meet this ratio, you may end up with no train or test sets after the run.
- We built the dataset using **Ghidra 10.4**. If you wish to use a different version of Ghidra, please update the Ghidra download link in the Dockerfile accordingly.
- In some cases there is a license popup which should be accepted before you can successfully run IDA in docker.
- Disable type casts for more efficient variable matching. (we disabled it while building VarCorpus).


Sample Function:
```json
{   
    "id": 5,
    "language": "C",
    "func": "__int64 __fastcall sub_13DF(_QWORD *@@var_1@@a1@@, _QWORD *@@var_0@@Ancestors@@)\n{\n  while ( @@var_0@@Ancestors@@ )\n  {\n    if ( @@var_0@@Ancestors@@[1] == @@var_1@@a1@@[1] && @@var_0@@Ancestors@@[2] == *@@var_1@@a1@@ )\n      return 1LL;\n    @@var_0@@Ancestors@@ = *@@var_0@@Ancestors@@;\n  }\n  return 0LL;\n}",
    "type_stripped_vars": {"Ancestors": "dwarf", "a1": "ida"},
    "stripped_vars": ["a2", "a1"],
    "mapped_vars": {"a2": "Ancestors", "a1": "a1"},
    "func_name_dwarf": "is_ancestor",
    "hash": "2998f4a10a8f052257122c23897d10b7",     
    "func_name": "8140277b36ef8461df62b160fed946cb_(00000000000013DF)",
    "norm_func": "__int64 __fastcall sub_13DF(_QWORD *@@var_1@@a1@@, _QWORD *@@var_0@@ancestors@@)\n{\n  while ( @@var_0@@ancestors@@ )\n  {\n    if ( @@var_0@@ancestors@@[1] == @@var_1@@a1@@[1] && @@var_0@@ancestors@@[2] == *@@var_1@@a1@@ )\n      return 1LL;\n    @@var_0@@ancestors@@ = *@@var_0@@ancestors@@;\n  }\n  return 0LL;\n}",
    "vars_map": [["Ancestors", "ancestors"]], 
    "fid": "5-8140277b36ef8461df62b160fed946cb-(00000000000013DF)"
}
```
