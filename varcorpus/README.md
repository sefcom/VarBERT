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
    -joern <path_to_joern_directory>
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
