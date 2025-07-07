# VarBERT
VarBERT is a BERT-based model which predicts meaningful variable names and variable origins in decompiled code. Leveraging the power of transfer learning, VarBERT can help you in software reverse engineering tasks. VarBERT is pre-trained on 5M human-written source code functions, and then it is fine-tuned on decompiled code from IDA and Ghidra, spanning four compiler optimizations (*O0*, *O1*, *O2*, *O3*). 
We built two data sets: (a) Human Source Code data set (HSC) and (b) VarCorpus (for IDA and Ghidra).
This work is developed for IEEE S&P 2024 paper ["Len or index or count, anything but v1": Predicting Variable Names in Decompilation Output with Transfer Learning](https://www.atipriya.com/files/papers/varbert_oakland24.pdf) 

Key Features

- Pre-trained on 5.2M human-written source code functions.
- Fine-tuned on decompiled code from IDA and Ghidra.
- Supports four compiler optimizations: O0, O1, O2, O3. 
- Achieves an accuracy of 54.43% for IDA and 54.49% for Ghidra on O2 optimized binaries.
- A total of 16 models are available, covering two decompilers, four optimizations, and two splitting strategies.

### Table of Contents
- [Overview](#overview)
- [VarBERT Model](#varbert-model)
- [Using VarBERT](#use-varbert)
- [Training and Inference](#training-and-inference)
- [Data sets](#data-sets)
- [Installation Instructions](#installation)
- [Cite](#citing)

### Overview
This repository contains details on generating a new dataset, and training and running inference on existing VarBERT models from the paper. To use VarBERT models in your day-to-day reverse engineering tasks, please refer to [Use VarBERT](#use-varbert). 


### VarBERT Model
We take inspiration for VARBERT from the concepts of transfer learning generally and specifically Bidirectional Encoder Representations from Transformers (BERT).

- **Pre-training**: VarBERT is pre-trained on HSC functions using Masked Language Modeling (MLM) and Constrained Masked Language Modeling (CMLM).
- **Fine-tuning**: VarBERT is then further fine-tuned on top of the previously pre-trained model using VarCorpus (decompilation output of IDA and Ghidra). It can be further extended to any other decompiler capable of generating C-Style decompilation output.

### Use VarBERT
- The VarBERT API is a Python library to access and use the latest models. It can be used in three ways:
    1. From the CLI, directly on decompiled text (without an attached decompiler).
    2. As a scripting library.
    3. As a decompiler plugin with [DAILA](https://github.com/mahaloz/DAILA) for enhanced decompiling experience.

For a step-by-step guide and a demo on how to get started with the VarBERT API, please visit [VarBERT API](https://github.com/binsync/varbert_api/tree/main). 

### Training and Inference
For training a new model or running inference on existing models, see our detailed guide at [Training VarBERT](./varbert/README.md)

Models available for download:
- [Pre-trained models](https://www.dropbox.com/scl/fo/anibfmk6j8xkzi4nqk55f/h?rlkey=fw6ops1q3pqvsbdy5tl00brpw&dl=0)
- [Fine-tuned models](https://www.dropbox.com/scl/fo/socl7rd5lsv926whylqpn/h?rlkey=i0x74bdipj41hys5rorflxawo&dl=0)

(A [README](https://www.dropbox.com/scl/fi/13s9z5z08u245jqdgfsdc/readme.md?rlkey=yjo33al04j1d5jrwc5pz2hhpz&dl=0) containing all the necessary links for the model is also available.)

### Data sets
- **HSC**: Collected from C source files from the Debian APT repository, totaling 5.2M functions.

- **VarCorpus**: Decompiled functions from C and C++ binaries, built from Gentoo package repository for four compiler optimizations: O0, O1, O2, and O3.

Additionally, we have two splits: (a) Function Split (b) Binary Split.
- Function Split: Functions are randomly distributed between the test and train sets.
- Binary Split: All functions from a single binary are exclusively present in either the test set or the train set.
To create a new data, follow detailed instuctions at [Building VarCorpus](./varcorpus/README.md)

Data sets available at:
- [HSC](https://www.dropbox.com/scl/fo/4cu2fmuh10c4wp7xt53tu/h?rlkey=mlsnkyed35m4rl512ipuocwtt&dl=0)
- [VarCorpus](https://www.dropbox.com/scl/fo/3thmg8xoq2ugtjwjcgjsm/h?rlkey=azgjeq513g4semc1qdi5xyroj&dl=0)


The fine-tuned models and their corresponding datasets are named `IDA-O0-Function` and `IDA-O0`, respectively. This naming convention indicates that the models and data set are based on functions decompiled from O0 binaries using the IDA decompiler.

> [!NOTE]
> Our existing data sets have been generated using IDA Pro 7.6 and Ghidra 10.4.

You can access the Gentoo binaries used to create VarCorpus here: https://www.dropbox.com/scl/fo/awtitjnc48k224373vcrx/h?rlkey=muj6t1watc6vn2ds6du7egoha&e=1&st=eicpyqln&dl=0

### Installation
Prerequisites for training model or generating data set

    Linux with Python 3.8 or higher
    torch ≥ 1.9.0
    transformers ≥ 4.10.0

#### Docker

```
docker build -t . varbert
```

#### Without Docker
```bash
pip install -r requirements.txt

# joern requires Java 11
sudo apt-get install openjdk-11-jdk

# Ghidra 10.4 requires Java 17+
sudo apt-get install openjdk-17-jdk

git clone git@github.com:rhelmot/dwarfwrite.git
cd dwarfwrite
pip install .
```
Note: Ensure you install the correct Java version required by your specific Ghidra version.



### Citing
```
TODO
```
