### Table of Contents

1. [Training VarBERT](#training-varbert)
2. [Fine-tuning VarBERT](#fine-tune)
3. [Vocab Files](#vocab-files)
4. [Tokenizer](#tokenizer)
5. [Masked Language Modeling (MLM)](#masked-language-modeling-mlm)
6. [Constrained Masked Language Modeling (CMLM) in VarBERT](#constrained-masked-language-modeling-cmlm-in-varbert)
7. [Resize Model](#resize-model)

### Training VarBERT

In our paper, we follow a two-step training process:
- **Pre-training**: VarBERT is pre-trained on source code functions (HSC data set) using Masked Language Modeling (MLM) followed by Constrained Masked Language Modeling (CMLM).
- **Fine-tuning**: Subsequently, VarBERT is fine-tuned on top of the pre-trained model using VarCorpus (decompilation output of IDA and Ghidra) to predict variable names and variable origins (i.e., whether a variable originates from source code or is decompiler-generated).

Training Process Overview (from the paper):

    BERT-Base → MLM → CMLM → Fine-tune


This approach can be adapted for use with any other decompiler capable of generating C-Style decompilation output. Use the pre-trained model from step one and fine-tune with a new (or existing) decompiler. jump to []

**Essential Components for Training:**
1. Base Model: Choose from BERT-Base, MLM Model or CMLM Model. Remember to resize the Base Model if the vocab size on subsequent model is different ([How to Resize Model](#resize-model)). 
2. [Tokenizer](#tokenizer)
3. Train and Test sets
4. [Vocab Files](#vocab-files): Contains the most frequent variable names from your training dataset. 


### Fine-tune
Fine-tuning is generally performed on top of a pre-trained model (MLM + CMLM Model in our case). However, fine-tuning can also be directly applied to an MLM model or a BERT-Base model.
During this phase, the model learns to predict variable names and their origins.
VarBERT predicts the `Top-N` variables (where N can be 1, 3, 5, 10) for variable names. For their origin, the output dwarf indicates a source code origin, while `ida` or `ghidra` suggest that the variable is decompiler-generated.

Access our fine-tuned model from our paper: [Models](https://www.dropbox.com/scl/fo/socl7rd5lsv926whylqpn/h?rlkey=i0x74bdipj41hys5rorflxawo&dl=0)

To train a new model follow these steps:

- **Base Model**: [CMLM Model](https://www.dropbox.com/scl/fi/72ku0tf3o93kn67k60d7d/CMLM_MODEL.tar.gz?rlkey=8kwlfwc87uwcsab86np4bhub0&dl=0)
- **Tokenizer**: Refer to [Tokenizer](#tokenizer)
- **Train and Test sets**: Each VarCorpus data set tarball has both pre-processed and non-processed train and test sets. If you prefer to use pre-processed sets please jump to step 2 (i.e. training a model), otherwise, start with step 1. [VarCorpus](https://www.dropbox.com/scl/fo/3thmg8xoq2ugtjwjcgjsm/h?rlkey=azgjeq513g4semc1qdi5xyroj&dl=0). Alternatively, if you created a new data set using `generate.py` in [Building VarCorpus](./varcorpus/README.md) use files saved in `data dir`
- **Vocab Files**: You can find our vocab files in tarball of each trained model available at [link](https://www.dropbox.com/scl/fo/socl7rd5lsv926whylqpn/h?rlkey=i0x74bdipj41hys5rorflxawo&dl=0) or refer to [Vocab Files](#vocab-files) to create new vocab files.


1. Preprocess data set for training

```python
python3 preprocess.py \
--train_file <path_to_final_train_file.jsonl> \
--test_file <path_to_final_test_file.jsonl> \
--tokenizer <path_to_tokenizer> \
--vocab_word_to_idx  <path_to_vocabfiles/word_to_idx.json> \
--vocab_idx_to_word  <path_to_vocabfiles/idx_to_word.json> \
--decompiler <ida_or_ghidra> \
--max_chunk_size 800 \
--workers 2 \
--out_train_file <path_of_preprocessed_files/preprocessed_train.json> \
--out_test_file <path_of_preprocessed_files/preprocessed_test.json>
```

2. Fine-tune a model


```python
python3 -m torch.distributed.launch --nproc_per_node=2 training.py \
--overwrite_output_dir \
--train_data_file <path_of_preprocessed_files/preprocessed_train.json> \
--output_dir <path_to_save_trained_model> \
--block_size 800 \
--tokenizer_name <path_to_tokenizer> \
--model_type roberta \
--model_name_or_path <path_of_cmlm_model> \
--vocab_path <path_to_vocabfiles/word_to_idx.json> \
--do_train \
--num_train_epochs 4 \
--save_steps 10000 \
--logging_steps 1000 \
--per_gpu_train_batch_size 4 \
--mlm;
```
This base model path can be any model. Either previously trained CMLM model or MLM model or BERT-base model.

Note: To run training without distributed, use `python3 training.py`, with the same arguments.

3. Run Inference

```python
python3 eval.py \
--model_name <path_to_ft_model> \
--tokenizer_name <path_to_tokenizer> \
--block_size 800 \
--data_file <path_of_preprocessed_files/preprocessed_test.json> \
--prefix ft_bintoo_test \
--batch_size 16 \
--pred_path resultdir \
--out_vocab_map <path_to_vocabfiles/idx_to_word.json>
```


### Vocab Files
Vocab files consist of the top N most frequently occurring variable names from the chosen training dataset. Specifically, we use the top 50K variable names from the Human Source Code (HSC) dataset and the top 150K variable names from the VarCorpus dataset. These are variables our model learns upon. 

(We use `50001` and `150001` as vocab size for CMLM model and Fine-tuning respectively. top 50K variable names + 1 for UNK)

Use existing vocab files from our paper:

Please note that we have different vocab files for each model. You can find our vocab files in tarball of each trained model available at [link](https://www.dropbox.com/scl/fo/socl7rd5lsv926whylqpn/h?rlkey=i0x74bdipj41hys5rorflxawo&dl=0). 

To create new vocab files:

```python
python3 generate_vocab.py \
    --dataset_type <hsc_or_varcorpus> \
    --train_file <path_to_train_file.jsonl> \
    --test_file <path_to_test_file.jsonl> \
    --vocab_size <vocab_size> \
    --output_dir <path_to_save_generated_vocab_files> \
```


### Tokenizer
To adapt model to the unique nature of the new data (source code), we use a Byte-Pair Encoding (BPE) tokenizer to learn a new source vocabulary. We train a tokenizer with a vocabulary size of 50K, similar to RoBERTa's 50,265.

Use tokenizer trained on HSC data set (train set): [Tokenizer](https://www.dropbox.com/scl/fi/i8seayujpqdc0egavks18/tokenizer.tar.gz?rlkey=fnhorh3uo2diqv0v1qaymzo2r&dl=0)
```bash
wget -O tokenizer.tar.gz https://www.dropbox.com/scl/fi/i8seayujpqdc0egavks18/tokenizer.tar.gz?rlkey=fnhorh3uo2diqv0v1qaymzo2r&dl=0
``` 

Training a new Tokenizer:
1. Prepare train set: The input to tokenizer should be in text format.
(If using [HSC data set](https://www.dropbox.com/scl/fi/1eekwcsg7wr7cux6y34xb/hsc_data.tar.gz?rlkey=s3kjroqt7a27hoeoc56mfyljw&dl=0))

```python
python3 preprocess.py \
--input_file <path_to_hsc_files/train.jsonl> \
--output_file <path_to_processed_hsc_file_for_tokenization/train.txt> 
```

2. Training Tokenizer

```python
python3 train_bpe_tokenizer.py \
 --input_path <path_to_processed_hsc_file_for_tokenization/train.txt> \
 --vocab_size 50265 \
 --min_frequency 2 \
 --output_path <path_to_tokenizer>
```


**To pre-train model from scratch**

### Masked Language Modeling (MLM)

Learn the representation of code tokens using BERT from scratch through a Masked Language Modeling approach similar to the one used in RoBERTa. In this process, some tokens are randomly masked, and the model learns to predict these masked tokens, thereby gaining a deeper understanding of code-token representations.

#### Using a Pre-trained MLM Model

Access our pre-trained MLM model, trained on 5.2M functions, from our paper: [MLM Model](https://www.dropbox.com/scl/fi/72ku0tf3o93kn67k60d7d/CMLM_MODEL.tar.gz?rlkey=8kwlfwc87uwcsab86np4bhub0&dl=0)

#### To train a new model, follow these steps:

1. Preprocess data set for training
[HSC data set](https://www.dropbox.com/scl/fi/1eekwcsg7wr7cux6y34xb/hsc_data.tar.gz?rlkey=s3kjroqt7a27hoeoc56mfyljw&dl=0)

```python
python3 preprocess.py \
--train_file <path_to_hsc_files/train.jsonl> \
--test_file <path_to_hsc_files/test.jsonl> \
--output_train_file <path_to_processed_hsc_file_for_mlm/train.json> \
--output_test_file <path_to_processed_hsc_file_for_mlm/test.json>
```

2. Train MLM Model

- The MLM model is trained on top of the BERT-Base model using pre-processed HSC train and test files from step 1.
- For the BERT-Base model, refer to [BERT-Base Model](https://www.dropbox.com/scl/fi/18p37f5drph8pekv8kcj2/BERT_Base.tar.gz?rlkey=3x4mpr4hmkyndunhg9fpu0p3b&dl=0)


```python
python3 training.py \
    --model_name_or_path <BERT_Base_Model> \
    --model_type roberta \
    --tokenizer_name <path_to_tokenizer> \
    --train_file <path_to_processed_hsc_file_for_mlm/train.json> \
    --validation_file <path_to_processed_hsc_file_for_mlm/test.json>  \
    --max_seq_length 800 \
    --mlm_probability 0.15 \
    --num_train_epochs 40 \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 44 \
    --per_device_eval_batch_size 44 \
    --output_dir <path_to_save_mlm_model> \
    --save_steps 10000 \
    --logging_steps 5000 \
    --overwrite_output_dir
```

### Constrained Masked Language Modeling (CMLM)
Constrained MLM is a variation of MLM. In this approach, tokens are not randomly masked; instead, we specifically mask certain tokens, which in our case are variable names in source code functions.

#### Using a Pre-trained CMLM Model:
Access our pre-trained CMLM model from the paper: [CMLM Model](https://www.dropbox.com/scl/fi/72ku0tf3o93kn67k60d7d/CMLM_MODEL.tar.gz?rlkey=8kwlfwc87uwcsab86np4bhub0&dl=0)

To train a new model follow these steps:
- **Base Model**: [MLM Model](https://www.dropbox.com/scl/fi/a0i61xeij0bogkusr4yf7/MLM_MODEL.tar.gz?rlkey=pqenu7f851sgdn6ofcfp6dxoa&dl=0)
- **Tokenizer**: Refer to [Tokenizer](#tokenizer)
- **Train and Test sets**: [CMLM Data set](https://www.dropbox.com/scl/fi/q0itko6fitpxx3dx71qhv/cmlm_dataset.tar.gz?rlkey=51j9iagvg8u3rak79euqjocml&dl=0)
- **Vocab Files**: To generate new vocab refer to [Vocab Files](#vocab-files) or use exisiting at [link](https://www.dropbox.com/scl/fi/yot7urpeem53dttditg7p/cmlm_dataset.tar.gz?rlkey=cned7sgijladr1pu5ery82z8a&dl=0)


1. Preprocess data set for training
[HSC CMLM data set](https://www.dropbox.com/scl/fi/q0itko6fitpxx3dx71qhv/cmlm_dataset.tar.gz?rlkey=51j9iagvg8u3rak79euqjocml&dl=0)

```python
python3 preprocess.py \
    --train_file <path_to_hsc_files/train.jsonl> \
    --test_file <path_to_hsc_files/test.jsonl> \
    --tokenizer <path_to_tokenizer>  \
    --vocab_word_to_idx <path_to_vocabfiles/word_to_idx.json> \
    --vocab_idx_to_word <path_to_vocabfiles/idx_to_word.json> \
    --max_chunk_size 800 \
    --out_train_file <path_to_processed_hsc_file_for_cmlm/train.json> \
    --out_test_file <path_to_processed_hsc_file_for_cmlm/test.json> \
    --workers 4 \
    --vocab_size 50001
```

2. Train CMLM Model
The CMLM model is trained on top of the MLM model using pre-processed HSC train and test files from step 1.
[MLM Model](https://www.dropbox.com/scl/fi/a0i61xeij0bogkusr4yf7/MLM_MODEL.tar.gz?rlkey=pqenu7f851sgdn6ofcfp6dxoa&dl=0)

```python
python3 training.py  \
 --overwrite_output_dir \
 --train_data_file <path_to_processed_hsc_file_for_cmlm/train.json> \
 --output_dir <path_to_save_trained_cmlm_model> \
 --block_size 800 \
 --model_type roberta \
 --model_name_or_path <path_to_mlm_model> \
 --tokenizer_name <path_to_tokenizer> \
 --do_train \
 --num_train_epochs 30 \
 --save_steps 50000 \
 --logging_steps 5000 \
 --per_gpu_train_batch_size 32 \
 --mlm;
```

3. Run Inference

```python
python run_cmlm_scoring.py \
 --model_name <path_to_save_trained_cmlm_model> \
 --data_file <path_to_processed_hsc_file_for_cmlm/test.json> \
 --prefix cmlm_hsc_5M_50K \
 --pred_path <path_to_save_results> \
 --batch_size 40;
```

### Resize Model

It's necessary to resize the model when the vocab size changes. For example, if you're fine-tuning over a CMLM model initially trained with a 50K vocab size using a 150K vocab size for the fine-tuned, the CMLM model needs to be resized.

```python
python3 resize_model.py \
    --old_model <path_to_old_model> \ 
    --vocab_path <path_to_new_vocab_file/word_to_idx.json> \ 
    --out_model_path <path_to_new_model>
```