from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaTokenizerFast

import argparse
import os

def main(agrs):
    paths = [str(x) for x in Path(args.input_path).glob("**/*.txt")]

    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer(add_prefix_space=True)

    # Customize training
    tokenizer.train(files=paths, 
                    vocab_size=args.vocab_size, 
                    min_frequency=args.min_frequency, 
                    special_tokens=[   "<s>",
                                        "<pad>",
                                        "</s>",
                                        "<unk>",
                                        "<mask>",
                                    ])
    os.makedirs(args.output_path,exist_ok=True)
    tokenizer.save_model(args.output_path)
    
    
    # Test the tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained(args.output_path, max_len=1024)
    ghidra_func = "undefined * FUN_001048d0(void)\n{\n  undefined8 uVar1;\n  long lVar2;\n  undefined *puVar3;\n  \n  uVar1 = PyState_FindModule(&readlinemodule);\n  lVar2 = PyModule_GetState(uVar1);\n  if (*(lVar2 + 0x18) == 0) {\n    FUN_00103605(&_Py_NoneStruct);\n    puVar3 = &_Py_NoneStruct;\n  }\n  else {\n    uVar1 = PyState_FindModule(&readlinemodule);\n    lVar2 = PyModule_GetState(uVar1);\n    FUN_00103605(*(lVar2 + 0x18));\n    uVar1 = PyState_FindModule(&readlinemodule);\n    lVar2 = PyModule_GetState(uVar1);\n    puVar3 = *(lVar2 + 0x18);\n  }\n  return puVar3;\n}"
    
    tokens = tokenizer.tokenize(ghidra_func)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    # print("Normal Tokens:",str(tokens))
    # print("Normal Token Ids:",str(token_ids))
    
    human_func = 'int fast_s_mp_sqr(const mp_int *a, mp_int *b)\n{\n   int       olduse, res, pa, ix, iz;\n   mp_digit   W[MP_WARRAY], *tmpx;\n   mp_word   W1;\n   /* grow the destination as required */\n   pa = a->used + a->used;\n   if (b->alloc < pa) {\n      if ((res = mp_grow(b, pa)) != 0 /* no error, all is well */) {\n         return res;\n      }\n   }\n   /* number of output digits to produce */\n   W1 = 0;\n   for (ix = 0; ix < pa; ix++) {\n      int      tx, ty, iy;\n      mp_word  _W;\n      mp_digit *tmpy;\n      /* clear counter */\n      _W = 0;\n      /* get offsets into the two bignums */\n      ty = MIN(a->used-1, ix);\n      tx = ix - ty;\n      /* setup temp aliases */\n      tmpx = a->dp + tx;\n      tmpy = a->dp + ty;\n      /* this is the number of times the loop will iterrate, essentially\n         while (tx++ < a->used && ty-- >= 0) { ... }\n       */\n      iy = MIN(a->used-tx, ty+1);\n      /* now for squaring tx can never equal ty\n       * we halve the distance since they approach at a rate of 2x\n       * and we have to round because odd cases need to be executed\n       */\n      iy = MIN(iy, ((ty-tx)+1)>>1);\n      /* execute loop */\n      for (iz = 0; iz < iy; iz++) {\n         _W += (mp_word)*tmpx++ * (mp_word)*tmpy--;\n      }\n      /* double the inner product and add carry */\n      _W = _W + _W + W1;\n      /* even columns have the square term in them */\n      if (((unsigned)ix & 1u) == 0u) {\n         _W += (mp_word)a->dp[ix>>1] * (mp_word)a->dp[ix>>1];\n      }\n      /* store it */\n      W[ix] = _W & ((((mp_digit)1)<<((mp_digit)MP_DIGIT_BIT))-((mp_digit)1));\n      /* make next carry */\n      W1 = _W >> (mp_word)(CHAR_BIT*sizeof(mp_digit));\n   }\n   /* setup dest */\n   olduse  = b->used;\n   b->used = a->used+a->used;\n   {\n      mp_digit *tmpb;\n      tmpb = b->dp;\n      for (ix = 0; ix < pa; ix++) {\n         *tmpb++ = W[ix] & ((((mp_digit)1)<<((mp_digit)MP_DIGIT_BIT))-((mp_digit)1));\n      }\n      /* clear unused digits [that existed in the old copy of c] */\n      for (; ix < olduse; ix++) {\n         *tmpb++ = 0;\n      }\n   }\n   mp_clamp(b);\n   return 0 /* no error, all is well */;\n}'
    
    tokens = tokenizer.tokenize(human_func)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    # print("Normal Tokens:",str(tokens))
    # print("Normal Token Ids:",str(token_ids))
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help='path to the input text files')
    parser.add_argument('--vocab_size', type=int, default=50265, help='size of tokenizer vocabulary')
    parser.add_argument('--min_frequency', type=int, default=2, help='minimum frequency')
    parser.add_argument('--output_path', type=str, help='path to the output text files')


    args = parser.parse_args()

    