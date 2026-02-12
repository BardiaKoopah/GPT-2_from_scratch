# Intro

#### Here is my from scratch implementation of GPT-2 based on the paper [Language Models Are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). Vocab and merges file was created with [OpenText dataset](https://huggingface.co/datasets/Skylion007/openwebtext/viewer/plain_text/train), and model was trained on [Wikitext](https://huggingface.co/datasets/mindchain/wikitext2), both v1 and 103 large. I was able to get ~35 valid PPL on the smaller Wikitext dataset, and just a few points lower on the large set. Check out model params below for more details and also notes so that your computer doesn't get fried.


# Model Params

####
- **Layers (Transformer Blocks):** 12
- **d_model:** 768
- **num_heads (Attention Heads):** 12
- **Head Dimension (`d_model / num_heads`):** 64
- **MLP Inner Dimension:** 3072  
- **Vocabulary Size:** 50,257
- **Special Token:** `<|endoftext|>` (document separator)
- **seq_len (Context Length):** 1024
- **Batch Size:** 512
- **Training Sequence Length:** 1024
- **Optimizer:** AdamW (i know the paper said just normal Adam but cmon we got AdamW now)
- **Learning Rate Schedule:**
  - Linear warmup over first 2000 updates
  - Cosine decay to zero
- **Weight Decay:** 0.01  
  - Applied to non-bias parameters
- **Dropout:** 0.1

# Notes

#### Ok very important, the above is the model parameters based ~EXACTLY off the paper, however one iteration of batch size of 512 with a context length of 1024 literally crashed my MPS. Not saying this will happen to you, but I personally started with smaller batch size of 8 and context length of 128 and that gave me 100 batches processed in ~21 seconds, and is how I got the scores mentioned in Intro. If you can afford it, by all means scale up to the stats in Model Params. And if you want to go even further, check out page 4 of the GPT-2 pdf to see other model sizes they played around with (context and batch stayed the same, just changed the d_model and # of layers)



