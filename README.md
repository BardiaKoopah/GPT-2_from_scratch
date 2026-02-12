# Intro

#### Here is my from scratch implementation of GPT-2 based on the paper [Language Models Are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). Vocab and merges file was created with [OpenText dataset](https://huggingface.co/datasets/Skylion007/openwebtext/viewer/plain_text/train), and model was trained on [Wikitext](https://huggingface.co/datasets/mindchain/wikitext2), both v1 and 103 large. I was able to get ~35 valid PPL on the smaller Wikitext dataset, and just a few points lower on the large set. Check out GPT-2 model params below for more details and also notes so that your computer doesn't get fried.


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
- **Training Sequence Length:** 128
- **Optimizer:** AdamW (i know the paper said just normal Adam but cmon we got AdamW now)
- **Learning Rate Schedule:**
  - Linear warmup over first 2000 updates
  - Cosine decay to zero
- **Weight Decay:** 0.01  
  - Applied to non-bias parameters
- **Dropout:** 0.1
