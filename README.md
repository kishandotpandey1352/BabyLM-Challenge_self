# BabyLM-Challenge

Team NA1 model implementation for the BabyLM Challenge. This repo trains a GPT-2 style decoder with a proxy-driven curriculum scheduler, then exports the model in a Hugging Face compatible format for inference.

Note: Large artifacts (datasets, tokenizers, logs, environments) are intentionally ignored in git. You will need to supply or generate your own tokenized data files.

## What is in this repo
- A GPT-2 style decoder and training loop in [models/gpt.py](models/gpt.py).
- Proxy model training and curriculum scheduling in [main/proxy_main.py](main/proxy_main.py) and [curriculum/scheduler.py](curriculum/scheduler.py).
- Full model training in [main/gpt_model.py](main/gpt_model.py).
- Export to Hugging Face format in [export_custom_to_hf_compat.py](export_custom_to_hf_compat.py).

## Training overview
1) Tokenize your data into a `.pkl` file and place it under `tokenizers/`.
2) Run proxy training to compute learnability scores.
3) Train the main GPT model using the curriculum scheduler (or baseline training without it).
4) Export trained weights to Hugging Face format for inference.

## Creating tokenized `.pkl` files
This repo expects a pickled object containing tokenized data that `data/loader.py` can read. A minimal workflow is:

1) Prepare a plain text file (one sample per line).
2) Use a tokenizer to convert text to token IDs.
3) Save the token IDs list to a `.pkl` file in `tokenizers/`.

Example script (adjust paths as needed):

```python
import pickle
from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

tokens = []
with open("data/train.txt", "r", encoding="utf-8") as f:
   for line in f:
      line = line.strip()
      if not line:
         continue
      ids = tokenizer.encode(line)
      tokens.extend(ids)

with open("tokenizers/10M_data_token.pkl", "wb") as f:
   pickle.dump(tokens, f)
```

If you already have a tokenized dataset in another format, convert it to a list of token IDs and pickle it the same way.

### Proxy training
The proxy model learns on a holdout split and produces per-sample scores used to order the curriculum.

```bash
python main/proxy_main.py \
   --data_path tokenizers/10M_data_token.pkl \
   --data_size 10M \
   --n_tokens 500000
```

This writes proxy weights and score files to `trained_models/`.

### Main model training
The main model can use the curriculum (scores from proxy) or run as a baseline.

```bash
# Curriculum training
python main/gpt_model.py \
   --data_path tokenizers/10M_data_token.pkl \
   --scoring comp_score \
   --curriculum on \
   --data_size 10M \
   --n_tokens 500000

# Baseline training
python main/gpt_model.py \
   --data_path tokenizers/10M_data_token.pkl \
   --scoring loss \
   --curriculum off \
   --data_size 10M \
   --n_tokens 500000
```

Model weights are saved to `trained_models/`.

## Export to Hugging Face format
After training, export to a Hugging Face compatible directory:

```bash
python export_custom_to_hf_compat.py
```

This writes a model and tokenizer to `hf_main_baseline_10M/`.

## Next-word generation (inference)
Use the exported folder with `transformers` to generate the next token(s):

```python
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

model = GPT2LMHeadModel.from_pretrained("hf_main_baseline_10M")
tokenizer = GPT2TokenizerFast.from_pretrained("hf_main_baseline_10M")

prompt = "The quick brown fox"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
      **inputs,
      max_new_tokens=1,
      do_sample=False
)

print(tokenizer.decode(outputs[0]))
```

Set `max_new_tokens` higher to generate longer continuations.

## HPC notes
- Adjust the batch scripts under `scripts/` for your username and environment paths.
- If using CUDA, ensure your PyTorch build matches the CUDA version on the machine.

If you run into issues, open an issue or ping the author.
