# lfm2-vl-finetune-guide

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Absurd7550/lfm2-vl-finetune-guide/blob/main/train_colab.ipynb)

Colab guide for fine-tuning **LiquidAI/LFM2-VL-1.6B** with **LoRA (PEFT)** and **4-bit quantization**.

---

## Run it in Colab

1) Click **Open in Colab** (button above).
2) In Colab: **Runtime → Change runtime type → GPU**.
3) Run cells top-to-bottom. When prompted, upload your `train_data.zip`.
4) After training, download `lfm2_adapter.zip`.  
   Optional: upload `probe.7z` (or any `.7z` with images, e.g. `probe.7z` / `prob.7z`, or upload images directly) to test and get `preds.jsonl`.
---
Latest release: https://github.com/Absurd7550/lfm2-vl-finetune-guide/releases/tag/v1.0
## What you will get

- A small LoRA adapter you can share:
  - `out/final_adapter/` (folder)
  - `lfm2_adapter.zip` (archive)
- A working inference recipe:
  - Base model: `LiquidAI/LFM2-VL-1.6B`
  - + your adapter
  - Run on new images

---

## Requirements

- Google Colab with GPU (recommended).
- Your dataset packed as a ZIP archive (`train_data.zip`) in the format below.

---

## Dataset format

Create `train_data.zip` with this structure:
dataset/
images/
00001.jpg
00002.jpg
...
metadata.jsonl



`metadata.jsonl` must be **JSON Lines** (one JSON object per line):

```json
{"file_name":"00001.jpg","text":"Your target text / caption / answer"}
{"file_name":"00002.jpg","text":"..."}
Notes:

file_name must exist inside dataset/images/.
text is what you want the model to learn (captions, instructions, structured JSON answers, etc.).
Start small (200–1000 examples) and iterate.
A minimal example is provided in dataset_example/.

Why this notebook is stable
This guide uses a training setup discovered through trial-and-error that avoids common pitfalls:

Correct model class for training: Lfm2VlForConditionalGeneration (not AutoModel).
Avoids fragile vision patches (no SigLIP2 hacks).
Forces single-crop by resizing images to 384×384 before passing them to the processor.
This reduces multi-crop shape issues that can break training.
Trainer configuration uses remove_unused_columns=False (important for multimodal batches).
W&B is disabled to avoid interactive prompts in Colab.
Quick start overview (what the notebook does)
The notebook train_colab.ipynb contains:

Install dependencies
Upload + unzip train_data.zip
Load LiquidAI/LFM2-VL-1.6B in 4-bit
Attach LoRA and fine-tune on your dataset
Save the adapter to lfm2_adapter.zip
Optional: test on new images / probe.7z and save results to preds.jsonl
Common issues
The model repeats the prompt or returns empty output during inference
Use:

processor.apply_chat_template(..., add_generation_prompt=True)
Decode only generated tokens: out[0][input_len:]
W&B asks interactive questions in Colab
Disable it:

WANDB_DISABLED=true
report_to="none"
Multi-crop / vision shape errors
Use forcing single-crop:

img.resize((384,384))
Publishing your adapter
You can publish lfm2_adapter.zip via:

GitHub Releases (simple)
Hugging Face Hub (recommended for adapters)
Note about Raspberry Pi 5
Fine-tuning should be done on GPU (Colab/desktop/server).
Running a 1.6B vision-language model on Raspberry Pi 5 is usually slow; Pi is best used as an edge capture/control device, while inference runs on a stronger machine or a heavily quantized runtime.

Attribution / license
Base model: LiquidAI/LFM2-VL-1.6B (see the model card/license on Hugging Face).
This repository provides training code and guidance; users are responsible for respecting dataset/model licenses.
