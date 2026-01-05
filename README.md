# lfm2-vl-finetune-guide
“Colab guide for fine-tuning LiquidAI/LFM2-VL-1.6B with LoRA”

# Fine-tuning LiquidAI/LFM2-VL-1.6B on your own images (Colab, LoRA, 4-bit)

This repository is a practical, step-by-step guide to fine-tuning **LiquidAI/LFM2-VL-1.6B** on your own image+text dataset using **Google Colab**, **LoRA (PEFT)** and **4-bit quantization**.

The goal is reproducibility: a user should be able to take their own dataset, run the notebook cells in order, and obtain a small **LoRA adapter** (`lfm2_adapter.zip`) that can be attached to the base model.

---

## What you will get

- A trained **LoRA adapter** (small, easy to share) saved as:
  - `out/final_adapter/` (folder)
  - `lfm2_adapter.zip` (archive for download/publishing)
- A working inference recipe:
  - base model `LiquidAI/LFM2-VL-1.6B`
  - + your adapter
  - run on new images

---

## Requirements

- Google Colab with GPU (recommended).
- A dataset ZIP archive with the structure below.

---

## Dataset format

Create `train_data.zip` with this structure:
dataset/
images/
00001.jpg
00002.jpg
...
metadata.jsonl

text


`metadata.jsonl` must be **JSON Lines** (one JSON object per line):

```json
{"file_name":"00001.jpg","text":"Your target text / caption / answer"}
{"file_name":"00002.jpg","text":"..."}
Notes:

file_name must exist inside dataset/images/.
text is what you want the model to learn (captions, instructions, structured JSON answers, etc.).
Start small (200–1000 examples) and iterate.
Why this guide works (important details)
This project uses a stable training setup discovered through trial and error:

Correct model class: for training you must load Lfm2VlForConditionalGeneration (not AutoModel).
No risky vision patches: avoid fragile patches (e.g. SigLIP2 positional embedding hacks).
Forcing single-crop: we resize images to a fixed size (384×384) before passing them to the processor to reduce multi-crop shape issues.
Trainer configuration:
remove_unused_columns=False is critical for multimodal batches
disable W&B to avoid interactive prompts in Colab
Quick start (Google Colab)
1) Install dependencies + disable W&B
Python

!pip -q install -U "pillow<12" transformers peft accelerate bitsandbytes

import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
2) Upload and unzip your dataset
Upload your train_data.zip in Colab.

Python

from google.colab import files
import os, zipfile

if not os.path.exists("dataset"):
    up = files.upload()  # upload train_data.zip
    zip_name = list(up.keys())[0]
    with zipfile.ZipFile(zip_name, "r") as z:
        z.extractall("dataset")

assert os.path.exists("dataset/metadata.jsonl"), "dataset/metadata.jsonl not found"
assert os.path.exists("dataset/images"), "dataset/images not found"

print("✅ dataset ready")
3) Load model (4-bit) + attach LoRA
Python

import torch
from transformers import AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers.models.lfm2_vl import Lfm2VlForConditionalGeneration

MODEL_ID = "LiquidAI/LFM2-VL-1.6B"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

model = Lfm2VlForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

model.config.use_cache = False

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

lora = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora)
model.print_trainable_parameters()

print("✅ model ready")
4) Dataset + Trainer (single-crop)
Python

import os, json
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer

FIX_SIZE = 384

class MyDataset(Dataset):
    def __init__(self, jsonl_path, images_dir, processor):
        self.items = [json.loads(l) for l in open(jsonl_path, "r", encoding="utf-8")]
        self.images_dir = images_dir
        self.processor = processor

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img_path = os.path.join(self.images_dir, it["file_name"])
        img = Image.open(img_path).convert("RGB")

        # For stability: force single-crop behavior
        img = img.resize((FIX_SIZE, FIX_SIZE))

        # Minimal stable prompt format
        prompt = f"<image>\nDescribe the image.\nAnswer: {it['text']}"

        enc = self.processor(text=prompt, images=img, return_tensors="pt")
        enc["labels"] = enc["input_ids"].clone()

        return {k: v.squeeze(0) for k, v in enc.items()}

def collate_fn(features):
    out = {}
    for k in features[0]:
        v0 = features[0][k]
        out[k] = torch.stack([f[k] for f in features]) if isinstance(v0, torch.Tensor) else [f[k] for f in features]
    return out

train_ds = MyDataset("dataset/metadata.jsonl", "dataset/images", processor)

args = TrainingArguments(
    output_dir="out",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="no",
    report_to="none",
    remove_unused_columns=False,
    dataloader_pin_memory=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    data_collator=collate_fn,
)

trainer.train()
print("✅ training done")
5) Save the adapter (lfm2_adapter.zip)
Python

import os, zipfile

OUT_DIR = "out/final_adapter"
os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR)

zip_name = "lfm2_adapter.zip"
with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files_ in os.walk(OUT_DIR):
        for f in files_:
            full = os.path.join(root, f)
            z.write(full, os.path.relpath(full, OUT_DIR))

print("✅ Saved:", zip_name)

# optional download
try:
    from google.colab import files
    files.download(zip_name)
except Exception:
    pass
Inference on new images (base + adapter)
1) Extract adapter
Python

import os, zipfile
os.makedirs("adapter", exist_ok=True)
with zipfile.ZipFile("lfm2_adapter.zip", "r") as z:
    z.extractall("adapter")
print("✅ adapter extracted")
2) Load base model + adapter
Python

import torch
from transformers import AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from transformers.models.lfm2_vl import Lfm2VlForConditionalGeneration

MODEL_ID = "LiquidAI/LFM2-VL-1.6B"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

base = Lfm2VlForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    quantization_config=bnb,
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base, "adapter").eval()
print("✅ inference model ready")
3) Run a single image (chat template + correct decoding)
Python

import torch
from PIL import Image

@torch.no_grad()
def infer(image_path: str, question: str):
    img = Image.open(image_path).convert("RGB").resize((384, 384))

    messages = [
        {"role": "system", "content": "Answer briefly."},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": question},
        ]},
    ]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    enc = processor(text=prompt, images=img, return_tensors="pt")

    for k, v in enc.items():
        if isinstance(v, torch.Tensor):
            enc[k] = v.to(model.device)

    out = model.generate(**enc, max_new_tokens=128, do_sample=False)

    # decode ONLY newly generated tokens (avoid prompt echo confusion)
    gen = out[0][enc["input_ids"].shape[-1]:]
    return processor.decode(gen, skip_special_tokens=True).strip()

print(infer("dataset/images/00001.jpg", "What is shown in the image?"))
Common issues
1) AutoModel loads but training/inference is broken
Use:

from transformers.models.lfm2_vl import Lfm2VlForConditionalGeneration
2) The model repeats the prompt or returns empty output
Use processor.apply_chat_template(..., add_generation_prompt=True)
Decode only generated tokens: out[0][input_len:]
3) W&B asks interactive questions in Colab
Set:

WANDB_DISABLED=true
report_to="none"
4) Multi-crop / vision shape errors
Use forcing single-crop:

img.resize((384,384))
Publishing your adapter
You can publish lfm2_adapter.zip on:

GitHub Releases (simple)
Hugging Face Hub (recommended for adapters)
If you’re new to publishing, start with GitHub (this repo), then optionally upload the adapter to HF.

Disclaimer about edge devices (Raspberry Pi 5)
Fine-tuning should be done on GPU (Colab/desktop/server).
Running a 1.6B vision-language model on Raspberry Pi 5 is usually slow; Pi is best used as an edge capture/control device, while inference runs on a GPU machine or a quantized runtime.

License / Attribution
Base model: LiquidAI/LFM2-VL-1.6B (see model card/license on Hugging Face).
This repo provides training code and guidance; users are responsible for respecting dataset/model licenses.
