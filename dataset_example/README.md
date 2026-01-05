[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Absurd7550/lfm2-vl-finetune-guide/blob/main/train_colab.ipynb)
# Dataset example (how to prepare train_data.zip)

The Colab notebook expects a single ZIP file (for example `train_data.zip`) with this structure:
dataset/
images/
00001.jpg
00002.jpg
...
metadata.jsonl

text


## metadata.jsonl format

`metadata.jsonl` is JSON Lines: one JSON object per line.

Each line must contain:
- `file_name` — image filename located in `dataset/images/`
- `text` — the target text/answer you want the model to learn

Example:

```json
{"file_name":"00001.jpg","text":"A character is standing in the center of the screen."}
{"file_name":"00002.jpg","text":"Fishing is active (the UI shows 'Fishing')."}
How to create the ZIP
From the folder that contains dataset/:

Linux / macOS
Bash

zip -r train_data.zip dataset
Windows
Right click the dataset folder → “Send to” → “Compressed (zipped) folder”
Rename the result to train_data.zip.

Common mistakes
The zip does NOT contain the top-level dataset/ folder (must be included).
file_name in metadata.jsonl does not match the real image name (case-sensitive).
metadata.jsonl is not valid JSONL (must be one JSON object per line).
text


---

# 4) `dataset_example/metadata.jsonl`
(JSONL):

```jsonl
{"file_name":"00001.jpg","text":"A game screenshot. A character is standing in the center of the screen."}
{"file_name":"00002.jpg","text":"A game screenshot. The UI shows a fishing activity in progress."}
{"file_name":"00003.jpg","text":"A game screenshot. A vendor NPC is visible and an interaction prompt may be present."}
