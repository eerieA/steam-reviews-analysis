import argparse
import os
import time
import json
import logging

import pandas as pd
import matplotlib.pyplot as plt

from itertools import chain
from datasets import Dataset
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax

# Argument parsing
start = time.time()
parser = argparse.ArgumentParser(description="Analyze Steam review sentiments")
parser.add_argument(
    "--filename",
    type=str,
    required=True,
    help="Path to the downloaded reviews JSON file",
)
args = parser.parse_args()
filename = args.filename

# Check if the file exists
if not os.path.isfile(filename):
    print(f"Error: File not found → {filename}")
    exit(1)

# Try to detect a GPU; fall back to CPU
try:
    import torch
    torch._dynamo.config.suppress_errors = True # See comments in test_trfm_gpu.py about why these configs
    torch._dynamo.disable()
    logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
    device = 0 if torch.cuda.is_available() else -1
except ImportError:
    device = -1

# Config constants
local_path = "./models/cirimus-modernbert-base-go-emo"
batch_size = 8
top_k_labels = 5

# Load JSON and extract reviews
with open(filename, "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract only the review and language fields to avoid Arrow schema issues
clean_reviews = [
    {"review": r["review"], "language": r.get("language", "unknown")}
    for r in data["reviews"]
    if isinstance(r.get("review"), str)  # Filter out non-string reviews
]

# Convert to Hugging Face Dataset
dataset = Dataset.from_list(clean_reviews)

# Load tokenizer & model from offline model files
tokenizer = AutoTokenizer.from_pretrained(local_path)
model = AutoModelForSequenceClassification.from_pretrained(local_path)

classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=device,
    top_k=None,
    batch_size=batch_size,
)

# Run classification using batched mapping
""" With manual tokenizer + model, just to suppress a warning.
So actually this:
def classify(batch):
    result = classifier(batch["review"])  # <- this triggers the warning
    return {"raw_scores": result}
would have the same result and speed XD"""
def classify(batch):
    encodings = tokenizer(batch["review"], padding=True, truncation=True, return_tensors="pt")
    encodings = {k: v.to(device) for k, v in encodings.items()}
    with torch.no_grad():   # Only prediction (inference), does not need grads
        outputs = model(**encodings)
        probs = softmax(outputs.logits, dim=1)
    probs = probs.cpu().numpy() # Have to move back to CPU bcz NumPy needs it

    top_k = probs.argsort(axis=1)[:, -top_k_labels:][:, ::-1]
    labels = model.config.id2label

    batch_results = []
    for row_probs, row_topk in zip(probs, top_k):
        batch_results.append([
            {"label": labels[i], "score": float(row_probs[i])}
            for i in row_topk
        ])

    """ Output would look like:
    {
        "raw_scores": [
            [ {"label": "joy", "score": 0.84}, {"label": "admiration", "score": 0.10}, ... ],  # review 1
            [ {"label": "amusement", "score": 0.65}, {"label": "joy", "score": 0.21}, ... ],  # review 2
            ...
        ]
    }
    . Uncomment the debug prints below to see some actual samples.
    """    
    # import pprint
    # print("\nSample raw_scores from classify:")
    # pprint.pprint(batch_results[:3])

    return {"raw_scores": batch_results}

dataset = dataset.map(classify, batched=True, batch_size=batch_size)

# Extract top-k emotion labels
def extract_top_labels(example):
    sorted_scores = sorted(example["raw_scores"], key=lambda x: x["score"], reverse=True)
    top_labels = [entry["label"] for entry in sorted_scores[:top_k_labels]]
    return {"top_labels": top_labels}

dataset = dataset.map(extract_top_labels)

# Flatten top labels and count
all_top_labels = list(chain.from_iterable(dataset["top_labels"]))
label_counts = pd.Series(all_top_labels).value_counts().sort_values(ascending=False)

# Plot and save
plt.figure(figsize=(12, 6))
plt.bar(label_counts.index, label_counts.values, color='skyblue')
plt.xticks(rotation=45, ha="right")
plt.title(f"Top-{top_k_labels} Emotion Distribution in Steam Reviews")
plt.xlabel("Emotion")
plt.ylabel("Number of Appearances")
plt.tight_layout()
plt.savefig(f"emotion_distribution_top{top_k_labels}.png", dpi=300)
print(f"Plot saved as emotion_distribution_top{top_k_labels}.png")
print("Elapsed time:", time.time() - start)