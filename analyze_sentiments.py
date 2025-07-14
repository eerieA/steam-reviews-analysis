import argparse
import os

import json
import pandas as pd
import matplotlib.pyplot as plt

# Argument parsing
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
    import logging
    logging.getLogger("torch._dynamo").setLevel(logging.ERROR)

    device = 0 if torch.cuda.is_available() else -1
except ImportError:
    device = -1

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Config constants
local_path = "./models/cirimus-modernbert-base-go-emo"
batch_size = 16
top_k_labels = 5

# Load reviews from JSON
with open(filename, "r", encoding="utf-8") as f:
    data = json.load(f)
reviews = [r["review"] for r in data["reviews"]]

# Load tokenizer & model from offline files
tokenizer = AutoTokenizer.from_pretrained(local_path)
model = AutoModelForSequenceClassification.from_pretrained(local_path)

# Inference pipeline
classifier = pipeline(
    "text-classification", model=model, tokenizer=tokenizer, device=device, top_k=None
)

# Start collecting top-k labels from each review
all_top_labels = []

for i in range(0, len(reviews), batch_size):
    batch = reviews[i : i + batch_size]
    results = classifier(batch)  # list of lists of {label, score}
    for single_result in results:
        # Sort by score descending, and get top N
        sorted_result = sorted(single_result, key=lambda x: x['score'], reverse=True)
        top_labels = [entry['label'] for entry in sorted_result[:top_k_labels]]
        all_top_labels.extend(top_labels)

# Count label frequency
label_counts = pd.Series(all_top_labels).value_counts().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
plt.bar(label_counts.index, label_counts.values, color='skyblue')
plt.xticks(rotation=45, ha="right")
plt.title(f"Top-{top_k_labels} Emotion Distribution in Steam Reviews")
plt.xlabel("Emotion")
plt.ylabel("Number of Appearances")
plt.tight_layout()
plt.savefig(f"emotion_distribution_top{top_k_labels}.png", dpi=300)
