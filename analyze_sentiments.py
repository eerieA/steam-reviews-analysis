import argparse
import os
from pathlib import Path
import time
import json
import logging

import pandas as pd
import matplotlib.pyplot as plt

from itertools import chain
from datasets import Dataset
from transformers import pipeline
import torch
from torch.nn.functional import softmax

# Central language config module
from utils.language_reg import get_registry, validate_language


def detect_device():
    """Detect GPU/CPU device for processing"""
    try:
        import torch
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.disable()
        logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
        return 0 if torch.cuda.is_available() else -1
    except ImportError:
        return -1


def load_and_clean_reviews(filename: str) -> Dataset:
    """Load reviews from JSON file and return clean dataset"""
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract only the review and language fields to avoid Arrow schema issues
    clean_reviews = [
        {"review": r["review"], "language": r.get("language", "unknown")}
        for r in data["reviews"]
        if isinstance(r.get("review"), str)  # Filter out non-string reviews
    ]

    return Dataset.from_list(clean_reviews)


def create_classify_function(tokenizer, model, device: int, top_k_labels: int):
    """Create classification function based on tokennizer and model (for a language), and device.
       The return is a function pointer, the function also sets up data post-proc parameters before the plot,
       such as top_k_labels. """

    # Need to know the correct device, GPU or CPU
    target_device = f"cuda:{device}" if device >= 0 else "cpu"
    model.to(target_device)

    def classify(batch):
        encodings = tokenizer(
            batch["review"], padding=True, truncation=True, return_tensors="pt"
        )
        encodings = {k: v.to(target_device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = model(**encodings)
            probs = softmax(outputs.logits, dim=1)

        probs = probs.cpu().numpy()

        top_k = probs.argsort(axis=1)[:, -top_k_labels:][:, ::-1]
        labels = model.config.id2label
        return {
            "raw_scores": [
                [{"label": labels[i], "score": float(
                    row_probs[i])} for i in row_topk]
                for row_probs, row_topk in zip(probs, top_k)
            ]
        }

    return classify


def extract_top_labels_function(top_k_labels: int):
    """Create the top labels extraction function"""
    def extract_top_labels(example):
        sorted_scores = sorted(
            example["raw_scores"], key=lambda x: x["score"], reverse=True)
        top_labels = [entry["label"] for entry in sorted_scores[:top_k_labels]]
        return {"top_labels": top_labels}

    return extract_top_labels


def plot_emotion_distribution(label_counts: pd.Series, output_image: str, top_k_labels: int, sample_size: int = -1):
    """Create and save emotion distribution plot"""
    sample_size_str = f" (sample size {sample_size})" if sample_size and sample_size > 0 else ""
    plt.figure(figsize=(12, 6))
    plt.bar(label_counts.index, label_counts.values, color='skyblue')
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Top-{top_k_labels} Emotion Distribution in Steam Reviews{sample_size_str}")
    plt.xlabel("Emotion")
    plt.ylabel("Number of Appearances")
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Plot saved as {output_image}")


def main():
    start = time.time()

    # Get language registry
    registry = get_registry()
    supported_languages = ", ".join(registry.get_supported_languages())

    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Analyze Steam review sentiments")
    parser.add_argument(
        "--filename",
        type=str,
        required=True,
        help="Path to the downloaded reviews JSON file",
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        default="english",
        help=f"Review language. Available: {supported_languages}",
    )
    parser.add_argument(
        "--appid",
        type=int,
        required=True,
        help="Steam App ID of the game, for naming the output file"
    )
    parser.add_argument(
        "-s",
        "--sample-size",
        type=int,
        default=None,
        help="Number of reviews to randomly sample from the dataset (default: use all reviews)",
    )
    args = parser.parse_args()

    # Validate inputs
    if not os.path.isfile(args.filename):
        print(f"Error: File not found → {args.filename}")
        exit(1)

    try:
        language = validate_language(args.language)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)

    # Load configuration
    SCRIPT_DIR = Path(__file__).resolve().parent
    CONFIG_PATH = SCRIPT_DIR / "config.json"
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    batch_size = config["batch_size"]
    top_k_labels = config["top_k_labels"]
    output_dir = config["output_dir"]
    output_image = f"{output_dir}{args.appid}_emo_distrib_{language}.png"

    device = detect_device()

    # Load and process data
    print(f"Loading reviews from {args.filename}...")
    dataset = load_and_clean_reviews(args.filename)
    # Sample a subset of random reviews if requested
    if args.sample_size is not None and args.sample_size < len(dataset):
        total_count = len(dataset)
        dataset = dataset.shuffle().select(range(args.sample_size))
        print(f"Sampled {args.sample_size} reviews from {total_count} total reviews.")

    print(
        f"Setting up sentiment analysis for {registry.get_config(language).name}...")
    tokenizer, model = registry.get_model_components(language)
    if device >= 0:
        model = model.to(f"cuda:{device}")
    else:
        model = model.to("cpu")

    # Create classification function
    classify_fn = create_classify_function(
        tokenizer, model, device, top_k_labels)

    print("Running sentiment analysis...")
    dataset = dataset.map(classify_fn, batched=True, batch_size=batch_size)

    # Extract top labels
    extract_fn = extract_top_labels_function(top_k_labels)
    dataset = dataset.map(extract_fn)

    # Analyze results
    print("Analyzing emotion distribution...")
    all_top_labels = list(chain.from_iterable(dataset["top_labels"]))
    label_counts = pd.Series(
        all_top_labels).value_counts().sort_values(ascending=False)

    # Plot
    if args.sample_size is not None:
        plot_emotion_distribution(label_counts, output_image, top_k_labels, args.sample_size)
    else:
        plot_emotion_distribution(label_counts, output_image, top_k_labels)

    print("Elapsed time:", time.time() - start)


if __name__ == "__main__":
    main()
