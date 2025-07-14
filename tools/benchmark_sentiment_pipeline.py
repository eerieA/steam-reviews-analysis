import time
import torch
torch._dynamo.config.suppress_errors = True # Disable compilation that would use TorchInductor and Triton etc
torch._dynamo.disable()

import random
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# --- Config ---
MODEL_PATH = "./models/cirimus-modernbert-base-go-emo"
NUM_SAMPLES = 100  # Number of synthetic reviews to test
REVIEW_TEXTS = [
    "This game is so good, I can't stop playing it!",
    "I'm really disappointed in the graphics and gameplay.",
    "Feels relaxing and peaceful. Great soundtrack.",
    "Too many bugs. Constant crashes. Waste of money.",
    "The multiplayer is super fun with friends.",
    "UI is clunky, but the story is decent.",
    "Absolutely stunning visuals! 10/10 would play again.",
    "The tutorial is confusing and doesn't help at all.",
    "Combat system feels responsive and satisfying.",
    "Takes forever to load. Needs optimization."
]
# Duplicate to simulate load
REVIEW_TEXTS *= NUM_SAMPLES // len(REVIEW_TEXTS)

# --- Helper to run classification ---
def run_pipeline_on_device(device_name: str, device_id: int):
    print(f"\nRunning on {device_name.upper()} (device={device_id})")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(
        "cuda" if device_id >= 0 else "cpu"
    )

    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device_id,
        top_k=None
    )

    if device_id == 0:
        # If device is GPU, do warm up. This is critical for observing GPU's advantage in parallel computing
        _ = classifier(["warm up"] * 16, batch_size=16)

    start_time = time.time()
    results = classifier(REVIEW_TEXTS, batch_size=16)
    elapsed = time.time() - start_time

    print(f"Time taken: {elapsed:.2f} seconds")
    return results, elapsed

# --- Run on CPU ---
results_cpu, time_cpu = run_pipeline_on_device("cpu", device_id=-1)

# --- Run on GPU (if available) ---
if torch.cuda.is_available():
    results_gpu, time_gpu = run_pipeline_on_device("gpu", device_id=0)
    print(f"\n🔍 Summary:\nCPU time: {time_cpu:.2f}s\nGPU time: {time_gpu:.2f}s")
    speedup = time_cpu / time_gpu
    print(f"⚡ GPU is approximately {speedup:.2f}x faster")
else:
    print("\n⚠️ No GPU available. Skipping GPU benchmark.")
