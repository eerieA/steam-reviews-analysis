from pathlib import Path
import json
import torch
torch._dynamo.config.suppress_errors = True # Disable compilation that would use TorchInductor and Triton etc
torch._dynamo.disable()
import logging
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)  # Also don't display log messages that are not errors
                                                            # Without this, dynamo will print many "I won’t compile this frame" warnings in terminal

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Get the current script's directory
SCRIPT_DIR = Path(__file__).resolve().parent
# Path to config.json (one level above)
CONFIG_PATH = SCRIPT_DIR.parent / "config.json"
# Load config
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)
model_subfolder_name = config["model_subfolder_name"]
local_model_path = f"{SCRIPT_DIR.parent}/models/{model_subfolder_name}"

tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForSequenceClassification.from_pretrained(local_model_path)

# Use GPU (device=0); use device=-1 for CPU
classifier = pipeline(
    "text-classification", 
    model=model, 
    tokenizer=tokenizer, 
    top_k=None, 
    device=0
)

# Warm-up GPU to avoid startup lag
_ = classifier(["Warm-up sentence!"] * 16, batch_size=16)  # This primes CUDA, compiles kernels, etc.

# Test
print(classifier("Gotta love a hidden object game with cats... unless you can't stand cats--which I understand. I do love them, but I also realize that they are indeed direct descendants of Satan."))
