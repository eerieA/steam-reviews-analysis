import torch
torch._dynamo.config.suppress_errors = True # Disable compilation that would use TorchInductor and Triton etc
torch._dynamo.disable()
import logging
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)  # Also don't display log messages that are not errors
                                                            # Without this, dynamo will print many "I won’t compile this frame" warnings in terminal

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

local_path = "./models/cirimus-modernbert-base-go-emo"

tokenizer = AutoTokenizer.from_pretrained(local_path)
model = AutoModelForSequenceClassification.from_pretrained(local_path)

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
