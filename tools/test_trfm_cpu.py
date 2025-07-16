from pathlib import Path
import json
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
    device=-1
)

# Test
print(classifier("3 days left to launch for EDENS ZERO the game! 🥳 Take off to the stars, and adventure! "))