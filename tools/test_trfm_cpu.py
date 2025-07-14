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
    device=-1
)

# Test
print(classifier("3 days left to launch for EDENS ZERO the game! 🥳 Take off to the stars, and adventure! "))