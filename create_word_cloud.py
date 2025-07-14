import argparse
import os

import nltk
nltk.download('stopwords', download_dir='./nltk_data')
nltk.data.path.append('./nltk_data')  # Tell NLTK to use your local folder

import json
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords

# Argument parsing
parser = argparse.ArgumentParser(description="Generate a word cloud from Steam reviews")
parser.add_argument(
    "--filename",
    type=str,
    required=True,
    help="Path to the downloaded Steam reviews JSON file",
)
args = parser.parse_args()
filename = args.filename

# Check if the file exists
if not os.path.isfile(filename):
    print(f"Error: File not found → {filename}")
    exit(1)

# Config constants
output_image = "wordcloud_reviews.png"

# Load reviews
with open(filename, "r", encoding="utf-8") as f:
    data = json.load(f)
    
reviews = [review["review"] for review in data["reviews"] if "review" in review]

# Text Preprocessing, with extra custom stop words list which will be excluded
stop_words = set(stopwords.words("english"))
custom_exclusions = {
    "game", "games", "theres", "there", "really", "thing", "things", "play", "playing", 
    "fun", "time", "get", "got", "make", "made", "like", "bit", "fits", "yes", "want"
    "another", "minutes", "always", "actually", "yet", "wouldve", "would", "im",
    "please", "hey"
}
stop_words.update(custom_exclusions)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)     # remove url like substrings
    text = re.sub(r"[^a-z\s]", "", text)    # remove non-alphabet parts
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)

cleaned_reviews = [clean_text(review) for review in reviews]
all_text = " ".join(cleaned_reviews)

# Generate Word Cloud
wc = WordCloud(
    width=1200,
    height=600,
    background_color='white',
    max_words=200,
    collocations=True
).generate(all_text)

# Plot
plt.figure(figsize=(14, 7))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Most Frequent Words in Steam Reviews", fontsize=16)
plt.tight_layout()
plt.savefig(output_image, dpi=300)
