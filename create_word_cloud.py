import argparse
import os
from pathlib import Path

import json
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords

import jieba    # Chinese word segmentation
import nltk
nltk.download('stopwords', download_dir='./nltk_data')
nltk.data.path.append('./nltk_data')  # Tell NLTK to use local folder

# TODO: separate preprocessing of different languages
# TODO: Add a constant dict somewhere to lookup available languages and report error if not supported

# Argument parsing
parser = argparse.ArgumentParser(description="Generate a word cloud from Steam reviews")
parser.add_argument(
    "--filename",
    type=str,
    required=True,
    help="Path to the downloaded Steam reviews JSON file",
)
parser.add_argument(
    "-l",
    "--language",
    type=str,
    default="english",
    help="Review language (e.g., 'english').",
)
parser.add_argument(
    "--appid",
    type=int,
    required=True,
    help="Steam App ID of the game, for naming the output file"
)
args = parser.parse_args()
filename = args.filename
language = args.language
appid = args.appid

# Check if the file exists
if not os.path.isfile(filename):
    print(f"Error: File not found → {filename}")
    exit(1)

# Config constants
# Get the current script's directory
SCRIPT_DIR = Path(__file__).resolve().parent
# Path to config.json (same directory)
CONFIG_PATH = SCRIPT_DIR / "config.json"
# Load config
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)
output_dir = config["output_dir"]
output_image = f"{output_dir}{appid}_wordcloud_{language}.png"

# Load reviews
with open(filename, "r", encoding="utf-8") as f:
    data = json.load(f)
    
reviews = [review["review"] for review in data["reviews"] if "review" in review]

# Text Preprocessing, with extra custom stop words list which will be excluded
stop_words = set(stopwords.words("english"))
custom_stop_words_eng = {
    "game", "games", "theres", "there", "really", "thing", "things", "play", "playing", 
    "fun", "time", "get", "got", "make", "made", "like", "bit", "fits", "yes", "want"
    "another", "minutes", "always", "actually", "yet", "wouldve", "would", "im",
    "please", "hey"
}
stop_words.update(custom_stop_words_eng)
stop_words.update({"游戏", "不是", "这个", "好玩", "就是", "但是", "可以", "虽然", "而且", "时候", "那个", "所以", "因为", "那么", "之前", "之后", "的话", "然后", "以及", "不过", "这样", "除了", "同时", "甚至", "这种", "那种", "觉得", "还有", "知道", "所以", "还是", "一下", "一个", "如果", "没有"})

def clean_text(text, language):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\u4e00-\u9fff\s]", "", text)  # keep Chinese chars
    if language == "schinese":
        tokens = jieba.cut(text)
    else:
        tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    return " ".join(tokens)

cleaned_reviews = [clean_text(review, language) for review in reviews]
all_text = " ".join(cleaned_reviews)

# Generate Word Cloud
wc = WordCloud(
    width=1200,
    height=600,
    background_color='white',
    max_words=200,
    collocations=True,
    font_path="./assets/NotoSansCJK-Regular.ttc"
).generate(all_text)

# Plot
plt.figure(figsize=(14, 7))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Most Frequent Words in Steam Reviews", fontsize=16)
plt.tight_layout()
plt.savefig(output_image, dpi=300)
