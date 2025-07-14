import nltk
nltk.download('stopwords', download_dir='./nltk_data')
nltk.data.path.append('./nltk_data')  # Tell NLTK to use your local folder

import json
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords

# Config constants
filename = "./downloaded/2954460_eng.json"
output_image = "wordcloud_reviews.png"

# Load reviews
with open(filename, "r", encoding="utf-8") as f:
    data = json.load(f)
    
reviews = [review["review"] for review in data["reviews"] if "review" in review]

# Text Preprocessing
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
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
