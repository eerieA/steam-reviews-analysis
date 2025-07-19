import argparse
import os
from pathlib import Path
import json
import re
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
import jieba    # Chinese word segmentation
import nltk

nltk.download('stopwords', download_dir='./nltk_data')
nltk.data.path.append('./nltk_data')  # Tell NLTK to use local folder


class LanguageProcessor(ABC):
    """Abstract base class for language-specific text processing"""

    def __init__(self):
        self.stop_words = set()
        self._setup_stop_words()

    @abstractmethod
    def _setup_stop_words(self):
        """Setup language-specific stop words"""
        pass

    @abstractmethod
    def tokenize(self, text):
        """Tokenize text according to language rules"""
        pass

    @abstractmethod
    def get_regex_pattern(self):
        """Return regex pattern for keeping valid characters"""
        pass

    def clean_text(self, text):
        """Clean and process text"""
        text = text.lower()
        text = re.sub(r"http\S+", "", text)  # Remove URLs
        text = re.sub(self.get_regex_pattern(), "",
                      text)  # Keep only valid chars

        tokens = self.tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 1]
        return " ".join(tokens)


class EnglishProcessor(LanguageProcessor):
    """English text processor"""

    def _setup_stop_words(self):
        self.stop_words = set(stopwords.words("english"))
        custom_stop_words = {
            "game", "games", "theres", "there", "really", "thing", "things",
            "play", "playing", "fun", "time", "get", "got", "make", "made",
            "like", "bit", "fits", "yes", "want", "another", "minutes",
            "always", "actually", "yet", "wouldve", "would", "im", "please", "hey"
        }
        self.stop_words.update(custom_stop_words)

    def tokenize(self, text):
        return text.split()

    def get_regex_pattern(self):
        return r"[^a-z\s]"  # Keep only English letters and spaces


class ChineseProcessor(LanguageProcessor):
    """Chinese text processor"""

    def _setup_stop_words(self):
        chinese_stop_words = {
            "游戏", "不是", "这个", "好玩", "就是", "但是", "可以", "虽然",
            "而且", "时候", "那个", "所以", "因为", "那么", "之前", "之后",
            "的话", "然后", "以及", "不过", "这样", "除了", "同时", "甚至",
            "这种", "那种", "觉得", "还有", "知道", "所以", "还是", "一下",
            "一个", "如果", "没有"
        }
        self.stop_words.update(chinese_stop_words)

    def tokenize(self, text):
        return list(jieba.cut(text))

    def get_regex_pattern(self):
        # Keep Chinese chars, English letters, and spaces
        return r"[^a-z\u4e00-\u9fff\s]"


# Language processor registry
LANGUAGE_PROCESSORS = {
    "english": EnglishProcessor,
    "schinese": ChineseProcessor,
}

def get_supported_langs():
    return ", ".join(LANGUAGE_PROCESSORS.keys())

def get_language_processor(language):
    """Factory function to get appropriate language processor"""
    if language not in LANGUAGE_PROCESSORS:
        raise ValueError(
            f"Unsupported language '{language}'. Available languages: {get_supported_langs()}")

    return LANGUAGE_PROCESSORS[language]()


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Generate a word cloud from Steam reviews")
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
        help=f"Review language. Available: {get_supported_langs()}",
    )
    parser.add_argument(
        "--appid",
        type=int,
        required=True,
        help="Steam App ID of the game, for naming the output file"
    )
    args = parser.parse_args()

    # Check if the file exists
    if not os.path.isfile(args.filename):
        print(f"Error: File not found → {args.filename}")
        exit(1)
    
    # Check if the language is supported
    if args.language not in LANGUAGE_PROCESSORS:
        print(f"Error: Unsupported language '{args.language}'. Available languages: {get_supported_langs()}")
        exit(1)

    # Get language processor
    try:
        processor = get_language_processor(args.language)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)

    # Config constants
    SCRIPT_DIR = Path(__file__).resolve().parent
    CONFIG_PATH = SCRIPT_DIR / "config.json"
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    output_dir = config["output_dir"]
    output_image = f"{output_dir}{args.appid}_wordcloud_{args.language}.png"

    # Load reviews
    with open(args.filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    reviews = [review["review"]
               for review in data["reviews"] if "review" in review]

    # Process text using language-specific processor
    cleaned_reviews = [processor.clean_text(review) for review in reviews]
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
    print(f"Word cloud saved to: {output_image}")


if __name__ == "__main__":
    main()
