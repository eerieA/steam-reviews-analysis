import argparse, importlib
import os
from pathlib import Path
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from preprocessors.base import LanguageProcessor


# Language processor registry, for dynamic module loading
# The values are fully qualified reference to a class or function as a string
LANGUAGE_PROCESSORS = {
    "english": "preprocessors.english.EnglishProcessor",
    "schinese": "preprocessors.schinese.ChineseProcessor",
}

def get_supported_langs():
    return ", ".join(LANGUAGE_PROCESSORS.keys())

def get_language_processor(language):
    """Factory function to get appropriate language processor"""
    qualname = LANGUAGE_PROCESSORS.get(language)
    if not qualname:
        raise ValueError(
            f"Unsupported language '{language}'. Available languages: {get_supported_langs()}")
    
    module_name, class_name = qualname.rsplit(".", 1)
    mod = importlib.import_module(module_name)
    PreprocClass = getattr(mod, class_name)
    if not issubclass(PreprocClass, LanguageProcessor):
        raise TypeError(f"{class_name} is not subclass of LanguageProcessor")

    return PreprocClass()


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
