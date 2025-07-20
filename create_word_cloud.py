import argparse
import os
from pathlib import Path
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Central language config module
from utils.language_reg import get_registry, validate_language, get_language_processor


def load_reviews(filename: str) -> list[str]:
    """Load reviews from JSON file"""
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [review["review"]
            for review in data["reviews"]
            if "review" in review]


def generate_wordcloud(text: str, output_path: str, language: str):
    """Generate and save word cloud"""
    registry = get_registry()
    language_name = registry.get_config(language).name

    script_dir = Path(__file__).resolve().parent
    font_path = script_dir / "assets" / "NotoSansCJK-Regular.ttc"

    wc = WordCloud(
        width=1200,
        height=600,
        background_color='white',
        max_words=200,
        collocations=True,
        font_path=str(font_path),  # WordCloud expects a string path
    ).generate(text)

    # Plot
    plt.figure(figsize=(14, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(
        f"Most Frequent Words in Steam Reviews ({language_name})", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Word cloud saved to: {output_path}")


def main():
    # Get language registry for help text
    registry = get_registry()
    supported_languages = ", ".join(registry.get_supported_languages())

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
        help=f"Review language. Available: {supported_languages}",
    )
    parser.add_argument(
        "--appid",
        type=int,
        required=True,
        help="Steam App ID of the game, for naming the output file"
    )
    args = parser.parse_args()

    # Validate inputs
    if not os.path.isfile(args.filename):
        print(f"Error: File not found → {args.filename}")
        exit(1)

    try:
        language = validate_language(args.language)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)

    # Load configuration
    SCRIPT_DIR = Path(__file__).resolve().parent
    CONFIG_PATH = SCRIPT_DIR / "config.json"
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    output_dir = config["output_dir"]
    output_image = f"{output_dir}{args.appid}_wordcloud_{language}.png"

    # Get language processor
    try:
        processor = get_language_processor(language)
    except (ImportError, AttributeError) as e:
        print(f"Error loading language processor: {e}")
        exit(1)

    # Load and process reviews
    print(f"Loading reviews from {args.filename}...")
    reviews = load_reviews(args.filename)

    print(f"Processing text for {registry.get_config(language).name}...")
    cleaned_reviews = [processor.clean_text(review) for review in reviews]
    all_text = " ".join(cleaned_reviews)

    # Plot
    print("Generating word cloud...")
    generate_wordcloud(all_text, output_image, language)


if __name__ == "__main__":
    main()
