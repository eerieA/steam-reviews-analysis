from .base import LanguageProcessor # Inside a top-level module imports need to use relative path
import nltk


# For English, NLTK provides some stop words so use that
# Ensure stopwords are downloaded
try:
    from nltk.corpus import stopwords
    _ = stopwords.words("english")  # Try loading to check availability
except LookupError:
    nltk.download("stopwords", download_dir="./nltk_data")
    nltk.data.path.append('./nltk_data')
    from nltk.corpus import stopwords  # Reload after download

class EnglishProcessor(LanguageProcessor):
    """English text processor"""

    def _setup_stop_words(self):
        self.stop_words = set(stopwords.words("english"))
        english_stop_words_extra = {
            "game", "games", "theres", "there", "really", "thing", "things",
            "play", "playing", "fun", "time", "get", "got", "make", "made",
            "like", "bit", "fits", "yes", "want", "another", "minutes",
            "always", "actually", "yet", "wouldve", "would", "im", "please", "hey"
        }
        self.stop_words.update(english_stop_words_extra)

    def tokenize(self, text):
        return text.split()

    def get_regex_pattern(self):
        # Pattern of characters to remove
        return r"[^a-z\s]"
