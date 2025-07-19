import re
from abc import ABC, abstractmethod


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
        """Return a regex that matches characters to remove"""
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
