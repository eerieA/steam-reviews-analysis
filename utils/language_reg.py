"""
Unified language support module for other scripts.
Provides a centralized registry for language-specific configurations and processors.
"""
import importlib
import json
from pathlib import Path
from typing import Dict, Any, Optional


class LanguageConfig:
    """Configuration container for language-specific settings"""

    def __init__(self,
                 code: str,
                 name: str,
                 model_subfolder: str,
                 processor_class: str,
                 **kwargs):
        self.code = code
        self.name = name
        self.model_subfolder = model_subfolder
        self.processor_class = processor_class
        self.extra_config = kwargs

    @property
    def local_model_path(self) -> Path:
        # Make it relative to the project root
        return (Path(__file__).resolve().parent.parent / "models" / self.model_subfolder).resolve()


class LanguageRegistry:
    """Central registry for all language configurations"""

    def __init__(self, config_path: Optional[Path] = None):
        self._configs: Dict[str, LanguageConfig] = {}
        self._processors: Dict[str, Any] = {}  # Cache for processor instances

        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.json"

        self._load_config(config_path)

    def _load_config(self, config_path: Path):
        """Load language configurations from JSON file"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)

            for lang_code, lang_data in config_data["languages"].items():
                self._configs[lang_code] = LanguageConfig(
                    code=lang_code,
                    **lang_data
                )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Language config file not found: {config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")

    def get_supported_languages(self) -> list[str]:
        """Get list of supported language codes"""
        return list(self._configs.keys())

    def get_language_names(self) -> Dict[str, str]:
        """Get mapping of language codes to human-readable names"""
        return {code: config.name for code, config in self._configs.items()}

    def is_supported(self, language: str) -> bool:
        """Check if a language is supported"""
        return language in self._configs

    def get_config(self, language: str) -> LanguageConfig:
        """Get configuration for a specific language"""
        if not self.is_supported(language):
            supported = ", ".join(self.get_supported_languages())
            raise ValueError(
                f"Unsupported language '{language}'. Available: {supported}")
        return self._configs[language]

    def get_processor(self, language: str):
        """Get language processor instance (cached)"""
        if language not in self._processors:
            config = self.get_config(language)

            # Import and instantiate processor class
            module_name, class_name = config.processor_class.rsplit(".", 1)
            try:
                module = importlib.import_module(module_name)
                processor_class = getattr(module, class_name)
                self._processors[language] = processor_class()
            except (ImportError, AttributeError) as e:
                raise ImportError(
                    f"Failed to load processor for {language}: {e}")

        return self._processors[language]

    def get_model_components(self, language: str):
        """Get tokenizer and model for sentiment analysis"""
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        config = self.get_config(language)
        model_path = config.local_model_path

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path)
            return tokenizer, model
        except Exception as e:
            raise RuntimeError(f"Failed to load model for {language}: {e}")


# Global registry instance
_registry = None


def get_registry() -> LanguageRegistry:
    """Get the global language registry instance"""
    global _registry
    if _registry is None:
        _registry = LanguageRegistry()
    return _registry


def validate_language(language: str) -> str:
    """Validate language and return it, or raise an error"""
    registry = get_registry()
    if not registry.is_supported(language):
        supported = ", ".join(registry.get_supported_languages())
        raise ValueError(
            f"Unsupported language '{language}'. Available: {supported}")
    return language


# Convenience functions
def get_supported_languages() -> list[str]:
    return get_registry().get_supported_languages()


def get_language_processor(language: str):
    return get_registry().get_processor(language)


def get_model_components(language: str):
    return get_registry().get_model_components(language)
