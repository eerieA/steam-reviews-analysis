from .base import LanguageProcessor # Inside a top-level module imports need to use relative path
import jieba    # Chinese word segmentation


class ChineseProcessor(LanguageProcessor):
    """Chinese text processor"""

    def _setup_stop_words(self):    # NLTK does not provide stop words for SCh so we write our own
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