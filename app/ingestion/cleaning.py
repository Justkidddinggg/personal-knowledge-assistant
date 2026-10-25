import re
import unicodedata

_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalize text without destroying meaning.

    Deliberately conservative: we do NOT lowercase or strip punctuation,
    because embedding models and the LLM both benefit from natural text.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _MULTI_SPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()