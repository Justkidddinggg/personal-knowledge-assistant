import re
import unicodedata

_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_HYPHEN_BREAK = re.compile(r"(?<=[A-Za-z])-\n(?=[a-z])")  # <-- ตัวแปรใหม่


def clean_text(text: str) -> str:
    """Normalize text without destroying meaning.

    Deliberately conservative: we do NOT lowercase or strip punctuation,
    because embedding models and the LLM both benefit from natural text.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u0e4d\u0e32", "\u0e33")  # นิคหิต + สระอา -> สระอำ
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _MULTI_SPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _HYPHEN_BREAK.sub("", text)  # <-- บรรทัดใหม่ ต้องอยู่หลังบรรทัด strip ด้านบน
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()

def test_clean_text_fixes_thai_sara_am() -> None:
    assert clean_text("จ\u0e4d\u0e32กัด") == "จำกัด"


def test_clean_text_joins_hyphenated_line_breaks() -> None:
    assert clean_text("capabili-\nties and taxon-\nomy") == "capabilities and taxonomy"