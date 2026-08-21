import re
import html

def clean_text(value: object) -> str:
    """Decode HTML, remove tags, and normalize whitespace."""

    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]*>", " ", text)
    return re.sub(r"\s+", " ", text).strip()
