import re
from unidecode import unidecode

ABBREV = {
    r"\bcorp\b": "corporation",
    r"\bpvt\b": "private",
    r"\bltd\b": "limited",
    r"\bco\b": "company",
    r"\binc\b": "incorporated",
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bnr\b": "near",
    r"\bopp\b": "opposite",
    r"\b&\b": "and",
}


def normalize(s):
    """Lowercase, strip accents, remove punctuation, expand abbreviations."""
    if not isinstance(s, str):
        return ""

    s = unidecode(s).lower()
    s = re.sub(r"[^\w\s]", " ", s)

    for pat, rep in ABBREV.items():
        s = re.sub(pat, rep, s)

    return re.sub(r"\s+", " ", s).strip()