"""Text preprocessing: normalisation, word tokenisation and detokenisation.

Choices (justified in the report, section 4):
* Unicode NFKC + unified quotes: the same character always gets the same code.
* Lowercasing: politeness does not depend on case, and lowercasing halves the number of word
  forms the model has to learn ("Could" / "could"). The interface re-capitalises the output.
* Word-level tokens; punctuation marks are separate tokens because "?" and "." carry meaning
  here (a polite request is usually a question). Contractions stay one token ("don't").
"""
import re
import unicodedata

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})
TOKEN_RE = re.compile(r"\w+(?:'\w+)?|[^\w\s]")
_NO_SPACE_BEFORE = re.compile(r" ([.,!?;:%)\]}])")
_NO_SPACE_AFTER = re.compile(r"([(\[{$]) ")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).translate(_QUOTES).lower()
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize(text))


def detokenize(tokens: list[str], capitalize: bool = False) -> str:
    """'could you help me ?' -> 'could you help me?' ; capitalize=True also restores sentence
    capitals and the pronoun 'I' (used for display only, metrics work on lowercase tokens)."""
    text = " ".join(tokens)
    text = _NO_SPACE_BEFORE.sub(r"\1", text)
    text = _NO_SPACE_AFTER.sub(r"\1", text)
    if capitalize:
        text = re.sub(r"(\w) - (\w)", r"\1-\2", text)          # "double - check" -> "double-check"
        text = re.sub(r"\bi\b", "I", text)
        text = re.sub(r"\bi'(m|d|ll|ve)\b", lambda m: "I'" + m.group(1), text)
        text = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    return text


def restore_case(tokens: list[str], source_text: str) -> list[str]:
    """Display helper: give output words the casing they had in the user's input
    ('tv' -> 'TV', 'sarah' -> 'Sarah'). Words not in the input keep their lowercase form."""
    raw = unicodedata.normalize("NFKC", str(source_text)).translate(_QUOTES)
    cased, prev = {}, "."
    for t in TOKEN_RE.findall(raw):
        starts_sentence = prev in (".", "!", "?", ":")
        special = len(t) > 1 and t[1:].lower() != t[1:]          # TV, McDonald, iPhone
        if t.lower() != t and (special or not starts_sentence):   # skip capitals due to sentence start only
            cased.setdefault(t.lower(), t)
        prev = t
    return [cased.get(t, t) for t in tokens]
