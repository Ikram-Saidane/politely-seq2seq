"""Two non-neural baselines that every model is compared with.

copy   output = input. Keeps all the content, but is as impolite as the input.
rules  a hand-written rule system: delete common markers of impatience / dismissiveness and add
       "please" at the beginning. It is the obvious "no machine learning" solution.
       The marker list was written from general knowledge of impolite English; it is NOT derived
       from the dataset's held-out templates.
"""
import re

from politely.preprocessing import detokenize, tokenize

RUDE_MARKERS = [
    "and stop wasting my time", "i am not asking twice", "i'm not asking twice", "no excuses",
    "deal with it", "not my problem", "stop asking me about it", "stop asking", "get it right",
    "i don't have all day", "answer me", "hurry up", "end of discussion", "get used to it",
    "this is ridiculous", "unbelievable", "for the last time", "how many times do i have to ask",
    "are you even listening", "it's annoying", "understood", "already", "right now", "immediately",
    "now", "seriously", "honestly", "obviously", "come on", "just", "look", "listen", "well",
    "whatever", "period", "do it", "no way", "forget it", "absolutely not", "typical", "again",
]
_RE = re.compile(r"\b(" + "|".join(re.escape(m) for m in sorted(RUDE_MARKERS, key=len, reverse=True)) + r")\b")


def copy_baseline(tokens):
    return list(tokens)


def rules_baseline(tokens):
    text = _RE.sub(" ", detokenize(tokens))
    out = tokenize(text)
    # remove punctuation left orphaned at the start or doubled by the deletions
    cleaned = []
    for t in out:
        if t in ",.!?" and (not cleaned or cleaned[-1] in ",.!?"):
            continue
        cleaned.append(t)
    return ["please", ","] + cleaned if cleaned else ["please", "."]
