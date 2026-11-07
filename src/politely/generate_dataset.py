"""Generate the synthetic impolite -> polite paraphrase dataset.

    python -m politely.generate_dataset            (from the project root, with src/ on the path)

Outputs (in data/):
    politeness_dataset.csv   every pair, with its split, category and template id
    train.csv / valid.csv / test.csv / test_unseen.csv   the same pairs, one file per split
    dataset_card.json        sizes and generation settings

How the data is built (see resources.py for the hand-written blocks):
1. For a category (request, refusal, ...) we draw the CONTENT of a message: an action and its
   object, a person, a time, a fact... ("slots").
2. We realise this content once with an impolite template and three times with three different
   polite templates: one impolite input, three polite references with exactly the same content.
3. Splits are made by CONTENT: the same slot combination never appears in two splits, so a test
   sentence is never a training sentence with a different wording.
4. test_unseen uses only impolite templates that are never used for train/valid/test: it measures
   whether the model generalises to new phrasings rather than memorising templates.
5. Every sentence is checked against a list of banned words (insults, swear words, hateful terms).
"""
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

from politely import resources as R

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SEED = 42

# Inputs per category and split (9 categories -> x9).
N_TRAIN, N_VALID, N_TEST, N_UNSEEN = 1600, 150, 150, 100
REFS_EVAL = 3          # references per input in valid / test / test_unseen
REFS_TRAIN = 2         # references per input used as training pairs


# --------------------------------------------------------------------------------------- slots
def _fill(text, s):
    """Fill nested slots ({doc}, {name}, ...) inside a resource string."""
    return text.format(**s)


def _base_slots(rng):
    out = {
        "doc": rng.choice(R.DOCS), "event": rng.choice(R.EVENTS), "place": rng.choice(R.PLACES),
        "item": rng.choice(R.ITEMS), "name": rng.choice(R.NAMES), "day": rng.choice(R.DAYS),
        "clock": rng.choice(R.CLOCK), "city": rng.choice(R.CITIES), "ordinal": rng.choice(R.ORDINALS),
        "price": rng.choice(R.PRICES), "topic": rng.choice(R.TOPICS),
    }
    out["place2"] = rng.choice([p for p in R.PLACES if p != out["place"]])
    # v2 slots: everyday objects, rooms, bills, facilities
    out.update({
        "openable": rng.choice(R.OPENABLE), "device": rng.choice(R.DEVICES), "loud": rng.choice(R.LOUD),
        "obj_s": rng.choice(R.OBJ_S), "obj_p": rng.choice(R.OBJ_P), "food": rng.choice(R.FOOD),
        "food_n": rng.choice(R.FOOD_N), "supplies": rng.choice(R.SUPPLIES), "room": rng.choice(R.ROOMS),
        "bill": rng.choice(R.BILLS), "facility": rng.choice(R.FACILITIES), "count": rng.choice(R.COUNT),
    })
    out["obj"] = rng.choice([out["obj_s"], out["obj_p"]])
    return out


# Actions after which "Do it / <Verb> it" sounds wrong (phrasal verbs, double objects...).
NO_IT = {"turn", "switch", "pick", "put", "take", "give", "fill", "tidy", "drop", "get", "lend", "save",
         "hand", "pass", "bring", "make", "walk", "feed", "set", "do", "empty", "buy", "order", "call",
         "reply", "remind", "schedule", "water", "move", "share", "explain", "carry", "wash", "find"}
P_REASON = 0.3          # share of requests / reminders / stops that come with an explanation


def _reason(group, b, rng):
    if group and rng.random() < P_REASON:
        return _fill(rng.choice(R.REASONS[group]), b)
    return ""


def sample_content(category, rng):
    """Return (content_key, slots): the slots used by the templates of this category."""
    b = _base_slots(rng)
    if category in ("request", "reminder"):
        action = rng.choice(R.ACTIONS)
        base, ing, pp, comp = action[:4]
        group = action[4] if len(action) > 4 else ("work" if base in R.WORK_VERBS else None)
        it_ok = base not in NO_IT and comp.startswith(("the ", "my ", "your "))
        comp = _fill(comp, b)
        t_imp, t_pol = rng.choice(R.TIMES)
        if category == "reminder" and t_imp in ("now", "right now", "immediately"):
            t_imp, t_pol = "today", "today"
        reason = _reason(group, b, rng)
        s = {"base": base, "ing": ing, "pp": pp, "comp": comp, "time": t_imp, "ptime": t_pol,
             "reason": reason, "it_ok": it_ok}
        key = (base, comp, reason, t_pol)
    elif category == "refusal":
        a_base, a_ing = rng.choice(R.ACTIVITIES)
        has_when = any(w in a_base for w in ("{day}", "week", "month", "weekend", "holiday"))
        when = "" if has_when else _fill(rng.choice(R.REFUSAL_WHEN), b)
        s = {"abase": _fill(a_base, b) + (" " + when if when else ""),
             "aing": _fill(a_ing, b) + (" " + when if when else "")}
        key = (s["abase"],)
    elif category == "criticism":
        kind = rng.choice(sorted(R.THINGS))
        thing = rng.choice(R.THINGS[kind])
        ctx = _fill(rng.choice(R.THING_CONTEXT_BY_KIND.get(kind, R.THING_CONTEXT)), b)
        thing = thing + (" " + ctx if ctx else "")
        harsh, soft, _ = rng.choice([i for i in R.ISSUES if kind in i[2]])
        s = {"thing": thing, "harsh": harsh, "soft": soft}
        key = (thing, harsh)
    elif category == "complaint":
        service = rng.choice(sorted(R.SERVICES))
        problem, tense = rng.choice(R.SERVICES[service])
        dur = _fill(rng.choice(R.DURATIONS[tense]), b)
        if service in R.NUMBERED_SERVICES and rng.random() < 0.5:
            service = f"{service} number {rng.randint(1000, 9999)}"
        s = {"service": service, "problem": problem, "dur": dur}
        key = (service, problem, dur)
    elif category == "disagreement":
        fact = _fill(rng.choice(R.FACTS), b)
        s = {"fact": fact}
        key = (fact,)
    elif category == "question":
        direct, embedded = rng.choice(R.QUESTIONS)
        s = {"direct": _fill(direct, b), "embedded": _fill(embedded, b)}
        key = (s["direct"],)
    elif category == "stop":
        beh = rng.choice(R.BEHAVIOURS)
        has_ctx = any(w in beh for w in (" in ", " during ", " at ", " on ", " after ", " in front",
                                         "without", "{place}", "{event}", " to "))
        ctx = "" if has_ctx else _fill(rng.choice(R.STOP_CONTEXT), b)
        s = {"behaviour": _fill(beh, b) + (" " + ctx if ctx else ""), "reason": _reason("stop", b, rng)}
        key = (s["behaviour"], s["reason"])
    elif category == "announcement":
        if rng.random() < 0.3:      # v2: facilities (pool, gym...) are announced too
            s = {"event": b["facility"], "change": _fill(rng.choice(R.FACILITY_CHANGES), b)}
        else:
            s = {"event": rng.choice(R.EVENTS), "change": _fill(rng.choice(R.CHANGES), b)}
        key = (s["event"], s["change"])
    else:
        raise ValueError(category)
    return (category,) + key, s


# --------------------------------------------------------------------------------------- realisation
def _cap(x):
    return x[:1].upper() + x[1:] if x else x


def realise(template, slots):
    """Fill a template; {Slot} (capitalised name) inserts the value with a capital first letter."""
    filled = {k: v for k, v in slots.items() if isinstance(v, str)}
    for k, v in list(filled.items()):
        filled[_cap(k)] = _cap(v)
    text = template.format(**filled)
    text = re.sub(r"\s+", " ", text)                    # empty slots leave double spaces
    text = re.sub(r"\s+([.,!?:;])", r"\1", text)        # no space before punctuation
    text = re.sub(r"([.!?])\s*\.", r"\1", text)         # "?." -> "?"
    text = text.strip()
    text = _cap(text)
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)


def banned_tokens(text, banned):
    return {w for w in re.findall(r"[a-z']+", text.lower()) if w in banned}


# --------------------------------------------------------------------------------------- generation
def generate(seed=SEED, extra_banned=None):
    """extra_banned: optional path to a larger list of offensive words, one per line, used only
    as an additional filter (it is not part of the dataset)."""
    rng = random.Random(seed)
    banned = set(R.BANNED_WORDS)
    if extra_banned:
        banned |= {w.strip().lower() for w in Path(extra_banned).read_text(encoding="utf-8").split()
                   if w.strip() and not w.startswith("#")}

    rows, used_keys = [], set()
    plan = [("train", N_TRAIN, False), ("valid", N_VALID, False), ("test", N_TEST, False),
            ("test_unseen", N_UNSEEN, True)]
    for category, tpl in R.TEMPLATES.items():
        seen_tpl = [t for t in tpl["impolite"] if not t[2]]
        held_tpl = [t for t in tpl["impolite"] if t[2]]
        for split, n, held in plan:
            templates = held_tpl if held else seen_tpl
            made, tries = 0, 0
            while made < n:
                tries += 1
                if tries > 200 * n:
                    raise RuntimeError(f"not enough distinct content for {category}/{split}")
                key, slots = sample_content(category, rng)
                usable = [t for t in templates if slots.get("it_ok", True) or "{Base} it" not in t[1]]
                tid, imp_tpl, _ = rng.choice(usable)
                if "time" in slots and "{time}" not in imp_tpl:
                    # The impolite sentence says no time, so the polite one must not invent one.
                    slots["time"], slots["ptime"] = "", ""
                    key = key[:-1] + ("",)
                if key in used_keys:                     # content already used in some split
                    continue
                impolite = realise(imp_tpl, slots)
                # A polite template that has no {ptime} slot would DROP the time given in the
                # impolite sentence: only templates that keep all the content are allowed.
                polite_ok = [p for p in tpl["polite"] if not slots.get("ptime") or "{ptime}" in p]
                refs = [realise(p, slots) for p in rng.sample(polite_ok, REFS_EVAL)]
                if slots.get("reason"):           # the explanation is carried over unchanged
                    extra = " " + _cap(slots["reason"]) + "."
                    impolite = impolite + extra
                    refs = [r + extra for r in refs]
                for sent in [impolite] + refs:
                    bad = banned_tokens(sent, banned)
                    if bad:
                        raise ValueError(f"banned word {bad} in: {sent}")
                used_keys.add(key)
                k = REFS_TRAIN if split == "train" else REFS_EVAL
                for r_i, ref in enumerate(refs[:k]):
                    rows.append({"id": f"{category[:3]}-{split}-{made:04d}", "split": split,
                                 "category": category, "template_id": tid, "ref_index": r_i,
                                 "impolite": impolite, "polite": ref})
                made += 1
    return rows


def write(rows):
    DATA.mkdir(exist_ok=True)
    fields = ["id", "split", "category", "template_id", "ref_index", "impolite", "polite"]
    with open(DATA / "politeness_dataset.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    for split in ("train", "valid", "test", "test_unseen"):
        with open(DATA / f"{split}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(r for r in rows if r["split"] == split)
    by_split = Counter(r["split"] for r in rows)
    inputs = Counter(r["split"] for r in rows if r["ref_index"] == 0)
    card = {
        "name": "Politely — synthetic impolite-to-polite paraphrase pairs (English)",
        "seed": SEED, "categories": R.CATEGORY_DESCRIPTIONS,
        "pairs_per_split": dict(by_split), "inputs_per_split": dict(inputs),
        "references_per_input": {"train": REFS_TRAIN, "valid/test/test_unseen": REFS_EVAL},
        "impolite_templates": {c: {"seen": sum(not t[2] for t in v["impolite"]),
                                   "held_out": sum(t[2] for t in v["impolite"])}
                               for c, v in R.TEMPLATES.items()},
        "polite_templates_per_category": {c: len(v["polite"]) for c, v in R.TEMPLATES.items()},
    }
    (DATA / "dataset_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    return card


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra-banned", help="optional extra list of offensive words to filter")
    a = ap.parse_args()
    card = write(generate(extra_banned=a.extra_banned))
    print(json.dumps({k: card[k] for k in ("pairs_per_split", "inputs_per_split")}, indent=2))
