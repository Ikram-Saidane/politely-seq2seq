# =============================================================================
# data/generate_dataset.py
# Builds the impolite -> polite dataset from the hand-written blocks of resources.py.
#   python data/generate_dataset.py               -> final dataset in data/
#   python data/generate_dataset.py --version 1   -> first version in data_v1/
#
# For each category we choose a content (action, object, person, time...), then write it
# with one impolite template and 3 polite templates (same content, different tone).
# Splits are made by content, and test_unseen only uses impolite templates never seen in training.
# =============================================================================
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import resources as R

R2 = R                         # final version of the dataset
R1 = SimpleNamespace(**R.V1)   # first version
V2 = True

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEED = 42

# number of inputs per category
N_TRAIN, N_VALID, N_TEST, N_UNSEEN = 1600, 150, 150, 100
N_TRAIN_V1 = 1200
REFS_EVAL = 3    # polite references per input in valid / test
REFS_TRAIN = 2   # polite references per input in train


def _fill(text, s):
    return text.format(**s)


def _base_slots(rng):
    out = {
        "doc": rng.choice(R.DOCS), "event": rng.choice(R.EVENTS), "place": rng.choice(R.PLACES),
        "item": rng.choice(R.ITEMS), "name": rng.choice(R.NAMES), "day": rng.choice(R.DAYS),
        "clock": rng.choice(R.CLOCK), "city": rng.choice(R.CITIES), "ordinal": rng.choice(R.ORDINALS),
        "price": rng.choice(R.PRICES), "topic": rng.choice(R.TOPICS),
    }
    out["place2"] = rng.choice([p for p in R.PLACES if p != out["place"]])
    if not V2:
        return out
    # new slots added in version 2
    out.update({
        "openable": rng.choice(R.OPENABLE), "device": rng.choice(R.DEVICES), "loud": rng.choice(R.LOUD),
        "obj_s": rng.choice(R.OBJ_S), "obj_p": rng.choice(R.OBJ_P), "food": rng.choice(R.FOOD),
        "food_n": rng.choice(R.FOOD_N), "supplies": rng.choice(R.SUPPLIES), "room": rng.choice(R.ROOMS),
        "bill": rng.choice(R.BILLS), "facility": rng.choice(R.FACILITIES), "count": rng.choice(R.COUNT),
    })
    out["obj"] = rng.choice([out["obj_s"], out["obj_p"]])
    return out


# verbs where "Do it" / "<Verb> it" sounds wrong
NO_IT = {"turn", "switch", "pick", "put", "take", "give", "fill", "tidy", "drop", "get", "lend", "save",
         "hand", "pass", "bring", "make", "walk", "feed", "set", "do", "empty", "buy", "order", "call",
         "reply", "remind", "schedule", "water", "move", "share", "explain", "carry", "wash", "find"}
P_REASON = 0.3   # probability to add a reason ("We are already late.")


def _reason(group, b, rng):
    if V2 and group and rng.random() < P_REASON:
        return _fill(rng.choice(R.REASONS[group]), b)
    return ""


def sample_content(category, rng):
    b = _base_slots(rng)
    if category in ("request", "reminder"):
        action = rng.choice(R.ACTIONS)
        base, ing, pp, comp = action[:4]
        group = action[4] if len(action) > 4 else ("work" if V2 and base in R.WORK_VERBS else None)
        it_ok = base not in NO_IT and comp.startswith(("the ", "my ", "your "))
        comp = _fill(comp, b)
        t_imp, t_pol = rng.choice(R.TIMES)
        if category == "reminder" and t_imp in ("now", "right now", "immediately"):
            t_imp, t_pol = "today", "today"
        reason = _reason(group, b, rng)
        s = {"base": base, "ing": ing, "pp": pp, "comp": comp, "time": t_imp, "ptime": t_pol,
             "reason": reason, "it_ok": it_ok or not V2}
        key = (base, comp, reason, t_pol) if V2 else (base, comp, t_pol)
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
        contexts = R.THING_CONTEXT_BY_KIND.get(kind, R.THING_CONTEXT) if V2 else R.THING_CONTEXT
        ctx = _fill(rng.choice(contexts), b)
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
        key = (s["behaviour"], s["reason"]) if V2 else (s["behaviour"],)
    elif category == "announcement":
        if V2 and rng.random() < 0.3:
            s = {"event": b["facility"], "change": _fill(rng.choice(R.FACILITY_CHANGES), b)}
        else:
            s = {"event": rng.choice(R.EVENTS), "change": _fill(rng.choice(R.CHANGES), b)}
        key = (s["event"], s["change"])
    else:
        raise ValueError(category)
    return (category,) + key, s


def _cap(x):
    return x[:1].upper() + x[1:] if x else x


def realise(template, slots):
    # {Slot} with a capital letter puts a capital on the value
    filled = {k: v for k, v in slots.items() if isinstance(v, str)}
    for k, v in list(filled.items()):
        filled[_cap(k)] = _cap(v)
    text = template.format(**filled)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,!?:;])", r"\1", text)
    text = re.sub(r"([.!?])\s*\.", r"\1", text)   # "?." -> "?"
    text = text.strip()
    text = _cap(text)
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)


def banned_tokens(text, banned):
    return {w for w in re.findall(r"[a-z']+", text.lower()) if w in banned}


def generate(seed=SEED, version=2):
    global R, V2
    if version == 2:
        R, V2 = R2, True
    else:
        R, V2 = R1, False
    n_train = N_TRAIN if V2 else N_TRAIN_V1
    rng = random.Random(seed)
    banned = set(R.BANNED_WORDS)

    rows, used_keys = [], set()
    plan = [("train", n_train, False), ("valid", N_VALID, False), ("test", N_TEST, False),
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
                    # no time in the impolite sentence, so no time in the polite one
                    slots["time"], slots["ptime"] = "", ""
                    key = key[:-1] + ("",)
                if key in used_keys:   # this content is already in a split
                    continue
                impolite = realise(imp_tpl, slots)
                # the polite template must keep the time if there is one
                polite_ok = [p for p in tpl["polite"] if not slots.get("ptime") or "{ptime}" in p]
                refs = [realise(p, slots) for p in rng.sample(polite_ok, REFS_EVAL)]
                if slots.get("reason"):   # the reason is copied as it is
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


def write(rows, out=None):
    folder = Path(out) if out else DATA
    folder.mkdir(exist_ok=True)
    fields = ["id", "split", "category", "template_id", "ref_index", "impolite", "polite"]
    with open(folder / "politeness_dataset.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    for split in ("train", "valid", "test", "test_unseen"):
        with open(folder / f"{split}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(r for r in rows if r["split"] == split)
    by_split = Counter(r["split"] for r in rows)
    inputs = Counter(r["split"] for r in rows if r["ref_index"] == 0)
    card = {
        "name": "Politely — impolite-to-polite paraphrase pairs (English)",
        "version": 2 if V2 else 1, "seed": SEED, "categories": R.CATEGORY_DESCRIPTIONS,
        "pairs_per_split": dict(by_split), "inputs_per_split": dict(inputs),
        "references_per_input": {"train": REFS_TRAIN, "valid/test/test_unseen": REFS_EVAL},
        "impolite_templates": {c: {"seen": sum(not t[2] for t in v["impolite"]),
                                   "held_out": sum(t[2] for t in v["impolite"])}
                               for c, v in R.TEMPLATES.items()},
        "polite_templates_per_category": {c: len(v["polite"]) for c, v in R.TEMPLATES.items()},
    }
    (folder / "dataset_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    return card


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, default=2, choices=[1, 2])
    a = ap.parse_args()
    out = None if a.version == 2 else ROOT / "data_v1"
    card = write(generate(version=a.version), out=out)
    print(json.dumps({k: card[k] for k in ("pairs_per_split", "inputs_per_split")}, indent=2))
