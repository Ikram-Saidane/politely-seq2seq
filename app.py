# =============================================================================
# app.py
# User interface (Streamlit). It uses the models trained by train.py and saved in checkpoints/.
#
#   streamlit run app.py
# =============================================================================
import json
import time

import altair as alt
import pandas as pd
import streamlit as st

from evaluate import PUNCT, RUDE_MARKERS, STOP, copy_baseline, has_repetition, rules_baseline, stem
from inference import FINAL_MODEL, beam_search, greedy_decode, load_model
from politeness import PolitenessClassifier
from preprocessing import ROOT, detokenize, restore_case, tokenize

MAX_CHARS, MAX_TOKENS = 300, 40

# name shown in the interface + short description, in the order of the report
MODELS = {
    "A_lstm": ("A · LSTM encoder–decoder", "Baseline: the whole sentence is squeezed into one vector."),
    "B_attention": ("B · + attention", "The decoder looks back at every input word at each step."),
    "C_copy_nodrop": ("C · + copy mechanism (final)", "Can copy input words (names, numbers, unknown words)."),
    "C_copy": ("C' · copy + word dropout", "Model C trained with 10% of input words hidden behind <UNK>."),
    "D_gru_attention": ("D · GRU + attention", "Model B with GRU cells instead of LSTM cells."),
    "E_attention_h128": ("E · attention, hidden 128", "Model B with a smaller hidden state."),
    "E_attention_h512": ("E · attention, hidden 512", "Model B with a larger hidden state."),
    "F_attention_tf0.5": ("F · attention, teacher forcing 0.5", "Model B trained on its own predictions half of the time."),
}
DECODINGS = {"Beam search (5)": "beam", "Greedy": "greedy", "Sampling": "sample"}
RUDE_WORDS = {w for m in RUDE_MARKERS for w in m.split()}
TEAL, ORANGE, RUST, GREY = "#176F63", "#E0A33A", "#B4532A", "#9AA3A0"

st.set_page_config(page_title="Politely", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1150px;}
.hero h1 {font-size: 2.4rem; margin-bottom: 0.2rem;}
.hero p {font-size: 1.05rem; opacity: 0.8; margin-top: 0;}
.result {border-left: 5px solid #176F63; background: rgba(23,111,99,0.08); padding: 1rem 1.3rem;
         border-radius: 8px; font-size: 1.35rem; margin: 0.5rem 0 1rem 0;}
.word {display: inline-block; padding: 0.1rem 0.55rem; margin: 0.15rem; border-radius: 999px;
       font-size: 0.9rem;}
.kept {background: rgba(23,111,99,0.15); color: #176F63;}
.lost {background: rgba(180,83,42,0.15); color: #B4532A; text-decoration: line-through;}
.tok {display: inline-block; font-family: monospace; padding: 0.1rem 0.4rem; margin: 0.12rem;
      border: 1px solid rgba(128,128,128,0.35); border-radius: 4px; font-size: 0.85rem;}
.small {opacity: 0.7; font-size: 0.9rem;}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Loading (cached so that it is done only once)
# -----------------------------------------------------------------------------
def available_models():
    return [m for m in MODELS if (ROOT / "checkpoints" / f"{m}.pt").exists()]


@st.cache_resource
def get_model(name):
    return load_model(name)


@st.cache_resource
def get_classifier():
    return PolitenessClassifier.load()


@st.cache_data
def read_json(path, default=None):
    p = ROOT / path
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


@st.cache_data
def read_csv(path):
    return pd.read_csv(ROOT / path, keep_default_na=False)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def check_input(text):
    # returns (tokens, error message)
    text = (text or "").strip()
    if not text:
        return None, "Please type a sentence."
    if len(text) > MAX_CHARS:
        return None, f"Please enter at most {MAX_CHARS} characters (you typed {len(text)})."
    if not any(c.isalpha() for c in text):
        return None, "The sentence must contain words."
    tokens = tokenize(text)
    if len(tokens) > MAX_TOKENS:
        return None, f"The model handles up to {MAX_TOKENS} tokens; your sentence has {len(tokens)}."
    return tokens, None


def run(name, tokens, decoding="beam", beam_size=5, temperature=0.8):
    model, vocab = get_model(name)
    if decoding == "beam":
        return beam_search(model, tokens, vocab, k=beam_size)
    if decoding == "sample":
        return greedy_decode(model, [tokens], vocab, sample={"temperature": temperature, "seed": int(time.time())})[0]
    return greedy_decode(model, [tokens], vocab)[0]


def describe(tokens, out_tokens):
    # politeness before / after and the content words kept by the rewrite
    keys = [t for t in dict.fromkeys(tokens) if t not in STOP and t not in PUNCT and t not in RUDE_WORDS]
    out_stems = {stem(t) for t in out_tokens}
    kept = [k for k in keys if stem(k) in out_stems]
    p_in, p_out = get_classifier().predict_proba([tokens, out_tokens or ["."]])
    return {"politeness_in": float(p_in), "politeness_out": float(p_out) if out_tokens else 0.0,
            "keys": keys, "kept": kept, "lost": [k for k in keys if k not in kept],
            "content_kept": len(kept) / len(keys) if keys else 1.0,
            "repetition": has_repetition(out_tokens)}


def label(system):
    # "C_copy_nodrop__beam5" -> "C · + copy mechanism (final) + beam 5"
    base, _, dec = system.partition("__")
    names = {"baseline_copy": "Copy baseline", "baseline_rules": "Rule-based baseline"}
    name = names.get(base, MODELS.get(base, (base,))[0])
    return name + (f" + {dec.replace('beam', 'beam ')}" if dec else "")


# -----------------------------------------------------------------------------
# Sidebar: settings
# -----------------------------------------------------------------------------
models = available_models()
if not models:
    st.error("No trained model in checkpoints/. Train one first: python train.py --config configs/C_copy_nodrop.json")
    st.stop()

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Model", models, index=models.index(FINAL_MODEL) if FINAL_MODEL in models else 0,
                              format_func=lambda m: MODELS[m][0])
    st.caption(MODELS[model_name][1])
    decoding = DECODINGS[st.radio("Decoding", list(DECODINGS))]
    beam_size, temperature = 5, 0.8
    if decoding == "beam":
        beam_size = st.slider("Beam size", 1, 10, 5)
    if decoding == "sample":
        temperature = st.slider("Temperature", 0.1, 1.5, 0.8, 0.1,
                                help="Low = safe and repetitive, high = varied but more mistakes")
    st.divider()
    st.markdown("**Politely** rewrites an impolite English sentence into a polite one with the same meaning.")
    st.caption("NLP course project · LSTM encoder–decoder with attention and copy mechanism, "
               "implemented and trained from scratch with PyTorch.")

st.markdown('<div class="hero"><h1>Politely</h1><p>From an impolite sentence to a polite one with the same '
            'meaning. Task: paraphrase generation (impolite → polite). Model: an LSTM encoder–decoder with '
            'attention and a copy mechanism, trained from scratch on 28,800 sentence pairs.</p></div>',
            unsafe_allow_html=True)

tab_rewrite, tab_compare, tab_results, tab_data, tab_how = st.tabs(
    ["Rewrite", "Compare models", "Training & results", "Dataset", "How it works"])


# -----------------------------------------------------------------------------
# Tab 1: rewrite a sentence
# -----------------------------------------------------------------------------
with tab_rewrite:
    if "text" not in st.session_state:
        st.session_state.text = "Close the window now."

    # demo examples: one test sentence and one hand-written sentence per category
    examples = []
    test = read_csv("data/test.csv").drop_duplicates("impolite")
    for cat, g in test.groupby("category"):
        examples.append(f"[test · {cat}] " + g.sample(1, random_state=7).iloc[0]["impolite"])
    challenge = read_csv("data/challenge.csv")
    for cat, g in challenge.groupby("category"):
        examples.append(f"[hand-written · {cat}] " + g.iloc[0]["impolite"])

    def pick_example():
        if st.session_state.example:
            st.session_state.text = st.session_state.example.split("] ", 1)[1]

    st.selectbox("Try an example", examples, index=None, key="example", on_change=pick_example,
                 placeholder="Choose a test sentence or a sentence written by hand…")

    text = st.text_area("Impolite sentence", key="text", height=90, max_chars=MAX_CHARS)
    go = st.button("Rewrite politely", type="primary")

    if go:
        tokens, error = check_input(text)
        if error:
            st.warning(error)
        else:
            t0 = time.time()
            r = run(model_name, tokens, decoding, beam_size, temperature)
            ms = (time.time() - t0) * 1000
            out = r["tokens"]
            polite = detokenize(restore_case(out, text), capitalize=True)
            d = describe(tokens, out)
            vocab = get_model(model_name)[1]

            st.markdown(f'<div class="result">{polite or "(empty output)"}</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Politeness before", f"{d['politeness_in']:.0%}")
            c2.metric("Politeness after", f"{d['politeness_out']:.0%}",
                      delta=f"{(d['politeness_out'] - d['politeness_in']) * 100:+.0f} pts")
            c3.metric("Content kept", f"{d['content_kept']:.0%}")
            c4.metric("Time", f"{ms:.0f} ms")

            chips = "".join(f'<span class="word kept">{w}</span>' for w in d["kept"])
            chips += "".join(f'<span class="word lost">{w}</span>' for w in d["lost"])
            st.markdown("**Content words of your sentence** (green = kept, red = lost): " + (chips or "none"),
                        unsafe_allow_html=True)

            unknown = [t for t in dict.fromkeys(tokens) if t not in vocab and t not in PUNCT]
            if unknown:
                note = "the copy mechanism can still reuse them." if get_model(model_name)[0].cfg.copy \
                    else "they are read as <UNK> and may be lost."
                st.info(f"Words never seen in training: {', '.join(unknown)} — {note}")
            if d["repetition"]:
                st.warning("The output repeats a word (a known weakness of small seq2seq models).")
            if out and d["politeness_out"] < 0.5:
                st.warning("The politeness classifier does not find the output polite enough.")

            with st.expander("Attention map: where the decoder looked for each word", expanded=True):
                if r.get("attention") is None:
                    st.caption("Model A has no attention mechanism.")
                else:
                    att = r["attention"]
                    rows_lbl = (out + ["<EOS>"])[:att.shape[0]]
                    cols_lbl = tokens[:att.shape[1]]
                    cells = [{"output": f"{i:02d} {rows_lbl[i]}", "input": f"{j:02d} {cols_lbl[j]}",
                              "weight": float(att[i, j])}
                             for i in range(att.shape[0]) for j in range(att.shape[1])]
                    chart = alt.Chart(pd.DataFrame(cells)).mark_rect().encode(
                        x=alt.X("input:O", title="input word", sort=None, axis=alt.Axis(labelAngle=-40)),
                        y=alt.Y("output:O", title="generated word", sort=None),
                        color=alt.Color("weight:Q", scale=alt.Scale(scheme="greens", domain=[0, 1])),
                        tooltip=["output", "input", alt.Tooltip("weight:Q", format=".2f")],
                    ).properties(height=max(220, 24 * len(rows_lbl)))
                    st.altair_chart(chart, width="stretch")
                    st.caption("Each row is a generated word; dark cells are the input words used to write it.")

            if r.get("p_gen"):
                with st.expander("Copy switch: generate or copy?"):
                    pg = pd.DataFrame({"word": [f"{i:02d} {w}" for i, w in enumerate((out + ["<EOS>"])[:len(r["p_gen"])])],
                                       "copy probability": [1 - p for p in r["p_gen"]]})
                    st.altair_chart(alt.Chart(pg).mark_bar(color=TEAL).encode(
                        x=alt.X("word:O", sort=None, axis=alt.Axis(labelAngle=-40)),
                        y=alt.Y("copy probability:Q", scale=alt.Scale(domain=[0, 1]))), width="stretch")
                    st.caption("1 − p_gen: high bars are words copied from your sentence.")

            with st.expander("Tokens and ids"):
                ids = vocab.encode(tokens, add_eos=True)
                st.markdown("Input tokens: " + "".join(f'<span class="tok">{t}</span>' for t in tokens + ["<EOS>"]),
                            unsafe_allow_html=True)
                st.markdown("Input ids: " + "".join(f'<span class="tok">{i}</span>' for i in ids),
                            unsafe_allow_html=True)
                st.markdown("Generated tokens: " + "".join(f'<span class="tok">{t}</span>' for t in out + ["<EOS>"]),
                            unsafe_allow_html=True)
                if r.get("token_probs"):
                    st.caption("Probability of each generated word: " +
                               " · ".join(f"{w} {p:.2f}" for w, p in zip(out + ["<EOS>"], r["token_probs"])))


# -----------------------------------------------------------------------------
# Tab 2: compare every model on the same sentence
# -----------------------------------------------------------------------------
with tab_compare:
    st.subheader("Same sentence, every system")
    st.caption("All models use greedy decoding here, the baselines need no training.")
    cmp_text = st.text_input("Impolite sentence", value="Turn the music down, it's too loud.", key="cmp")
    if st.button("Compare", type="primary"):
        tokens, error = check_input(cmp_text)
        if error:
            st.warning(error)
        else:
            rows = [("Copy baseline", copy_baseline(tokens)), ("Rule-based baseline", rules_baseline(tokens))]
            for m in models:
                rows.append((MODELS[m][0], run(m, tokens, "greedy")["tokens"]))
            table = []
            for name, out in rows:
                d = describe(tokens, out)
                table.append({"System": name, "Output": detokenize(restore_case(out, cmp_text), capitalize=True),
                              "Polite": round(d["politeness_out"], 2), "Content kept": round(d["content_kept"], 2)})
            st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch",
                         column_config={"Polite": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                                        "Content kept": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})


# -----------------------------------------------------------------------------
# Tab 3: training curves and evaluation results
# -----------------------------------------------------------------------------
with tab_results:
    results = read_json("reports/results.json", [])
    st.subheader("Success rate on the three test sets")
    st.caption("Success = judged polite AND all content words kept AND no repeated word. "
               "test = same phrasings as training · test_unseen = new impolite phrasings · challenge = 72 sentences written by hand.")
    systems = ["baseline_copy", "baseline_rules"] + [m for m in MODELS] + ["C_copy_nodrop__beam5"]
    rows = []
    for r in results:
        if r["system"] in systems and r["split"] in ("test", "test_unseen", "challenge"):
            rows.append({"system": label(r["system"]), "order": systems.index(r["system"]), "split": r["split"],
                         "success": r["success_rate"], "BLEU": r["BLEU"], "content kept": r["content_preservation"],
                         "polite": r["polite_rate"]})
    res = pd.DataFrame(rows)
    if not res.empty:
        st.altair_chart(alt.Chart(res).mark_bar().encode(
            y=alt.Y("system:N", sort=alt.EncodingSortField("order"), title=None, axis=alt.Axis(labelLimit=300)),
            x=alt.X("success:Q", scale=alt.Scale(domain=[0, 1])),
            yOffset=alt.YOffset("split:N", sort=["test", "test_unseen", "challenge"]),
            color=alt.Color("split:N", sort=["test", "test_unseen", "challenge"],
                            scale=alt.Scale(range=[TEAL, ORANGE, RUST])),
            tooltip=["system", "split", alt.Tooltip("success:Q", format=".3f")],
        ).properties(height=620), width="stretch")
        table = res.pivot_table(index=["order", "system"], columns="split", values="success").reset_index()
        table = table.sort_values("order").drop(columns="order")
        st.dataframe(table[["system", "test", "test_unseen", "challenge"]].round(3), hide_index=True,
                     width="stretch", height=35 * (len(table) + 1) + 3)
        st.success("Final system: model C + beam search 5, chosen on the validation set "
                   "(success 0.999 on test, 0.787 on test_unseen, 0.764 on challenge).")

    st.subheader("Human evaluation")
    human = read_json("reports/human_eval_summary.json", {})
    hrows = [{"system": label(k.split(" (")[0]),
              "sentences": k.split(" (")[1].rstrip(")") if " (" in k else "challenge, all 72",
              "acceptable": v["counts"].get("acceptable", 0),
              "partial": v["counts"].get("partial", 0), "failed": v["counts"].get("failed", 0),
              "acceptable rate": v["acceptable_rate"], "automatic success": v["automatic_success"]}
             for k, v in human.items() if k.startswith(("C_copy_nodrop", "B_attention"))]
    if hrows:
        st.dataframe(pd.DataFrame(hrows), hide_index=True, width="stretch")
        st.caption("One annotator only; the labels were produced with the help of an AI assistant.")

    st.subheader("Learning curves")
    chosen = st.multiselect("Models", models, default=[m for m in ("A_lstm", "B_attention", "C_copy_nodrop") if m in models],
                            format_func=lambda m: MODELS[m][0])
    curves = []
    for m in chosen:
        for e in read_json(f"experiments/{m}/history.json", []):
            curves.append({"model": MODELS[m][0], "epoch": e["epoch"], "loss": e["train_loss"], "set": "train"})
            curves.append({"model": MODELS[m][0], "epoch": e["epoch"], "loss": e["valid_loss"], "set": "validation"})
    if curves:
        st.altair_chart(alt.Chart(pd.DataFrame(curves)).mark_line(point=True).encode(
            x="epoch:Q", y=alt.Y("loss:Q", title="cross-entropy per token"), color="model:N",
            strokeDash=alt.StrokeDash("set:N", sort=["train", "validation"]),
            tooltip=["model", "set", "epoch", alt.Tooltip("loss:Q", format=".3f")]).properties(height=360),
            width="stretch")
        st.caption("Without the copy mechanism the training loss ends slightly below the validation loss "
                   "(A: 0.11 vs 0.17): mild overfitting, controlled by early stopping (3 epochs without improvement). "
                   "With copy (C), both losses stay equal.")

    st.subheader("Decoding strategies (validation set, model C)")
    dec = [r for r in results if r["split"] == "valid" and r["system"].startswith("C_copy_nodrop")]
    if dec:
        st.dataframe(pd.DataFrame([{"decoding": r["system"].partition("__")[2] or "greedy",
                                    "success": r["success_rate"], "BLEU": r["BLEU"],
                                    "content kept": r["content_preservation"], "polite": r["polite_rate"],
                                    "repetitions": r["repetition_rate"]} for r in dec]).round(3),
                     hide_index=True, width="stretch")

    with st.expander("All the figures of the report"):
        for f in ("learning_curves", "metrics_test", "decoding", "errors_by_category",
                  "attention_1", "attention_2", "attention_3"):
            path = ROOT / "reports" / "figures" / f"{f}.png"
            if path.exists():
                st.image(str(path), width="stretch")


# -----------------------------------------------------------------------------
# Tab 4: the dataset
# -----------------------------------------------------------------------------
with tab_data:
    stats = read_json("reports/dataset_stats.json", {})
    st.subheader("A balanced dataset without insults")
    st.write("The dataset was built for the project by a program (data/generate_dataset.py) that combines "
             "situations, people, objects and phrasings written by hand (data/resources.py). "
             "Each impolite sentence has polite references with exactly the same content. "
             "It was improved once: the first version (data_v1/) only covered office life and had incoherent pairs.")
    if stats:
        st.dataframe(pd.DataFrame([
            {"split": s, "impolite inputs": stats["splits"][s]["inputs"], "pairs": stats["splits"][s]["pairs"],
             "used for": use}
            for s, use in (("train", "training"), ("valid", "choosing the model and the decoding"),
                           ("test", "test, same phrasings as training"),
                           ("test_unseen", "test, impolite phrasings never seen in training"),
                           ("challenge", "test, sentences written by hand"))]), hide_index=True, width="stretch")
        cats = pd.DataFrame([{"category": c, "inputs": n} for c, n in stats["splits"]["train"]["categories"].items()])
        st.altair_chart(alt.Chart(cats).mark_bar(color=TEAL).encode(
            x=alt.X("category:N", sort=None, axis=alt.Axis(labelAngle=-30)), y="inputs:Q"
        ).properties(height=260, title="Training inputs per category"), width="stretch")
        checks = stats["checks"]
        shared = sum(checks["inputs_shared_between_splits"].values())
        st.caption(f"Sentences shared between splits: {shared} · duplicate pairs: {checks['duplicate_pairs']} · "
                   f"mean length: {stats['splits']['train']['impolite_len_mean']} tokens (impolite), "
                   f"{stats['splits']['train']['polite_len_mean']} tokens (polite)")

    st.subheader("Examples (one per category)")
    train = read_csv("data/train.csv")
    ex = [g.sample(1, random_state=3).iloc[0] for _, g in train.groupby("category")]
    st.table(pd.DataFrame([{"category": r["category"], "impolite": r["impolite"], "polite": r["polite"]}
                           for r in ex]).set_index("category"))

    st.subheader("Vocabulary")
    vstats = read_json("reports/vocab_stats.json", {})
    if vstats:
        st.write(f"**{vstats['vocab_size']:,} entries**, built on the training set only (words seen at least "
                 f"{vstats['min_freq']} times) + `<PAD>` `<SOS>` `<EOS>` `<UNK>`.")
        st.dataframe(pd.DataFrame([{"split": s, "unknown input tokens": v["impolite_tokens"],
                                    "inputs with an unknown word": v["impolite_sentences_with_unk"]}
                                   for s, v in vstats["unk_rate"].items()]), hide_index=True, width="stretch")

    st.subheader("Version 1 → version 2")
    comp = read_json("reports/dataset_v1_vs_v2.json", {})
    if comp:
        st.dataframe(pd.DataFrame([
            {"": "vocabulary size", "version 1": comp["v1"]["vocab_size"], "version 2": comp["v2"]["vocab_size"]},
            {"": "training inputs", "version 1": comp["v1"]["inputs"], "version 2": comp["v2"]["inputs"]},
            {"": "unknown tokens in the hand-written set", "version 1": comp["v1"]["challenge_unk_token_rate"],
             "version 2": comp["v2"]["challenge_unk_token_rate"]},
        ]), hide_index=True, width="stretch")


# -----------------------------------------------------------------------------
# Tab 5: how it works
# -----------------------------------------------------------------------------
with tab_how:
    st.subheader("The pipeline")
    st.markdown("""
| Step | What happens | File |
|---|---|---|
| 1. Preprocessing | Unicode normalisation, lowercasing, words and punctuation become tokens | `preprocessing.py` |
| 2. Vocabulary | words seen ≥ 2 times in train + `<PAD>` `<SOS>` `<EOS>` `<UNK>` | `preprocessing.py` |
| 3. Representation | each id becomes a 128-dimensional embedding (shared by encoder and decoder) | `model.py` |
| 4. Encoder | an LSTM (hidden size 256) reads the sentence, one state per word | `model.py` |
| 5. Attention | the decoder scores every input state, softmax → weights → context vector | `model.py` |
| 6. Decoder | an LSTM writes one word at a time, from `<SOS>` until `<EOS>` (max 40) | `model.py` |
| 7. Copy | p_gen mixes "generate a vocabulary word" and "copy an input word" | `model.py` |
| 8. Training | cross-entropy, Adam, teacher forcing, early stopping | `train.py` |
| 9. Generation | greedy, beam search or sampling | `inference.py` |
| 10. Evaluation | BLEU, chrF, exact match, content kept, politeness, success | `evaluate.py` |
""")

    st.subheader("A worked example with the final model")
    model_c, vocab_c = get_model(models[0] if FINAL_MODEL not in models else FINAL_MODEL)
    sentence = "Close the window now."
    toks = tokenize(sentence)
    res = beam_search(model_c, toks, vocab_c, k=5)
    out_ids = [vocab_c.stoi.get(t, -1) for t in res["tokens"]]
    st.markdown(f"""
1. **Input:** {sentence}
2. **Tokens:** {" · ".join(toks)}
3. **Ids:** `{" ".join(map(str, vocab_c.encode(toks, add_eos=True)))}`
4. **Generated ids:** `{" ".join(map(str, out_ids))}`
5. **Output:** {detokenize(res["tokens"], capitalize=True)}
""")

    st.subheader("Inside one decoding step")
    st.markdown("""
- the decoder receives the **previous word** (the true one during training = teacher forcing, its own prediction when generating);
- **attention**: score_j = vᵀ tanh(W_enc h_j + W_dec s), softmax over the input words, context c = Σ α_j h_j;
- the LSTM gets [word embedding ; context] and produces a new state;
- a linear layer + softmax gives a probability for each of the vocabulary words;
- with the **copy mechanism**: P(w) = p_gen · P_vocab(w) + (1 − p_gen) · attention on w.
""")

    st.subheader("The models")
    mrows = []
    for m in models:
        s = read_json(f"experiments/{m}/train_summary.json", {})
        params = s.get("params")
        mrows.append({"model": MODELS[m][0], "what changes": MODELS[m][1],
                      "parameters": f"{params / 1e6:.2f} M" if params else "—",
                      "best epoch": s.get("best_epoch")})
    st.dataframe(pd.DataFrame(mrows), hide_index=True, width="stretch")
