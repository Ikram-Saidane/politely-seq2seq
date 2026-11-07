import pandas as pd
import pytest
import torch

from politely.config import Config
from politely.dataset import PairDataset, collate, extended_ids
from politely.decode import beam_search, greedy_decode
from politely.models import build_model, count_parameters
from politely.preprocessing import detokenize, tokenize
from politely.vocab import EOS, PAD, SOS, UNK, Vocab


@pytest.fixture
def vocab():
    return Vocab.build([["send", "me", "the", "report", "now", "."],
                        ["could", "you", "send", "me", "the", "report", "?"]], min_freq=1)


def test_tokenize_roundtrip():
    toks = tokenize("Send me the Report NOW, please!")
    assert toks == ["send", "me", "the", "report", "now", ",", "please", "!"]
    assert tokenize(detokenize(toks)) == toks
    assert detokenize(["i", "think", "it's", "fine", "."], capitalize=True) == "I think it's fine."


def test_special_tokens_and_unk(vocab):
    assert vocab.itos[:4] == ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
    assert (PAD, SOS, EOS, UNK) == (0, 1, 2, 3)
    assert vocab.encode(["zorglub"]) == [UNK]
    ids = vocab.encode(["send", "me"], add_sos=True, add_eos=True)
    assert ids[0] == SOS and ids[-1] == EOS and vocab.decode(ids) == ["send", "me"]


def test_extended_ids_copy_oov(vocab):
    src_ext, tgt_ext, oovs = extended_ids(["send", "order", "4521"], ["order", "4521", "?"], vocab)
    assert oovs == ["order", "4521"]
    assert src_ext[2] == len(vocab) + 1
    assert vocab.decode(tgt_ext, oovs=oovs) == ["order", "4521", "?"]


def test_collate_padding(vocab):
    df = pd.DataFrame({"impolite": ["send me the report now.", "send it"],
                       "polite": ["could you send me the report?", "could you"]})
    batch = collate([PairDataset(df, vocab)[i] for i in range(2)])
    assert batch["src"].shape == (2, 6) and batch["src_len"].tolist() == [6, 2]
    assert batch["src"][1, 2:].tolist() == [PAD] * 4
    assert batch["tgt"][:, 0].tolist() == [SOS, SOS]


def test_truncation(vocab):
    df = pd.DataFrame({"impolite": ["send me the report now ."], "polite": ["could you send me the report ?"]})
    ex = PairDataset(df, vocab, max_len=3)[0]
    assert len(ex["src"]) == 3 and len(ex["tgt"]) == 3 + 2


@pytest.mark.parametrize("kw", [{}, {"attention": True}, {"attention": True, "copy": True},
                                {"attention": True, "rnn_type": "gru"},
                                {"attention": True, "copy": True, "bidirectional": True},
                                {"rnn_type": "gru", "bidirectional": True, "n_layers": 2}])
def test_shapes_and_masking(kw):
    V, B, S, T = 40, 3, 6, 5
    cfg = Config(emb_dim=8, hid_dim=12, attn_dim=8, **kw)
    model = build_model(cfg, V)
    torch.manual_seed(0)
    src = torch.randint(4, V, (B, S)); src_len = torch.tensor([6, 4, 2])
    for i, l in enumerate(src_len):
        src[i, l:] = PAD
    tgt = torch.randint(4, V, (B, T)); tgt[:, 0] = SOS
    max_oov = 1 if cfg.copy else 0
    logp, attn = model(src, src_len, tgt, 1.0, src.clone(), max_oov)
    assert logp.shape == (B, T - 1, V + max_oov)
    assert torch.allclose(logp.exp().sum(-1), torch.ones(B, T - 1), atol=1e-4)
    if cfg.attention:
        assert attn.shape == (B, T - 1, S) and attn[2, :, 2:].abs().max() == 0   # padding masked
    count_parameters(model)


def test_copy_requires_attention():
    with pytest.raises(ValueError):
        build_model(Config(attention=False, copy=True), 10)


def test_beam1_equals_greedy():
    torch.manual_seed(0)
    vocab = Vocab.build([["a", "b", "c", "d"]], min_freq=1)
    for kw in ({"attention": True, "copy": True}, {"rnn_type": "gru"}):
        model = build_model(Config(emb_dim=8, hid_dim=8, attn_dim=8, **kw), len(vocab)).eval()
        src = ["a", "b", "zz", "c"]
        g = greedy_decode(model, [src], vocab, max_len=8)[0]["tokens"]
        assert beam_search(model, src, vocab, k=1, max_len=8, alpha=0.0)["tokens"] == g
        assert len(beam_search(model, src, vocab, k=3, max_len=8)["tokens"]) <= 8
        s = greedy_decode(model, [src], vocab, max_len=8, sample={"temperature": 0.7, "top_k": 3, "seed": 1})
        assert len(s[0]["tokens"]) <= 8


def test_bleu_matches_sacrebleu_and_hand_case():
    from politely.metrics import corpus_bleu, sacre_bleu, content_words
    hyps = [["could", "you", "send", "me", "the", "report", "?"], ["thanks", "for", "the", "help", "."]]
    refs = [[["could", "you", "send", "me", "the", "report", "today", "?"], ["please", "send", "the", "report", "."]],
            [["thanks", "for", "your", "help", "."]]]
    assert abs(corpus_bleu(hyps, refs) - sacre_bleu(hyps, refs)) < 1e-6
    assert corpus_bleu([refs[1][0]], [refs[1]]) == pytest.approx(100.0)
    assert content_words(["send", "me", "the", "report", "now"],
                         [["could", "you", "send", "me", "the", "report", "?"]]) == {"send", "report"}
    # light stemming: a change of verb form keeps the content word
    assert content_words(["close", "the", "window"], [["would", "you", "mind", "closing", "the", "window", "?"]]) \
        == {"clos", "window"}


def test_rules_baseline_removes_markers():
    from politely.baselines import rules_baseline
    out = rules_baseline(tokenize("Send me the report right now, seriously."))
    assert out[:2] == ["please", ","] and "now" not in out and "seriously" not in out and "report" in out


def test_restore_case_for_display():
    from politely.preprocessing import restore_case
    out = restore_case(["send", "sarah", "the", "tv", "remote"], "Give the TV remote to Sarah. Send it now.")
    assert out == ["send", "Sarah", "the", "TV", "remote"]
