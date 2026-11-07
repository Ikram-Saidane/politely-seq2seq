"""Tests of the web API: valid input, invalid inputs, unknown model. Skipped if no checkpoint exists."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if not any((ROOT / "checkpoints").glob("[A-Z]_*.pt")):
    pytest.skip("no trained model yet", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

from app.server import app  # noqa: E402

client = TestClient(app)


def test_index_and_info():
    assert client.get("/").status_code == 200
    info = client.get("/api/info").json()
    assert info["models"] and info["default_model"] and info["examples"]


def test_rewrite_ok():
    r = client.post("/api/rewrite", json={"text": "Send me the report now."})
    assert r.status_code == 200
    d = r.json()
    assert d["output"] and 0 <= d["politeness_out"] <= 1 and 0 <= d["content_kept"] <= 1


@pytest.mark.parametrize("dec", ["beam", "sample"])
def test_other_decodings(dec):
    assert client.post("/api/rewrite", json={"text": "Call Hugo back today.", "decoding": dec}).status_code == 200


@pytest.mark.parametrize("text", ["", "   ", "123 !!!", "x" * 301, "word " * 50])
def test_invalid_inputs(text):
    r = client.post("/api/rewrite", json={"text": text})
    assert r.status_code == 422 and isinstance(r.json()["detail"], str)


def test_unknown_model_and_decoding():
    assert client.post("/api/rewrite", json={"text": "Send it.", "model": "Z"}).status_code == 404
    assert client.post("/api/rewrite", json={"text": "Send it.", "decoding": "magic"}).status_code == 422


def test_unknown_words_warning():
    d = client.post("/api/rewrite", json={"text": "Send the zxqvortex to Grzegorz now."}).json()
    assert "zxqvortex" in d["unknown_words"] and d["warnings"]


def test_compare_and_dashboard():
    c = client.post("/api/compare", json={"text": "Stop interrupting me."}).json()
    assert {s["id"] for s in c["systems"]} >= {"baseline_copy", "baseline_rules"}
    assert "dataset" in client.get("/api/dashboard").json()
