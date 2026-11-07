# Tests of the Streamlit interface (skipped if there is no trained model)
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "checkpoints" / "C_copy_nodrop.pt").exists():
    pytest.skip("no trained model yet", allow_module_level=True)

from streamlit.testing.v1 import AppTest


def start_app():
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()


def click(at, label):
    [b for b in at.button if b.label == label][0].click().run()


def test_app_starts():
    at = start_app()
    assert not at.exception
    assert len(at.tabs) == 5


def test_rewrite_with_the_trained_model():
    at = start_app()
    at.text_area[0].set_value("Close the window now.").run()
    click(at, "Rewrite politely")
    assert not at.exception
    result = [m.value for m in at.markdown if 'class="result"' in m.value]
    assert result and "window" in result[0].lower()


def test_empty_input_shows_a_warning():
    at = start_app()
    at.text_area[0].set_value("   ").run()
    click(at, "Rewrite politely")
    assert at.warning and not at.exception


def test_compare_models():
    at = start_app()
    click(at, "Compare")
    assert not at.exception
    table = at.dataframe[0].value
    assert "Copy baseline" in list(table["System"])
