"""Smoke test: every dashboard page renders without an exception, on the default data
and after changing the risk weights.

Run: python -m unittest tests.test_app      (needs outputs/; takes a minute or two)
"""

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["executive", "live", "transaction", "account", "rings", "network", "behavior",
         "explain", "alerts", "performance", "guide"]
SCRIPT = """
import sys
sys.path.insert(0, {root!r})
import streamlit as st
from ui.components import inject_css
from ui.pages import {page}
inject_css()
{page}.render()
"""


class AppSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:  # pragma: no cover
            raise unittest.SkipTest("streamlit testing API not available")
        if not (ROOT / "outputs" / "transaction_scores.csv.gz").exists():
            raise unittest.SkipTest("run `python -m fraud.pipeline` first")
        cls.AppTest = AppTest

    def render(self, page, weights=None):
        at = self.AppTest.from_string(SCRIPT.format(root=str(ROOT), page=page),
                                      default_timeout=240)
        if weights:
            at.session_state["weights"] = weights
        at.run()
        return at

    def test_every_page_renders(self):
        for page in PAGES:
            with self.subTest(page=page):
                at = self.render(page)
                self.assertFalse(at.exception, f"{page}: {[e.value for e in at.exception]}")

    def test_pages_render_with_custom_weights(self):
        w = {"transaction": 0.6, "behavioral": 0.1, "network": 0.3}
        for page in ("executive", "transaction", "performance"):
            with self.subTest(page=page):
                at = self.render(page, w)
                self.assertFalse(at.exception, f"{page}: {[e.value for e in at.exception]}")

    def test_full_app_starts(self):
        at = self.AppTest.from_file(str(ROOT / "app.py"), default_timeout=240)
        at.run()
        self.assertFalse(at.exception, [e.value for e in at.exception])


if __name__ == "__main__":
    unittest.main()
