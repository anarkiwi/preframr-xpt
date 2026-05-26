"""Unit tests for the tokenizer-health metric extractors (long-tail
fractions read straight from tokens.csv -- no docker, no spec)."""

from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from preframr_experiments.metrics import (
    _longtail_frac,
    _worst_family_longtail_frac,
)


def _write_tokens(path: Path) -> None:
    """60-atom family A (30 long-tail, 30 frequent) + 10-atom family B
    + a count=0 sentinel row. Real atoms=70, long-tail=30; the only
    family with >=50 atoms is A at 0.5 long-tail."""
    rows = [("0", "-1", "-1", "0", "0", "0")]  # sentinel: count=0, excluded
    n = 1
    for i in range(30):
        rows.append(("0", "1", "0", str(i), "5", str(n)))  # A, long-tail (<10)
        n += 1
    for i in range(30):
        rows.append(("0", "1", "0", str(100 + i), "100", str(n)))  # A, frequent
        n += 1
    for i in range(10):
        rows.append(("45", "21", "4", str(i), "100", str(n)))  # B, small family
        n += 1
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["op", "reg", "subreg", "val", "count", "n"])
        w.writerows(rows)


class TestTokenizerHealthMetrics(unittest.TestCase):
    def test_longtail_and_worst_family(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "tokens.csv"
            _write_tokens(p)
            art = SimpleNamespace(tokens_csv=p)
            self.assertAlmostEqual(_longtail_frac(art), 30 / 70, places=6)
            self.assertAlmostEqual(_worst_family_longtail_frac(art), 0.5, places=6)

    def test_absent_tokens_csv_is_nan(self):
        art = SimpleNamespace(tokens_csv=Path("/nonexistent/tokens.csv"))
        self.assertTrue(math.isnan(_longtail_frac(art)))
        self.assertTrue(math.isnan(_worst_family_longtail_frac(art)))


if __name__ == "__main__":
    unittest.main()
