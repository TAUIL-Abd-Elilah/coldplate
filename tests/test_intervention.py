# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "orchestrator"))

from intervention_test import balanced_topk_direction  # noqa: E402


def test_balanced_topk_direction_uses_equal_material_budget():
    g = np.array([[4.0, -3.0, 2.0], [-2.0, 0.0, 1.0]])
    d, add, remove = balanced_topk_direction(g, 2)
    assert d.sum() == 0
    assert np.count_nonzero(d == 1) == 2
    assert np.count_nonzero(d == -1) == 2
    assert set(add) == {1, 3}
    assert set(remove) == {0, 2}
    assert np.sum(g * d) < 0


def test_balanced_topk_direction_rejects_invalid_k():
    with pytest.raises(ValueError):
        balanced_topk_direction(np.ones((2, 2)), 0)
    with pytest.raises(ValueError):
        balanced_topk_direction(np.ones((2, 2)), 3)
