import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tap_detect as t  # noqa: E402


def test_selftest_patterns_and_sides():
    # Real room noise plus synthetic taps: singles/doubles/triples, left/right,
    # and the ambiguous or mixed-side patterns that must be ignored.
    assert t.selftest()


def test_calibration_recording_sides():
    # Real labelled desk taps, close to and away from the laptop.
    d = np.load(os.path.join(t.DATA_DIR, "calib_20261027_203700.npz"))
    for clip, label in zip(d["clips"], d["labels"]):
        f = t.features(clip[-t.EVENT:], 1e-6)
        assert t.side(f) == label.split("-")[1], (label, f["side_db"])


def test_gcc_phat_recovers_lag():
    th = t.thump(1.0, 2, lag=6)
    x = np.zeros((t.EVENT + 400, 2), np.float32)
    x[t.PRE:t.PRE + len(th)] = th
    assert t.gcc_phat_lag(x[:1024, 0], x[:1024, 1]) == 6
