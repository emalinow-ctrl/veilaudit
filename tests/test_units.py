"""Fast unit tests — no model download, no video I/O.

Covers the pure-logic pieces: IoU tracker bookkeeping, blur-metric behaviour on
synthetic crops, GT schedule helpers, and config loading.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from blurcheck import laplacian_var, highfreq_ratio  # noqa: E402
from tracker import _iou, FaceTrack  # noqa: E402


# ---------------------------------------------------------------- tracker ---
class TestIoU:
    def test_identical_boxes(self):
        assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_disjoint(self):
        assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_half_overlap(self):
        # two 10x10 boxes offset by 5 -> intersection 50
        assert _iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)

    def test_touching_edges(self):
        assert _iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0


class TestFaceTrack:
    def test_sharp_votes_accumulate(self):
        t = FaceTrack(1, (0, 0, 10, 10), 0, window=3)
        for f in range(1, 4):
            t.update((0, 0, 10, 10), f, sharp=True)
        assert t.n_sharp == 3
        assert t.n_censored == 0
        assert list(t.recent) == [True, True, True]

    def test_window_keeps_recent_only(self):
        t = FaceTrack(1, (0, 0, 10, 10), 0, window=3)
        t.update((0, 0, 10, 10), 1, sharp=True)
        t.update((0, 0, 10, 10), 2, sharp=True)
        t.update((0, 0, 10, 10), 3, sharp=False)
        t.update((0, 0, 10, 10), 4, sharp=False)
        assert list(t.recent) == [True, False, False]

    def test_missed_counter(self):
        t = FaceTrack(1, (0, 0, 10, 10), 0, window=3)
        t.missed += 1
        t.missed += 1
        assert t.missed == 2
        t.update((0, 0, 10, 10), 5, sharp=False)
        assert t.missed == 0


# ------------------------------------------------------------- blurcheck ---
def _sharp_face(size=120, seed=1):
    rng = np.random.default_rng(seed)
    img = rng.integers(60, 200, (size, size), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0.8)
    # add crisp edges
    cv2.rectangle(img, (20, 20), (size - 20, size - 20), 255, 2)
    return img


def _blurred_face(size=120, seed=1):
    img = _sharp_face(size, seed)
    return cv2.GaussianBlur(img, (0, 0), size / 10)


def _pixelated_face(size=120, seed=1):
    img = _sharp_face(size, seed)
    small = cv2.resize(img, (size // 10, size // 10), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (size, size), interpolation=cv2.INTER_NEAREST)


class TestMetrics:
    def test_sharp_has_more_highfreq_than_blurred(self):
        assert highfreq_ratio(_sharp_face()) > highfreq_ratio(_blurred_face())

    def test_pixelation_has_flat_blocks(self):
        # pixelated faces: large flat regions -> flat fraction high
        from blurcheck import flat_frac  # if exposed
        assert flat_frac(_pixelated_face()) > flat_frac(_sharp_face())

    def test_laplacian_sharp_beats_blur(self):
        assert laplacian_var(_sharp_face()) > laplacian_var(_blurred_face())

    def test_tiny_roi_safe(self):
        assert highfreq_ratio(np.zeros((4, 4), np.uint8)) == 0.0


# ---------------------------------------------------------------- config ---
class TestConfig:
    def test_repo_config_loads(self):
        import yaml
        cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
        assert "video" in cfg and "blur_check" in cfg
        assert cfg["video"]["sample_every_n_frames"] >= 1

    def test_test_config_points_at_tmp(self):
        import yaml
        cfg = yaml.safe_load((ROOT / "tests" / "test_config.yaml").read_text())
        assert str(cfg["paths"]["input_dir"]).startswith("/tmp/")
