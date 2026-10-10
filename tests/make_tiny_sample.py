#!/usr/bin/env python3
"""Dataset-free tiny synthetic sample for CI / smoke tests.

Builds a short, low-resolution "nightclub" clip by re-compositing real face
photos harvested from the committed synthetic fixtures (testdata/videos +
testdata/ground_truth) onto a dark background, with KNOWN blur schedules and
ground-truth JSON in the same schema as tests/make_test_videos.py — so the
same evaluate.py scores it. No downloads, no external dataset.

Usage:
    python tests/make_tiny_sample.py --out /tmp/tiny --gt /tmp/tiny_gt
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
W, H, FPS = 960, 540, 24
DUR = 12
NFRAMES = DUR * FPS
FACE_PX = 110

# actor 0: blur -> sharp (violation) -> blur ; actor 1: sharp the whole time
SCHEDULES = [
    [(0, 5, "blur"), (5, 10, "sharp"), (10, DUR, "blur")],
    [(0, DUR, "sharp")],
]


def harvest_faces(count: int, seed: int = 3) -> list[np.ndarray]:
    """Crop real 'sharp' face photos out of the committed fixture videos."""
    gt_dir = ROOT / "testdata" / "ground_truth"
    vid_dir = ROOT / "testdata" / "videos"
    crops: list[np.ndarray] = []
    for gt_file in sorted(gt_dir.glob("*.json")):
        gt = json.loads(gt_file.read_text())
        vid = vid_dir / gt["video"]
        if not vid.exists():
            continue
        cap = cv2.VideoCapture(str(vid))
        for fr in gt["frame_data"][::20]:
            for a in fr["actors"]:
                if a["state"] != "sharp":
                    continue
                x1, y1, x2, y2 = a["bbox"]
                if x2 - x1 < 120 or y2 - y1 < 120:
                    continue
                cap.set(cv2.CAP_PROP_POS_MSEC, fr["t"] * 1000)
                ok, frame = cap.read()
                if not ok:
                    continue
                pad = int((x2 - x1) * 0.06)
                crop = frame[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
                if crop.size:
                    crops.append(cv2.resize(crop, (FACE_PX, FACE_PX),
                                            interpolation=cv2.INTER_AREA))
                break
            if len(crops) >= count:
                break
        cap.release()
        if len(crops) >= count:
            break
    return crops[:count]


def soft_mask(size: int) -> np.ndarray:
    m = np.ones((size, size), np.float32)
    edge = max(1, size // 10)
    ramp = np.linspace(0.0, 1.0, edge, dtype=np.float32)
    m[:edge, :] *= ramp[:, None]
    m[-edge:, :] *= ramp[::-1][:, None]
    m[:, :edge] *= ramp[None, :]
    m[:, -edge:] *= ramp[::-1][None, :]
    return m


def variants(face: np.ndarray) -> dict:
    size = face.shape[0]
    blur = cv2.GaussianBlur(face, (0, 0), size / 12)
    small = max(3, size // 10)
    blur = cv2.resize(blur, (small, small), interpolation=cv2.INTER_AREA)
    blur = cv2.resize(blur, (size, size), interpolation=cv2.INTER_NEAREST)
    return {"blur": blur, "sharp": face}


def state_at(schedule, t: float) -> str:
    for t0, t1, state in schedule:
        if t0 <= t < t1:
            return state
    return schedule[-1][2]


def render(out_path: Path, gt_dir: Path, faces: list[np.ndarray], seed: int) -> None:
    rng = np.random.default_rng(seed)
    xs = np.linspace(40, 20, W, dtype=np.uint8)
    bg = np.stack([xs * 0.4, xs * 0.7, xs], axis=-1).astype(np.uint8)
    bg = np.tile(bg[None, :, :], (H, 1, 1))
    variants_list = [variants(f) for f in faces]
    masks = [soft_mask(FACE_PX) for _ in faces]
    anchors = [(0.28, 0.45), (0.66, 0.52), (0.47, 0.30), (0.85, 0.30)]
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    gt_frames = []
    for fi in range(NFRAMES):
        t = fi / FPS
        frame = bg.astype(np.int16) + rng.integers(-4, 5, (H, W, 3)).astype(np.int16)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        f_actors = []
        for ai, sched in enumerate(SCHEDULES):
            ax, ay = anchors[ai % len(anchors)]
            cx = (ax + 0.03 * math.sin(2 * math.pi * t / 5 + ai)) * W
            cy = (ay + 0.025 * math.sin(2 * math.pi * t / 6 + ai * 2)) * H
            state = state_at(sched, t)
            img = variants_list[ai][state]
            x1, y1 = int(cx - FACE_PX / 2), int(cy - FACE_PX / 2)
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(W, x1 + FACE_PX), min(H, y1 + FACE_PX)
            if x2c > x1c and y2c > y1c:
                m = masks[ai][y1c - y1:y2c - y1, x1c - x1:x2c - x1][..., None]
                roi = frame[y1c:y2c, x1c:x2c].astype(np.float32)
                pasted = img[y1c - y1:y2c - y1, x1c - x1:x2c - x1].astype(np.float32)
                frame[y1c:y2c, x1c:x2c] = (roi * (1 - m) + pasted * m).astype(np.uint8)
            f_actors.append({"actor": ai, "bbox": [x1c, y1c, x2c, y2c], "state": state})
        gt_frames.append({"t": round(t, 3), "actors": f_actors})
        vw.write(frame)
    vw.release()
    gt = {"video": out_path.name, "fps": FPS, "frames": NFRAMES,
          "actors": [{"actor": ai, "schedule": s} for ai, s in enumerate(SCHEDULES)],
          "frame_data": gt_frames}
    with open(gt_dir / f"{out_path.stem}.json", "w") as fh:
        json.dump(gt, fh)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--gt", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.gt.mkdir(parents=True, exist_ok=True)
    faces = harvest_faces(len(SCHEDULES), args.seed)
    if len(faces) < len(SCHEDULES):
        raise SystemExit("could not harvest enough face crops from testdata fixtures")
    render(args.out / "tiny_sample.mp4", args.gt, faces, args.seed)
    print(f"wrote {args.out / 'tiny_sample.mp4'} + ground truth "
          f"({NFRAMES} frames @ {W}x{H}, {len(faces)} real face crops)")


if __name__ == "__main__":
    main()
