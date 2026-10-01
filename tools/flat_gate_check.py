#!/usr/bin/env python
"""flat_gate_check.py -- measure what `flat_only` buys, through the NODE's own code path.

    python tools/flat_gate_check.py --dir <folder of side-by-side pair images> [--radius 4]
    python tools/flat_gate_check.py                      # synthetic smoke test

It calls StereoStabilizeSBS.run() exactly as ComfyUI does, at several `flat_only` values, and
reports the two numbers that matter against each other:

    boil     the per-pixel temporal sizzle (what the median removes)
    detail   the Laplacian variance (what the median costs) -- split into FLAT and EDGE
             pixels, because that split is the whole point of the gate

and it verifies the two contracts that make the gate safe:

  * `flat_only = 0` is BIT-IDENTICAL to the ungated median (an old graph is unchanged)
  * outside the mask the output is BIT-IDENTICAL to the level-normalised input, and inside
    the mask it is BIT-IDENTICAL to the ungated median (the gate moves nothing by itself)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nodes.stereo_stabilize import (StereoStabilizeSBS, _flat_mask,  # noqa: E402
                                    _level_normalise, _temporal_median)


def luma(f):
    return cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32)


def _m2(mask):
    """Mask as a plain (H,W) bool array."""
    return np.asarray(mask).reshape(np.asarray(mask).shape[:2])


def boil(seq, mask=None):
    """Per-pixel temporal sd.  `seq` may be one eye or the whole SBS; `mask` is one eye wide,
    so with a full SBS it is applied to each half and the halves are averaged."""
    def one(sub, m=None):
        sd = np.stack([luma(f)[::4, ::4] for f in sub]).std(axis=0)
        return float(sd.mean() if m is None else sd[_m2(m)[::4, ::4]].mean())
    if mask is None:
        return one(seq)
    w = seq.shape[2] // 2
    return float(np.mean([one(seq[:, :, s], mask) for s in (slice(0, w), slice(w, None))]))


def sharpness(seq, mask=None):
    """Laplacian variance of the TEMPORAL MEAN -- structure, with the boil averaged out.

    Plain per-frame Laplacian variance is a bad quality proxy here: the boil IS
    high-frequency noise, so removing it lowers that number, and a filter that keeps the
    noise looks sharper than one that removes it.  Averaging over the clip cancels the
    noise (it is not correlated) and leaves the structure, so this number answers the
    question that matters: did the filter eat anything real?
    """
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)

    def one(sub, m=None):
        mean = np.mean([luma(f)[::2, ::2] for f in sub], axis=0)
        lap = cv2.filter2D(mean, -1, k)
        return float(lap.var() if m is None else lap[_m2(m)[::2, ::2]].var())

    if mask is None:
        return one(seq)
    w = seq.shape[2] // 2
    return float(np.mean([one(seq[:, :, s], mask) for s in (slice(0, w), slice(w, None))]))


def synthetic(n=24, h=192, w=256):
    """Smoke-test clip: flat background with boil plus a still panel (so there ARE edges).

    Deliberately motionless -- a moving object would be smeared by the median (that is the
    documented radius rule), which would swamp the numbers this tool is here to produce.
    Real numbers come from --dir.
    """
    rng = np.random.default_rng(0)
    out = []
    for t in range(n):
        f = np.full((h, w, 3), 90.0) + rng.normal(0, 8, (h, w, 3))
        f[h // 4:3 * h // 4, w // 4:3 * w // 4] = 175.0 + rng.normal(0, 8, (h // 2, w // 2, 3))
        out.append(np.clip(f, 0, 255).astype(np.uint8))
    return np.array(out)


def load(d):
    fs = sorted(Path(d).glob("*.png")) or sorted(Path(d).glob("*.jpg"))
    if not fs:
        raise SystemExit(f"no images in {d}")
    return np.array([cv2.imread(str(p))[..., ::-1] for p in fs], np.uint8)


def same(x, y):
    """Equal to within float32 round-trip noise (< 0.05 of a grey level).

    The node returns the array as /255 in float32, so an exact `==` would fail on the
    round-trip alone; anything the gate got wrong would move pixels by whole levels.
    """
    return float(np.abs(np.asarray(x, np.float32) - np.asarray(y, np.float32)).max()) < 0.05


def per_eye_gate(frames, gate, feather):
    """The node builds the mask from the left eye and applies it to both halves."""
    w = frames.shape[2] // 2
    return _flat_mask(frames[:, :, :w].astype(np.float32), gate, feather), w


def same_masked(x, y, sel2d):
    """`same` on the pixels selected by a (H,W) mask, per eye (the mask indexes one eye)."""
    w = x.shape[2] // 2
    full = np.broadcast_to(sel2d, (x.shape[0],) + sel2d.shape)   # (N,H,W), no copy
    return all(same(x[:, :, s][full], y[:, :, s][full])
               for s in (slice(0, w), slice(w, None)))


def ungated_reference(frames, radius, level):
    w = frames.shape[2] // 2
    return np.concatenate([
        _temporal_median(_level_normalise(frames[:, :, :w].astype(np.float32), level), radius),
        _temporal_median(_level_normalise(frames[:, :, w:].astype(np.float32), level), radius)],
        axis=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    ap.add_argument("--radius", type=int, default=4)
    ap.add_argument("--level", type=float, default=1.0)
    ap.add_argument("--gates", default="0,60,85")
    ap.add_argument("--feather", type=int, default=9)
    ap.add_argument("--max-pairs", type=int, default=0)
    a = ap.parse_args()

    frames = load(a.dir) if a.dir else synthetic()
    if a.max_pairs:
        frames = frames[:a.max_pairs]
    n, h, w = frames.shape[:3]
    print(f"{n} pairs @ {w // 2}x{h} per eye | radius {a.radius} | feather {a.feather} | "
          f"{'synthetic' if not a.dir else a.dir}")

    node = StereoStabilizeSBS()
    sbs = torch.from_numpy(frames.astype(np.float32) / 255.0)
    ref = ungated_reference(frames, a.radius, a.level)      # what "gated in" must equal
    fixed = ungated_reference(frames, 0, a.level)           # what "gated out" must equal

    # Because both of those are BIT-IDENTICAL, the gate's whole effect is determined by (a) how
    # much of the frame it covers and (b) what the median does there.  So measure exactly those:
    # inside the gate = the ungated median, outside = untouched.  No other column can surprise.
    print(f"{'flat_only':>9} {'covers':>7} | {'boil whole frame':>17} | {'boil in the gate':>17} | "
          f"{'structure whole':>17} | contract")
    print(f"{'':>9} {'':>7} | {'original':>8} {'gated':>8} | {'original':>8} {'gated':>8} | "
          f"{'original':>8} {'gated':>8} |")
    bad = []
    for g in [int(x) for x in a.gates.split(",")]:
        out, rep, npairs = node.run(sbs, a.radius, a.level, 0, False, g, a.feather)
        arr = out.numpy() * 255.0
        if g == 0:
            sel = np.ones((frames.shape[1], frames.shape[2] // 2), bool)   # gate off = everything
            ok = "bit-identical to the ungated median" if same(arr, ref) else "FAILED"
            if not same(arr, ref):
                bad.append("flat_only=0 is not the ungated median")
        else:
            sel = per_eye_gate(frames, g, a.feather)[0]
            a_ok = same_masked(arr, fixed, np.logical_not(sel))  # untouched region
            b_ok = same_masked(arr, ref, sel)                    # gated region
            ok = ("exact outside / exact inside" if a_ok and b_ok
                  else f"FAILED (outside {a_ok}, inside {b_ok})")
            if not (a_ok and b_ok):
                bad.append(f"gate {g} did not preserve one of its own regions")
        print(f"{g:>9} {100 * sel.mean():6.0f}% | {boil(frames):8.2f} {boil(arr):8.2f} | "
              f"{boil(frames, sel):8.2f} {boil(arr, sel):8.2f} | "
              f"{sharpness(frames):8.1f} {sharpness(arr):8.1f} | {ok}")

    print("\nBoil is on the WHOLE frame (comparable across rows) and then only inside the gate \n"
          "(how much of it the gate can reach).  Structure is the whole frame.  gate 0 filters\n"
          "everything: maximum cleanup, maximum cost.  The gate's job is to keep the whole-frame\n"
          "boil near the gate-0 row while its `covers` column stays below 100%.")

    # print the node's own report for the last gate, so the numbers in one place agree
    print("\n--- node report (flat_only %d) ---\n%s" % (int(a.gates.split(",")[-1]), rep))
    if bad:
        for b in bad:
            print("FAILED:", b)
        raise SystemExit(1)
    print("OK -- gate contracts hold")


if __name__ == "__main__":
    main()
