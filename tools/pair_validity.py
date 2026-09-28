#!/usr/bin/env python
"""
pair_validity.py -- "is this generation actually a stereo pair, and is it moving?"

Works directly on the production pair images (ComfyUI/output/Stereo/pair_NNNNN_.png),
so it can be pointed at any run without my test harness.

Two independent questions per run:

 1. IS THE PAIR REAL?  For a genuine same-instant pair the right half is a
    horizontally shifted copy of the left half, so warping by the measured disparity
    removes most of the difference:
        mean|L-R|            raw difference between the halves
        warp ratio           residual after the best horizontal warp / raw difference
    Calibrated on this project's own outputs:
        real same-instant pair : ratio 0.42-0.50 , mean|L-R| 11-13/255 , spread 7-10 px
        broken config (no stereo, same view twice) :
                                 ratio 0.70-0.90 , mean|L-R|  2.5-4.2/255 , spread 1-2 px
    thresholds: < 0.55 REAL · 0.55-0.68 WEAK · > 0.68 NOT A PAIR
    (caveat: a genuinely low-depth scene also reads high -- pair it with the spread)

 2. IS IT MOVING?  Consecutive pair images give the same eye one instant apart
    (pair k's left half -> pair k+1's left half), so the instant motion is measurable
    straight from the pair files. The stride-1 model sat at ~0.15-0.35 px; the training
    data's own median instant carries 0.34 px at stride 1 and 3.66 px at stride 3.

HOW THIS RELATES TO THE NODE'S OWN READOUT (added 2026-09-27)

`Stereo Pair Frames + head trim / QC` prints `eye_step_px` while it runs. That number and
this tool's numbers describe the same disparity field from two directions and are NOT
interchangeable -- printed together on one Wan 2.2 clip (768 px eye width) they read:

    node `eye_step_px`   -13.1 px    a RIGID global fit: one shift for the whole frame,
                                     weighted toward the strongest structure, so it lands
                                     near the STRONG (near-field) end of the range
    this tool, median dx  -7.4 px    the middle of the per-pixel distribution
    this tool, spread     14.2 px     p95 - p5: the full range, i.e. how much the
                                     disparity varies across the frame (depth variation)
    p5 / p95            -14.3 / -0.1  the two ends of that range

So: **use the node's number to ask "is there parallax at all, and how much on the dominant
structure"**, and this tool's **spread** to ask "how much depth variation is in the frame".
A clip can be strong on both, and a clip with a big uniform shift and no depth variation
(a cardboard cut-out) scores high on the first and low on the second. Per the film-class
scale, 10-20 px of eye step at 1024 px eye width (median 14.8, i.e. 1.0-2.0 % of width) is
what real 3D films measure.

The node also returns a one-line VERDICT (`verdict` / the first line of `report`): OK, OK (PATCHY),
or REJECT - MIXED EYE ORDER / WARPED / NO STEREO. This tool measures the pair, the node judges the
clip; when they disagree, this tool's warp ratio and spread are the second opinion.

BOTH SIDES NOW REPORT BANDS (top / middle / bottom third), and they do not agree by design:
this tool's bands are the **area-weighted** per-pixel median, the node's are **structure-weighted**
rigid fits with a fit-quality (`explained`) per band. On one Wan clip this tool's bands read
-5.6 / -11.4 / -5.5 px (mid-heavy) while the node's rigid fits read -13.1 / -14.0 / -6.6 px
(top/mid, bottom weak). Same clip, same question, two weightings. Use either one to spot a band
that is EMPTY -- that verdict agrees -- but compare each tool's bands against its own whole-frame
number, never across tools.

Note also that the node may trim frames before pairing, so a run's pair numbering can start
later than the render's frame 0 -- that does not affect any number here.

Usage:
  pair_validity.py --file 3602
  pair_validity.py --range 3602-3641
  pair_validity.py --dir <Stereo dir> --runs          # group by mtime and report all
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import time
from collections import defaultdict

import cv2
import numpy as np

DEFAULT_DIR = "output/Stereo"   # relative to ComfyUI's root; override with --dir


def gray(x):
    return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32)


def pair_metrics(im):
    h, w = im.shape[:2]
    hw = w // 2
    L, R = gray(im[:, :hw]), gray(im[:, hw:])
    f = cv2.calcOpticalFlowFarneback(L, R, None, 0.5, 3, 21, 3, 7, 1.5, 0)
    dx, dy = f[..., 0], f[..., 1]
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    t = np.sqrt(gx ** 2 + gy ** 2)
    m = t > np.percentile(t, 65)
    hh, ww = np.mgrid[0:h, 0:hw].astype(np.float32)
    zero = float(np.abs(L - R)[m].mean())
    best = float(np.abs(cv2.remap(L, ww - dx, hh, cv2.INTER_LINEAR) - R)[m].mean())
    y1, y2 = h // 3, 2 * h // 3
    bands = [float(np.median(dx[0:y1][m[0:y1]])) if m[0:y1].any() else 0.0,
             float(np.median(dx[y1:y2][m[y1:y2]])) if m[y1:y2].any() else 0.0,
             float(np.median(dx[y2:h][m[y2:h]])) if m[y2:h].any() else 0.0]
    return dict(spread=float(np.percentile(dx[m], 95) - np.percentile(dx[m], 5)),
                dx_med=float(np.median(dx[m])), dy=float(np.median(np.abs(dy[m]))),
                bands=bands, zero=zero, ratio=best / zero if zero > 0 else 1.0, L=L, eye_w=hw)


def instant_motion(Ls):
    out = []
    for i in range(len(Ls) - 1):
        f = cv2.calcOpticalFlowFarneback(Ls[i], Ls[i + 1], None, 0.5, 3, 21, 3, 7, 1.5, 0)
        mag = np.sqrt((f ** 2).sum(-1))
        m = np.ones_like(mag, bool)
        out.append(float(np.median(mag[m])))
    return float(np.median(out)) if out else 0.0


def verdict(ratio):
    if ratio < 0.55:
        return "REAL PAIR"
    if ratio < 0.68:
        return "WEAK"
    return "NOT A PAIR"


def report(name, paths):
    res, Ls = [], []
    for p in paths:
        im = cv2.imread(p)
        if im is None:
            continue
        r = pair_metrics(im)
        res.append(r)
        Ls.append(r["L"])
    if not res:
        return
    med = lambda k: float(np.median([r[k] for r in res]))
    v = verdict(med("ratio"))
    spct = med("spread") / max(1.0, med("eye_w")) * 100.0
    bs = [float(np.median([r["bands"][j] for r in res])) for j in range(3)]
    print("%-18s %3d pairs  spread %5.1f (%4.2f%% of eye)  |dy| %5.2f  mean|L-R| %5.2f  "
          "warp %4.2f  eye step %+6.2f px  bands t/m/b %s  INSTANT %5.2f px  %s"
          % (name, len(res), med("spread"), spct, med("dy"), med("zero"), med("ratio"),
             med("dx_med"), " ".join("%+6.2f" % x for x in bs), instant_motion(Ls), v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--file", type=int, help="single pair index, e.g. 3602")
    ap.add_argument("--range", help="pair index range, e.g. 3602-3641")
    ap.add_argument("--runs", action="store_true", help="group every file by mtime")
    ap.add_argument("--tail", type=int, default=6, help="with --runs: show only the last N runs")
    a = ap.parse_args()

    def path(i):
        return os.path.join(a.dir, "pair_%05d_.png" % i)

    print("directory: %s" % a.dir)
    print("%-18s %10s %24s %8s %12s %6s %18s %22s %15s  %s" %
          ("run", "", "spread", "|dy|", "mean|L-R|", "warp", "eye step (median)", "bands top/mid/bottom",
           "INSTANT", "verdict"))
    print("-" * 160)
    if a.file:
        report("pair_%05d" % a.file, [path(a.file)])
    elif a.range:
        lo, hi = (int(x) for x in a.range.split("-"))
        report("%05d-%05d" % (lo, hi), [path(i) for i in range(lo, hi + 1) if os.path.exists(path(i))])
    else:
        runs = defaultdict(list)
        for f in sorted(glob.glob(os.path.join(a.dir, "pair_*.png"))):
            m = re.search(r"pair_(\d+)_", os.path.basename(f))
            if not m:
                continue
            runs[round(os.path.getmtime(f))].append((int(m.group(1)), f))
        groups = sorted((sorted(v) for v in runs.values() if len(v) >= 3), key=lambda v: v[0][0])
        for g in groups[-a.tail:]:
            report("%05d-%05d" % (g[0][0], g[-1][0]), [f for _, f in g])
    print("-" * 160)
    print("spread   = p95-p5 of the per-pixel disparity: how much the depth VARIES across the frame.")
    print("eye step = the median per-pixel disparity (the middle of that range, not its strong end).")
    print("bands    = the same median, measured separately in the top / middle / bottom third of the")
    print("           frame. A band far below the others means the depth does not cover the frame --")
    print("           the failure the node reports as 'incoherent'.")
    print("           CAUTION: that is AREA-weighted. The node's bands are RIGID fits (structure-weighted,")
    print("           plus a fit-quality per band), so the two can differ in magnitude AND in which band")
    print("           looks strongest -- on the Wan run below this reads mid-heavy while the node reads")
    print("           top/mid. Compare each tool's bands against its own whole-frame number, not across tools.")
    print("REAL PAIR: right half is a horizontal shift of the left      "
          "(calibrated: good runs 0.42-0.50, no-stereo run 0.70-0.90)")
    print("Film class for the eye step: 10-20 px at 1024 px eye width (median 14.8 = 1.45% of width).")
    print("The node's `eye_step_px` is a RIGID global fit and lands near the strong end of that")
    print("range -- on the Wan run in the docstring: -13.1 px against p5 -14.3 / p95 -0.1.")


if __name__ == "__main__":
    main()