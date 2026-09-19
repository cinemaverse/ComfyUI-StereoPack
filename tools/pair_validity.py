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
    return dict(spread=float(np.percentile(dx[m], 95) - np.percentile(dx[m], 5)),
                dx_med=float(np.median(dx[m])), dy=float(np.median(np.abs(dy[m]))),
                zero=zero, ratio=best / zero if zero > 0 else 1.0, L=L)


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
    print("%-20s %3d pairs  spread %5.1f  |dy| %5.2f  mean|L-R| %5.2f  warp ratio %5.2f  "
          "INSTANT %5.2f px   %s"
          % (name, len(res), med("spread"), med("dy"), med("zero"), med("ratio"),
             instant_motion(Ls), v))


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
    print("%-20s %10s %14s %11s %14s %15s   %s" %
          ("run", "", "spread", "|dy|", "mean|L-R|", "warp ratio", "verdict"))
    print("-" * 108)
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
    print("-" * 108)
    print("REAL PAIR: right half is a horizontal shift of the left      "
          "(calibrated: good runs 0.42-0.50, no-stereo run 0.70-0.90)")


if __name__ == "__main__":
    main()