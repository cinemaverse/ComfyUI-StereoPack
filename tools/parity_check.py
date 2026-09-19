#!/usr/bin/env python
"""
parity_check.py -- protect the one property that must never silently flip:
which physical eye is frame 0 (depth ordering / pseudoscopy).

Three checks, because parity can break in three different places:

  clips   the EXTRACTION side. Verifies an emitted training clip against the
          source movie at pixel level:
             frame0 == source RIGHT half at instant t
             frame1 == source LEFT  half of the SAME instant
             frame2 == source RIGHT half at instant t+K        (K = stride)
          This is the check that a stride implementation must not break.

  outputs the GENERATION side, absolute. Compares a run's disparity field with
          the ground-truth pair of the anchor (eye-R/eye-L of the same source
          frame, laid out as [R|L] to match the generated pair image). A parity
          flip negates the disparity field, so the correlation goes negative.

  diff    the GENERATION side, differential. Compares a run against a
          known-good run (same seed/prompt/anchor, e.g. the verified
          new_lora_new_phrase baseline). This is the guard to run after ANY
          post-process (disparity scale / HIT): a sign or wrong-panel error
          shows up as a negative correlation, with no vision needed.

Disparity convention used throughout (established and verified in STATUS.md 8.2):
for a converged horizontal rig laid out [R|L],
    dx = x_L - x_R = +d,   d > 0 for content nearer than the convergence plane.
A parity flip negates this, so a positive correlation with a known-good
reference means the ordering survived; negative means pseudoscopic.

Usage:
  parity_check.py clips  --movie M --clip C --t0 T [--stride K]
  parity_check.py outputs --dir D [--tag T ...] [--ref-tag T] [--near x,y,w,h]
  parity_check.py diff   --dir D --ref-tag T [--tag T ...]
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess

import cv2
import numpy as np

GT_DIR = "input"                # relative to ComfyUI's root; override with --dir
GT_R = GT_DIR + "/cross__00055__eye-R.png"
GT_L = GT_DIR + "/cross__00055__eye-L.png"
FPS = 24000 / 1001


# ------------------------------------------------------------------ helpers
def gray(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)


def flow(a, b):
    return cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 21, 3, 7, 1.5, 0)


def tex_mask(a, pct=70):
    gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    t = np.sqrt(gx ** 2 + gy ** 2)
    return t > np.percentile(t, pct)


def corr(a, b, m):
    x, y = a[m].astype(np.float64), b[m].astype(np.float64)
    x = x - x.mean(); y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else 0.0


def verdict(c):
    if c > 0.15:
        return "PARITY OK"
    if c < -0.15:
        return "INVERTED (pseudoscopic)"
    return "inconclusive"


def src_frames(movie, t0, n, stride=1, w=3840, h=1080):
    """n source frames from t0, every `stride`-th one."""
    vf = "select='not(mod(n\\,%d))'" % stride if stride > 1 else None
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-ss", "%.3f" % t0, "-i", movie]
    if vf:
        cmd += ["-vf", vf, "-fps_mode", "passthrough"]
    cmd += ["-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    fsz = w * h * 3
    c = len(raw) // fsz
    return np.frombuffer(raw[:c * fsz], np.uint8).reshape(c, h, w, 3)


def clip_frames(path, n):
    cap = cv2.VideoCapture(path)
    out = []
    while len(out) < n:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def pair_fields(root, tag, nhint=8, wmax=768):
    """mean disparity field + list over the first nhint pairs of a run, from the raw seq."""
    fs = sorted(glob.glob(os.path.join(root, tag + "_pair_seq_*.png")))
    if not fs:
        fs = sorted(glob.glob(os.path.join(root, tag + "_seq_*.png")))
    g = [gray(cv2.imread(p)) for p in fs[:2 * nhint + 1]]
    g = [x for x in g if x is not None]
    if len(g) < 4:
        return None, None
    h, w = g[0].shape
    s = wmax / float(w)
    if s != 1.0:
        g = [cv2.resize(x, (wmax, int(round(h * s))), interpolation=cv2.INTER_AREA) for x in g]
    fields, mags = [], []
    for i in range(0, len(g) - 1, 2):
        f = flow(g[i], g[i + 1])
        fields.append(f[..., 0] / s)          # disparity in reference px
        mags.append(f[..., 0] / s)
    m = tex_mask(g[0])
    return np.mean(fields, axis=0), m


# ------------------------------------------------------------------ clips
def cmd_clips(a):
    sf = src_frames(a.movie, a.t0, 4, 1)          # t0, +1, +2, +3
    if len(sf) < 3:
        print("could not decode source"); return 1
    cf = clip_frames(a.clip, 3)
    if len(cf) < 3:
        print("could not decode clip"); return 1
    # expected source frame index for each clip frame, given the stride
    want = [(0, "RIGHT"), (0, "LEFT"), (a.stride, "RIGHT")]
    print("clip %s   stride=%d" % (os.path.basename(a.clip), a.stride))
    ok = True
    for ci, (off, eye) in enumerate(want):
        if off >= len(sf):
            print("  frame%d: source frame %d unavailable" % (ci, off)); ok = False; continue
        s = sf[off]
        hw = s.shape[1] // 2
        c = cf[ci]
        cs = cv2.resize(c, (hw, s.shape[0]), interpolation=cv2.INTER_AREA)
        dR = float(np.abs(gray(cs) - gray(s[:, hw:])).mean())
        dL = float(np.abs(gray(cs) - gray(s[:, :hw])).mean())
        got = "RIGHT" if dR < dL else "LEFT"
        good = got == eye
        ok &= good
        print("  frame%d -> source instant +%d, %-5s eye (mad R=%.2f L=%.2f)  %s"
              % (ci, off, got, dR, dL, "OK" if good else "MISMATCH (expected %s)" % eye))
    print("  => %s" % ("PARITY OK (R,L,R over instants 0,0,%d)" % a.stride if ok else "PARITY BROKEN"))
    return 0 if ok else 2


# ------------------------------------------------------------------ outputs
def cmd_outputs(a):
    # reference: ground-truth anchor pair, laid out [R|L] like the generated pair image
    R = cv2.imread(GT_R); L = cv2.imread(GT_L)
    if R is None or L is None:
        print("ground-truth anchor pair not found in %s" % GT_DIR); return 1
    h = min(R.shape[0], L.shape[0]); w = min(R.shape[1], L.shape[1])
    ref_pair = np.hstack([cv2.resize(R[:h, :w], (w, h)), cv2.resize(L[:h, :w], (w, h))])
    hh, ww = ref_pair.shape[:2]
    rp = cv2.resize(ref_pair, (2 * a.wmax, int(round(hh * (2 * a.wmax) / ww))), interpolation=cv2.INTER_AREA)
    f = flow(gray(rp[:, :rp.shape[1] // 2]), gray(rp[:, rp.shape[1] // 2:]))
    ref_field = f[..., 0]

    tags = a.tag or sorted(set(re.sub(r"_pair_seq_\d+_?\.png$", "", os.path.basename(p))
                                 for p in glob.glob(os.path.join(a.dir, "*_pair_seq_*.png"))))
    print("reference: ground-truth anchor pair [eye-R | eye-L]  (parity: near content has dx>0)")
    print("\n%-26s %8s %10s %8s   %s" % ("tag", "corr", "dx_med", "spread", "verdict"))
    print("-" * 78)
    for t in tags:
        fld, m = pair_fields(a.dir, t, wmax=a.wmax)
        if fld is None:
            print("%-26s  (no output yet)" % t); continue
        c = corr(fld.ravel(), cv2.resize(ref_field, (fld.shape[1], fld.shape[0])).ravel(), m.ravel())
        spread = float(np.percentile(fld[m], 95) - np.percentile(fld[m], 5))
        print("%-26s %+8.3f %10.1f %8.1f   %s" % (t, c, float(np.median(fld[m])), spread, verdict(c)))
    return 0


def cmd_diff(a):
    rf, m = pair_fields(a.dir, a.ref_tag, wmax=a.wmax)
    if rf is None:
        print("reference tag %s has no output" % a.ref_tag); return 1
    tags = a.tag or sorted(set(re.sub(r"_pair_seq_\d+_?\.png$", "", os.path.basename(p))
                               for p in glob.glob(os.path.join(a.dir, "*_pair_seq_*.png"))))
    print("reference (known good): %s" % a.ref_tag)
    print("\n%-26s %8s %10s %8s   %s" % ("tag vs reference", "corr", "dx_med", "spread", "verdict"))
    print("-" * 78)
    for t in tags:
        if t == a.ref_tag:
            continue
        fld, mm = pair_fields(a.dir, t, wmax=a.wmax)
        if fld is None:
            print("%-26s  (no output yet)" % t); continue
        k = mm.ravel() & m.ravel() if mm.shape == m.shape else m.ravel()
        c = corr(fld.ravel(), cv2.resize(rf, (fld.shape[1], fld.shape[0])).ravel(), k)
        print("%-26s %+8.3f %10.1f %8.1f   %s" % (t, c, np.median(fld[mm]),
              float(np.percentile(fld[mm], 95) - np.percentile(fld[mm], 5)), verdict(c)))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("clips")
    c.add_argument("--movie", required=True); c.add_argument("--clip", required=True)
    c.add_argument("--t0", type=float, required=True); c.add_argument("--stride", type=int, default=1)
    c.set_defaults(fn=cmd_clips)

    o = sub.add_parser("outputs")
    o.add_argument("--dir", required=True); o.add_argument("--tag", action="append")
    o.add_argument("--ref-tag"); o.add_argument("--near")
    o.add_argument("--wmax", type=int, default=768)
    o.set_defaults(fn=cmd_outputs)

    d = sub.add_parser("diff")
    d.add_argument("--dir", required=True); d.add_argument("--ref-tag", required=True)
    d.add_argument("--tag", action="append"); d.add_argument("--wmax", type=int, default=768)
    d.set_defaults(fn=cmd_diff)

    a = ap.parse_args()
    raise SystemExit(a.fn(a))