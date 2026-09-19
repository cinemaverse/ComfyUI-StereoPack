#!/usr/bin/env python
"""
eye_order_check.py -- which half of a stereo pair is the LEFT eye?

Answer for the DDD LoRA (read from ComfyUI-DDD/nodes.py, Create3DImage):

    left_eye  = images[5]      # the LATER frame of the pair
    right_eye = images[4]      # the EARLIER frame of the pair

So in a 1-in-2 alternation the EVEN indices (0,2,4,...) are the RIGHT eye and
the ODD indices (1,3,5,...) are the LEFT eye.  "frame1 = right, frame2 = left"
is therefore the trained convention, not a reversal.  (The node's own comment
says "indices 7 and 8" -- stale, ignore it; the code is the authority.)

This script verifies that convention EMPIRICALLY on rendered output.

Sign convention (standard pinhole stereo, right camera at +B):
    x_R = x_L - fx*B/Z            near disparity is positive
    dx(half0 -> half1) = x_1 - x_0
    half0 is the LEFT eye  <=>  dx_near < 0    (near content moves left)

WHY NOT |dx| MAGNITUDE RANKING ("near = biggest shift")
-------------------------------------------------------
It is WRONG on a DDD pair. The DDD second eye is *generated*, not a warp, so
block/flow matching on the far planet/window produced confident false matches
at +26 px, and a magnitude-ranked test then reports "[R|L]" for a pair that is
"[L|R]". A known-near object, or a depth mask, is required. This tool uses the
depth mask when given one and refuses to give a verdict that violates the
near/far magnitude ordering.

Usage
-----
  # SBS clip (each frame already holds both eyes) + DA3 metric depth
  python eye_order_check.py --video mono81_pf2_headset.mp4 \
      --depth-npz path/to/depth.npz

  # a specific known-near object instead of a depth map  (x0,y0,w,h, 768-space)
  python eye_order_check.py --video mono81_pf2_headset.mp4 --near-box 240,430,130,170

  # raw alternation clip: consecutive whole frames are the two eyes
  python eye_order_check.py --video wan22_65__00099.mp4 --mode alternate --depth-npz ...
"""
import argparse
import glob
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

S_DEFAULT = 768


# ---------------------------------------------------------------- utilities
def gray(x):
    return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32)


def read_frames(path):
    d = tempfile.mkdtemp()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path,
                    os.path.join(d, "f%05d.png")], check=True)
    return [cv2.imread(p) for p in sorted(glob.glob(os.path.join(d, "*.png")))]


def flow_dx(a, b):
    """dx such that x_b = x_a + dx (Farneback, prev=a, next=b)."""
    return cv2.calcOpticalFlowFarneback(gray(a), gray(b), None,
                                        0.5, 3, 25, 3, 5, 1.1, 0)[..., 0]


def selftest():
    """A=left, B=right, near disparity 4..18 px -> near dx must be < 0."""
    rng = np.random.default_rng(0)
    h, w = 384, 768
    img = np.full((h, w, 3), 60, np.uint8)
    for _ in range(90):
        x = int(rng.integers(0, w - 60)); y = int(rng.integers(0, h - 60))
        cv2.rectangle(img, (x, y),
                      (x + int(rng.integers(10, 50)), y + int(rng.integers(10, 50))),
                      tuple(int(c) for c in rng.integers(40, 230, 3)), -1)
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    disp = 18.0 - 14.0 * (yy / h)                 # near at top
    B = cv2.remap(img, (xx + disp).astype(np.float32), yy, cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_REPLICATE)   # B(x) = A(x + disp)
    dx = flow_dx(img, B)
    near = yy < h / 3
    n, f = float(np.median(dx[near])), float(np.median(dx[~near]))
    ok = n < 0 and abs(n) > abs(f)
    print(f"[selftest] synthetic A=LEFT B=RIGHT  dx_near {n:+.2f} dx_far {f:+.2f}"
          f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def load_depth(npz, fidx, size):
    z = np.load(npz)
    d = z["depth"] if "depth" in z else list(z.values())[0]
    k = fidx if fidx < d.shape[0] else d.shape[0] - 1
    return cv2.resize(np.asarray(d[k], np.float32), (size, size),
                      interpolation=cv2.INTER_LINEAR), d


def shift_of_box(a, b, box, maxs=60, blur=1.5):
    """dx for one known-near box, taken from `a`, found in `b`."""
    from scipy.ndimage import gaussian_filter

    def ncc(p, q):
        p = p - p.mean(); q = q - q.mean()
        return float((p * q).mean() /
                     (np.sqrt((p * p).mean() * (q * q).mean()) + 1e-9))
    x0, y0, w, h = box
    P = gaussian_filter(gray(a)[y0:y0 + h, x0:x0 + w], blur)
    best = (0, -2.0)
    for dx in range(-maxs, maxs + 1):
        xs = x0 + dx
        if xs < 0 or xs + w > a.shape[1]:
            continue
        v = ncc(P, gaussian_filter(gray(b)[y0:y0 + h, xs:xs + w], blur))
        if v > best[1]:
            best = (dx, v)
    return best


# ---------------------------------------------------------------- reporting
def report(label, d_near, d_far, n_near, n_far, coherent=None):
    if d_near is None:
        print(f"  {label:38s}  could not measure")
        return None
    verdict = "half0 = LEFT  [L|R]" if d_near < 0 else "half0 = RIGHT [R|L]"
    extra = ""
    if coherent is not None and not coherent:
        extra = "   <-- INCOHERENT (near |dx| < far |dx|): do NOT trust"
    print(f"  {label:38s}  dx_near {d_near:+6.2f}  dx_far {d_far:+6.2f}  "
          f"(n {n_near}/{n_far})  -> {verdict}{extra}")
    if coherent is not None and not coherent:
        return None
    return verdict


def check_clip(path, mode, depth_npz, near_box, max_pairs):
    frames = read_frames(path)
    if not frames:
        print(f"  {path}: unreadable")
        return None
    S = frames[0].shape[0]           # square eyes
    pairs = []
    if mode == "sbs":
        for f in frames:
            pairs.append((f[:, :S], f[:, S:2 * S]))
    else:
        for i in range(0, len(frames) - 1, 2):
            pairs.append((frames[i], frames[i + 1]))
    pairs = pairs[:max_pairs]

    print(f"\n=== {path}  [{mode}]  {S}x{S} per eye  {len(pairs)} pairs")
    votes = []
    if near_box:
        ds, vs = [], []
        for i, (a, b) in enumerate(pairs):
            dx, v = shift_of_box(a, b, near_box)
            if v > 0.3:
                ds.append(dx); vs.append(v)
        if ds:
            from collections import Counter
            common = Counter(ds).most_common(1)[0]
            votes.append(report(f"known-near box {near_box}", float(np.median(ds)), float("nan"),
                                 len(ds), 0, coherent=True))
            print(f"      vote: dx={common[0]:+d} in {common[1]}/{len(ds)} pairs, "
                  f"ncc {np.median(vs):.3f}")
    if depth_npz:
        ns, fs = [], []
        for i, (a, b) in enumerate(pairs):
            d, allz = load_depth(depth_npz, i, S)
            zn, zf = np.percentile(d, 15), np.percentile(d, 85)
            dx = flow_dx(a, b)                    # a = half0, b = half1
            ns.append(np.median(dx[d <= zn]))
            fs.append(np.median(dx[d >= zf]))
        ns, fs = np.array(ns), np.array(fs)
        coherent = abs(np.median(ns)) >= abs(np.median(fs))
        votes.append(report("depth-anchored (DA3 metric)", float(np.median(ns)),
                            float(np.median(fs)), len(ns), len(fs), coherent))
    if not near_box and not depth_npz:
        print("  !! give --depth-npz or --near-box; a magnitude-ranked test is")
        print("     invalid on a generated (non-warped) second eye (see docstring).")
    return votes[-1] if votes else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", action="append", required=True)
    ap.add_argument("--mode", choices=["sbs", "alternate"], default="sbs")
    ap.add_argument("--depth-npz", default=None)
    ap.add_argument("--near-box", default=None, help="x0,y0,w,h in eye pixels")
    ap.add_argument("--max-pairs", type=int, default=20)
    a = ap.parse_args()

    if not selftest():
        print("sign self-test failed -- refusing to report")
        sys.exit(2)

    box = tuple(int(v) for v in a.near_box.split(",")) if a.near_box else None
    out = []
    for v in a.video:
        r = check_clip(v, a.mode, a.depth_npz, box, a.max_pairs)
        if r:
            out.append((r, v))
    print("\n---- summary (half0 = first half / first frame of each pair) ----")
    for r, v in out:
        print(f"  {r:20s}  {v}")
    print("\nDDD convention: even indices = RIGHT eye, odd indices = LEFT eye")
    print("(ComfyUI-DDD/nodes.py: left_eye = images[5], right_eye = images[4]).")
    print("To flip the viewing, swap the halves -- no second generation needed.")


if __name__ == "__main__":
    main()
