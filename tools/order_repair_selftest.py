"""Selftest for the per-pair, trust-gated order repair in StereoPairFrames.

Run with the ComfyUI embedded interpreter (needs torch + cv2 + the node):

    python_embeded/python.exe tools/order_repair_selftest.py

Cases
-----
1. REAL render 387 (the clip that motivated the change). 8 near-dominant opening pairs measure
   -9.8..-13.1 px with fits 0.01-0.13, the far-dominant body +1.5..+7.7 px with fits 0.08-0.19.
   Nothing clears the trust bar, so the node must ship the clip EXACTLY as generated -- the plain
   frame-even|frame-odd pairing -- and say so. This is the arrangement the user confirmed by eye
   (with `measure` off) after every repair attempt had inverted a group that was already correct.
2. Synthetic controls with clean fits (a translated texture: the rigid fit explains everything),
   so the repair path itself is exercised: trained-sign clip -> no swap; mirrored clip -> one
   uniform swap; mixed clip -> ONLY the contradicting pairs, pair by pair, with the majority
   deliberately on the other side (the case where a count-based rule picks the wrong group).
"""
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nodes.stereo_pair import StereoPairFrames, _global_shift  # noqa: E402

VIEW_CROSS = "cross-eyed  (left = frame 1, right = frame 2)"
SRC387 = (r"D:/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable/ComfyUI/input/"
          r"MiniMax_H3_00387_.mp4")
FAILED = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (("  -> " + detail) if detail else ""))
    if not cond:
        FAILED.append(name)


def node(images, viewing=VIEW_CROSS, **kw):
    args = dict(skip_first=0, auto_extend=True, expected_min_px=3, measure=True,
                phase_lock=False, reject_bad_seed=False, auto_honor=True)
    args.update(kw)
    return StereoPairFrames().pair(images, viewing, "drop (leave it alone)", True, **args)


def report_lines(report, *keys):
    return [ln for ln in report.splitlines() if any(k in ln for k in keys)]


# --------------------------------------------------------------------------- 1. the real render
print("\n=== 1. render 387 (real clip) ===")
cap = cv2.VideoCapture(SRC387)
frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
cap.release()
src = torch.from_numpy(np.stack(frames)).float().div_(255.0)          # B,H,W,3 in 0..1
print("  clip: %d frames, %dx%d" % (src.shape[0], src.shape[2], src.shape[1]))

sbs, second, pairs, dropped, step, phase, report, verdict, ok = node(src)
print("  verdict: %s" % verdict)
for ln in report_lines(report, "auto-honor", "order repair", "clip order", "sampled steps"):
    print("    | " + ln)

check("387: 62 pairs", pairs == 62, "pairs=%d" % pairs)
check("387: repair skipped (nothing cleared the trust bar)",
      "auto-honor SKIPPED" in report)
check("387: no pair was swapped", "auto-honor (per-pair)" not in report
      and "came out MIRRORED" not in report)

# emitted halves must be the plain pairing: left half == frame 2k, right half == frame 2k+1
plain, worst = True, 0.0
for k in range(pairs):
    left = (sbs[k, :, :sbs.shape[2] // 2, :].numpy() * 255).round()
    right = (sbs[k, :, sbs.shape[2] // 2:, :].numpy() * 255).round()
    e0 = np.abs(left - frames[2 * k]).mean()
    e1 = np.abs(right - frames[2 * k + 1]).mean()
    worst = max(worst, e0, e1)
    if e0 > 2 or e1 > 2:
        plain = False
check("387: shipped as generated (left=frame-even, right=frame-odd) on all 62 pairs",
      plain, "worst half error %.3f/255" % worst)

# the gates the skip is based on: every pair is weak by fit, the split is content-driven
steps = []
for k in range(pairs):
    a = frames[2 * k].astype(np.float32)
    b = frames[2 * k + 1].astype(np.float32)
    steps.append(_global_shift(a, b))
opening, body = steps[:8], steps[8:]
med_open = float(np.median([d for d, _, _ in opening]))
med_body = float(np.median([d for d, _, _ in body]))
check("387: opening reads negative, body positive (content-driven split)",
      med_open < -5.0 and med_body > 2.0,
      "median opening %+.1f px, median body %+.1f px" % (med_open, med_body))
check("387: no pair has fit >= 0.20 (so no swap is supportable)",
      max(e for _, _, e in steps) < 0.20, "best fit %.2f" % max(e for _, _, e in steps))

# ------------------------------------------------------------ 2. synthetic controls, clean fits
print("\n=== 2. synthetic controls (clean fits) ===")
rng = np.random.default_rng(7)
base = rng.normal(128, 45, (96, 160)).astype(np.float32)
base = cv2.GaussianBlur(base, (0, 0), 1.2)


def shifted(dx, n=2):
    """`n` frames of a real 3-D-ish pair: a textured plane shifted by dx between the eyes."""
    out = []
    for i in range(n):
        dxs = dx if i % 2 == 0 else 0.0            # frame 2k has the content at `dx`, 2k+1 at 0
        img = np.stack([np.roll(base, int(round(dxs)), axis=1)] * 3, axis=-1)
        out.append(img)
    return out


def clip(sequence):
    return torch.from_numpy(np.stack(sequence)).float().div_(255.0)


def emitted_lr(sbs, k):
    w = sbs.shape[2] // 2
    return (sbs[k, :, :w].numpy() * 255).round(), (sbs[k, :, w:].numpy() * 255).round()


# `_global_shift(frame_even, frame_odd)` is the shift of the even frame onto the odd one, so a roll
# of +12 on the even frame measures -12 px = the trained sign, and a roll of -12 measures +12 px.
trained = clip(shifted(+12.0) * 8)
sbs, _, pairs, _, _, _, report, verdict, _ = node(trained)
check("trained-sign clip: nothing swapped", "Swapped:" not in report,
      "; ".join(report_lines(report, "order repair", "auto-honor")[:1]))
check("trained-sign clip: left half is the even frame (as generated)",
      np.abs(emitted_lr(sbs, 0)[0] - trained[0].numpy() * 255).mean() < 2)

mirrored = clip(shifted(-12.0) * 8)
sbs, _, pairs, _, _, _, report, verdict, _ = node(mirrored)
check("mirrored clip: all 8 pairs swapped in one uniform swap",
      "came out MIRRORED" in report and "Swapped: 0-7" in report,
      "; ".join(report_lines(report, "auto-honor")[:1]))
check("mirrored clip: left half is now the odd frame",
      np.abs(emitted_lr(sbs, 0)[0] - mirrored[1].numpy() * 255).mean() < 2)

# mixed, with the MIRRORED side in the majority: 12 mirrored pairs + 6 trained ones. The count
# rule that shipped the 387 regression swapped the minority (here: the 6 correct pairs); the
# convention-anchored rule swaps the 12 mirrored ones and leaves the 6 trained pairs alone.
mixed = clip(shifted(-12.0) * 12 + shifted(+12.0) * 6)
sbs, _, pairs, _, _, _, report, verdict, _ = node(mixed)
swapped_line = report_lines(report, "Swapped:")
check("mixed clip: the 12 mirrored pairs were swapped and the 6 trained pairs left alone "
      "(no count involved)",
      bool(swapped_line) and swapped_line[0].strip().endswith("0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11")
      and "12 of 18 pairs" in report,
      swapped_line[0] if swapped_line else "no swap line")
check("mixed clip: report says per-pair, not majority",
      "auto-honor (per-pair)" in report)

# auto_honor off -> documented fallback (trained order assumed, no swapping at all)
sbs, _, _, _, _, _, report, _, _ = node(mirrored, auto_honor=False)
# legacy phase_lock must not fill the gap when the auto-honor deliberately abstained
sbs, _, _, _, _, _, report_pl, _, _ = node(src, phase_lock=True)
check("387: phase_lock ON cannot re-invert the clip while auto_honor is on",
      "Swapped:" not in report_pl and "auto-honor SKIPPED" in report_pl,
      "; ".join(report_lines(report_pl, "auto-honor")[:1]))

check("auto_honor off: no swapping", "Swapped:" not in report and "auto-honor:" not in report,
      "; ".join(report_lines(report, "clip order")[:1]))

print("\n%s (%d failure%s)" % ("SELFTEST PASS" if not FAILED else "SELFTEST FAIL",
                              len(FAILED), "" if len(FAILED) == 1 else "s"))
sys.exit(1 if FAILED else 0)
