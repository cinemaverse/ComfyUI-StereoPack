"""Scale the DEPTH (disparity) of a side-by-side stereo pair -- without touching motion.

Why this exists
---------------
A generated stereo pair is usually a little shallower than the real thing: measured on
this project, 8.9 px of disparity spread at a 768 px eye where the true same-instant
reference pair carried 10.4 px (+17%). Depth is an *interpretation* choice, not a
property of the render, so it can be corrected afterwards:

    x_L -> x_L + delta,   x_R -> x_R - delta,   delta = (factor - 1) * disparity / 2

Both eyes move by the same amount in OPPOSITE directions, about the same centre. That
scales the disparity by `factor` while leaving the zero-parallax plane where it was. The
temporal axis is never touched, so instant-to-instant motion is preserved exactly --
unlike StereoStabilize, which trades motion for cleanliness.

The disparity field is measured per pair from the two halves themselves (optical flow),
so no depth map is needed. Scale to a fixed factor, or aim at a target spread measured on
a real reference clip. `target_spread_pct` is in PERCENT OF EYE WIDTH, which is
resolution independent: a 1920 px eye carrying 26 px of spread is 1.35%, the same depth
as a 768 px eye carrying 10.4 px.

The measurement is deliberately identical to tools/pair_validity.py (flow from the left
half to the right, over textured pixels only, spread = p95 - p5), so the numbers the node
prints are directly comparable with that tool's.

Caveats, honestly
-----------------
* The disparity field is smoothed (median + gaussian) before use. Raw optical flow is
  noisy at occlusion boundaries and in flat regions, and a noisy correction field tears
  the picture; depth scaling only needs its low-frequency structure.
* Scaling UP creates disocclusion: pixels only one eye ever saw get stretched, not
  invented. Keep the factor small (< 1.3 is visually clean) and keep `max_shift_px` sane.
* Scaling up also increases divergence for near objects; beyond the reference depth
  expect eye strain rather than a better picture. Match, don't maximise.
* The three reported numbers are for YOU -- the correction maths never reads them.
  `spread_before_pct` is the uncorrected input (batch median), `spread_after_pct` is the
  result, and `spread_pass1_pct` is the spread between the two refinement passes (it
  equals `spread_before_pct` when `refine` is off, since there is only one measurement).

No dependencies beyond torch, numpy and opencv, all already in the ComfyUI environment.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

_CAT = "Stereo"

ARRANGEMENTS = ("parallel / headset  [L|R]", "cross-eyed  [R|L]")


def _first_half_is_left(arrangement: str) -> bool:
    """'cross' anywhere in the value -> the first half is the RIGHT eye."""
    return "cross" not in str(arrangement).lower()


def _gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)


def _measure(left: np.ndarray, right: np.ndarray):
    """Disparity field taking the first half onto the second, plus median and spread.

    Same convention as tools/pair_validity.py: flow is computed first -> second, the
    statistics use textured pixels only (top 35% gradient magnitude), and spread is
    p95 - p5 of the horizontal component. Returns (field, median, spread).

    Note this is symmetric in the two halves: swapping them negates `field`, and since
    the correction is also negated, the resulting depth change is identical. That is why
    the `arrangement` widget cannot make this node do the wrong thing.
    """
    f = cv2.calcOpticalFlowFarneback(left, right, None, 0.5, 3, 21, 3, 7, 1.5, 0)
    dx = f[..., 0].astype(np.float32)
    gx = cv2.Sobel(left, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(left, cv2.CV_32F, 0, 1, ksize=3)
    m = np.sqrt(gx * gx + gy * gy) > np.percentile(np.sqrt(gx * gx + gy * gy), 65)
    if not m.any():
        return np.zeros_like(dx), 0.0, 0.0
    vals = dx[m]
    med = float(np.median(vals))
    spread = float(np.percentile(vals, 95) - np.percentile(vals, 5))
    # flat regions have no reliable flow: use the pair's global shift there
    return np.where(m, dx, med).astype(np.float32), med, spread


def _shift(img: np.ndarray, amount) -> np.ndarray:
    """Move image CONTENT right by `amount` px (map_x = x - amount).

    `amount` is a scalar or a full (H, W) field, so the y-map must be passed explicitly:
    cv2.remap(img, map_x, None, ...) silently returns garbage.
    """
    h, w = img.shape[:2]
    xs = np.arange(w, dtype=np.float32)[None, :] - np.asarray(amount, dtype=np.float32)
    xs = np.ascontiguousarray(np.broadcast_to(xs, (h, w)))
    ys = np.ascontiguousarray(np.broadcast_to(np.arange(h, dtype=np.float32)[:, None], (h, w)))
    return cv2.remap(img, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _gray_of(rgb_float: np.ndarray) -> np.ndarray:
    """float RGB (0..255) -> grayscale float32, the form _measure expects."""
    return cv2.cvtColor(rgb_float.clip(0, 255).astype(np.uint8),
                        cv2.COLOR_RGB2GRAY).astype(np.float32)


def _smooth(field: np.ndarray, smooth_px: int) -> np.ndarray:
    """Raw flow is noisy at occlusion edges; depth scaling needs only low frequencies."""
    if smooth_px >= 3:
        return cv2.GaussianBlur(cv2.medianBlur(field, 5), (0, 0), smooth_px / 2.0)
    return field


class StereoDepthScale:
    """Scale (or auto-match) the disparity of an SBS pair batch. Motion is untouched."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "sbs": ("IMAGE", {"tooltip": "Side-by-side pairs, in order."}),
            "arrangement": (ARRANGEMENTS, {"default": ARRANGEMENTS[0],
                "tooltip": "Informational: which half is the left eye. Depth scaling is "
                           "symmetric, so this does NOT change the result (the "
                           "measurement and the correction flip together). Kept so the "
                           "graph reads like the other stereo nodes."}),
            "mode": (["match a target spread (%)", "scale by a fixed factor"], {
                "default": "match a target spread (%)",
                "tooltip": "'match' measures each batch and aims it at target_spread_pct. "
                           "'fixed' multiplies the disparity by depth_factor."}),
            "target_spread_pct": ("FLOAT", {"default": 1.35, "min": 0.05, "max": 8.0,
                "step": 0.01,
                "tooltip": "Target disparity spread as PERCENT OF EYE WIDTH, so it is "
                           "resolution independent. 1.35% is a real same-instant pair "
                           "from a stereo feature (26 px at a 1920 px eye = 10.4 px at "
                           "a 768 px eye)."}),
            "depth_factor": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 3.0, "step": 0.01,
                "tooltip": "Used in 'fixed' mode. 1.0 = unchanged, 1.17 = +17% depth."}),
            "smooth_px": ("INT", {"default": 9, "min": 0, "max": 51, "step": 2,
                "tooltip": "Gaussian smoothing of the disparity field, in pixels. Depth "
                           "scaling needs only its low-frequency structure; raw flow is "
                           "noisy at occlusion edges and would tear the picture."}),
            "max_shift_px": ("INT", {"default": 12, "min": 1, "max": 200, "step": 1,
                "tooltip": "Clamp on the per-pixel correction, guarding against flow "
                           "outliers. +17% depth needs only a few px."}),
            },
            "optional": {
                "convergence_px": ("FLOAT", {"default": 0.0, "min": -64.0, "max": 64.0,
                    "step": 0.5,
                    "tooltip": "Move BOTH eyes the same way: shifts where the scene sits "
                               "relative to the screen plane without changing its depth."}),
                "check": ("BOOLEAN", {"default": True,
                    "tooltip": "Re-measure the spread after warping and report it. It "
                               "should land on the target; this is the check that the "
                               "symmetric correction really is symmetric. With it off, "
                               "spread_after_pct is simply a copy of spread_before_pct."}),
                "refine": ("BOOLEAN", {"default": True,
                    "tooltip": "'match' mode only: measure after the first correction and "
                               "apply the residual, so the result actually lands on the "
                               "target. Flow measurement has a slight bias (~6% low), "
                               "which this removes."}),
            }}

    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("sbs", "report", "spread_before_pct", "spread_after_pct",
                    "spread_pass1_pct")
    FUNCTION = "run"
    CATEGORY = _CAT

    def run(self, sbs, arrangement, mode, target_spread_pct, depth_factor, smooth_px,
            max_shift_px, convergence_px=0.0, check=True, refine=True):
        x = (sbs.clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
        first_is_left = _first_half_is_left(arrangement)
        half = x.shape[2] // 2
        first, second = x[:, :, :half], x[:, :, half:]
        left, right = (first, second) if first_is_left else (second, first)

        out = np.empty_like(x)
        before, after, mid, factors = [], [], [], []
        for i in range(x.shape[0]):
            eye_w = float(half) if half else 1.0
            cur_l = left[i].astype(np.float32)
            cur_r = right[i].astype(np.float32)
            total = 1.0
            input_pct = mid_pct = None

            n_pass = 2 if (mode.startswith("match") and refine) else 1
            for it in range(n_pass):
                field, _, spread = _measure(_gray_of(cur_l), _gray_of(cur_r))
                pct = 100.0 * spread / eye_w
                if it == 0:
                    input_pct = pct        # the number the user asked "how deep is it?"
                else:
                    mid_pct = pct          # spread after pass 1 (refine only)
                if mode.startswith("match"):
                    step = (target_spread_pct / pct) if pct > 1e-6 else 1.0
                else:
                    step = float(depth_factor)
                step = float(np.clip(step, 0.2, 3.0))
                if it and abs(step - 1.0) < 0.03:      # second pass: already there
                    break
                # disparity d = -(flow L->R); symmetric -> delta_L = -(step-1)*d/2
                delta = np.clip(-0.5 * (step - 1.0) * _smooth(field, smooth_px),
                                -max_shift_px, max_shift_px)
                if it == 0 and convergence_px:
                    delta = delta + convergence_px
                cur_l = _shift(cur_l, delta)
                cur_r = _shift(cur_r, -delta)
                total *= step

            if check:
                _, _, spread2 = _measure(_gray_of(cur_l), _gray_of(cur_r))
                after.append(100.0 * spread2 / eye_w)
            before.append(input_pct if input_pct is not None else pct)
            mid.append(mid_pct)
            factors.append(total)

            pair = (np.concatenate([cur_l, cur_r], axis=1) if first_is_left
                    else np.concatenate([cur_r, cur_l], axis=1))
            out[i] = np.clip(pair, 0, 255).round().astype(np.uint8)

        n = len(before)
        mb = float(np.median(before)) if n else 0.0
        ma = float(np.median(after)) if after else mb
        mm = (float(np.median([m for m in mid if m is not None]))
              if any(m is not None for m in mid) else mb)
        mf = float(np.median(factors)) if n else 1.0
        rep = ("%d pairs | eye %dpx | factor %.3f | spread %.2f%% -> %.2f%% of eye width"
               % (n, half, mf, mb, ma))
        if mm is not None and abs(mm - mb) > 0.02:
            rep += " (after pass 1: %.2f%%)" % mm
        if mode.startswith("match") and n and abs(ma - target_spread_pct) > 0.25 * target_spread_pct:
            rep += ("  NOTE: overshot/undershot the target -- the disparity field may be "
                    "too noisy for this content, or max_shift_px is clamping it.")
        print("[StereoDepthScale] " + rep)
        return (torch.from_numpy(out).to(torch.float32) / 255.0, rep, mb, ma, mm)


NODE_CLASS_MAPPINGS = {"StereoDepthScale": StereoDepthScale}
NODE_DISPLAY_NAME_MAPPINGS = {"StereoDepthScale": "Stereo Depth Scale (disparity, motion-safe)"}
