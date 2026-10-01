"""ComfyUI-StereoStabilize -- remove the temporal BOIL from a side-by-side pair.

One node:

  StereoStabilizeSBS   temporal median, per eye, of an SBS image batch

Purely additive: nothing else is modified, and no new dependencies (torch, numpy and
opencv are already in the ComfyUI environment).

Why a temporal MEDIAN and not a temporal mean or a motion-compensated average -- all
measured on real output, stereo_8fps_00019.mp4 (80 pairs, 768x576 per eye):

  * the camera is NOT shaking.  ECC similarity pose over all 80 frames: total drift
    0.116 px, jitter sd 0.14 px, rotation 0.004 deg.  A translation-only matcher hides
    this entirely.
  * a back-wall patch where nothing moves changes by 3.29 grey levels every frame
    (median 2.06, p90 6.68, p99 20.7).  56% of that variance is FAST.  The model
    re-synthesises every frame, so texture sizzles.  That is the "shaky hands".
  * motion compensation removes only 24-28% of the frame-to-frame change, so it is
    re-rendering, not sub-pixel drift -- aligning cannot fix it.

  filter                 boil      detail
  temporal MEAN   N=5   -44%      -26%
  temporal MEDIAN N=5   -35%      -10%
  temporal MEDIAN N=9   -47%      -13%
  temporal MEDIAN N=11  -51%      -14%

  The median costs half the detail of the mean for the same boil reduction, and detail
  loss SATURATES near -13% while the boil keeps falling: the median throws away the
  frames that disagree instead of averaging structure away.

Two things that must be right:

  * SELECT WHOLE PIXELS BY LUMA.  A per-channel median invents colours the model never
    rendered; picking the tap whose luma is the median keeps real pixels.
  * EACH EYE IS FILTERED INDEPENDENTLY.  Eye A and eye B are separate sequences (source
    frames 2i and 2i+1), so filtering them separately cannot change the disparity
    between them.  On the real clip: disparity sd 0.22 -> 0.00, range 12.76 -> 12.92,
    vertical 0.00.  The node re-measures this every run and shouts if it moved.

THE QUALITY COST IS AVOIDABLE: `flat_only`.  Filtering EVERY pixel is what costs detail -- at
an edge the median can pick a tap from a neighbouring instant and smear it.  So filter only
where there is no edge to smear.  The gate is the clip's own median gradient magnitude
(static, built once from frames spread across the clip, never per-frame -- a per-frame mask
flickers, and that flicker is worse than the boil it removes), feathered with a small blur
before thresholding.  Measured through the node on stereo_12fps_00001 (62 pairs,
1248x1664/eye, radius 4, `tools/flat_gate_check.py`):

                         boil (whole)   boil (inside the gate)   detail/frame   structure
    every pixel          8.63 -> 7.02      8.63 -> 7.02           -19%           +24%
    flat_only 60         8.63 -> 7.80      4.80 -> 3.46            +2%            +2%
    flat_only 85         8.63 -> 7.42      6.41 -> 4.94            -1%            +4%

`detail/frame` is the per-frame Laplacian variance the node prints, and most of what it
"loses" is the boil it has just removed, so it overstates the cost.  `structure` is the same
measure on the clip's TEMPORAL MEAN, where the boil averages out -- that one moves only if
something real was eaten, and it never drops.  Read the gate this way: inside the mask the
output is BIT-IDENTICAL to the ungated median (same cleanup -- 4.80 -> 3.46), outside the
mask it is BIT-IDENTICAL to the input.  So the gate costs nothing anywhere; all it decides is
HOW MUCH OF THE FRAME gets the cleanup.  `0` = off (filter everything), which is why the
default changes nothing for a graph saved before the gate existed.  Only the MEDIAN is gated:
the exposure fix is one gain + offset per frame, so it costs no detail anywhere and stays
global.

NOTHING RIGID CAN DO THIS (2026-09-30).  A whole-frame (rigid) stabiliser was measured
against the same clip and is nearly free -- an integer shift interpolates nothing at all
(boil 8.64 -> 8.64, detail 370.4 -> 370.4, bit-identical), a sub-pixel warp costs -0.3%
detail -- but it does not remove the jitter, it DOUBLES it: locking the lower two thirds with
a trajectory measured on those same two thirds moved the wobble 0.49 -> 0.93 px, and locking
the top third with its own trajectory moved it 1.42 -> 2.73 px.  The control explains why:
build a synthetic right eye by shifting the left eye by a CONSTANT -8.5 px but take the
frames from consecutive source frames -- exactly what a real pair is -- and the eye step
still measures 0.38 px of "wobble" that does not exist.  What looks like a moving disparity
is mostly the estimator reading the content difference between two independently
re-synthesised frames, and no transform can shift that away.  The real jitter is per-pixel,
so only a per-pixel filter removes it -- which is what this node does, and why the fix for
its quality cost is a gate, not a transform.

Tested and rejected (do not re-try): a variance gate that blends the median back only
where pixels are unstable is WORSE (-31% vs -42% boil at N=7) because it re-injects the
flicker it just removed.  Motion-compensated averaging is worse than a plain median
(detail -38% at 5 taps).

ONE THING THIS NODE CANNOT DO.  The pair rate is half the source frame rate (two source
frames make one pair), so the video output node must be set to HALF the source fps --
the wan22 source runs at 16 fps, so VHS_VideoCombine must be 8, and the clip should be
10.00 s for 80 pairs.  If it plays back at 16 fps the pair rate doubles and this node's
whole point is halved, because flicker frequency doubles with it.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

_CAT = "Stereo"

# --------------------------------------------------------------------------- self-test
def _self_test() -> None:
    """Cheap checks that CAN fail.  A filter that silently does nothing, or that blurs
    a moving edge, is the failure mode worth catching."""
    rng = np.random.default_rng(0)
    base = np.clip(rng.normal(128, 40, (48, 48)), 0, 255).astype(np.uint8)
    const = np.repeat(base[None].repeat(3, 2)[..., None], 3, axis=3)

    out = _temporal_median(const, 0)
    assert np.array_equal(out, const), "radius=0 is not the identity"

    flat = np.full((7, 32, 32, 3), 100, np.uint8)
    assert np.array_equal(_temporal_median(flat, 2), flat), "constant was changed"

    noisy = np.repeat(
        np.stack([np.clip(base.astype(np.float32) + rng.normal(0, 8, base.shape),
                          0, 255) for _ in range(5)]).astype(np.uint8)[..., None],
        3, axis=3)
    a = (noisy.astype(np.float32) - base[None, ..., None]).std()
    b = (_temporal_median(noisy, 2).astype(np.float32) - base[None, ..., None]).std()
    assert b < a * 0.95, f"median did not reduce noise ({a:.2f} -> {b:.2f})"

    sweep = np.zeros((9, 16, 64, 3), np.uint8)
    for t in range(9):
        sweep[t, :, 20 + t:] = 255
    step = np.abs(np.diff(_temporal_median(sweep, 4)[4, 8, :, 0].astype(int)))
    assert step.max() > 200, "median blurred a moving edge"

    # the flat gate must prefer the flat region and leave the busy one alone
    x = np.arange(64)
    amp = 20.0 + 40.0 * x / 63.0                       # a grating whose gradient varies, so the
    row = 128.0 + amp * np.sin(2 * np.pi * x / 3.0)     # threshold never sits ON a value
    tex = np.repeat(np.clip(row, 0, 255).astype(np.uint8)[None, :, None], 64, axis=0)
    tex = np.repeat(tex, 3, axis=2)
    tex[16:48, 16:48, :] = 128                         # flat square, a quarter of the frame
    m = _flat_mask(np.repeat(tex[None], 5, axis=0), 40, 3)
    assert m.dtype == np.bool_ and m.shape == (64, 64), "gate is not a bool (H,W) mask"
    flat_sel, busy_sel = m[20:44, 20:44].mean(), m[:8, :].mean()
    assert flat_sel > 0.9, f"gate did not pick the flat region ({flat_sel:.2f})"
    assert busy_sel < 0.4 * flat_sel, f"gate did not prefer the flat region ({busy_sel:.2f})"
    # ... and the gate must be exact outside the mask, in both directions
    f = tex.astype(np.float32) * 0 + 7.0
    assert np.array_equal(_gate(f, tex, np.ones((64, 64, 1), bool)), f), \
        "gate with an all-True mask is not the filter"
    assert np.array_equal(_gate(f, tex, np.zeros((64, 64, 1), bool)), tex), \
        "gate with an all-False mask is not the identity"

    print(f"[StereoStabilize] self-test OK "
          f"(radius=0 identity, constant kept, noise {a:.2f}->{b:.2f}, "
          f"moving edge kept {step.max()}, flat gate picks "
          f"{100 * flat_sel:.0f}% of a flat quarter vs {100 * busy_sel:.0f}% of a busy grating)")


# ------------------------------------------------------------------------------ filter
def _temporal_median(seq: np.ndarray, radius: int) -> np.ndarray:
    """Temporal median of one eye, selecting whole pixels.

    seq: (N,H,W,3) uint8.  For each pixel the tap whose LUMA is the median is chosen and
    that tap's full BGR is taken, so no colour is invented.
    """
    n = len(seq)
    if radius <= 0 or n < 3:
        return seq.copy()
    luma = np.stack([_luma(seq[i]) for i in range(n)])          # (N,H,W) uint8
    out = np.empty_like(seq)
    for t in range(n):
        lo, hi = max(0, t - radius), min(n - 1, t + radius)
        win = luma[lo:hi + 1]                                    # (m,H,W) float32
        med = np.median(win, axis=0)
        # the median of an odd-length window IS one of the samples, so the nearest tap
        # is exactly the median and its BGR is a real pixel
        pick = np.argmin(np.abs(win - med[None]), axis=0)
        out[t] = np.take_along_axis(seq[lo:hi + 1], pick[None, ..., None],
                                    axis=0)[0]
    return out


def _luma(img: np.ndarray) -> np.ndarray:
    """ITU-R 601 luma of an **RGB** image, in float32.

    ComfyUI IMAGE tensors are RGB, not BGR -- getting that backwards looks harmless
    (it is still a perceptual proxy) but it makes this node disagree with the offline
    tool about WHICH tap is the median, and at a high-contrast edge picking a
    neighbouring tap changes the pixel by 150+ levels.

    Routed through cv2 rather than a hand-written weighted sum on purpose: the offline
    tool uses cv2, and weights written by hand differ from cv2's in the last float bit,
    which flips the odd median tie-break at a high-contrast edge and changes that pixel
    by 150+ levels.  Identical maths on both sides is what makes the measured trade-offs
    reproducible (verified: max |diff| 0.0001 grey levels against the offline tool).
    """
    return cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY)


def _level_normalise(seq: np.ndarray, amount: float, clamp: float = 0.03) -> np.ndarray:
    """Per-frame luma match to the clip median, killing exposure pulsing.

    Gain AND offset, fitted on the MEDIAN and the median absolute deviation -- both
    robust, so a bright object entering frame cannot drag the correction.  Per eye, and
    clamped to +-3%, far below anything that could disturb the stereo (the eye-to-eye
    gain was measured at 1.008 +- 0.005).
    """
    if amount <= 0:
        return seq
    n = len(seq)
    g = np.stack([_luma(seq[i]) for i in range(n)])
    mu = np.median(g.reshape(n, -1), axis=1)
    mad = np.median(np.abs(g - mu[:, None, None]).reshape(n, -1), axis=1)
    tm, ts = float(np.median(mu)), float(np.median(mad))
    out = seq.astype(np.float32).copy()
    for i in range(n):
        a = float(np.clip(ts / max(mad[i], 1e-3), 1 - clamp, 1 + clamp))
        b = tm - a * mu[i]
        out[i] = np.clip(seq[i] * (1 + amount * (a - 1)) + amount * b, 0, 255)
    return out


def _flat_mask(seq: np.ndarray, pct: float, feather: int = 9,
               max_samples: int = 12) -> np.ndarray:
    """True where the clip is FLAT enough that the temporal median is safe to apply.

    The median only costs detail at edges, so this is the mask that keeps the detail and
    throws away only the sizzle.  Built from the MEDIAN spatial-gradient map over
    `max_samples` frames spread across the clip: a static mask, so the boundary between
    filtered and untouched pixels cannot itself flicker (a per-frame mask does, and the
    flicker it adds is worse than the boil it removes).

    `pct` is the share of the frame kept: 60 = filter the flattest 60%, 85 = protect only the
    strongest 15% of edges.  `feather` is the blur applied before thresholding, in pixels
    (1 = a hard boundary).  Static, and it does not touch the picture by itself.
    """
    n = len(seq)
    idx = np.linspace(0, n - 1, min(int(max_samples), n)).astype(int)
    g = []
    for i in idx:
        l = _luma(seq[i])
        g.append(np.abs(cv2.Sobel(l, cv2.CV_32F, 1, 0, ksize=3)) +
                 np.abs(cv2.Sobel(l, cv2.CV_32F, 0, 1, ksize=3)))
    gm = np.median(np.stack(g), axis=0)                     # (H,W) float32
    if feather > 1:
        gm = cv2.GaussianBlur(gm, (int(feather), int(feather)), 0)
    return gm < np.percentile(gm, float(pct))


def _gate(filtered: np.ndarray, original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """`filtered` where the mask is True, `original` everywhere else.  mask: (H,W,1) bool.

    Split out from the node so it can be tested: with an all-False mask the output must be
    the input EXACTLY, not approximately -- that is the promise `flat_only` makes.
    """
    return np.where(mask, filtered, original)


# ----------------------------------------------------------------------------- metrics
def _luma_sd(seq: np.ndarray) -> float:
    """Mean per-pixel temporal sd -- 'the boil'.  Computed on a stride so the node stays
    fast; it is a diagnostic, not a deliverable."""
    g = np.stack([_luma(seq[i])[::3, ::3] for i in range(len(seq))])
    return float(g.std(axis=0).mean())


def _detail(seq: np.ndarray) -> float:
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)
    n = min(len(seq), 24)
    v = [cv2.filter2D(_luma(seq[i][::2, ::2]).astype(np.float32), -1, k).var()
         for i in range(n)]
    return float(np.mean(v))


def _structure(seq: np.ndarray) -> float:
    """Laplacian variance of the clip's TEMPORAL MEAN -- the picture with the boil averaged out.

    `_detail` above is measured PER FRAME, so the boil is part of what it counts: removing the
    sizzle lowers it, and a filter that keeps the noise looks sharper than one that removes it.
    That makes it a poor answer to "did this cost me quality?", which is the question the gate
    exists for.  Averaging over the clip first cancels the boil (it is not correlated between
    frames) and leaves the structure, so this number only moves when something REAL was eaten.
    """
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)
    n = min(len(seq), 24)
    m = np.mean([_luma(seq[i][::2, ::2]).astype(np.float32) for i in range(n)], axis=0)
    return float(cv2.filter2D(m, -1, k).var())


def _disparity(seq: np.ndarray, max_pairs: int = 6) -> tuple:
    """Coarse median background disparity of the SBS pairs, via 40 px tile matching.

    This is the check that matters: the node must not move the stereo.  Measured on the
    real clip, before -> after was sd 0.22 -> 0.00 with the range unchanged.
    """
    ts, step, search = 40, 40, 36
    dxs, dys, spread = [], [], []
    idx = np.linspace(0, len(seq) - 1, min(max_pairs, len(seq))).astype(int)
    for i in idx:
        f = seq[i]
        w = f.shape[1] // 2
        L, R = _luma(f[:, :w]), _luma(f[:, w:])
        h = L.shape[0]
        dx, dy = [], []
        for y in range(search, h - ts - search, step):
            for x in range(search, w - ts - search, step):
                t = L[y:y + ts, x:x + ts].astype(np.float32)
                if t.std() < 6:
                    continue
                sub = R[y - search:y + ts + search,
                        x - search:x + ts + search].astype(np.float32)
                _, mx, _, ml = cv2.minMaxLoc(
                    cv2.matchTemplate(sub, t, cv2.TM_CCOEFF_NORMED))
                if mx < 0.75:
                    continue
                dx.append(ml[0] - search)
                dy.append(ml[1] - search)
        if len(dx) >= 8:
            dxs.append(np.median(dx))
            dys.append(np.median(dy))
            spread.append(np.percentile(dx, 95) - np.percentile(dx, 5))
    if not dxs:
        return dict(ok=False, n=0, mean=0.0, sd=0.0, range=0.0, vert=0.0)
    dxs = np.array(dxs, float)
    return dict(ok=True, n=len(dxs), mean=float(dxs.mean()), sd=float(dxs.std()),
                range=float(np.mean(spread)), vert=float(np.mean(dys)))


# ------------------------------------------------------------------------------- node
class StereoStabilizeSBS:
    """Temporal median stabiliser for a side-by-side pair batch.

    Put it after StereoPairFrames and before the video/image output.  radius=0 returns
    the input unchanged, which is both an implementation check and a way to A/B.
    `flat_only` gates the median to the flattest N% of the frame, which keeps the boil
    reduction in the areas where it is visible and the detail in the areas where it lives.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "sbs": ("IMAGE", {"tooltip": "The side-by-side pairs, in order. "
                                         "Left half = one eye, right half = the other."}),
            "radius": ("INT", {"default": 4, "min": 0, "max": 8, "step": 1,
                "tooltip": "Half-window of the temporal median in PAIRS (N = 2r+1). "
                           "0 = off (pass-through). Measured boil / detail cost: "
                           "1 -> -23% / -7%, 2 -> -35% / -10%, 3 -> -42% / -12%, "
                           "4 -> -47% / -13%, 5 -> -51% / -14%. Detail loss saturates, "
                           "so 4 or 5 is the useful end. Radius 4 spans +-0.5 s at "
                           "8 fps: fine for slow scenes, use 1-2 if things really move "
                           "or objects will stick and jump."}),
            "level": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "How much of the per-frame luma/exposure correction to "
                           "apply. Removed the pulsing (level drift sd 0.93 -> 0.11) "
                           "at no measurable cost. Per eye, clamped to +-3%."}),
        },
            "optional": {
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1,
                    "tooltip": "0 = all pairs. Truncate for a quick look."}),
                "check_stereo": ("BOOLEAN", {"default": True,
                    "tooltip": "Re-measure the disparity before and after and report "
                               "it. It must not move; this is the check that the "
                               "per-eye filtering really is order-preserving."}),
                "flat_only": ("INT", {"default": 0, "min": 0, "max": 100, "step": 5,
                    "tooltip": "0 = filter EVERY pixel (the original behaviour, so an old graph "
                               "is unchanged). Above 0 = filter only the flattest N% of pixels "
                               "and leave the rest EXACTLY as generated. The median only costs "
                               "detail at edges -- it can pick a tap from a neighbouring "
                               "instant and smear them -- and this is the gate that stops "
                               "that. Measured through the node at radius 4 on a real "
                               "62-pair clip (1248x1664/eye): every pixel -> per-frame "
                               "detail -19%; flat_only 60 -> +2%, and inside the gate the "
                               "boil still drops 4.80 -> 3.46, bit-identical to the "
                               "ungated median there, while the other 40% of the frame is "
                               "left untouched bit-for-bit. LOWER it to clean more of the "
                               "frame, RAISE it to protect more edges."}),
                "flat_feather": ("INT", {"default": 9, "min": 1, "max": 31, "step": 2,
                    "tooltip": "Blur applied to the clip's gradient map before it is "
                               "thresholded, in pixels (odd). 1 = a hard boundary between "
                               "filtered and untouched pixels; larger = softer, which is "
                               "safer because the boundary itself then cannot read as an "
                               "edge. Only used when flat_only > 0."}),
            }}

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("sbs", "report", "pairs")
    FUNCTION = "run"
    CATEGORY = _CAT

    def run(self, sbs, radius, level, max_frames=0, check_stereo=True,
            flat_only=0, flat_feather=9):
        x = sbs
        n = int(x.shape[0])
        if n < 2:
            raise ValueError(f"StereoStabilizeSBS needs at least 2 pairs, got {n}.")
        if max_frames:
            x = x[:max_frames]
            n = int(x.shape[0])

        H, W = int(x.shape[1]), int(x.shape[2])
        if W % 2:
            raise ValueError(f"SBS width must be even to split into two eyes, got {W}. "
                             f"StereoPairFrames pads to even -- enable its pad_to_even.")
        hw = W // 2
        # stay in float32 throughout: the median SELECTS source pixels, so quantising
        # to uint8 first would throw away precision for nothing
        arr = np.clip(x.detach().cpu().numpy().astype(np.float32) * 255.0, 0, 255)

        L = np.ascontiguousarray(arr[:, :, :hw])
        R = np.ascontiguousarray(arr[:, :, hw:])

        before = dict(boil=_luma_sd(L), detail=_detail(L), struct=_structure(L))
        before["disp"] = _disparity(arr) if check_stereo else dict(ok=False, n=0)

        if radius <= 0 and level <= 0:
            return (x, self._report(radius, level, before, before, n, hw, H,
                                    check_stereo, skipped=True), n)

        # each eye filtered INDEPENDENTLY -- this is what keeps the disparity intact
        Ln, Rn = _level_normalise(L, level), _level_normalise(R, level)
        Ls, Rs = _temporal_median(Ln, radius), _temporal_median(Rn, radius)
        gate = None
        if flat_only > 0:
            # only the MEDIAN is gated.  The exposure fix is one gain + offset per frame, so
            # it costs no detail anywhere and stays global.
            mask = _flat_mask(L, flat_only, flat_feather)[..., None]
            Ls, Rs = _gate(Ls, Ln, mask), _gate(Rs, Rn, mask)
            gate = (flat_only, flat_feather, 100.0 * float(mask.mean()))

        out = np.concatenate([Ls, Rs], axis=2)
        after = dict(boil=_luma_sd(Ls), detail=_detail(Ls), struct=_structure(Ls))
        after["disp"] = _disparity(out) if check_stereo else dict(ok=False, n=0)

        rep = self._report(radius, level, before, after, n, hw, H, check_stereo, gate=gate)
        print("[StereoStabilize] " + rep.replace("\n", "\n[StereoStabilize] "))

        t = torch.from_numpy(out.astype(np.float32) / 255.0)
        if x.is_cuda:
            t = t.to(x.device)
        return (t, rep, n)

    @staticmethod
    def _report(radius, level, before, after, n, hw, H, check_stereo, skipped=False, gate=None):
        N = 2 * radius + 1
        lines = []
        if skipped:
            lines.append(f"radius=0 and level=0 -> PASS-THROUGH, input returned "
                         f"unchanged ({n} pairs @ {hw}x{H}, N={N}).")
        else:
            lines.append(f"{n} pairs @ {hw}x{H}/eye | temporal median N={N} "
                         f"(radius {radius}), level {level:g}")
        bd = before["disp"]
        ad = after["disp"]
        if gate is not None:
            pct, feather, cov = gate
            lines.append(f"  flat_only {pct}: filtered the flattest {cov:.0f}% of pixels "
                         f"(feather {feather} px) -- the other {100 - cov:.0f}% are exactly as "
                         f"generated. Lower it to clean more of the frame, raise it to "
                         f"protect more edges.")
        if not skipped:
            lines.append(f"  boil (mean per-pixel temporal sd) "
                         f"{before['boil']:5.2f} -> {after['boil']:5.2f}  "
                         f"({100 * (after['boil'] / max(before['boil'], 1e-6) - 1):+3.0f}%)")
            lines.append(f"  detail per frame {before['detail']:7.1f} -> {after['detail']:7.1f}  "
                         f"({100 * (after['detail'] / max(before['detail'], 1e-6) - 1):+3.0f}%)  "
                         f"<- this one counts the boil it removed as detail lost")
            lines.append(f"  STRUCTURE (detail of the clip's temporal mean, boil averaged out) "
                         f"{before['struct']:7.1f} -> {after['struct']:7.1f}  "
                         f"({100 * (after['struct'] / max(before['struct'], 1e-6) - 1):+3.0f}%)  "
                         f"<- the honest quality number: only real structure moves it")
        if check_stereo and bd.get("ok") and ad.get("ok"):
            d = abs(ad["mean"] - bd["mean"])
            lines.append(f"  stereo geometry (must not move): disparity "
                         f"{bd['mean']:+.2f} (sd {bd['sd']:.2f}, range {bd['range']:.2f})"
                         f" -> {ad['mean']:+.2f} (sd {ad['sd']:.2f}, "
                         f"range {ad['range']:.2f}), vertical {ad['vert']:+.2f}")
            if d > 0.5:
                lines.append(f"  !! WARNING: disparity mean moved {d:.2f} px "
                             f"(> 0.5). The per-eye filtering is supposed to be "
                             f"order-preserving -- report this.")
        lines.append("  REMINDER: the pair rate is half the source frame rate. "
                     "Set VHS_VideoCombine frame_rate to HALF the source "
                     "(wan22 source 16 fps -> 8). At 16 fps the clip plays 2x fast "
                     "and the flicker frequency doubles, which halves this fix.")
        return "\n".join(lines)


NODE_CLASS_MAPPINGS = {"StereoStabilizeSBS": StereoStabilizeSBS}
NODE_DISPLAY_NAME_MAPPINGS = {
    "StereoStabilizeSBS": "Stereo Stabilize (temporal median, per eye)"}

_self_test()
