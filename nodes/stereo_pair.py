"""Turn a generated video into a stereo (side-by-side) video, with a head trim and a QC readout.

The trick this implements: a generated clip whose consecutive frames are *similar but
not identical* can be read as the two eyes of a stereo pair -- frame N as one eye,
frame N+1 as the other. Playing those pairs side by side gives a stereoscopic clip
with no depth estimation involved.

`StereoPairFrames` groups a batch two frames at a time and stitches each pair
horizontally. It is deliberately length-safe:

  * 81 frames -> 40 pairs, and frame 81 is left alone (dropped, never paired with a
    repeat of itself). `ImageStitch` cannot be used for this: when one input batch is
    shorter it pads by REPEATING that batch's last frame, which would emit a bogus
    (frame81 | frame80) pair pointing backwards.
  * an even count gives an exact 1:1 pairing with nothing dropped.

`second_eye` is returned already aligned 1:1 with `sbs`, so it can be fed straight into
ComfyUI-DDDDeflicker's `partner` input.

EYE ORDER, as measured on real output: pairing `frame1 | frame2` reads as a CROSS-EYED
view, and `frame2 | frame1` is the one a headset needs. The default is therefore
cross-eyed -- it is what the labels say, so they cannot be misread.

HEAD TRIM (2026-09-27). The opening frames of a generated clip are the weakest: the
alternation has to establish itself before it reaches full strength, and a dead opening
pair ships to the viewer as a mono instant. Measured on the project's renders (frame pairs
0, 1, 2, 3, 4 ... = frames 0-1, 2-3, 4-5, ...):

    render        ramp (px per pair)                 clip median
    356 (H3)      -14.8 -17.4 -14.3 -15.5 ...        -15.6        strong from frame 0
    347 (H3)       -9.8  -7.9  -9.1  -7.5 ...         -7.6
    355 (H3)       -6.3  ...                         -14.7
    359 (H3)       -0.3  -0.0  -1.7  -3.2  -5.2 ...  -15.8        DEAD for 8 frames
    WAN 2.2       -11.5 -12.6 -12.2 -12.2 ...        -13.1        strong from frame 0

`skip_first` drops frames before pairing (2 = one instant, 4 = one whole latent token for a
4x-temporal VAE such as H3's or Wan's -- `patch_size_t = 4`), and `auto_extend` keeps
dropping instants while the measured pair is still weak, so a clip like 359 loses its dead
opening instead of shipping it. A clip that is strong from frame 0 (every WAN render
measured, most H3 ones) is left completely alone -- with `skip_first` at its default of 0
the node is a no-op on those.

The QC readout exists so a shallow clip can be told from a broken one without leaving
ComfyUI. `eye_step_px` is the eye separation in pixels, `phase_ok` is the fraction of pairs
that agree on the eye order:

    eye_step_px    10-20 px at 1024 wide   film class (real 3D films: 10.8-20.1, median 14.8)
                   ~3 px                   shallow, but visible stereo
                   < 1 px                  no measurable disparity in the sampled pairs
    phase_ok       1.00                    consistent eye order; < 1.00 = the eyes swap mid-clip

The measurement is the same algorithm the project's own tool uses (`tools/h3_verify_pair.py
:: global_shift`): a cv2 Sobel gradient mask at the 60th percentile, masked mean |a - b|,
coarse-to-fine over +-rng with sub-pixel parabolic refinement. Numbers therefore match the
project's ledger exactly. On this pipeline the correct sign is NEGATIVE (frame 0 = RIGHT eye).
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

_CAT = "Stereo"
_VIEWING = ["cross-eyed  (left = frame 1, right = frame 2)",
            "parallel / headset  (left = frame 2, right = frame 1)"]
_UNPAIRED = ["drop (leave it alone)", "black partner", "duplicate last frame"]


def _left_is_first(viewing: str) -> bool:
    """True if the pair's first frame should end up on the left.

    Measured: `frame1 | frame2` is the CROSS-EYED arrangement, `frame2 | frame1` is what a
    headset needs. Tolerant of wording so older saved graphs and literal phrasing still
    resolve instead of silently flipping the depth.
    """
    v = (viewing or "").lower()
    if "parallel" in v or "headset" in v:
        return False
    if "cross" in v or "crosseye" in v or "cross-eye" in v:
        return True
    return "second" not in v          # legacy fallback: 'first frame of pair' etc.


# ------------------------------------------------------------------------------------------------
# QC measurement -- the project's own algorithm, vendored so that numbers read inside the graph are
# identical to the ones in tools/h3_verify_pair.py and out/si74f_ladder.json.
# ------------------------------------------------------------------------------------------------
def _global_shift(a: np.ndarray, b: np.ndarray, rng: int = 48, mask_pct: int = 60):
    """Best single horizontal shift of `a` onto `b`, sub-pixel, plus (d, raw, explained).

    d positive = the SECOND frame's content sits at +d relative to the first, so a correctly
    ordered same-instant pair reads NEGATIVE on this pipeline (frame 0 = RIGHT eye).
    """
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(ga, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(ga, cv2.CV_32F, 0, 1, ksize=3)
    t = np.sqrt(gx ** 2 + gy ** 2)
    m = t > np.percentile(t, mask_pct)
    rawh = float(np.abs(ga - gb)[m].mean())

    def resid(d):
        wb = cv2.warpAffine(ga, np.float32([[1, 0, d], [0, 1, 0]]), (ga.shape[1], ga.shape[0]),
                            borderMode=cv2.BORDER_REPLICATE)
        return float(np.abs(wb - gb)[m].mean())

    coarse = range(-rng, rng + 1, 4)
    best = min(coarse, key=resid) if rng >= 4 else 0
    res = {d: resid(d) for d in range(best - 4, best + 5)}
    bd = min(res, key=res.get)
    d = float(bd)
    if best - 4 < bd < best + 4:
        y0, y1, y2 = res[bd - 1], res[bd], res[bd + 1]
        den = y0 - 2 * y1 + y2
        if den != 0:
            d += 0.5 * (y0 - y2) / den
    explained = (rawh - res[bd]) / rawh if rawh > 1e-6 else 0.0
    return d, rawh, float(explained)


def _np_frames(images: torch.Tensor, idx) -> np.ndarray:
    """Frames at the pair indices in `idx`, as HxWx3 uint8 RGB (what cv2 wants), in clip order."""
    order = [2 * k + j for k in idx for j in (0, 1)]
    sel = images[order]
    return (sel.clamp(0, 1) * 255.0).round().to(torch.uint8).numpy()


def _steps_for(images: torch.Tensor, idx, rng: int = 48) -> list:
    """Full-res eye step for the pair indices in `idx` (identical numbers to the project's tool)."""
    arr = _np_frames(images, idx)
    return [float(_global_shift(arr[2 * i], arr[2 * i + 1], rng=rng)[0]) for i in range(len(idx))]


def measure_eye_step(images: torch.Tensor, n_prefix: int = 12, n_spread: int = 12) -> dict:
    """Clip QC: the first `n_prefix` pairs (the ramp) plus `n_spread` pairs across the clip."""
    npairs = int(images.shape[0]) // 2
    if npairs < 1:
        return dict(steps=[], median=0.0, phase_ok=0.0, prefix=[], sampled=0)
    prefix = _steps_for(images, list(range(min(n_prefix, npairs))))
    spread_idx = sorted({int(round(i * (npairs - 1) / max(1, n_spread - 1)))
                         for i in range(min(n_spread, npairs))})
    spread = _steps_for(images, spread_idx)
    med = float(sorted(spread)[len(spread) // 2])
    hard = [s for s in spread if abs(s) >= 1.0]                # |s| < 1 px carries no order info
    if hard:
        dom = 1.0 if sum(1 for s in hard if s > 0) > len(hard) / 2 else -1.0
        phase_ok = sum(1 for s in hard if s * dom > 0) / len(hard)
    else:
        phase_ok = 0.0
    return dict(steps=spread, median=med, phase_ok=phase_ok, prefix=prefix, sampled=len(spread),
                band_idx=spread_idx[::3])


def _first_strong(prefix, floor_px: float, median: float) -> int:
    """Index of the first pair in `prefix` that is strong enough to start the clip on.

    The relative part is deliberately mild (0.25 of the clip's own median): the target is a DEAD
    opening pair, not a merely gentler one. A 6.3 px opening on a 14.7 px clip (render 355) is real
    stereo and must NOT be trimmed; a 0.3 px opening (render 359) must be.
    """
    need = max(abs(floor_px), 0.25 * abs(median))
    for k, s in enumerate(prefix):
        if abs(s) >= need:
            return k
    return 0


def measure_bands(images: torch.Tensor, idx, rng: int = 48):
    """Rigid eye step per third of the frame -- [top, middle, bottom] px, plus explained per band.

    The whole-frame number is a single rigid fit; when the disparity field is not uniform (it is not,
    on these LoRAs) it hides a field that is strong in one band and dead in another. Measured on a
    square-canvas render whose whole-frame step read -3.2 px: top -12.7 / mid -2.3 / bottom -3.2 px
    with `explained` 0.44 / 0.11 / 0.35 -- film-class stereo in the top third, nothing measurable in
    the middle. That is what "I can see stereo but your number says almost none" looks like.

    A band whose fit explains <= 0.15 of its horizontal structure, or whose step is under ~1 px, has
    NO measurable disparity -- read it as empty, not as small.
    """
    h = int(images.shape[1])
    y1, y2 = h // 3, 2 * h // 3
    idx = list(idx)
    arr = _np_frames(images, idx)
    got = [[_global_shift(arr[2 * i][s:e], arr[2 * i + 1][s:e], rng=rng)
            for s, e in ((0, y1), (y1, y2), (y2, h))] for i in range(len(idx))]
    d = np.mean([[g[j][0] for j in range(3)] for g in got], axis=0)
    e = np.mean([[g[j][2] for j in range(3)] for g in got], axis=0)
    return [float(x) for x in d], [float(x) for x in e]


def band_verdict(bands, explained) -> str:
    """Is the depth field coherent across the frame, or is one band carrying everything?

    Three outcomes, and the third is the one that ruins a clip: bands can be EMPTY (no
    measurable disparity), the field can be carried by ONE band, or the bands can disagree in
    SIGN. Measured on a real render (SBS 140, instants 0-9): top -6 px, mid +4 px, bottom
    +14 px -- the eye images warped against each other, which the viewer sees as a twist or
    doubling that no re-pairing can remove. Worth flagging loudly, because it looks like an
    eye-order problem and is not one.
    """
    names = ("top", "mid", "bottom")
    empty = [name for name, d, e in zip(("top", "mid", "bottom"), bands, explained)
             if abs(d) < 1.0 or e <= 0.15]
    if len(empty) == 3:
        return "no measurable disparity in ANY band"
    if empty:
        return "incoherent: %s empty" % ",".join(empty)
    strong = [(names[i], bands[i]) for i in range(3)
              if abs(bands[i]) >= 3.0 and explained[i] >= 0.2]
    signs = {1 if d > 0 else -1 for _, d in strong}
    if len(signs) > 1:
        return ("SIGNS DISAGREE: the bands are shifted in OPPOSITE directions (%s) -- the two "
                "eye images are WARPED relative to each other, not merely offset, so nothing "
                "fuses cleanly. No pairing / swap / viewing setting can fix that. Re-render."
                % ", ".join("%s %+.1f px" % (n, d) for n, d in strong))
    return "coherent across top/mid/bottom"


def phase_lock_mask(images: torch.Tensor, npairs: int, min_px: float = 3.0) -> list:
    """Which pairs to flip so the clip is internally CONSISTENT, plus the sign it settled on.

    Two rules, both learned the hard way on real renders:

    1. **Follow the clip's own majority, not the trained sign.** An earlier version forced every
       pair to the trained order (frame 0 = RIGHT eye, negative step) on the theory that the
       conditioned opening frames are the reliable ones. Measured on render 378: the opening was
       negative for 10 pairs, the body positive for 21 -- the MAJORITY was positive, so that rule
       flipped 21 of 31 strong pairs and inverted a clip the viewer had been reading correctly.
       A repair must not change which way round the clip reads; it must only remove the minority
       pairs that disagree with it. Global arrangement is the `viewing` setting's job.

    2. **Only touch pairs whose order is measurable.** A pair under `min_px` has a sign that is
       noise; swapping it can turn a locally-visible disparity around and is how a "flat" stretch
       starts reading as an inverted one. Those pairs are left exactly as generated.

    Returns (mask, dominant_sign) where mask[k] is True when pair k should have its halves swapped.
    """
    steps = _steps_for(images, list(range(npairs)))
    strong = [s for s in steps if abs(s) >= min_px]
    if len(strong) < 2:
        return [False] * npairs, 0
    dominant = 1 if sum(strong) > 0 else -1
    return [(abs(s) >= min_px and (1 if s > 0 else -1) != dominant) for s in steps], dominant


def _mixed_order(prefix, steps, min_px: float = 3.0, ratio_min: float = 0.6,
                 share_min: float = 0.15) -> bool:
    """Does this clip really contain BOTH eye orders, or just some weak failed pairs?

    Counting pairs whose sign differs is not enough: a pair that fails to produce the eye separation
    reads 2-4 px in whatever direction its noise happens to point, and calling that an inversion
    rejects clips that look fine. Measured on three renders of one prompt:

        386 (fine by eye)      pos 25 median 4.3 px  vs neg  8 median 11.2 px  -> ratio 0.38
        378 (visible reversal) pos 18 median 5.0 px  vs neg 10 median  6.3 px  -> ratio 0.78
        370 (visible reversal) pos 51 median 6.2 px  vs neg  3 median  7.7 px  -> ratio 0.81

    So: only call it mixed when BOTH directions carry comparable separation (the weaker median at
    least `ratio_min` of the stronger) AND the weaker group is not a handful of pairs.
    """
    d = [x for x in list(prefix) + list(steps)]
    pos = [abs(x) for x in d if x >= min_px]
    neg = [abs(x) for x in d if x <= -min_px]
    if not pos or not neg:
        return False
    mp, mn = float(np.median(pos)), float(np.median(neg))
    weak, strong = (mp, mn) if mp <= mn else (mn, mp)
    n_weak = len(pos) if mp <= mn else len(neg)
    return bool(strong > 0 and weak / strong >= ratio_min
                and n_weak >= max(2, int(share_min * len(d))))


class StereoPairFrames:
    """Pair consecutive frames of a clip as the two eyes of a stereo image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "The generated clip's frames, in order (the RAW "
                                            "alternating render, not the side-by-side file)."}),
            "viewing": (_VIEWING, {
                "tooltip": "Which eye order matches how you watch it. Measured on real "
                           "output: frame1|frame2 reads as CROSS-EYED, so that is the "
                           "default. A headset needs the halves swapped -- pick "
                           "parallel / headset for that."}),
            "unpaired_last": (_UNPAIRED, {
                "tooltip": "What to do when the clip has an odd frame count, so the "
                           "final frame has no partner. 'drop' simply leaves it out."}),
            "pad_to_even": ("BOOLEAN", {"default": True,
                "tooltip": "Pad to an even width/height if needed. h264 with yuv420p "
                           "cannot encode an odd dimension."}),
            "skip_first": ("INT", {"default": 0, "min": 0, "max": 64, "step": 2,
                "tooltip": "Frames dropped before pairing. 2 = one instant; 4 = one whole latent "
                           "token on a 4x-temporal VAE (H3, Wan) -- use 4 if a fixed 2 is not "
                           "enough. Must be EVEN so frame 0 of each pair stays the RIGHT eye. With "
                           "auto_extend on this is the MINIMUM trim; 0 lets auto decide alone."}),
            "auto_extend": ("BOOLEAN", {"default": True,
                "tooltip": "Keep dropping instants while the measured pair is still weak (below "
                           "expected_min_px, or below a quarter of the clip's own median), so a "
                           "dead opening does not ship as a mono instant. A clip that is strong "
                           "from frame 0 is left alone."}),
            "expected_min_px": ("INT", {"default": 3, "min": 0, "max": 200, "step": 1,
                "tooltip": "A pair below this many pixels counts as weak for auto_extend, and the "
                           "report flags it. 3 px at 1024 wide is already shallow; film class "
                           "starts around 10 px."}),
            "measure": ("BOOLEAN", {"default": True,
                "tooltip": "Measure the eye step / phase / bands for the QC outputs (about 24 pairs "
                           "plus 12 band fits; roughly 3-6 s). Turn off to skip the measurement."}),
            "reject_bad_seed": ("BOOLEAN", {"default": False,
                "tooltip": "Off: a bad render is reported and still paired (watch the `verdict` / "
                           "`report`). On: a render that is MIXED, WARPED or has no stereo at all "
                           "makes this node ERROR OUT -- the node turns red and the graph stops, so "
                           "you see the bad seed immediately instead of at the end of a watch. The "
                           "raw render is already saved by then; just change the seed and re-run."}),
            "phase_lock": ("BOOLEAN", {"default": False,
                "tooltip": "OFF by default. Turns on an automatic eye-order repair: measure every "
                           "pair and swap the halves of those whose order disagrees with the clip's "
                           "majority. Tested on real renders and REJECTED as a default -- on a clip "
                           "whose disparity is patchy (some pairs 6 px, others flat) every variant of "
                           "the repair, even one restricted to the solid pairs, introduced visible "
                           "disturbances for the viewer, while the untouched pairing read fine. "
                           "The node now reports the inconsistency instead of acting on it: read "
                           "`phase_ok` / the ramp line, and re-render on another seed."}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT", "FLOAT", "FLOAT", "STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("sbs", "second_eye", "pairs", "dropped", "eye_step_px", "phase_ok", "report",
                    "verdict", "ok")
    FUNCTION = "pair"
    CATEGORY = _CAT

    def pair(self, images, viewing, unpaired_last, pad_to_even,
             skip_first=0, auto_extend=True, expected_min_px=3, measure=True, phase_lock=False,
             reject_bad_seed=False):
        skip_first = int(skip_first)
        if skip_first % 2:
            raise ValueError("skip_first must be even (frame 0 of each pair = RIGHT eye); got %d"
                             % skip_first)
        n0 = int(images.shape[0])
        if n0 < 2:
            raise ValueError("StereoPairFrames needs at least 2 frames, got %d." % n0)

        raw_skip, qc, bd, be = skip_first, None, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        if measure:
            qc = measure_eye_step(images)
            if auto_extend and qc["prefix"]:
                k = _first_strong(qc["prefix"], expected_min_px, qc["median"])
                raw_skip = max(skip_first, 2 * k)
        limit = n0 - 2 if (n0 - 2) % 2 == 0 else n0 - 3
        skip = max(0, min(raw_skip, limit))

        trimmed = images[skip:]
        n = int(trimmed.shape[0])
        first, second = trimmed[0::2], trimmed[1::2]
        pairs = min(first.shape[0], second.shape[0])
        dropped = n - 2 * pairs

        if dropped:
            # The unpaired frame is always the trailing one; here `first` holds it.
            longer, shorter = (first, second) if first.shape[0] > second.shape[0] \
                else (second, first)
            want = longer.shape[0] - shorter.shape[0]
            if unpaired_last == "black partner":
                extra = torch.zeros((want,) + tuple(shorter.shape[1:]),
                                    dtype=shorter.dtype)
                shorter = torch.cat([shorter, extra], dim=0)
            elif unpaired_last == "duplicate last frame":
                shorter = torch.cat([shorter, shorter[-1:].repeat(want, 1, 1, 1)], dim=0)
            else:                                   # "drop (leave it alone)"
                longer = longer[:shorter.shape[0]]
            if first.shape[0] > second.shape[0]:
                first, second = longer, shorter
            else:
                first, second = shorter, longer
            pairs = first.shape[0]

        if first.shape[1:3] != second.shape[1:3]:
            raise ValueError(
                f"the two streams have different frame sizes: {tuple(first.shape[1:3])} "
                f"vs {tuple(second.shape[1:3])}. Load the clip at a fixed size.")

        L, R = (first, second) if _left_is_first(viewing) else (second, first)

        swapped, lock_note = 0, ""
        if qc is not None and phase_lock and qc["phase_ok"] < 1.0:
            # Gate: lock only when there is real disparity to lock ONTO, judged from the sampled
            # pairs, not from the whole-frame median. Two sampled pairs at >= 5 px means a real eye
            # step exists somewhere; a clip whose strongest sampled pair is 3-4 px (measured: render
            # 375, whole-frame +0.1 px) is noise, and swapping there is a coin flip per pair.
            strong = sorted((abs(s) for s in qc["steps"]), reverse=True)
            lockable = len(strong) >= 2 and strong[1] >= 5.0
            if not lockable:
                lock_note = ("phase_lock SKIPPED: no sampled pair reaches 5 px (strongest %.1f px, "
                             "whole-frame %+.1f px) -- there is no eye step to tell the order from, "
                             "so swapping would be a coin flip per pair. Re-render this one rather "
                             "than re-pairing it." % (strong[0] if strong else 0.0, qc["median"]))
            else:
                # measured on the TRIMMED frames: pair k of the output is pair k of images[skip:],
                # not of images -- using the untrimmed batch leaves an off-by-skip pair unswapped
                mask, dominant = phase_lock_mask(images[skip:], pairs)
                if any(mask):
                    m = torch.tensor(mask, dtype=torch.bool).view(-1, 1, 1, 1)
                    swapped = int(sum(mask))
                    lock_sign = ("cross-eyed order (frame 0 = RIGHT eye)" if dominant < 0
                                 else "PARALLEL order (frame 0 = LEFT eye -- mirrored relative to "
                                      "the training convention; set `viewing` to whatever reads "
                                      "correctly)")
                    L, R = torch.where(m, R, L), torch.where(m, L, R)
        second_out = R                                  # the right half = the partner of the left
        sbs = torch.cat([L, R], dim=2)                   # BHWC -> concat on width

        if pad_to_even:
            h, w = sbs.shape[1], sbs.shape[2]
            ph, pw = h % 2, w % 2
            if ph or pw:
                sbs = torch.nn.functional.pad(sbs, (0, 0, 0, pw, 0, ph), mode="replicate")

        step, phase, report = 0.0, 0.0, ("measurement off; trimmed %d frames -> %d pairs"
                                         % (skip, pairs))
        verdict, ok = "UNKNOWN (measurement off)", True
        if qc is not None:
            step, phase = qc["median"], qc["phase_ok"]
            W = int(images.shape[2])
            bd, be = measure_bands(images, qc["band_idx"])
            if swapped:
                # the clip is now the mirror of what was measured for the inverted majority, so the
                # reported field is negated -- printed and returned values must agree
                step, bd = -step, [-x for x in bd]
            # "present in PARTS" is reserved for the case where the whole-frame number UNDERSTATES
            # the clip: a strong band or a strong opening while the median says nothing (measured on
            # render 377: median 1.8 px, bottom band 12.7 px, opening -6.3 px). When the median is
            # already honest about a shallow clip, say shallow -- the band line covers the rest.
            # two strong opening pairs, not one: a single spike in an otherwise dead clip (render
            # 375: -6.2 px at pair 2, everything else under 5) is noise, not partial stereo.
            parts = (abs(step) < 2.0
                     and (max([abs(x) for x in bd] + [0.0]) >= 5.0
                          or sum(1 for x in qc["prefix"][:10] if abs(x) >= 5.0) >= 2))
            state = ("film class (>=10 px at 1024 wide)" if abs(step) >= 10
                     else "present in PARTS: see the band and ramp lines" if parts
                     else "shallow -- visible stereo, ~1/4 of a film's separation" if abs(step) >= 2
                     else "almost no disparity")
            # Which global order did this render come out in, and therefore which `viewing`
            # setting reads correctly? One prompt, three seeds: 377/378 came out with the opening
            # in the trained order (negative, frame 0 = RIGHT eye) and the body mirrored; the next
            # seed came out mirrored for the WHOLE clip (+7.5 px opening, SBS 139). A whole-clip
            # mirror is fixed by `viewing` -- a uniform swap, with nothing per-pair to disturb.
            _strong = [x for x in qc["steps"] if abs(x) >= 3.0]
            _mixed = _mixed_order(qc["prefix"], qc["steps"])
            order = 0 if not _strong else (1 if sum(_strong) > 0 else -1)
            order_note = ("clip order: MIXED -- this render is not in one eye order (the ramp "
                          "line shows which pairs disagree); no `viewing` setting can fix a mix, "
                          "so re-render on another seed") if _mixed else {
                1: ("clip order: PARALLEL (frame 0 = LEFT eye) -- for cross-eyed viewing "
                              "set `viewing` = 'parallel / headset'; the 'cross-eyed' setting "
                              "assumes the opposite order"),
                          -1: ("clip order: cross-eyed (frame 0 = RIGHT eye) -- the default `viewing` "
                               "setting is the right one for this clip"),
                0: "clip order: undetermined (no pair reaches 3 px)"}[order]
            sampled = qc["steps"]
            strength_note = ("strongest sampled pair %.1f px; %d of %d sampled under 1 px"
                             % (max([abs(x) for x in sampled] + [0.0]),
                                sum(1 for x in sampled if abs(x) < 1.0), len(sampled)))
            note = []
            if auto_extend and skip != skip_first:
                note.append("auto_extend: trimmed %d frames (not %d) to start on the first strong "
                            "pair" % (skip, skip_first))
            elif qc["prefix"] and abs(qc["prefix"][0]) < float(expected_min_px):
                note.append("WARNING: the opening pair is still %.1f px (< %d) -- raise skip_first"
                            % (qc["prefix"][0], int(expected_min_px)))
            if phase and phase < 1.0 and not swapped:
                note.append("WARNING: eye order is not consistent (%.0f%% of pairs share the "
                            "dominant sign) -- this render changes eye order part-way through. "
                            "Nothing was re-arranged (phase_lock is off): watch it, and if the "
                            "reversal bothers you, re-render on another seed rather than letting "
                            "the node swap halves." % (phase * 100))
            if swapped:
                note.append("phase_lock: the clip's own majority order is %s; swapped the halves of "
                            "%d of %d pairs so the whole clip agrees with it (weak pairs left as "
                            "generated)" % (lock_sign, swapped, pairs))
            if lock_note:
                note.append(lock_note)
            # ----------------------------------------------------------------------------------
            # THE VERDICT FIRST. Read one line: OK, or REJECT with the reason. Everything else in
            # the report is detail for whoever wants to check the call.
            # ----------------------------------------------------------------------------------
            bandv = band_verdict(bd, be)
            warped = bandv.startswith("SIGNS DISAGREE")
            # NO STEREO before MIXED: a clip with no disparity (render 375: whole-frame 0.1 px) has no
            # order to be mixed in, and calling it MIXED sends the user looking for an order problem
            # that does not exist. Two strong opening pairs, not one -- a single spike is noise.
            flat = (max([abs(x) for x in bd] + [0.0]) < 5.0
                    and sum(1 for x in qc["prefix"][:10] if abs(x) >= 5.0) < 2
                    and abs(step) < 2.0)
            n_flat = sum(1 for x in qc["steps"] if abs(x) < 1.0)
            if warped:
                verdict = "REJECT - the two eyes are WARPED against each other (bands disagree in sign)"
            elif flat:
                verdict = "REJECT - NO STEREO (no band and fewer than two opening pairs reach 5 px)"
            elif _mixed:
                verdict = "REJECT - MIXED EYE ORDER (the render changes order part-way through)"
            elif n_flat > len(qc["steps"]) // 3 or bandv.startswith("incoherent") or phase < 0.999:
                verdict = ("OK (PATCHY) - one consistent eye order where the disparity is strong; the "
                           "rest is weak or flat (%s). Weak pairs reading the other way are failed "
                           "flips, not an inversion"
                           % (bandv if bandv.startswith("incoherent")
                              else "%d of %d sampled pairs carry no depth"
                                   % (n_flat, len(qc["steps"]))))
            else:
                verdict = "OK - usable stereo"
            ok = verdict.startswith("OK")
            report = ("*** %s ***\n"
                      "trimmed %d frames -> %d pairs  |  eye step %+.1f px (median of %d pairs; %s)\n"
                      "at %d px wide = %.2f%% of width -- %s\n"
                      "bands t/m/b %s px (%.2f / %.2f / %.2f %% of width)  expl %s  -> %s\n"
                      "%s\n"
                      "first %d pairs (the ramp): %s\n"
                      "sampled steps: %s\n"
                      "%s" % (verdict, skip, pairs, step, qc["sampled"], strength_note, W,
                              abs(step) / max(1, W) * 100.0, state,
                              " / ".join("%+.1f" % x for x in bd),
                              abs(bd[0]) / max(1, W) * 100.0, abs(bd[1]) / max(1, W) * 100.0,
                              abs(bd[2]) / max(1, W) * 100.0,
                              " / ".join("%.2f" % x for x in be), bandv,
                              order_note, len(qc["prefix"]), " ".join("%+.1f" % s for s in qc["prefix"]),
                              " ".join("%+.1f" % s for s in qc["steps"]),
                              ("\n".join(note) if note else "")))
            if reject_bad_seed and not ok:
                raise ValueError(
                    "*** CHANGE THE SEED AND TRY AGAIN ***\n"
                    "this render was REJECTED: %s\n"
                    "evidence: opening pairs %s   |   bands t/m/b %s px\n"
                    "The render itself is saved (upstream SaveVideo); only the side-by-side was "
                    "skipped. Change the seed and re-run -- and note that re-running the SAME seed "
                    "will fail the same way (ComfyUI serves the sampler from cache).\n"
                    "To pair it anyway and judge for yourself, set `reject_bad_seed = false`.\n"
                    % (verdict,
                       " ".join("%+.1f" % s for s in qc["prefix"][:8]),
                       " / ".join("%+.1f" % x for x in bd)))
        return (sbs, second_out, pairs, dropped, float(step), float(phase), report, verdict, ok)


NODE_CLASS_MAPPINGS = {
    "StereoPairFrames": StereoPairFrames,     # the original key: every saved graph still loads
}
# Display name only. The class name, the mapping key and the three original widgets are unchanged so
# that a graph saved before 2026-09-27 opens with its settings intact; this title is what the node
# menu says now, because the node does more than pair frames.
NODE_DISPLAY_NAME_MAPPINGS = {
    "StereoPairFrames": "Stereo Pair Frames + head trim / QC (consecutive -> L|R)",
}
