# Node reference — every setting explained

Seven nodes, what each setting does, what to set it to, and the traps. If you only read
one section, read [the presets](#recommended-presets) at the bottom.

| node | one line |
|---|---|
| **Stereo Pair Frames + head trim / QC** | consecutive frames of a clip → one side-by-side stereo pair, minus the weak opening frames |
| **Split Stereo Pair** | SBS image → two separate eye images |
| **Split Stereo Pair → batch of 2** | same, as one 2-image batch |
| **Join Stereo Pair** | two eye images → one SBS image |
| **Swap Stereo Pair** | flip `[L\|R]` ↔ `[R\|L]` |
| **Stereo Stabilize** | removes the per-pixel temporal "boil" (and costs motion — see below) |
| **Stereo Depth Scale** | makes a pair deeper or shallower, without touching motion |

---

## First, the one concept everything depends on: eye order

```
parallel / headset   [L|R]   left eye in the left half   — what a VR headset wants
cross-eyed           [R|L]   right eye in the left half  — what you view by crossing your eyes
```

Two different inputs can be misread here, so be precise:

- **Stereo Pair Frames + head trim / QC's `viewing`** asks *how you watch it*. Its default (`cross-eyed`)
  puts **frame 1 in the left half**. For a frame-interleaved source where frame 1 is the
  **right** eye, the resulting image is therefore `[R|L]` — cross-eyed, consistent with the
  table above.
- **The `arrangement` setting** on Split/Join/Swap and Depth Scale describes the image in
  front of you: is the first half the left eye (`parallel / headset`) or the right eye
  (`cross-eyed`)?

Getting this wrong on **Split/Join** silently swaps the eyes. It does not crash and the
picture looks fine in 2D — it only reads as "inside-out" depth in a headset. If you are
unsure, view one pair cross-eyed: if the scene looks inverted (near things look far), flip
it with **Swap Stereo Pair**.

---

## 1. Stereo Pair Frames + head trim / QC (consecutive → L|R)

Groups an image batch two frames at a time and stitches each pair side by side. Length
safe: with an odd frame count the last frame is left out rather than paired with a repeat
of itself. Also drops the weak opening frames and measures the result (see
[the head trim](#head-trim-and-the-qc-readout) below).

### Inputs

| setting | default | what it does | what to set |
|---|---|---|---|
| `images` | — | the clip's frames, in order (the RAW alternating render, not an SBS file) | your decoded clip |
| `viewing` | `cross-eyed (left = frame 1, right = frame 2)` | which eye order matches how you will watch it. Measured on real output: `frame1\|frame2` reads as **cross-eyed**, which is why it is the default | leave default for cross-eyed/desktop viewing; pick `parallel / headset` for a VR headset |
| `unpaired_last` | `drop (leave it alone)` | what to do with a final odd frame that has no partner | `drop`. The other options (`black partner`, `duplicate last frame`) invent a frame and are for special cases |
| `pad_to_even` | `True` | pads to even width/height if needed | leave `True` — h264/yuv420p cannot encode an odd dimension |
| `skip_first` | `0` | frames dropped before pairing. **2 = one instant, 4 = one whole latent token** on a 4x-temporal VAE (H3, Wan). Must be even, so frame 0 of each pair stays the right eye. With `auto_extend` on this is the *minimum* | `0` and let `auto_extend` decide. Set `2` or `4` when you want a fixed trim regardless of the measurement |
| `auto_extend` | `True` | keeps dropping instants while the measured pair is still weak (below `expected_min_px`, or below a quarter of the clip's own median) | **leave `True`** — it is a no-op on a clip whose opening is already strong |
| `expected_min_px` | `3` | a pair below this counts as weak for `auto_extend`, and the report warns about it | `3`. Film class starts around 10 px, so 3 px is already shallow; raise it only if you want a stricter opening |
| `measure` | `True` | measures the eye step / phase / bands for the QC outputs (~3–6 s per run) | `True` while you are dialling in a model; `False` for batch rendering once you trust it |
| `reject_bad_seed` | `False` | when on, a render that is MIXED / WARPED / has no stereo makes this node **error out** — red node, graph stops, message says *CHANGE THE SEED AND TRY AGAIN* with the opening pairs and bands as evidence | `False` normally; `True` when you are hunting for a good seed and would rather be told immediately than discover it while watching |
| `phase_lock` | `True` | when the clip's eye order is inconsistent (some pairs mirrored), measures every pair and swaps the halves of the inverted ones, so the whole clip is in the trained order | **leave `True`** — it costs an extra pass only on clips that fail the check, and those are the clips you cannot watch |

### Outputs

| output | what it is |
|---|---|
| `sbs` | the side-by-side pair batch — wire this onward |
| `second_eye` | the RIGHT half of each `sbs` pair, 1:1 aligned with it (the partner stream). Identical to "the second frame of each pair" unless `phase_lock` swapped that pair |
| `pairs` | how many pairs were made |
| `dropped` | how many frames were left over (1 if the clip had an odd frame count) |
| `eye_step_px` | the eye separation in pixels, median over 12 sampled pairs. `10–20 px at 1024 px wide` is film class; `~3 px` is shallow but visible; `< 1 px` is no measurable disparity |
| `phase_ok` | fraction of sampled pairs agreeing on the eye order. `1.00` clean; lower means the eyes swap mid-clip and the depth inverts there |
| `report` | all of it as text, **starting with the one-line verdict** (`*** OK … ***` or `*** REJECT … ***`), then the numbers: eye step, band profile with per-band fit quality, the opening ramp, and any warning |
| `verdict` | the same one-line verdict as a string, for wiring somewhere else |
| `ok` | `True` / `False` — the verdict as a boolean |

**Example:** 81 frames → 40 pairs, `dropped = 1`. With a dead opening, 124 frames → 116 → 58 pairs
and `report` says `auto_extend: trimmed 8 frames (not 0) to start on the first strong pair`.

### The verdict — read this one line

Every run is judged before it is described, so you can accept or drop a seed at a glance:

```
*** OK - usable stereo ***
*** OK (PATCHY) - one consistent eye order where the disparity is strong; the rest is weak or flat
    (incoherent: mid empty). Weak pairs reading the other way are failed flips, not an inversion ***
*** REJECT - MIXED EYE ORDER (the render changes order part-way through) ***
*** REJECT - the two eyes are WARPED against each other (bands disagree in sign) ***
*** REJECT - NO STEREO (no band and fewer than two opening pairs reach 5 px) ***
```

`OK` / `OK (PATCHY)` are usable; `REJECT` means the render cannot be fixed by any pairing, swap or
`viewing` setting — change the seed. All three REJECT cases are things this pipeline actually produced:
a render that changes eye order mid-clip (measured: instants 0–9 negative, 10+ positive), renders whose
top and bottom bands are shifted in opposite directions (the eyes warped rather than offset), and
renders with no disparity at all.

**A mix is decided by STRENGTH, not by counting signs** — this is the rule that keeps `OK (PATCHY)`
from firing on good clips. A pair that fails to produce the eye separation reads 2–4 px in whatever
direction its noise points; that is a failed flip, not an inversion. So the node compares the two
directions' medians:

```
weaker direction's median / stronger direction's median   >= 0.6  ->  MIXED (a real reversal)
386: opposite pairs median 4.3 px vs matching 11.2 px     -> 0.38   OK (PATCHY), and it is a fine clip
378: 5.0 vs 6.3 px -> 0.78,  370: 6.2 vs 7.7 px -> 0.81   -> both real mixes
```

Detection needs `|step| >= ~3 px`; below that the sign is noise, so nothing is concluded from it.

With `reject_bad_seed` on, a REJECT makes the node error out — the red node *is* the flag, and you see
it right after the render instead of at the end of a watch:

```
*** CHANGE THE SEED AND TRY AGAIN ***
this render was REJECTED: REJECT - MIXED EYE ORDER (the render changes order part-way through)
evidence: opening pairs -6.3 -7.2 -6.9 -7.0 -6.3 -6.0 -6.0 -5.9   |   bands t/m/b +2.9 / +0.9 / -12.7 px
The render itself is saved (upstream SaveVideo); only the side-by-side was skipped. Change the seed
and re-run -- and note that re-running the SAME seed will fail the same way (ComfyUI serves the
sampler from cache).
To pair it anyway and judge for yourself, set `reject_bad_seed = false`.
```

The evidence line is deliberate: you can check the call yourself instead of trusting it — that is how a
false rejection on render 386 was caught and the rule above was fixed.

### Head trim and the QC readout

The opening frames of a generated clip are the weakest: the alternation has to establish itself before
it reaches full strength, and a dead opening pair ships as a mono instant. Measured per pair (frames
0-1, 2-3, 4-5 …) against the clip's own median:

```
H3 render 356   -14.8 -17.4 -14.3 -15.5 ...   median -15.6   strong from frame 0
H3 render 347    -9.8  -7.9  -9.1  -7.5 ...   median  -7.6
H3 render 355    -6.3  ...                    median -14.7   gentler, still real stereo
H3 render 359    -0.3  -0.0  -1.7  -3.2 ...   median -15.8   DEAD for 8 frames
WAN 2.2         -11.5 -12.6 -12.2 -12.2 ...   median -13.1   strong from frame 0
```

With `skip_first = 0` and `auto_extend = True`: 359 trims 8 frames and starts on a real pair, 347 /
355 / 356 / WAN trim **nothing**, and the output is byte-identical to what the fixed pairing produced.
The trim only ever removes whole instants, so the pairing phase is preserved.

A `report` worth reading on every new model or prompt:

```
trimmed 0 frames -> 62 pairs
eye step -15.6 px (median of 12 pairs) at 896 px wide = 1.74% of width -- film class
bands t/m/b -17.5 / -17.4 / -5.6 px (1.96 / 1.94 / 0.62 % of width)  expl 0.51 / 0.51 / 0.37  -> coherent across top/mid/bottom
first 12 pairs (the ramp): -14.8 -17.4 -14.3 -15.5 -18.0 -17.1 -18.5 -15.3 -18.0 -15.7 -17.4 -12.4
sampled steps: -14.8 -18.5 -12.4 -17.9 -15.6 -13.3 -15.3 -13.2 -16.8 -18.0 -17.4 -15.8
```

### Eye order that flips mid-clip (`phase_lock`)

A render can break the eye order part-way through, which the viewer sees as the depth snapping from
cross-eyed to parallel a second or two in. Measured on one 1376×768 render:

```
pairs  0- 6   frames   0- 13   step  -7 px   correct order
pairs  7-61   frames  14-123   step  +6 px   inverted  <- 55 of 62 pairs
```

The two frames inside each pair are still the **same instant** — a pairing offset would instead move
the disparity into the other slot, and it does not: in the body the pair slots read +6.4 px while the
temporal slots read -6.5 px, which is the swapped-order signature. So the repair is a **per-pair half
swap**, not a re-cut of the pairing. With `phase_lock` on:

```
phase_lock: swapped the halves of 55 of 61 pairs so the whole clip is in the trained eye order (frame 0 = RIGHT)
```

Pairs whose step is under 1 px carry no order information and inherit the majority of their
neighbours (±2 pairs), so the repair does not jitter at the weak pairs. `phase_ok` still reports the
**before** figure, which is why a repaired clip shows `phase 0.83` next to a clean output — check the
output itself, or the `phase_lock:` line, to see whether a repair happened. With `phase_lock` off the
node only warns, and the clip is unusable as delivered.

The **band line** is the one to read when a clip looks stereoscopic but the eye step is small: the
whole-frame number is a single rigid fit, and a field that is film-class in one third of the frame and
empty in another averages down to something that looks broken. A band with under ~1 px, or a fit
explaining <= 0.15 of its structure, is reported as **empty** -- no measurable disparity, which is not
the same as a small one:

```
bands t/m/b  -11.5 / -0.6 / -0.9 px (1.12 / 0.06 / 0.09 % of width)  expl 0.49 / 0.05 / 0.14
             -> incoherent: mid,bottom empty
```

The measurement is the same algorithm as `tools/pair_validity.py`'s siblings in the project it came
from (cv2 Sobel gradient mask, masked mean difference, coarse-to-fine, sub-pixel), so its numbers match
external tools exactly. Negative is the correct sign for the documented eye order (frame 0 = right eye).

---

## 2 & 3. Split Stereo Pair, and Split Stereo Pair → batch of 2

An SBS image → its two eyes. Both nodes are the same operation; the second returns them as
one 2-image batch for nodes that take a batch instead of two inputs.

| setting | default | what it does |
|---|---|---|
| `image` | — | the side-by-side stereo image |
| `arrangement` | `parallel / headset [L\|R]` | which half is the left eye — see the eye-order section |

| node | output | what it is |
|---|---|---|
| Split Stereo Pair | `left`, `right` | two separate images |
| Split Stereo Pair → batch of 2 | `eyes` | a batch of 2 images, `[left, right]` |

Note: an odd image width loses its last column, because half of an odd number is not a
half.

---

## 4. Join Stereo Pair

Two eye images → one side-by-side image. The right eye is resized to match the left if
their sizes differ, and a single image is repeated to match the other's batch size, so a
static left image can be paired with a moving right sequence.

| setting | default | what it does |
|---|---|---|
| `left` / `right` | — | the two eye images |
| `arrangement` | `parallel / headset [L\|R]` | which side the left eye goes on — see the eye-order section |

Output: `image` — the combined SBS image.

---

## 5. Swap Stereo Pair

| setting | what it does |
|---|---|
| `image` | an SBS image; the two halves are exchanged |

Output: `image`. Use it to convert between headset and cross-eyed viewing, or to rescue a
pair whose eyes were declared the wrong way round.

---

## 6. Stereo Stabilize (temporal median, per eye)

Removes the per-pixel temporal sizzle that reads as "shaky hands", plus per-frame exposure
pulsing. Each eye is filtered independently, so the disparity between them cannot change —
the node re-measures it and reports if it did.

### Inputs

| setting | default | what it does | what to set |
|---|---|---|---|
| `sbs` | — | the side-by-side pairs, in order | — |
| `radius` | **4** | half-window of the temporal median, in pairs (N = 2r+1). `0` = off (pass-through) | see the warning below — **1 for moving content, 4–5 only for static shots** |
| `level` | `1.0` | how much of the per-frame exposure/luma correction to apply; removed the pulsing (level drift sd 0.93 → 0.11) at no measurable cost. Per eye, clamped to ±3% | `1.0` |
| `max_frames` | `0` | `0` = process all pairs; truncate for a quick look | `0`, or a small number while testing |
| `check_stereo` | `True` | re-measures the disparity before and after and reports it — the check that per-eye filtering really is order-preserving | `True` |

### Outputs

`sbs` (the filtered pairs), `report` (text: boil removed, disparity check, pair count),
`pairs` (how many were processed).

### Boil vs detail — and the reason `radius` is a trap

Measured on a slow scene:

| radius | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| boil removed | −23% | −35% | −42% | −47% | −51% |
| detail lost | −7% | −10% | −12% | −13% | −14% |

Detail loss saturates while the boil keeps falling, so 4–5 looks like the sweet spot. **It
is not, for anything that moves.** Measured again on a walking subject (40 pairs, 768 px
per eye):

| radius | detail kept | **motion kept** |
|---|---|---|
| 4 | 6% | **14%** |
| 1 | — | **72%** |

At radius 4 the walk visibly freezes — the exact failure this pipeline exists to avoid.
The median assumes each pixel is the same thing over the window; a walking figure is not,
so it gets smeared away.

> **Rule: `radius = 1` when things move, `4–5` only on locked-off or slow shots.**
> Verify with `tools/pair_validity.py` — it reports instant motion straight from the
> output. If the motion number drops, your radius is too high.

---

## 7. Stereo Depth Scale (disparity, motion-safe)

Makes a pair deeper or shallower by moving each eye's content horizontally, in **opposite**
directions about the same centre. The disparity scales; the zero-parallax plane does not
move. The temporal axis is never touched, so **motion is preserved exactly** (measured:
3.29 → 3.22 px/instant, against 0.47 for Stabilize at radius 4).

The "how much does each pixel move" map is measured from the pair itself by optical flow
between the two halves, so **no depth map is needed**.

### Inputs

| setting | default | what it does | what to set |
|---|---|---|---|
| `sbs` | — | the side-by-side pairs, in order | — |
| `arrangement` | `parallel / headset [L\|R]` | informational only. Depth scaling is symmetric: the measurement and the correction flip together, so this **cannot** make it scale the wrong way (tested — declaring it wrong gives an identical result) | set it to match your pair, for readability |
| `mode` | `match a target spread (%)` | how the depth is chosen: `match` measures the batch and aims it at `target_spread_pct`; `scale by a fixed factor` multiplies the disparity by `depth_factor` | `match` for "same depth as this reference"; `fixed` for "a bit more, don't care how much" |
| `target_spread_pct` | `1.35` | the depth you want, as a **percent of eye width** — so it is resolution independent (1.35% = 10.4 px at a 768 px eye = 26 px at a 1920 px eye). Only used in `match` mode | `1.35` = a real same-instant pair from a stereo feature. See "choosing numbers", and **the trap below**: if your render already measures 1.35, this setting does nothing |
| `depth_factor` | `1.0` | plain multiplier for `fixed` mode: 1.0 = unchanged, 1.17 = +17% deeper, 0.85 = shallower | `1.0` unless using `fixed` mode |
| `smooth_px` | `9` | how much the disparity map is blurred before use. Raw flow is noisy at occlusion edges (where one eye sees something the other does not) and a noisy map tears the picture | `9`. `0` = sharper but can show local warping; 15–25 = safer/smoother |
| `max_shift_px` | `12` | safety clamp on the largest correction any single pixel may receive — stops one bad measurement (shiny edge, glass) from smearing a stripe | `12` is generous; a 1.17× change needs ~2 px. Drop to `6` if you ever see local smearing |
| `convergence_px` | `0.0` | moves **both** eyes the *same* way: does not change depth, changes **where the scene sits** relative to the screen (pushed in vs popping out) | `0.0`. Touch it only if the whole scene feels too far back or too far forward |
| `check` | `True` | re-measures the spread after warping and reports it | `True` — this is what tells you it worked |
| `refine` | `True` | in `match` mode, measures after the first correction and applies the remainder (flow measurement has a ~6% bias, which this removes) | `True`; disable only to halve the time on long clips |

### Outputs

| output | what it is |
|---|---|
| `sbs` | the depth-adjusted pairs |
| `report` | text, e.g. `40 pairs \| eye 768px \| factor 1.324 \| spread 1.36% -> 1.75% of eye width` |
| `spread_before_pct` | the **uncorrected input**, batch median, in percent of eye width — the number that answers "how deep was this pair already?" |
| `spread_after_pct` | the result, measured after the correction. With `check` off it is a copy of `spread_before_pct` |
| `spread_pass1_pct` | the spread between the two refinement passes (equal to `spread_before_pct` when `refine` is off, because there is only one measurement). Diagnostic only |

All three numbers are **informational** — the correction maths never reads them, so nothing
about the picture depends on this table. Wire `report` into a **Preview as Text** node
(search `preview as text`) to see it without watching the console; the same line is printed
to `ComfyUI/user/comfyui.log` on every run, so `grep "\[StereoDepthScale\]"` works too.

### Limits

- **A target you have already met is a no-op.** `match` mode aims at `target_spread_pct`,
  it does not boost by it. A generated pair that already measures 1.35% gives
  `factor 0.990` and an unchanged picture — correct behaviour, and the most common
  surprise. **`1.35` is the reference depth of a real stereo feature, not an upgrade.**
  To actually go deeper, set the target *above* the measured input (measured on one
  production render: `1.60 -> 1.56%`, `1.80 -> 1.74%`, `2.20 -> 2.11%`, `2.70 -> 2.57%`;
  in `fixed` mode `1.30 -> 1.61%`, `1.50 -> 1.80%`). Note the realized change runs ~10-15%
  below nominal because the flow measurement reads slightly low — ask for a little more
  than you want, or use `fixed` mode and read `spread_after_pct` to confirm.
- **Nothing upstream of this node changes.** Whatever you tap *before* it — a `SaveImage`,
  a preview, a second branch — keeps the uncorrected pairs. Only this node's `sbs` output
  carries the depth change.
- **Keep the change below ~1.3×.** Scaling up stretches pixels that only one eye ever saw
  (disocclusion); small changes are invisible, large ones smear.
- **Do not chase big numbers.** Past ~1.8% of eye width you get divergence — the eyes point
  outward and it hurts to fuse — rather than a better picture. Match a reference; do not
  maximise.
- **It cannot fix vertical misalignment.** Measured: applying a sub-pixel vertical
  correction left `|dy|` unchanged at 0.49 px, because that error is per-pixel content
  variance, not a global offset. There is nothing to align.

---

## Recommended presets

**Convert an existing clip (real footage or a generated one) to stereo SBS**

```
Load Video → Stereo Pair Frames + head trim / QC (defaults)
           → Stereo Stabilize   (radius 1 if anything moves, else 4)
           → Stereo Depth Scale (mode = match, target_spread_pct = 1.35)
           → Save Image / Video Combine
```

If all four of your outputs look right but the model is new to you, set `skip_first = 4` once and
compare: that is one whole latent token on a 4x-temporal VAE, and it is the structural boundary the
`auto_extend` measurement is guessing at.

**Generated clip from an image/video model, with the same-instant LoRA**

```
... → Stereo Pair Frames + head trim / QC (defaults: skip_first 0, auto_extend on, measure on)
    → Stereo Depth Scale (mode = match, target_spread_pct = 1.35)
    → Stereo Stabilize   (radius 1 for moving scenes)
    → Save Image / Video Combine
```

Read the `report` from Stereo Pair Frames + head trim / QC before anything else: if `eye_step_px` is under ~2 px the
LoRA did not take on that seed — change the seed rather than the sliders. If `phase_ok` is below
`1.00`, the depth inverts mid-clip and the clip needs a re-render or a per-segment eye fix.

Depth Scale before Stabilize for two reasons: the stabilizer filters each eye
independently, so it cannot alter the depth you just set, and its built-in check will
confirm that.

---

## Choosing the numbers by measuring, not guessing

`tools/pair_validity.py` reports the disparity spread of any pair images:

```bash
python tools/pair_validity.py --dir <ComfyUI>/output/Stereo --runs
```

To match a reference clip, take its **spread** and divide by the eye width, ×100 →
that is your `target_spread_pct`. Measured reference values on real material:

| source | spread (768 px eye) | as % of eye width |
|---|---|---|
| real same-instant pair from a stereo feature | 10.4 px | **1.35%** |
| generated, closest scene | 12.6 px | 1.64% |
| generated, wide/walking scene | 8.9 px | 1.16% |
| a pair that is *not* really stereo | 1–2 px | 0.1–0.3% |

Two diagnostics the same tool gives you, which are worth more than any setting:

- **warp ratio** — `> 0.68` means the two halves are *not* a stereo pair (they are two
  independent views), `0.26–0.46` is a real pair.
- **instant motion** — the pacing. If this collapses when you add Stabilize, radius is too
  high.
