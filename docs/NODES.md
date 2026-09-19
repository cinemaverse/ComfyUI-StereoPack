# Node reference — every setting explained

Seven nodes, what each setting does, what to set it to, and the traps. If you only read
one section, read [the presets](#recommended-presets) at the bottom.

| node | one line |
|---|---|
| **Stereo Pair Frames** | consecutive frames of a clip → one side-by-side stereo pair |
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

- **Stereo Pair Frames' `viewing`** asks *how you watch it*. Its default (`cross-eyed`)
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

## 1. Stereo Pair Frames (consecutive → L|R)

Groups an image batch two frames at a time and stitches each pair side by side. Length
safe: with an odd frame count the last frame is left out rather than paired with a repeat
of itself.

### Inputs

| setting | default | what it does | what to set |
|---|---|---|---|
| `images` | — | the clip's frames, in order | your decoded clip |
| `viewing` | `cross-eyed (left = frame 1, right = frame 2)` | which eye order matches how you will watch it. Measured on real output: `frame1\|frame2` reads as **cross-eyed**, which is why it is the default | leave default for cross-eyed/desktop viewing; pick `parallel / headset` for a VR headset |
| `unpaired_last` | `drop (leave it alone)` | what to do with a final odd frame that has no partner | `drop`. The other options (`black partner`, `duplicate last frame`) invent a frame and are for special cases |
| `pad_to_even` | `True` | pads to even width/height if needed | leave `True` — h264/yuv420p cannot encode an odd dimension |

### Outputs

| output | what it is |
|---|---|
| `sbs` | the side-by-side pair batch — wire this onward |
| `second_eye` | the second frame of each pair, 1:1 aligned with `sbs` (for nodes that want the partner/other-eye stream) |
| `pairs` | how many pairs were made |
| `dropped` | how many frames were left over (1 if the clip had an odd frame count) |

**Example:** 81 frames → 40 pairs, `dropped = 1`.

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
Load Video → Stereo Pair Frames (defaults)
           → Stereo Stabilize   (radius 1 if anything moves, else 4)
           → Stereo Depth Scale (mode = match, target_spread_pct = 1.35)
           → Save Image / Video Combine
```

**Generated clip from an image/video model, with the same-instant LoRA**

```
... → Stereo Pair Frames (defaults)
    → Stereo Depth Scale (mode = match, target_spread_pct = 1.35)
    → Stereo Stabilize   (radius 1 for moving scenes)
    → Save Image / Video Combine
```

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
