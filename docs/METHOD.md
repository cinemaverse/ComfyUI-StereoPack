# How the same-instant trick works, and how to check it

This is the measurement side of the project. It exists because "this clip looks 3D to
me" is not a result — and because the failure mode is *silent*: a clip whose pairs are
time-steps rather than eyes still looks plausible in motion, just flat.

## The problem

A video model renders a clip as time steps. Reading consecutive frames as the two eyes
requires the model to render frame 2 as *the other eye of the same instant* instead of
*the next moment*. Left alone it does the latter, and the "stereo pair" is a small
camera move:

| pair type | disparity spread | \|dy\| | mean\|L−R\| | warp ratio |
|---|---|---|---|---|
| time-step pair (no LoRA) | ~2 px | 0.09 | 2.4–4.2 | 0.70–0.90 |
| true same-instant pair | 12.1 px | 0.12 | ~13 | 0.37–0.46 |

Both are two similar images. Only one of them is stereo.

## Four measurements

All four are cheap and independent. `tools/pair_validity.py` computes them on any
side-by-side pair image.

**1. Disparity spread.** Optical flow between the two halves, restricted to textured
pixels (top 35% gradient magnitude — flat areas have no opinion). Spread =
p95 − p5 of the horizontal flow: how much *differential* shift exists across the frame.
Near zero means every pixel moved together, which is a pan, not depth.

**2. Vertical mismatch `|dy|`.** The median absolute vertical flow between the halves. A
stereo pair from a properly aligned rig is vertically aligned; a temporal pair is not,
because perspective changes as the camera or subject moves. Real pairs measure ~0.1–0.4 px.

**3. Warp ratio.** The decisive one. Warp the left half horizontally by the measured
disparity field and compare with the right half; divide the residual by the raw
difference. A true stereo pair is a horizontal *shift* of the same content, so the warp
removes most of the difference:

```
warp ratio < 0.55   real stereo pair
           0.55–0.68 weak / borderline
           > 0.68   not a pair (independent views)
```

**4. Instant motion.** Consecutive pair images give the same eye one instant apart — pair
*k*'s left half vs pair *k+1*'s left half. This measures pacing straight from the output,
which is how the frozen-scene defect was found (0.12 px/instant, where the training data
itself carried 0.34 px).

## What the numbers looked like over time

| configuration | spread | \|dy\| | mean\|L−R\| | warp ratio | instant motion | verdict |
|---|---|---|---|---|---|---|
| ground-truth pair from the source film | 12.1 | 0.12 | ~13 | 0.37–0.46 | — | REAL PAIR |
| earlier 3rd-party stereo LoRA | ~6–9 | 0.57–0.61 | — | 0.63–0.81 | — | phrase-sensitive, unreliable |
| same-instant LoRA, stride-1 data | 9.5 | 0.17 | 13.5 | 0.42 | 0.12 | REAL PAIR |
| …lower strength (0.83) | 6.9 | 0.14 | 12.0 | 0.37 | 0.16 | REAL PAIR |
| same-instant LoRA, stride-3 data, ep20 | 8.3 | 0.15 | 12.2 | 0.42 | 0.23 | REAL PAIR |
| **stride-3 data, ep15 (best)** | **12.6** | 0.35 | 22.5 | **0.34** | **0.35** | **REAL PAIR** |
| mixing up the low/high LoRA files | 1.2 | 0.09 | 3.0 | 0.74 | 0.21 | NOT A PAIR |

The last row is the trap: loading the low-noise LoRA into the high-noise slot. It looks
like a plausible clip, and it measures as a temporal continuation. If a run is flat,
measure it before you change anything else.

## Two findings worth more than the weights

**1. Train less, get more.** Earlier checkpoints won every comparison — epoch 10 beat
epoch 20 beat epoch 40 in *both* disparity and motion. The adapter only has to learn the
eye-flip convention; every bit of delta beyond that also encodes the model's usual
intrusions and its freeze. This is also why scaling an over-trained model's strength down
is the wrong repair: it costs stereo *and* prompt adherence together (measured efficiency
7.3 at ep10 versus 5.1 for ep40 at strength 0.83). Prefer an earlier checkpoint.

**2. Motion is not a linear dial.** Rebuilding the dataset so each training instant
carried 10.6× more action (median 0.34 → 3.66 px) raised the output motion by only ~2.9×
(0.12 → 0.35 px). The freeze is learned as a *content rule* — "same instant, therefore
nearly identical" — not as a magnitude to be scaled. If you want more movement, change
the data's stride or train fewer epochs; adding epochs makes it worse.

## Parity: keeping the two eyes apart

The dataset is built by splitting a stereoscopic source into per-eye frames and
interleaving them `R, L, R@+stride, …`. Parity means the *registration* of that
interleave is correct: frame 0 is the right half of instant *t*, frame 1 is the left half
of the same instant *t*, frame 2 is the right half of instant *t+stride*. Get it wrong
and the model learns the wrong relationship while the loss still looks fine — that is why
`tools/parity_check.py` is run on the built dataset rather than trusting the builder.

Rules that must not drift:

- `frame_stride = 1` in the dataset config (the pair is *adjacent* frames; the *source*
  stride that changed pacing is a builder argument, not this)
- `frame_extraction = "head"` (parity-safe)
- `source_fps` must be a float
- `max_frames` must match `target_frames` for a 9-frame clip

## Reproducing the checks

```bash
# is this output actually stereo?
python tools/pair_validity.py --dir <ComfyUI>/output/Stereo --runs

# is the extracted dataset registered correctly?
python tools/parity_check.py clips --dir <dataset>/videos --stride 3

# are the eyes the right way round?
python tools/eye_order_check.py --help
```
