# Train your own same-instant stereo LoRA

The LoRA is **not required** to use the nodes in this repo, and it is not distributed
here. This is the complete recipe if you want to make one — it is a two-stage job, and
the second stage exists because of a defect the first stage has.

Read [METHOD.md](METHOD.md) first for how the measurements work; every decision below is
based on them.

---

## What you need

- **Your own two-eye source material.** A stereoscopic film (left/right eye in one frame,
  side by side or stacked), a stereo camera rig, or any footage whose consecutive frames
  are near-duplicates. Anything you have the rights to use — check this before you
  publish anything trained on it.
- **A Wan 2.2 I2V base** (`wan2.2_i2v_low_noise_14B` / `..._high_noise_14B`), fp16 or
  fp8 — and the matching VAE, the umt5 text encoder, and the `wan_2.1_vae`.
- **[musubi-tuner](https://github.com/kohya-ss/musubi-tuner)** for the training itself.
- Optionally an existing stereo LoRA to start from. Starting from one converges much
  faster, but only use one you have the rights to build on.
- ~16 GB VRAM works (`--fp8_base` + `--blocks_to_swap 20`), and one model at a time.

---

## Stage 1 — build a statement-of-fact dataset (stride 1)

Pairs must be *exactly* one instant: frame 0 = right eye of instant *t*, frame 1 = left
eye of the same instant, frame 2 = right eye of instant *t+1*. The builder handles this;
what you choose is how much time an instant carries.

```bash
python training/sbs_segments_to_dataset.py \
  --movie <your-stereo-source.mkv> \
  --candidates candidates.json \
  --out <dataset_dir> \
  --source-layout lr --first-eye right \
  --frames 81 --source-stride 1 --auto-align
```

- `--source-layout lr` — which half of the source frame is which eye. Verify it, do not
  assume: `tools/eye_order_check.py`.
- `--first-eye right` — emit the right eye first. This is the order the nodes and the
  workflow expect; keep it.
- `--source-stride 1` — how many source frames one instant advances. **Stage 1 uses 1.**
- `--auto-align` — removes residual vertical misalignment from the source rig.

**Verify parity before training.** This is the step that silently ruins a run:

```bash
python tools/parity_check.py clips --dir <dataset_dir>/videos --stride 1
```

It must report `R, L, R@+1` on every clip.

---

## Stage 1 — train

```bash
# from a base LoRA (recommended)         or from scratch (drop INIT)
INIT=<your-base-lora.safetensors> ./training/train.sh low  <dataset_dir> 40 sameinstant_low
INIT=<your-base-lora.safetensors> ./training/train.sh high <dataset_dir> 40 sameinstant_high
```

Do the two noise stages as two separate runs — `--offload_inactive_dit` needs ~96 GB of
system RAM, so a combined run is not an option on a normal machine.

Expected: ~6 s/step at 512×512 with 9-frame clips, so 40 epochs over ~40 clips is about
2.5 h per stage. Save every 5 epochs — you will want the earlier checkpoints.

**Now look at the pacing before you go further.** The model will faithfully reproduce the
data's action per instant. At stride 1 that is ~0.34 px at 768 wide, which reads as slow
motion, and at high strength the model can satisfy "same instant" by simply *freezing*.
That is stage 2's job.

---

## Stage 2 — fix the pacing (stride 3)

Rebuild the same dataset with more time per instant, then **continue** from a stage-1
checkpoint:

```bash
python training/sbs_segments_to_dataset.py ... --source-stride 3 --out <dataset_stride3> ...
python tools/parity_check.py clips --dir <dataset_stride3>/videos --stride 3   # R, L, R@+3

INIT=<your-dir>/sameinstant_low-000020.safetensors  ./training/train.sh low  <dataset_stride3> 20 sameinstant_s3_low
INIT=<your-dir>/sameinstant_high-000020.safetensors ./training/train.sh high <dataset_stride3> 20 sameinstant_s3_high
```

Three deliberate choices, each measured:

- **Continue from epoch 20 of stage 1**, not from the final checkpoint: later checkpoints
  measured worse on both stereo and motion (see METHOD.md).
- **Half the learning rate** (the script defaults to `1.5e-4` for exactly this reason).
  The delta keeps growing ~1.1× per 10 epochs and the freeze grows with it, so a full-rate
  continuation inflates intrusion while fixing pacing.
- **Stride 3, not more.** 3 source frames = 1/8 s per instant = real-time motion at 8 fps
  playback. Expect the output motion to rise by roughly 3×, not by the 10× the data
  gained — motion is not a linear dial.

Do not mix strides within one dataset.

---

## Accept it with measurements, not vibes

Generate a clip through `workflows/03_i2v_sameinstant_full.json` at strength 1.0/1.0,
then:

```bash
python tools/pair_validity.py --dir <ComfyUI>/output/Stereo --runs
```

What good looks like (measured on this project's best run): disparity spread ~10–13 px,
`|dy|` ≤ 0.4 px, warp ratio ~0.34–0.42, instant motion higher than stage 1's. The verdict
must read **REAL PAIR**.

Red flags:

| symptom | cause |
|---|---|
| warp ratio > 0.68, `mean|L−R|` ≈ 3 | the pair is a temporal continuation, not stereo |
| spread ≈ 1–2 px | same thing, seen as flatness |
| instant motion ≈ 0.1 px | the scene is frozen — the stride is too small or the checkpoint is too late |
| spread collapses at a lower strength | strength is the wrong dial; use an earlier checkpoint |

---

### Did my fine-tune actually change much?

```bash
python tools/lora_overlap.py <base-lora.safetensors> <your-checkpoint.safetensors> [...]
```

It reports the energy-weighted cosine between the two updates: `1.00` = only refined,
`0.6` = two thirds of the original direction kept, `~0` = rewritten. Measured on this
project: 10 epochs of training already moved the update by 64% of the base's own
magnitude, and 40 epochs by 121% — which is the mechanism behind "earlier checkpoints
score better". A *continuation* accumulates, so compare the checkpoint before a stage
against the one after it to isolate that stage's own drift.

---

## Pitfalls that cost real time

- **`frame_stride = 1` must stay 1** in the dataset config. The *source* stride is a
  builder argument; changing the config's `frame_stride` breaks the adjacency the pair
  depends on.
- **`frame_extraction = "head"`** — parity-safe. Don't experiment.
- **`source_fps` must be a float** (`16.0`, not `16`) or the config is rejected.
- **`max_frames` must equal `target_frames`** for a 9-frame clip.
- **Never load the low-noise LoRA in the high-noise slot.** It produces a plausible clip
  that measures as NOT A PAIR. This mistake cost a full debugging cycle here.
- **A LoRA in kohya/musubi format loads in ComfyUI as-is** — no conversion step, and
  converting it only risks breaking it.
- **Don't run ComfyUI jobs while training.** They share VRAM, and the training run will
  silently slow to a crawl.
