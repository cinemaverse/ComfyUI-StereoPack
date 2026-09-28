# ComfyUI-StereoPack

Stereo video nodes for ComfyUI. Read a generated clip as a **stereo side-by-side pair**
(consecutive frames = the two eyes of one instant), trim the weak opening frames, measure the
result, split / join / swap the eyes, and remove the temporal boil.

Seven nodes, one install, **no extra dependencies** (torch, numpy and opencv are already in
ComfyUI).

---

## What's new — 2026-09-27

**`Stereo Pair Frames + head trim / QC` gained a head trim, a measurement, and an eye-order repair.**
All three come from work on a frame-interleaved 3D LoRA pipeline, where these kept showing up in
finished clips:

1. **The opening frames are weak.** A generated clip needs a moment before the eye alternation
   reaches full strength; a dead opening pair ships to the viewer as a mono instant — "the first
   two frames are not in stereo". `skip_first` and `auto_extend` drop those frames
   automatically, and only those frames: a clip that is strong from frame 0 is untouched.
2. **You could not tell shallow from broken without leaving ComfyUI.** The node now measures the
   eye step itself (`eye_step_px`), reports it as a percentage of frame width and per band, and
   reports `phase_ok` — the fraction of pairs that agree on the eye order.
3. **Some clips flipped eye order part-way through**, so the depth snapped from cross-eyed to
   parallel a second or two in (measured on one render: 7 pairs one way, 55 the other). `phase_lock`
   now swaps the halves of the inverted pairs, and only those, to give one consistent clip.

**Compatibility:** the node key, the class and the first three widgets are unchanged, so graphs
saved before today load without edits (the new inputs are appended after the old ones). Only the
title shown in the node menu changed, to name what the node now does.

---

## Nodes

**Every setting of every node, explained: [docs/NODES.md](docs/NODES.md)** — what each
one does, what to set it to, the recommended presets, and the traps (including the
stabilizer radius that silently freezes moving footage).

| Node | Output | What it does |
|---|---|---|
| **Stereo Pair Frames + head trim / QC (consecutive -> L\|R)** | `sbs`, `second_eye`, `pairs`, `dropped`, `eye_step_px`, `phase_ok`, `report` | Groups an image batch two frames at a time and stitches each pair side by side. Length-safe: an odd frame count leaves the last frame out instead of pairing it with a repeat of itself. Drops the weak opening frames before pairing (`skip_first` / `auto_extend`) and measures the eye step + eye-order consistency for you. |
| **Split Stereo Pair (SBS -> left + right)** | `left`, `right` | One SBS image → two eye images. |
| **Split Stereo Pair -> batch of 2** | `eyes` | Same, as a batch of 2 (for two-image nodes). |
| **Join Stereo Pair (left + right -> SBS)** | `image` | Two eye images → one SBS image (resizes the right eye if the sizes differ). |
| **Swap Stereo Pair (\|L\|R\| <-> \|R\|L\|)** | `image` | Flips the halves — fix a crossed pair, or switch between headset and cross-eyed. |
| **Stereo Stabilize (temporal median, per eye)** | `sbs`, `report`, `pairs` | Temporal median per eye removes the per-pixel sizzle ("boil"); also fixes per-frame exposure pulsing. Re-measures the disparity before/after and reports it. |
| **Stereo Depth Scale (disparity, motion-safe)** | `sbs`, `report`, `spread_before_pct`, `spread_after_pct` | Scales — or auto-matches — the depth of an SBS pair by moving the two eyes' content apart symmetrically. No depth map needed, and instant-to-instant motion is untouched. See below. |


## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/cinemaverse/ComfyUI-StereoPack.git
```

Restart ComfyUI. The nodes appear under the **Stereo** category (the eye-splitting nodes
under **image/stereo**). No `pip install` step, no requirements.

Also installable via ComfyUI-Manager once registered, or download the ZIP and drop the
folder into `custom_nodes`.

**Requires:** ComfyUI with torch, numpy and `opencv-python` (all standard). Nothing else.

---

## Full pipeline — Wan 2.2 I2V with the same-instant LoRA

`workflows/WAN_VideoWF_3D_Stereo_I2V.json` is the end-to-end graph: image → Wan 2.2 I2V
(low+high noise) → Stereo Pair Frames + head trim / QC → SBS video.

Set **`skip_first = 0`** on the pairing node. Wan clips measured so far are strong from frame 0
(14.8, 17.4, 16.1 … px on the first pairs of the test render), so `auto_extend` trims nothing and
the node is a no-op; the fixed default of 2 would have thrown away a good instant. The readout
from that graph is `eye step -13.1 px = 1.70 % of width -- film class`, `phase 1.00`.

Its **key is still `StereoPairFrames`**, so every saved graph opens unchanged — only the title in
the node menu changed, to **Stereo Pair Frames + head trim / QC**.

### The LoRA

Two files, one for each Wan 2.2 noise stage. Put them in
`ComfyUI/models/loras/wan/` **with these exact names**, which is what the workflow
expects or adapt:
https://civitai.red/models/2949659/real-stereo-depth-video-wan-22-i2v 

```
models/loras/wan/StereoDepth_Low_I2V.safetensors  <- i2v LOW noise  (node 159)
models/loras/wan/StereoDepth_High_I2V.safetensors    <- i2v HIGH noise (node 158)
```

## Full pipeline — H3 Minimax FL2VA with the same-instant LoRA

`workflows/H3_VideoWF_3D_Stereo_FL2VA.json` is the end-to-end graph: image → H3 Lora → Stereo Pair Frames + head trim / QC → SBS video.

Set **`skip_first = 0`** on the pairing node. Wan clips measured so far are strong from frame 0
(14.8, 17.4, 16.1 … px on the first pairs of the test render), so `auto_extend` trims nothing and
the node is a no-op; the fixed default of 2 would have thrown away a good instant. The readout
from that graph is `eye step -13.1 px = 1.70 % of width -- film class`, `phase 1.00`.

Its **key is still `StereoPairFrames`**, so every saved graph opens unchanged — only the title in
the node menu changed, to **Stereo Pair Frames + head trim / QC**.

### The LoRA

One file put it in
`ComfyUI/models/loras/wan/` **with these exact names**, which is what the workflow
expects or adapt:
https://civitai.red/models/2969930/real-stereo-depth-video-h3-minimax

```
models/loras/H3/H3_StereoDepth_FL2VA.safetensors
```


### Other models the workflow needs

| slot | file |
|---|---|
| UNET high / low | `wan/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors`, `wan/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors` |
| speed LoRA (4-step) | `wan/wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors` and its low-noise twin |
| text encoder | `wan/umt5_xxl_fp8_e4m3fn_scaled.safetensors` |
| VAE | `Wan2_1_VAE_fp32.safetensors` |
| CLIP vision | `clip_vision_h.safetensors` |

Sampler settings are the standard 4-step setup: 8 steps, cfg 1.0, euler / beta, split
across the two noise stages (0–4 and 4–end). Four `LoraLoaderModelOnly` slots in the
graph are **bypassed** placeholders for your own style LoRAs — un-bypass them (Ctrl+B)
and point them at your files if you want to stack a look on top.

Also required: [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)
(`VHS_*` nodes) and [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)
(`PathchSageAttentionKJ`, `PreviewAny`).

---

## Verify your own output

### In the graph — the node measures it for you (2026-09-27)

`Stereo Pair Frames + head trim / QC` reports its own numbers, so the first check needs no extra
tooling:

* **`eye_step_px`** — how far the image content slides between the two eyes of a pair, in pixels.
  This is the whole 3D signal: 0 px is flat, ~3 px is shallow but visible, **10–20 px at 1024 px
  wide is film class** (real 3D films measure 10.8–20.1 px, median 14.8).
* **`phase_ok`** — the fraction of sampled pairs that agree on the eye order. `1.00` is clean;
  anything lower means the eyes swap somewhere and the depth inverts there.
* **`verdict`** / **`ok`** — a one-line judgement of the render, and the same as a boolean: `OK`,
  `OK (PATCHY)` (usable where it has depth; weak opposite-order pairs are failed flips, not an
  inversion), or `REJECT` (mixed eye order / warped eyes / no stereo). It is the **first line of
  `report`** too, so the answer is the first thing you read.
* **`reject_bad_seed`** — with it on, a REJECT turns the node **red and stops the graph**, and the
  message says `*** CHANGE THE SEED AND TRY AGAIN ***` with the opening pairs and the band triple as
  evidence, and a note that re-running the same seed will fail identically (the sampler is cached).
  The flag, instead of a surprise while watching.
* **`report`** — the same numbers as text, plus the opening ramp (the first 12 pairs), the
  **band profile** (see below), and a note when a dead opening was trimmed or an inverted eye order
  was repaired.
* **`phase_lock`** — an eye order that flips part-way through the clip is repaired automatically by
  swapping the halves of the inverted pairs, so the depth stops snapping from cross-eyed to parallel
  mid-clip. It only runs on clips that fail the consistency check, so a clean clip pays nothing.

### The band profile — where the depth actually is

One whole-frame number hides a field that can be film-class in one third of the frame and empty
in another, which is exactly what "I can see stereo but your measurement says almost none"
looks like. The `report` therefore splits the rigid fit into three, and keeps a fit-quality
(`explained`) per band, so a band reads as *empty* rather than *small* when there is nothing
measurable to fit:

```
bands t/m/b  -11.5 / -0.6 / -0.9 px (1.12 / 0.06 / 0.09 % of width)  expl 0.49 / 0.05 / 0.14
             -> incoherent: mid,bottom empty
```

That clip measured `eye step -0.4 px` on the whole frame while its top third carried 1.12 % of
width (film class). A band is called empty when its step is under ~1 px or its fit explains
≤ 0.15 of that band's structure. `tools/pair_validity.py` prints bands too, from a per-pixel
flow rather than a rigid fit — see the note on the two weightings below.

### The head trim

The first frames of a generated clip are the weakest: the eye alternation has to establish itself
before it reaches full strength, and a dead opening pair ships to the viewer as a mono instant.
Measured on real renders (pixels per pair, then the clip median):

```
H3 render 356   -14.8 -17.4 -14.3 -15.5 ...   median -15.6   strong from frame 0
H3 render 347    -9.8  -7.9  -9.1  -7.5 ...   median  -7.6
H3 render 359    -0.3  -0.0  -1.7  -3.2 ...   median -15.8   DEAD for 8 frames
WAN 2.2         -11.5 -12.6 -12.2 -12.2 ...   median -13.1   strong from frame 0
```

`skip_first` drops frames before pairing — **2 = one instant, 4 = one whole latent token** on a
4x-temporal VAE (H3, Wan). It must be even, so frame 0 of each pair stays the right eye.
`auto_extend` (on by default) keeps dropping instants while the measured pair is still weak, so a
clip like 359 loses its dead opening instead of shipping it, while a clip that is strong from frame
0 — every Wan render measured, most H3 ones — is left completely alone. With `skip_first = 0` the
node is a **no-op on a clean clip**: 0 frames trimmed, the exact same pairs as before.

Measured with the two pipelines that produced those ramps, at the recommended settings:

```
H3  (portrait 896x1184)   trimmed 0 frames -> 62 pairs   eye step -15.6 px   1.74 % of width
H3  (1024x576)            trimmed 8 frames -> 58 pairs   eye step -15.8 px   1.54 % of width
WAN (768x1024)            trimmed 0 frames -> 40 pairs   eye step -13.1 px   1.70 % of width
```

**Tripping a false positive is the thing to watch:** auto_extend only fires on pairs below
`expected_min_px` (default 3 px) or below a quarter of the clip's own median, so a merely gentler
opening (6.3 px on a 14.7 px clip) is *not* trimmed. The `report` always states how many frames went,
so the decision is visible rather than silent.

---

### With the tools

**Which tool answers which question:**

| question | tool | what it prints |
|---|---|---|
| *is there parallax at all, and how much?* | the node itself (`eye_step_px`) | one rigid global shift per clip, in px and % of width |
| *did the eye order hold for the whole clip?* | the node (`phase_ok`) | fraction of pairs agreeing on the order; < 1.00 means the eyes swap mid-clip |
| *how much does the depth vary across the frame?* | `tools/pair_validity.py` (`spread`) | p95–p5 of the per-pixel disparity |
| *is the pair real at all?* | `tools/pair_validity.py` (`warp`) | residual after warping one half onto the other, with a REAL PAIR / WEAK / NOT A PAIR verdict |
| *is the eye convention right on a saved dataset or extracted clip?* | `tools/parity_check.py`, `tools/eye_order_check.py` | dataset-side checks, run offline |

**These numbers are not interchangeable, and comparing them without knowing why is confusing.**
Measured together on one Wan 2.2 clip (768 px eye width):

```
node eye_step_px        -13.1 px    rigid global fit -> the STRONG (near-field) end of the range
pair_validity median dx  -7.4 px    the middle of the per-pixel distribution
pair_validity spread     14.2 px    p95 - p5: the whole range, i.e. depth variation
p5 / p95            -14.3 / -0.1    the two ends of that range
```

So a clip with a large uniform shift but no depth variation (a cardboard cut-out) scores high on
the node's number and low on the tool's spread. Both are honest; they answer different questions.

The same applies to the new **bands** on both sides: the node's are structure-weighted rigid fits
(with a quality figure each), the tool's are area-weighted per-pixel medians. On the Wan clip
above the tool reads `bands -5.60 -11.41 -5.47` (mid-heavy) while the node reads
`-13.1 / -14.0 / -6.6` (top/mid, bottom weak). They agree on the useful verdict — *which band is
empty* — and disagree on magnitude by design. Never compare bands across the two.

---

Node packs rarely ship evidence that a claim is true. This one does:

```bash
python tools/pair_validity.py --dir "ComfyUI/output/Stereo" --runs
python tools/pair_validity.py --file 3722
```

It reports, per run: the disparity **spread** (in px and as % of the eye width -- how much the
depth varies across the frame), the **eye step** (median per-pixel disparity), vertical
misalignment `|dy|`, the raw difference
between the halves, the warp ratio, the **instant-to-instant motion**, and a verdict —
**REAL PAIR / WEAK / NOT A PAIR** -- calibrated on this project's own outputs. A Wan 2.2 run
reads `spread 14.2 (1.84 % of eye) ... eye step -6.19 px ... REAL PAIR`, against a film-class
eye step of 10-20 px at 1024 px eye width. Why those two numbers differ is explained in the
tool's own docstring.

Consecutive pair images also give you the motion for free (pair *k*'s left half and pair
*k+1*'s left half are the same eye one instant apart), so you can check pacing from the
outputs alone.

`tools/parity_check.py` and `tools/eye_order_check.py` are the dataset-side equivalents:
they verify that an extracted clip really is `R, L, R@+stride` and that the eyes are not
swapped.

---

## Train your own LoRA

The LoRA is not required to use these nodes, and it is not included in this repo. The
whole recipe is in **[docs/TRAIN_YOUR_OWN.md](docs/TRAIN_YOUR_OWN.md)**, with the dataset
builder and training configs in `training/`. It needs your own two-eye source material
(a 3D film, a stereo camera rig, or any clip whose consecutive frames are near-duplicates)
plus [musubi-tuner](https://github.com/kohya-ss/musubi-tuner) and a Wan 2.2 I2V base.

Two findings from that work that will save you time:

- **Train less, get more.** Earlier checkpoints won on every axis measured — more
  stereo *and* more motion at epoch 10–15 than at epoch 40. The adapter's real job is
  the eye-flip convention; everything the delta adds beyond that also adds intrusion and
  freezes the scene. Prefer an earlier checkpoint over a lower strength.
- **Motion is not a linear dial.** Putting 10× more action into each training instant
  raised the output motion only ~3×. If you need more movement, change the data (bigger
  stride) or train fewer epochs — not more.

---

## Licence

Apache-2.0 — see [LICENSE](LICENSE).
