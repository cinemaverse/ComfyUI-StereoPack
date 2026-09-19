# ComfyUI-StereoPack

Stereo video nodes for ComfyUI. Read a generated clip as a **stereo side-by-side pair**
(consecutive frames = the two eyes of one instant), split / join / swap the eyes, and
remove the temporal boil.

Three node sets, one install, **no extra dependencies** (torch, numpy and opencv are
already in ComfyUI).

---


## Nodes

**Every setting of every node, explained: [docs/NODES.md](docs/NODES.md)** — what each
one does, what to set it to, the recommended presets, and the traps (including the
stabilizer radius that silently freezes moving footage).

| Node | Output | What it does |
|---|---|---|
| **Stereo Pair Frames (consecutive -> L\|R)** | `sbs`, `second_eye`, `pairs`, `dropped` | Groups an image batch two frames at a time and stitches each pair side by side. Length-safe: an odd frame count leaves the last frame out instead of pairing it with a repeat of itself. |
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

`workflows/VideoWF_3D_Stereo_I2V.json` is the end-to-end graph: image → Wan 2.2 I2V
(low+high noise) → Stereo Pair Frames → SBS video.

### The LoRA

Two files, one for each Wan 2.2 noise stage. Put them in
`ComfyUI/models/loras/wan/` **with these exact names**, which is what the workflow
expects or adapt:

```
models/loras/wan/StereoDepth_Low_I2V.safetensors  <- i2v LOW noise  (node 159)
models/loras/wan/StereoDepth_High_I2V.safetensors    <- i2v HIGH noise (node 158)
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

Node packs rarely ship evidence that a claim is true. This one does:

```bash
python tools/pair_validity.py --dir "ComfyUI/output/Stereo" --runs
python tools/pair_validity.py --file 3722
```

It reports, per run: disparity spread, vertical misalignment `|dy|`, the raw difference
between the halves, the warp ratio, the **instant-to-instant motion**, and a verdict —
**REAL PAIR / WEAK / NOT A PAIR** — calibrated on this project's own outputs.

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
