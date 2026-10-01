"""Build WAN_V2V_Stereo.json from the Wan I2V stereo graph -- the drift-free route.

The point: with Wan the latent is a PLAIN video latent (WanVAE.encode), so the clip itself can
*be* the sampler's starting latent.  Denoise partway (`start_at_step`) and the timeline is kept
by construction while the stereo LoRA re-renders the second eye.  H3 cannot do this: its latent
is a nested (video, audio) tensor that only EmptyMiniMaxH3LatentAV / the conditioning nodes
build, so no shipped node can inject an encoded video.

Changes vs WAN_VideoWF_3D_Stereo_I2V.json:
  * VHS_LoadVideo loads a clip retimed to 8 fps (your trained instant spacing) and duplicates it
    to 16 fps via force_rate, so consecutive frames ARE the same-instant pair the LoRA expects;
  * VAEEncode (Wan VAE) turns those frames into the sampler's latent;
  * BOTH samplers stay wired exactly as Wan 2.2 wants: the HIGH-noise model owns the noisy
    steps (start -> 4) and hands the leftover noise to the LOW-noise model (4 -> end).  The
    drift dial is the high-noise sampler's start_at_step, where the noise is added.
    MEASURED at denoise 0.75 on this clip (spread is what makes 3D):
        high 2->4 + low 4->8   warp 0.48   spread 7.84 px (1.23% of eye)  REAL PAIR  <- default
        low  2->8 (B injects)  warp 0.53   spread 4.30 px (0.67%)         REAL PAIR
        high 2->2 + low 2->8   warp 0.95   spread 0.94 px                 NOT A PAIR (broken)
  * WanFirstLastFrameToVideo.length / width / height are set to match the encoded latent exactly
    (length must be 4k+1 and give the same latent T as the encoded clip);
  * start_image + clip vision now come from the clip's own first frame (ImageFromBatch).
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
SRC = WORK / "workflows" / "WAN_VideoWF_3D_Stereo_I2V.json"
DST = WORK / "workflows" / "WAN_V2V_Stereo.json"

CLIP = "gif_965734544_8fps.mp4"      # the GIF retimed to 8 fps (set this to your source)
W, H, LEN = 640, 768, 33             # 33 = 4k+1 -> latent T = 9, same as 33 encoded frames
DEFAULT_START_STEP = 2               # high-noise sampler's start_at_step (the dial)
SPLIT = 4                            # the two-model hand-off step (high ends, low begins)
                                     # measured: 4 -> no disparity at all (mono), 2 -> REAL PAIR

NOTE = f"""### Wan 2.2 video-to-video: your clip's timeline, generated stereo

The latent below is **your clip** (`VAEEncode`), not an empty latent, so the sampler cannot
drift away from it — denoising only rewrites what the LoRA is asked to rewrite.

**The drift dial is `start_at_step` on the HIGH-noise `KSamplerAdvanced`** (8 steps total).
That is where the noise is added, and the two-model split is untouched: high-noise model
start -> 4, low-noise model 4 -> end (its `add_noise` stays disabled and it continues from the
leftover noise, exactly as in your generation graph).

| high-noise start_at_step | denoise | what runs | effect |
|---|---|---|---|
| 0 | 1.00 | high 0->4 + low 4->8 | pure generation (drifts, like H3) |
| **2** | **0.75** | high 2->4 + low 4->8 | measured: REAL PAIR, the default |
| 4 | 0.50 | low 4->8 | half — measured: mono, no disparity |
| 6 | 0.25 | low 6->8 | very gentle, still shallow |

Keep the low-noise sampler at `4 -> 10000`, `add_noise = disable`; if you move the dial past 4
you must also give the *low-noise* sampler `add_noise = enable` and the same start step, or it
will sample from a latent that was never noised.

**Pacing.** `{CLIP}` is the source retimed to **8 fps** — your LoRA's trained instant spacing
(3 source frames = 1/8 s). `VHS_LoadVideo.force_rate = 16` duplicates each frame, so consecutive
frames are the *same instant* pair it was trained on. The SBS `VHS_VideoCombine` therefore runs at
**8 fps**. Change `force_rate` and that combine rate together, never one alone.

**Sizes must line up.** `WanFirstLastFrameToVideo.length` must give the same latent T as the
encoded clip: T = (length-1)/4 + 1. Here 33 frames -> T = 9 and length = 33. If you change the
frame count (`frame_load_cap`), change `length` with it. Width/height are {W}x{H} (Wan needs
multiples of 16) and must match the `custom_width`/`custom_height` VHS resizes to.

**The high-noise sampler is bypassed** because the noisiest steps are exactly what we don't want.
Un-bypass it (Ctrl+B) and reconnect its latent if you ever want the full generate-from-noise path
back — or simply lower `start_at_step`.

`LoadImage` (the still that used to feed the I2V start frame) is no longer connected: the clip's
own first frame feeds `start_image` and the clip-vision encode, via `ImageFromBatch`.
"""


def main() -> None:
    d = json.loads(SRC.read_text(encoding="utf-8"))
    by = {n["id"]: n for n in d["nodes"]}
    tmpl_vhs = next(n for n in d["nodes"] if n["type"] == "VHS_LoadVideoPath")

    last_link = d["last_link_id"]
    nid = d["last_node_id"]

    def new_id() -> int:
        nonlocal nid
        nid += 1
        return nid

    def new_link() -> int:
        nonlocal last_link
        last_link += 1
        return last_link

    # ------------------------------------------------------------------ new nodes
    vhs_id, enc_id, from_id = new_id(), new_id(), new_id()
    l_clip_enc, l_clip_from, l_vae_enc = new_link(), new_link(), new_link()

    vhs_widgets = dict(tmpl_vhs["widgets_values"])
    vhs_widgets.update({"video": CLIP, "force_rate": 16, "custom_width": W, "custom_height": H,
                        "frame_load_cap": LEN, "skip_first_frames": 0, "select_every_nth": 1})
    vhs_widgets["videopreview"] = {"hidden": False, "paused": False, "params": {}}

    def w(name, type_, **kw):
        e = {"localized_name": name, "name": name, "type": type_}
        e.update(kw)
        e.setdefault("link", None)
        return e

    vhs = {
        "id": vhs_id, "type": "VHS_LoadVideo", "pos": [-1350.0, 1180.0], "size": [470, 310],
        "flags": {}, "order": 30, "mode": 0,
        "inputs": [
            w("meta_batch", "VHS_BatchManager", shape=7), w("vae", "VAE", shape=7),
            w("video", "COMBO", widget={"name": "video"}),
            w("force_rate", "FLOAT", widget={"name": "force_rate"}),
            w("custom_width", "INT", widget={"name": "custom_width"}),
            w("custom_height", "INT", widget={"name": "custom_height"}),
            w("frame_load_cap", "INT", widget={"name": "frame_load_cap"}),
            w("skip_first_frames", "INT", widget={"name": "skip_first_frames"}),
            w("select_every_nth", "INT", widget={"name": "select_every_nth"}),
            w("format", "COMBO", widget={"name": "format"}),
        ],
        "outputs": [
            {"localized_name": "IMAGE", "name": "IMAGE", "type": "IMAGE",
             "links": [l_clip_enc, l_clip_from]},
            {"localized_name": "frame_count", "name": "frame_count", "type": "INT", "links": []},
            {"localized_name": "audio", "name": "audio", "type": "AUDIO", "links": []},
            {"localized_name": "video_info", "name": "video_info", "type": "VHS_VIDEOINFO",
             "links": []},
        ],
        "properties": {"cnr_id": "comfyui-videohelpersuite", "ver": "1.7.9",
                       "Node name for S&R": "VHS_LoadVideo"},
        "widgets_values": vhs_widgets,
    }
    encoder = {
        "id": enc_id, "type": "VAEEncode", "pos": [-800.0, 1180.0], "size": [210, 46],
        "flags": {}, "order": 31, "mode": 0,
        "inputs": [w("pixels", "IMAGE", link=l_clip_enc), w("vae", "VAE", link=l_vae_enc)],
        "outputs": [{"localized_name": "LATENT", "name": "LATENT", "type": "LATENT",
                     "links": [113]}],
        "properties": {"cnr_id": "comfy-core", "ver": "0.32.0", "Node name for S&R": "VAEEncode"},
        "widgets_values": [],
    }
    first_frame = {
        "id": from_id, "type": "ImageFromBatch", "pos": [-1350.0, 1580.0], "size": [210, 82],
        "flags": {}, "order": 32, "mode": 0,
        "inputs": [w("image", "IMAGE", link=l_clip_from),
                   w("batch_index", "INT", widget={"name": "batch_index"}),
                   w("length", "INT", widget={"name": "length"})],
        "outputs": [{"localized_name": "IMAGE", "name": "IMAGE", "type": "IMAGE",
                     "links": [186, 188]}],
        "properties": {"cnr_id": "comfy-core", "ver": "0.32.0",
                       "Node name for S&R": "ImageFromBatch"},
        "widgets_values": [0, 1],
    }
    # Stereo Depth Scale: the V2V output is a real pair but shallow (measured 0.4 px eye step
    # at 640 wide); this brings the disparity spread to the film-class target without another
    # render.  Their documented order is Depth Scale BEFORE Stabilize.
    scale_id = new_id()
    l_scale = new_link()
    scale = {
        "id": scale_id, "type": "StereoDepthScale", "pos": [1150.0, 1180.0], "size": [330, 260],
        "flags": {}, "order": 34, "mode": 0,
        "inputs": [
            w("sbs", "IMAGE", link=357),
            w("arrangement", "COMBO", widget={"name": "arrangement"}),
            w("mode", "COMBO", widget={"name": "mode"}),
            w("target_spread_pct", "FLOAT", widget={"name": "target_spread_pct"}),
            w("depth_factor", "FLOAT", widget={"name": "depth_factor"}),
            w("smooth_px", "INT", widget={"name": "smooth_px"}),
            w("max_shift_px", "INT", widget={"name": "max_shift_px"}),
            w("convergence_px", "FLOAT", shape=7, widget={"name": "convergence_px"}),
            w("check", "BOOLEAN", shape=7, widget={"name": "check"}),
            w("refine", "BOOLEAN", shape=7, widget={"name": "refine"}),
        ],
        "outputs": [
            {"localized_name": "sbs", "name": "sbs", "type": "IMAGE", "links": [l_scale]},
            {"localized_name": "report", "name": "report", "type": "STRING", "links": None},
            {"localized_name": "spread_before_pct", "name": "spread_before_pct", "type": "FLOAT",
             "links": None},
            {"localized_name": "spread_after_pct", "name": "spread_after_pct", "type": "FLOAT",
             "links": None},
            {"localized_name": "spread_pass1_pct", "name": "spread_pass1_pct", "type": "FLOAT",
             "links": None},
        ],
        "properties": {"Node name for S&R": "StereoDepthScale"},
        "widgets_values": ["parallel / headset  [L|R]", "match a target spread (%)",
                           1.35, 1.0, 9, 12, 0.0, True, True],
    }
    note = {
        "id": new_id(), "type": "MarkdownNote", "pos": [-1350.0, 400.0], "size": [760, 520],
        "flags": {}, "order": 33, "mode": 0, "inputs": [], "outputs": [],
        "title": "Wan 2.2 V2V stereo: how to drive it", "properties": {},
        "widgets_values": [NOTE], "color": "#432", "bgcolor": "#653",
    }
    d["nodes"] += [vhs, encoder, first_frame, scale, note]
    d["links"] += [
        [l_clip_enc, vhs_id, 0, enc_id, 0, "IMAGE"],
        [l_clip_from, vhs_id, 0, from_id, 0, "IMAGE"],
        [l_vae_enc, 39, 0, enc_id, 1, "VAE"],
        [l_scale, scale_id, 0, 202, 0, "IMAGE"],      # Depth Scale -> Stabilize
    ]
    by[39]["outputs"][0]["links"] = (by[39]["outputs"][0].get("links") or []) + [l_vae_enc]

    # ------------------------------------------------------------------ rewire
    # the clip becomes the latent that the HIGH-noise sampler starts from; the HIGH->LOW
    # hand-off (link 113) and both samplers stay exactly as in your generation graph.
    # links here are [id, origin_id, origin_slot, target_id, target_slot, type]
    link185 = next(l for l in d["links"] if l[0] == 185)       # was WanFirstLastFrameToVideo.latent
    link185[1], link185[2] = enc_id, 0
    by[79]["outputs"][2]["links"] = None                       # that empty latent is no longer used
    # the I2V start frame + clip vision come from the clip's own first frame
    for lid, slot in ((186, 5), (188, 1)):
        lk = next(l for l in d["links"] if l[0] == lid)
        lk[1], lk[2] = from_id, 0
    by[83]["outputs"][0]["links"] = None                        # the old still LoadImage, now unused
    # partial denoise: the HIGH-noise sampler adds the noise and starts later; it hands the
    # leftover noise to the LOW-noise sampler, which keeps its original 4 -> end run
    by[57]["widgets_values"] = ["enable", 123456791, "fixed", 8, 1, "euler", "beta",
                                DEFAULT_START_STEP, 4, "enable"]
    # default: the high-noise model does NOT sample (its end == start) -- it only injects the
    # noise; the low-noise model denoises 2->8.  Measured: this is the REAL PAIR configuration.
    by[57]["mode"] = 0
    by[57]["widgets_values"] = ["enable", 123456791, "fixed", 8, 1, "euler", "beta",
                                DEFAULT_START_STEP, SPLIT, "enable"]
    by[57]["title"] = "HIGH noise -- owns start -> 4; its start_at_step is the drift dial"
    by[58]["widgets_values"] = ["disable", 123456791, "fixed", 8, 1, "euler", "beta",
                                SPLIT, 10000, "disable"]
    by[58]["title"] = "LOW noise -- 4 -> end, continues the leftover noise (Wan 2.2 split)"
    # conditioning must describe the same latent shape as the encoded clip
    by[79]["widgets_values"] = [W, H, LEN, 1]
    # the pairing output now goes through Depth Scale first; the stabilizer takes its output
    lk357 = next(l for l in d["links"] if l[0] == 357)
    lk357[3], lk357[4] = scale_id, 0
    stab_in = next(i for i in by[202]["inputs"] if i["name"] == "sbs")
    stab_in["link"] = l_scale
    by[202]["outputs"][0]["links"] = [361]            # unchanged: stabilize -> SBS combine

    d["last_node_id"], d["last_link_id"] = nid, last_link

    bk = WORK / f"backup_{datetime.now():%Y%m%d_%H%M%S}" / "workflows"
    bk.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC, bk / SRC.name)
    DST.write_bytes(json.dumps(d, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    print(f"backup: {bk / SRC.name}")
    print(f"wrote:  {DST}  ({DST.stat().st_size} bytes, {len(d['nodes'])} nodes, "
          f"{len(d['links'])} links)")
    print(f"  {vhs_id} VHS_LoadVideo  {CLIP}  force_rate 16 -> {LEN} frames @ {W}x{H}")
    print(f"  {enc_id} VAEEncode      clip frames -> the high-noise sampler's latent (link 185)")
    print(f"  {from_id} ImageFromBatch frame 0 -> start_image + clip vision")
    print(f"  high-noise 57: add_noise enable, {DEFAULT_START_STEP} -> {SPLIT} "
          f"(denoise {1 - DEFAULT_START_STEP / 8:.2f}), leftover noise ->")
    print(f"  low-noise  58: {SPLIT} -> end, add_noise disable (Wan 2.2 two-model split kept)")
    print(f"  {scale_id} StereoDepthScale match 1.35% spread, before the stabilizer")
    print(f"  79 WanFirstLastFrameToVideo: {W}x{H}, length {LEN} (latent T = "
          f"{(LEN - 1) // 4 + 1}, same as {LEN} encoded frames)")


if __name__ == "__main__":
    main()
