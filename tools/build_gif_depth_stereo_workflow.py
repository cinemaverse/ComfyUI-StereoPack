"""Build GIF_DepthStereo_SBS.json -- the frame-exact route: every GIF frame is preserved.

Why this exists: MiniMax H3 (both ImageToVideo and ReferenceToVideo) *generates*, so the
output can only ever resemble the reference clip.  Measured on this machine: the reference
conditioning is context (Qwen sees the video sampled at 2 fps, the latents ride through every
sampling step) -- it is not a temporal template, and no prompt makes it one.  If the goal is
"the clip, with stereo added" then the only route that cannot drift is depth + warp, which is
what this graph does:

    VHS_LoadVideo (the GIF, native fps)
      -> VideoDepthAnythingDepth   (temporally consistent depth per frame)
      -> DepthToStereo             (one SBS pair per source frame)
      -> Stereo Stabilize          (optional: removes depth-map flicker)
      -> VHS_VideoCombine          (frame_rate = the GIF's own rate)

Output frame count == input frame count, in order, at the original timing.

Node/widget shapes are copied from the running ComfyUI (the VHS_VideoCombine node is cloned
from the H3 workflow, so the VHS schema is exact).
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
SRC = WORK / "workflows" / "H3_VideoWF_3D_Stereo_FL2VA.json"
DST = WORK / "workflows" / "GIF_DepthStereo_SBS.json"

GIF = "965734544.gif"      # 41 frames, 20 fps, 2.05 s, 286x338 (measured)
GIF_FPS = 20

NOTE = """### Frame-exact stereo from a GIF / video

Every frame of the source is preserved, in order, at the original timing — the second eye is
synthesised from a **depth map**, not re-rendered. That is the difference from the H3 routes:
H3 generates (ImageToVideo: 1-2 keyframes; ReferenceToVideo: context only), so it drifts.

* `VHS_LoadVideo` — `force_rate 0` keeps the source's own rate (this GIF: 41 frames @ 20 fps).
  Set `frame_load_cap` to render only the first N frames.
* `VideoDepthAnythingDepth` — `normalise = global (temporally consistent)`: per-frame
  normalisation makes the depth (and therefore the stereo) flicker. `input_size 518` is the
  trained default; higher = finer, much slower.
* `DepthToStereo` — `strength_pct` 1–2 % of eye width is comfortable viewing. `max_slope`
  limits how far an occlusion edge may stretch (lower = fewer artifacts, less depth).
  `disp_smooth` smooths the disparity map spatially.
* `layout` — `sbs_crosseyed` = [R|L], which is what the StereoPack's H3 chain delivers, so
  your existing viewing setup stays valid. Use `sbs (parallel, headset)` for a VR headset.
* `Stereo Stabilize` — bypass it (Ctrl+B) for a raw look; with `radius 1` + `flat_only 60` it
  removes depth-map flicker at no detail cost.
* `VHS_VideoCombine.frame_rate` must stay at the source rate (20 here), otherwise the clip
  plays fast or slow.
"""


def main() -> None:
    base = json.loads(SRC.read_text(encoding="utf-8"))
    # the VHS_VideoCombine node, as a template for a schema-exact clone
    tmpl = next(n for n in base["nodes"] if n["type"] == "VHS_VideoCombine")
    combine = json.loads(json.dumps(tmpl))
    combine["id"] = 5
    combine["pos"] = [1180.0, 40.0]
    # the clone carries the template's link ids -- none of them exist in this graph
    for i, inp in enumerate(combine["inputs"]):
        inp["link"] = None
    for out in combine.get("outputs") or []:
        if out.get("links"):
            out["links"] = []
    combine["widgets_values"]["frame_rate"] = GIF_FPS
    combine["widgets_values"]["filename_prefix"] = "Stereo/gif_depth_sbs"
    combine["widgets_values"]["videopreview"] = {"hidden": False, "paused": False, "params": {}}

    def w(name, type_, **kw):
        e = {"localized_name": name, "name": name, "type": type_}
        e.update(kw)
        e.setdefault("link", None)
        return e

    nodes = [
        {
            "id": 1, "type": "VHS_LoadVideo", "pos": [-620.0, 40.0], "size": [470, 310],
            "flags": {}, "order": 0, "mode": 0,
            "inputs": [
                w("meta_batch", "VHS_BatchManager", shape=7),
                w("vae", "VAE", shape=7),
                {"localized_name": "video", "name": "video", "type": "COMBO",
                 "widget": {"name": "video"}, "link": None},
                {"localized_name": "force_rate", "name": "force_rate", "type": "FLOAT",
                 "widget": {"name": "force_rate"}, "link": None},
                {"localized_name": "custom_width", "name": "custom_width", "type": "INT",
                 "widget": {"name": "custom_width"}, "link": None},
                {"localized_name": "custom_height", "name": "custom_height", "type": "INT",
                 "widget": {"name": "custom_height"}, "link": None},
                {"localized_name": "frame_load_cap", "name": "frame_load_cap", "type": "INT",
                 "widget": {"name": "frame_load_cap"}, "link": None},
                {"localized_name": "skip_first_frames", "name": "skip_first_frames", "type": "INT",
                 "widget": {"name": "skip_first_frames"}, "link": None},
                {"localized_name": "select_every_nth", "name": "select_every_nth", "type": "INT",
                 "widget": {"name": "select_every_nth"}, "link": None},
                {"localized_name": "format", "name": "format", "type": "COMBO",
                 "widget": {"name": "format"}, "link": None},
            ],
            "outputs": [
                {"localized_name": "IMAGE", "name": "IMAGE", "type": "IMAGE", "links": [1, 2]},
                {"localized_name": "frame_count", "name": "frame_count", "type": "INT", "links": []},
                {"localized_name": "audio", "name": "audio", "type": "AUDIO", "links": []},
                {"localized_name": "video_info", "name": "video_info", "type": "VHS_VIDEOINFO",
                 "links": []},
            ],
            "properties": {"cnr_id": "comfyui-videohelpersuite", "ver": "1.7.9",
                           "Node name for S&R": "VHS_LoadVideo"},
            "widgets_values": {
                "video": GIF, "force_rate": 0, "custom_width": 0, "custom_height": 0,
                "frame_load_cap": 0, "skip_first_frames": 0, "select_every_nth": 1,
                "format": "AnimateDiff", "choose video to upload": "image",
                "videopreview": {"hidden": False, "paused": False, "params": {
                    "filename": GIF, "type": "input", "format": "image/gif", "force_rate": 0,
                    "custom_width": 0, "custom_height": 0, "frame_load_cap": 0,
                    "skip_first_frames": 0, "select_every_nth": 1}},
            },
        },
        {
            "id": 2, "type": "VideoDepthAnythingEstimate", "pos": [-100.0, -260.0],
            "size": [420, 320], "flags": {}, "order": 2, "mode": 0,
            "inputs": [
                {"localized_name": "image", "name": "image", "type": "IMAGE", "link": 1},
                *[{"localized_name": n, "name": n, "type": t, "widget": {"name": n}, "link": None}
                  for n, t in (("encoder", "COMBO"), ("input_size", "INT"), ("normalise", "COMBO"),
                               ("far_percentile", "FLOAT"), ("near_percentile", "FLOAT"),
                               ("invert", "BOOLEAN"), ("fp32", "BOOLEAN"), ("device", "COMBO"))],
            ],
            "outputs": [{"localized_name": "IMAGE", "name": "IMAGE", "type": "IMAGE", "links": [3]}],
            "properties": {"Node name for S&R": "VideoDepthAnythingEstimate"},
            "widgets_values": ["Large (vitl, 382M)", 518, "global (temporally consistent)",
                               1.0, 99.5, False, False, "auto"],
        },
        {
            "id": 3, "type": "DepthToStereo", "pos": [390.0, 40.0], "size": [420, 340],
            "flags": {}, "order": 3, "mode": 0,
            "inputs": [
                {"localized_name": "image", "name": "image", "type": "IMAGE", "link": 2},
                {"localized_name": "depth", "name": "depth", "type": "IMAGE", "link": 3},
                *[{"localized_name": n, "name": n, "type": t, "widget": {"name": n}, "link": None}
                  for n, t in (("strength_pct", "FLOAT"), ("disp_smooth", "FLOAT"),
                               ("max_slope", "FLOAT"), ("layout", "COMBO"),
                               ("depth_is_normalised", "BOOLEAN"), ("far_percentile", "FLOAT"),
                               ("near_percentile", "FLOAT"), ("chunk", "INT"))],
            ],
            "outputs": [{"localized_name": "IMAGE", "name": "IMAGE", "type": "IMAGE", "links": [4]}],
            "properties": {"Node name for S&R": "DepthToStereo"},
            "widgets_values": [2.0, 1.0, 1.0, "sbs_crosseyed", True, 1.0, 99.5, 8],
        },
        {
            "id": 4, "type": "StereoStabilizeSBS", "pos": [860.0, 40.0], "size": [390, 210],
            "flags": {}, "order": 4, "mode": 0,
            "inputs": [
                {"localized_name": "sbs", "name": "sbs", "type": "IMAGE", "link": 4},
                *[{"localized_name": n, "name": n, "type": t,
                   **({"shape": 7} if opt else {}), "widget": {"name": n}, "link": None}
                  for n, t, opt in (("radius", "INT", False), ("level", "FLOAT", False),
                                    ("max_frames", "INT", True), ("check_stereo", "BOOLEAN", True),
                                    ("flat_only", "INT", True), ("flat_feather", "INT", True))],
            ],
            "outputs": [
                {"localized_name": "sbs", "name": "sbs", "type": "IMAGE", "links": [5]},
                {"localized_name": "report", "name": "report", "type": "STRING", "links": None},
                {"localized_name": "pairs", "name": "pairs", "type": "INT", "links": None},
            ],
            "properties": {"Node name for S&R": "StereoStabilizeSBS"},
            "widgets_values": [1, 1.0, 0, True, 60, 9],
        },
        combine,
        {
            "id": 6, "type": "MarkdownNote", "pos": [-620.0, 400.0], "size": [700, 420],
            "flags": {}, "order": 1, "mode": 0, "inputs": [], "outputs": [],
            "title": "Frame-exact stereo: how this works",
            "properties": {}, "widgets_values": [NOTE], "color": "#432", "bgcolor": "#653",
        },
    ]
    combine["inputs"][0]["link"] = 5
    combine["id"] = 5

    links = [
        [1, 1, 0, 2, 0, "IMAGE"],
        [2, 1, 0, 3, 0, "IMAGE"],
        [3, 2, 0, 3, 1, "IMAGE"],
        [4, 3, 0, 4, 0, "IMAGE"],
        [5, 4, 0, 5, 0, "IMAGE"],
    ]

    doc = {
        "id": base.get("id"), "revision": 0, "last_node_id": 6, "last_link_id": 5,
        "nodes": nodes, "links": links, "groups": [], "config": {},
        "extra": {"frontendVersion": (base.get("extra") or {}).get("frontendVersion", "1.48.7"),
                  "VHS_latentpreview": False, "VHS_latentpreviewrate": 0,
                  "VHS_MetadataImage": True, "VHS_KeepIntermediate": True,
                  "ds": {"scale": 1.0, "offset": [700, 420]}},
        "version": 0.4,
    }

    bk = WORK / f"backup_{datetime.now():%Y%m%d_%H%M%S}" / "workflows"
    bk.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC, bk / SRC.name)
    DST.write_bytes(json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    print(f"wrote: {DST}   ({DST.stat().st_size} bytes)")
    print(f"  {len(nodes)} nodes, {len(links)} links")
    for n in nodes:
        if n["type"] != "MarkdownNote":
            print(f"    {n['id']} {n['type']}")


if __name__ == "__main__":
    main()
