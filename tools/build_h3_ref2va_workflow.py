"""Build H3_Ref2VA_VideoWF_3D_Stereo.json from the existing H3 FL2VA stereo graph.

Changes (nothing is removed -- the old path stays in place, bypassed, for A/B):
  * inside the MiniMax H3 subgraph: a MiniMaxH3ReferenceToVideo node, wired to the same
    CLIP / video-VAE / audio-VAE / prompt / width / height / length, and to a NEW subgraph
    input `ref_video` (the reference frame batch);
  * its `positive` / `LATENT` outputs take over the two links that used to come from
    MiniMaxH3ImageToVideo (BasicGuider.conditioning and the sampler's latent_image);
  * MiniMaxH3ImageToVideo is set to bypass (Ctrl+B to flip the two back and forth);
  * at the top level: a VHS_LoadVideo loading the GIF at force_rate 24, linked into the
    subgraph's new `ref_video` input.

Representations were taken from the running frontend itself (autogrow inputs serialise as
`ref_videos.ref_video_0` with a `label`), not guessed.
"""
from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]     # repo root (this script lives in tools/)
SRC = WORK / "workflows" / "H3_VideoWF_3D_Stereo_FL2VA.json"
DST = WORK / "workflows" / "H3_Ref2VA_VideoWF_3D_Stereo.json"
GIF = "355932194.gif"

PROMPT = ("<Video 1>: preserve everything in this clip and continue with clip's natural "
          "movements.\n\n"
          "st3r30 footage: each moment of the scene is captured from a slightly shifted "
          "position at the same instant, frame by frame, while the action continues normally "
          "across the clip.")


def main() -> None:
    d = json.loads(SRC.read_text(encoding="utf-8"))
    sg = d["definitions"]["subgraphs"][0]
    inst = next(n for n in d["nodes"] if n["type"] == sg["id"])
    by_id = {n["id"]: n for n in sg["nodes"]}

    # ---------------------------------------------------------------- ids
    sg_link0 = max(l["id"] for l in sg["links"])
    ref_id = max(n["id"] for n in sg["nodes"]) + 1
    L = list(range(sg_link0 + 1, sg_link0 + 9))
    top_node = d["last_node_id"] + 1
    top_link = d["last_link_id"] + 1

    # width/height/length widgets: mirror the existing conditioning node
    old = by_id[104]
    w, h, length = old["widgets_values"][1:4]

    # ---------------------------------------------------------------- node
    def link(name, type_, **kw):
        e = {"localized_name": name, "name": name, "type": type_, "link": None}
        e.update(kw)
        return e

    ref = {
        "id": ref_id,
        "type": "MiniMaxH3ReferenceToVideo",
        "pos": [-1780.0, 4623.0],
        "size": [400, 304],
        "flags": {},
        "order": 13,
        "mode": 0,
        "inputs": [
            {"localized_name": "clip", "name": "clip", "type": "CLIP", "link": L[0]},
            {"localized_name": "vae", "name": "vae", "type": "VAE", "link": L[1]},
            {"localized_name": "audio_vae", "name": "audio_vae", "type": "VAE", "link": L[2]},
            {"label": "ref_image_0", "localized_name": "ref_images.ref_image_0",
             "name": "ref_images.ref_image_0", "shape": 7, "type": "IMAGE", "link": None},
            {"label": "ref_video_0", "localized_name": "ref_videos.ref_video_0",
             "name": "ref_videos.ref_video_0", "shape": 7, "type": "IMAGE", "link": L[7]},
            {"label": "ref_video_audio_0", "localized_name": "ref_video_audios.ref_video_audio_0",
             "name": "ref_video_audios.ref_video_audio_0", "shape": 7, "type": "AUDIO", "link": None},
            {"label": "ref_audio_0", "localized_name": "ref_audios.ref_audio_0",
             "name": "ref_audios.ref_audio_0", "shape": 7, "type": "AUDIO", "link": None},
            {"localized_name": "prompt", "name": "prompt", "type": "STRING",
             "widget": {"name": "prompt"}, "link": L[3]},
            {"localized_name": "width", "name": "width", "type": "INT",
             "widget": {"name": "width"}, "link": L[4]},
            {"localized_name": "height", "name": "height", "type": "INT",
             "widget": {"name": "height"}, "link": L[5]},
            {"localized_name": "length", "name": "length", "type": "INT",
             "widget": {"name": "length"}, "link": L[6]},
            {"localized_name": "ref_image_size", "name": "ref_image_size", "type": "COMBO",
             "widget": {"name": "ref_image_size"}, "link": None},
        ],
        "outputs": [
            {"localized_name": "positive", "name": "positive", "type": "CONDITIONING",
             "links": [187]},
            {"localized_name": "LATENT", "name": "LATENT", "type": "LATENT", "links": [188]},
        ],
        "properties": {"cnr_id": "comfy-core", "ver": "0.32.0",
                       "Node name for S&R": "MiniMaxH3ReferenceToVideo"},
        "widgets_values": [PROMPT, w, h, length, "match"],
    }
    sg["nodes"].append(ref)

    # The effective prompt is the subgraph INSTANCE's prompt widget: it feeds the internal
    # `prompt` input that both conditioning nodes read.  Set it to the ref2va wording (with the
    # `<Video 1>` tag the reference node needs) so it is still edited in one place at the top level.
    inst["widgets_values"][0] = PROMPT

    # ---------------------------------------------------------------- internal links
    sg["links"] += [
        {"id": L[0], "origin_id": 13, "origin_slot": 0, "target_id": ref_id, "target_slot": 0,
         "type": "CLIP"},
        {"id": L[1], "origin_id": 11, "origin_slot": 0, "target_id": ref_id, "target_slot": 1,
         "type": "VAE"},
        {"id": L[2], "origin_id": 24, "origin_slot": 0, "target_id": ref_id, "target_slot": 2,
         "type": "VAE"},
        {"id": L[3], "origin_id": -10, "origin_slot": 2, "target_id": ref_id, "target_slot": 7,
         "type": "STRING"},
        {"id": L[4], "origin_id": -10, "origin_slot": 3, "target_id": ref_id, "target_slot": 8,
         "type": "INT"},
        {"id": L[5], "origin_id": -10, "origin_slot": 4, "target_id": ref_id, "target_slot": 9,
         "type": "INT"},
        {"id": L[6], "origin_id": 107, "origin_slot": 1, "target_id": ref_id, "target_slot": 10,
         "type": "INT"},
        {"id": L[7], "origin_id": -10, "origin_slot": 11, "target_id": ref_id, "target_slot": 4,
         "type": "IMAGE"},
    ]
    for src_id, slot, lid in ((13, 0, L[0]), (11, 0, L[1]), (24, 0, L[2]), (107, 1, L[6])):
        out = by_id[src_id]["outputs"][slot]
        out["links"] = (out.get("links") or []) + [lid]

    # the two consumer links now start at the reference node; the old node keeps nothing
    for lid, slot in ((187, 0), (188, 1)):
        lk = next(x for x in sg["links"] if x["id"] == lid)
        lk["origin_id"] = ref_id
        lk["origin_slot"] = slot
    old["outputs"][0]["links"] = None
    old["outputs"][1]["links"] = None
    old["mode"] = 4                      # bypassed: Ctrl+B to A/B against the ref path

    # ---------------------------------------------------------------- new subgraph input
    sg["inputs"].append({"id": str(uuid.uuid4()), "name": "ref_video", "type": "IMAGE",
                         "linkIds": [top_link], "pos": [-2456, 4964]})
    inst["inputs"].append({"name": "ref_video", "shape": 7, "type": "IMAGE", "link": top_link})

    # ---------------------------------------------------------------- top level loader
    vhs = {
        "id": top_node,
        "type": "VHS_LoadVideo",
        "pos": [-2760.0, 5430.0],
        "size": [470, 310],
        "flags": {},
        "order": 15,
        "mode": 0,
        "inputs": [
            link("meta_batch", "VHS_BatchManager", shape=7),
            link("vae", "VAE", shape=7),
            link("video", "COMBO", widget={"name": "video"}),
            link("force_rate", "FLOAT", widget={"name": "force_rate"}),
            link("custom_width", "INT", widget={"name": "custom_width"}),
            link("custom_height", "INT", widget={"name": "custom_height"}),
            link("frame_load_cap", "INT", widget={"name": "frame_load_cap"}),
            link("skip_first_frames", "INT", widget={"name": "skip_first_frames"}),
            link("select_every_nth", "INT", widget={"name": "select_every_nth"}),
            link("format", "COMBO", widget={"name": "format"}),
        ],
        "outputs": [
            {"localized_name": "IMAGE", "name": "IMAGE", "type": "IMAGE", "links": [top_link]},
            {"localized_name": "frame_count", "name": "frame_count", "type": "INT", "links": []},
            {"localized_name": "audio", "name": "audio", "type": "AUDIO", "links": []},
            {"localized_name": "video_info", "name": "video_info", "type": "VHS_VIDEOINFO",
             "links": []},
        ],
        "properties": {"cnr_id": "comfyui-videohelpersuite", "ver": "1.7.9",
                       "Node name for S&R": "VHS_LoadVideo"},
        "widgets_values": {
            "video": GIF, "force_rate": 24, "custom_width": 0, "custom_height": 0,
            "frame_load_cap": 0, "skip_first_frames": 0, "select_every_nth": 1,
            "format": "AnimateDiff", "choose video to upload": "image",
            "videopreview": {"hidden": False, "paused": False, "params": {
                "filename": GIF, "type": "input", "format": "image/gif", "force_rate": 24,
                "custom_width": 0, "custom_height": 0, "frame_load_cap": 0,
                "skip_first_frames": 0, "select_every_nth": 1}},
        },
    }
    d["nodes"].append(vhs)
    d["links"].append([top_link, top_node, 0, inst["id"], len(inst["inputs"]) - 1, "IMAGE"])
    d["last_node_id"] = top_node
    d["last_link_id"] = top_link

    # ---------------------------------------------------------------- backup + write
    bk = WORK / f"backup_{datetime.now():%Y%m%d_%H%M%S}" / "workflows"
    bk.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC, bk / SRC.name)
    DST.write_bytes(json.dumps(d, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    print(f"backup of the original: {bk / SRC.name}")
    print(f"wrote: {DST}  ({DST.stat().st_size} bytes)")
    print(f"  subgraph node {ref_id} MiniMaxH3ReferenceToVideo  (links {L})")
    print(f"  top-level node {top_node} VHS_LoadVideo '{GIF}' force_rate 24 -> link {top_link}")
    print(f"  MiniMaxH3ImageToVideo (subgraph node {old['id']}) -> mode 4 (bypassed)")
    print(f"  length from the duration widget ({inst['widgets_values'][3]} s) -> "
          f"the snap expression gives 124 frames")


if __name__ == "__main__":
    main()
