"""Turn a generated video into a stereo (side-by-side) video.

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
"""
from __future__ import annotations

import torch

_CAT = "Stereo"


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


class StereoPairFrames:
    """Pair consecutive frames of a clip as the two eyes of a stereo image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "The generated clip's frames, in order."}),
            "viewing": (["cross-eyed  (left = frame 1, right = frame 2)",
                         "parallel / headset  (left = frame 2, right = frame 1)"], {
                "tooltip": "Which eye order matches how you watch it. Measured on real "
                           "output: frame1|frame2 reads as CROSS-EYED, so that is the "
                           "default. A headset needs the halves swapped -- pick "
                           "parallel / headset for that."}),
            "unpaired_last": (["drop (leave it alone)", "black partner",
                               "duplicate last frame"], {
                "tooltip": "What to do when the clip has an odd frame count, so the "
                           "final frame has no partner. 'drop' simply leaves it out."}),
            "pad_to_even": ("BOOLEAN", {"default": True,
                "tooltip": "Pad to an even width/height if needed. h264 with yuv420p "
                           "cannot encode an odd dimension."}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT")
    RETURN_NAMES = ("sbs", "second_eye", "pairs", "dropped")
    FUNCTION = "pair"
    CATEGORY = _CAT

    def pair(self, images, viewing, unpaired_last, pad_to_even):
        n = int(images.shape[0])
        if n < 2:
            raise ValueError(f"StereoPairFrames needs at least 2 frames, got {n}.")

        first = images[0::2]      # frame 1, 3, 5, ...  (indices 0, 2, 4)
        second = images[1::2]     # frame 2, 4, 5, ...  (indices 1, 3, 5)

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
        sbs = torch.cat([L, R], dim=2)                   # BHWC -> concat on width

        if pad_to_even:
            h, w = sbs.shape[1], sbs.shape[2]
            ph, pw = h % 2, w % 2
            if ph or pw:
                sbs = torch.nn.functional.pad(sbs, (0, 0, 0, pw, 0, ph), mode="replicate")

        return (sbs, second, pairs, dropped)


NODE_CLASS_MAPPINGS = {"StereoPairFrames": StereoPairFrames}
NODE_DISPLAY_NAME_MAPPINGS = {"StereoPairFrames": "Stereo Pair Frames (consecutive -> L|R)"}
