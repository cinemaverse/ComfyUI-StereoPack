"""Stereo eye pair nodes: split / join / swap a side-by-side stereo image.

Conventions
-----------
parallel / headset   [L|R]   left eye in the left half
cross-eyed           [R|L]   right eye in the left half

This is the same convention the DDD node uses (Create3DImage: `left_eye =
images[5]`, `right_eye = images[4]`, and its 'crosseyed' output is
`cat(right, left)`), and the one measured on real output in the comfy_ddd
project (near content moves right from the R eye to the L eye).

No dependencies beyond torch, which ComfyUI already has.
"""
import torch
import torch.nn.functional as F

ARRANGEMENTS = ("parallel / headset  [L|R]", "cross-eyed  [R|L]")


def _first_half_is_left(arrangement: str) -> bool:
    """'cross' anywhere in the value -> the first half is the RIGHT eye."""
    return "cross" not in str(arrangement).lower()


def _fit_batch(x: torch.Tensor, n: int) -> torch.Tensor:
    if x.shape[0] == n:
        return x
    if x.shape[0] == 1:
        return x.expand(n, -1, -1, -1)
    reps = (n + x.shape[0] - 1) // x.shape[0]
    return x.repeat(reps, 1, 1, 1)[:n]


class SplitStereoPair:
    """An SBS stereo image -> the two eye images on separate outputs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "arrangement": (ARRANGEMENTS, {"default": ARRANGEMENTS[0]}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("left", "right")
    FUNCTION = "split"
    CATEGORY = "image/stereo"
    DESCRIPTION = ("Split one side-by-side stereo image into its LEFT and RIGHT "
                   "eye images. Use the two outputs for first/last frame, or any "
                   "two-image node.")

    def split(self, image, arrangement):
        if image.shape[2] % 2 != 0:                 # h264 needs even dims anyway
            image = image[:, :, :image.shape[2] - 1, :]
        half = image.shape[2] // 2
        first, second = image[:, :, :half, :], image[:, :, half:, :]
        if _first_half_is_left(arrangement):
            return (first, second)
        return (second, first)


class SplitStereoBatch:
    """An SBS stereo image -> a batch of 2 images [left, right] on one output."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "arrangement": (ARRANGEMENTS, {"default": ARRANGEMENTS[0]}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("eyes",)
    FUNCTION = "split"
    CATEGORY = "image/stereo"
    DESCRIPTION = "Split an SBS stereo image into a batch of 2 images (left, right)."

    def split(self, image, arrangement):
        left, right = SplitStereoPair().split(image, arrangement)
        return (torch.cat((left, right), dim=0),)


class JoinStereoPair:
    """Two eye images -> one side-by-side stereo image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "left": ("IMAGE",),
                "right": ("IMAGE",),
                "arrangement": (ARRANGEMENTS, {"default": ARRANGEMENTS[0]}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "join"
    CATEGORY = "image/stereo"
    DESCRIPTION = "Join a LEFT and a RIGHT eye image into one SBS stereo image."

    def join(self, left, right, arrangement):
        if left.shape[1:3] != right.shape[1:3]:
            right = F.interpolate(right.movedim(-1, 1), size=tuple(left.shape[1:3]),
                                  mode="bilinear", align_corners=False).movedim(1, -1)
        n = max(left.shape[0], right.shape[0])
        left, right = _fit_batch(left, n), _fit_batch(right, n)
        if _first_half_is_left(arrangement):
            return (torch.cat((left, right), dim=2),)
        return (torch.cat((right, left), dim=2),)


class SwapStereoPair:
    """Flip a stereo image between [L|R] and [R|L] (or fix a crossed pair)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "swap"
    CATEGORY = "image/stereo"
    DESCRIPTION = "Swap the two halves of an SBS stereo image (L|R <-> R|L)."

    def swap(self, image):
        half = image.shape[2] // 2
        return (torch.cat((image[:, :, half:, :], image[:, :, :half, :]), dim=2),)


NODE_CLASS_MAPPINGS = {
    "SplitStereoPair": SplitStereoPair,
    "SplitStereoBatch": SplitStereoBatch,
    "JoinStereoPair": JoinStereoPair,
    "SwapStereoPair": SwapStereoPair,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SplitStereoPair": "Split Stereo Pair (SBS -> left + right)",
    "SplitStereoBatch": "Split Stereo Pair -> batch of 2",
    "JoinStereoPair": "Join Stereo Pair (left + right -> SBS)",
    "SwapStereoPair": "Swap Stereo Pair ([L|R] <-> [R|L])",
}
