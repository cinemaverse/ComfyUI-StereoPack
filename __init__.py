"""ComfyUI-StereoPack -- three small, dependency-free node sets for stereo video.

    StereoPairFrames      consecutive frames -> one side-by-side pair (the core node), with a
                          head trim (skip_first / auto_extend), an eye-step / phase / band QC
                          readout, a one-line VERDICT (OK / OK PATCHY / REJECT ...) and
                          reject_bad_seed, which stops the graph on an unusable render
    StereoEyes            split / join / swap the two eyes of a side-by-side image
    StereoStabilizeSBS    temporal median per eye, removes the per-pixel boil
    StereoDepthScale      scale / auto-match the disparity -- depth, without touching motion

Node class names and their NODE_CLASS_MAPPINGS keys are unchanged from the separate
packs these modules came from (ComfyUI-StereoPair / -StereoEyes / -StereoStabilize),
so workflows written against those keep loading without edits. `StereoPairFrames` keeps
its original key AND its original widget order (viewing, unpaired_last, pad_to_even), so a
graph saved before the trim existed still resolves its settings; the new inputs are appended
after them. Only the display name changed (it now says "+ head trim / QC").

No new dependencies: torch, numpy and opencv are already in the ComfyUI environment.
"""

from .nodes.stereo_pair import (
    NODE_CLASS_MAPPINGS as _PAIR_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _PAIR_NAMES,
)
from .nodes.stereo_eyes import (
    NODE_CLASS_MAPPINGS as _EYES_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _EYES_NAMES,
)
from .nodes.stereo_stabilize import (
    NODE_CLASS_MAPPINGS as _STAB_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _STAB_NAMES,
)
from .nodes.stereo_depth import (
    NODE_CLASS_MAPPINGS as _DEPTH_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _DEPTH_NAMES,
)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for _mapping in (_PAIR_CLASSES, _EYES_CLASSES, _STAB_CLASSES, _DEPTH_CLASSES):
    NODE_CLASS_MAPPINGS.update(_mapping)
for _mapping in (_PAIR_NAMES, _EYES_NAMES, _STAB_NAMES, _DEPTH_NAMES):
    NODE_DISPLAY_NAME_MAPPINGS.update(_mapping)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
