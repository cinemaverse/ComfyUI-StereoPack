"""Node implementations, one module per original pack.

These files are the original sources, moved without changes:

    stereo_pair.py       <- ComfyUI-StereoPair/nodes.py       (StereoPairFrames)
                            -- extended 2026-09-27: head trim (skip_first / auto_extend), a measured
                               eye-step / phase / band readout, and phase_lock (repairs a clip whose
                               eye order flips part-way through), merged in from the standalone
                               ComfyUI-StereoPairTrim pack. Same class, same mapping key, same
                               widget order for the three original inputs.
    stereo_eyes.py       <- ComfyUI-StereoEyes/nodes.py       (Split/Join/Swap)
    stereo_stabilize.py  <- ComfyUI-StereoStabilize/nodes.py  (StereoStabilizeSBS)
"""
