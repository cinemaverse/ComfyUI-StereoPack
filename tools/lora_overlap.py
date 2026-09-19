#!/usr/bin/env python
"""
lora_overlap.py -- how much of a base LoRA survives a fine-tune, as one number.

A LoRA's effective update for one module is  dW = (alpha/dim) * lora_up @ lora_down.
Comparing dW between two LoRAs answers "is this a refinement or a rewrite?":

    cos       <dW_new, dW_base> / (||dW_new|| ||dW_base||)
              1.00 = same update direction, only refined
              0.00 = an unrelated update that replaced the old one
    cos^2     the share of the normalised energy still doing the base's work
    |dW| rat  ||dW_new|| / ||dW_base||            did the update grow or shrink
    rel       ||dW_new - dW_base|| / ||dW_base||  how far it actually moved

Everything is aggregated by ENERGY over all modules (summing squared Frobenius norms),
not averaged per module, so large modules count for their real size. The Gram identities
avoid materialising 5120x5120 matrices:

    <dW_t, dW_0> = tr((B_t^T B_0)(A_0 A_t^T))
    ||dW||^2     = tr((B^T B)(A A^T))

Reminder when reading the numbers: a CONTINUATION accumulates. Training stage 2 from
your own stage-1 checkpoint means stage 2's file differs from the original base by both
stages' drift. To measure one stage's own drift, compare the checkpoint before that
stage against the one after it.

Usage: lora_overlap.py <base.safetensors> <trained.safetensors> [more.safetensors ...]
"""
import re
import sys
from collections import defaultdict

import numpy as np
import torch
from safetensors import safe_open


def load(path):
    out = {}
    with safe_open(path, "pt") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k).to(torch.float32).numpy()
    return out


def groups(d):
    g = defaultdict(dict)
    for k, v in d.items():
        m = re.match(
            r"(lora_unet_blocks_\d+)_(.+)\.(alpha|lora_down\.weight|lora_up\.weight)$", k)
        if m:
            g[m.group(1) + "_" + m.group(2)][m.group(3)] = v
    return {k: v for k, v in g.items() if "lora_down.weight" in v and "lora_up.weight" in v}


def scale(t):
    """alpha / dim, the factor the stored alpha implies (1.0 when alpha == dim)."""
    A = t["lora_down.weight"]
    alpha = t.get("alpha")
    return float(alpha.ravel()[0]) / A.shape[0] if alpha is not None else 1.0


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    base, targets = sys.argv[1], sys.argv[2:]
    G0 = groups(load(base))
    print("base: %s   (%d modules)" % (base.replace("\\", "/").split("/")[-1], len(G0)))

    # the base side is prepared once and reused for every target
    B0 = {k: t["lora_up.weight"] for k, t in G0.items()}
    A0 = {k: t["lora_down.weight"] for k, t in G0.items()}
    S0 = {k: scale(t) for k, t in G0.items()}
    n0 = {k: float(np.trace(((B0[k] * S0[k]).T @ (B0[k] * S0[k])).astype(np.float64)
                            @ (A0[k] @ A0[k].T).astype(np.float64))) for k in G0}

    print()
    print("%-30s %7s %7s %9s %8s %8s %8s" %
          ("trained file", "cos", "cos^2", "|dW| rat", "rel", "mod cos", "min cos"))
    print("-" * 86)
    for tp in targets:
        Gt = groups(load(tp))
        E0 = E1 = C = 0.0
        coss = []
        for k, t in Gt.items():
            if k not in G0:
                continue
            A, B = t["lora_down.weight"], t["lora_up.weight"]
            s = scale(t)
            Mt = ((B * s).T @ (B * s)).astype(np.float64)
            nt = float(np.trace(Mt @ (A @ A.T).astype(np.float64)))
            if n0[k] <= 0 or nt <= 0:
                continue
            cross = float(np.trace(
                ((B.T @ B0[k]).astype(np.float64) * (S0[k] * s))
                @ (A0[k] @ A.T).astype(np.float64)))
            E0 += n0[k]
            E1 += nt
            C += cross
            coss.append(cross / np.sqrt(n0[k] * nt))
        if not coss:
            print("%-30s  (no comparable modules)" % tp.split("/")[-1])
            continue
        cos = C / np.sqrt(E0 * E1)
        rel = np.sqrt(max(0.0, E1 + E0 - 2 * C)) / np.sqrt(E0)
        print("%-30s %7.3f %7.3f %9.3f %8.3f %8.3f %8.3f" %
              (tp.replace("\\", "/").split("/")[-1].replace(".safetensors", ""),
               cos, cos * cos, np.sqrt(E1 / E0), rel, float(np.mean(coss)),
               float(np.min(coss))))
    print("-" * 86)
    print("cos   = global update-direction overlap with the base (energy-weighted)")
    print("cos^2 = share of the normalised energy still doing the base's work")
    print("rel   = ||dW_new - dW_base|| / ||dW_base||   (how far it moved)")
    print("mod cos / min cos = the same cosine per module, averaged / worst case")


if __name__ == "__main__":
    main()
