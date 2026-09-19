#!/usr/bin/env python
"""
sbs_segments_to_dataset.py -- streaming SBS movie -> interleaved same-instant clips.

Unlike interleave_sbs_to_dataset.py (which lets ffmpeg dump EVERY frame of the
whole input to PNGs first), this seeks to individual segments and decodes only
the frames it needs, straight through the pipe into the encoder. A 57-minute
4K SBS source therefore costs seconds per clip instead of ~350 GB of temp PNGs.

Output per clip: <out>/videos/<name>.mp4 + <name>.txt, ordered
    eye1(t0), eye2(t0), eye1(t1), eye2(t1), ...
i.e. consecutive frames are the SAME INSTANT, two eyes.

Eye order
---------
--source-layout says what the file is ([L|R] = lr, [R|L] = rl).
--first-eye      says which eye becomes frame index 0 of the output.
The DDD/ComfyUI convention is even index = RIGHT eye (node: right_eye=images[4],
left_eye=images[5]), and the shipped stereo140epochs LoRA is in that phase, so
the dataset must be emitted with --first-eye right to stay compatible.
Naming the two axes separately stops a silent depth inversion.

QC per clip: vertical misalignment (dy), disparity spread, temporal motion, and
a FLAT flag -- a shot whose two halves are identical (a mono insert) has almost
zero disparity spread and must not be trained on.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np

FPS = 24000 / 1001


def probe(movie):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=width,height,r_frame_rate",
                          "-of", "csv=p=0", movie],
                         stdout=subprocess.PIPE, check=True).stdout.decode().strip()
    w, h, fr = out.split(",")
    return int(w), int(h), fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--movie", required=True)
    ap.add_argument("--candidates", required=True,
                    help="JSON list of {t0,t1} (as written by the shot-selection stage)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--caption", required=True)
    ap.add_argument("--source-layout", choices=["lr", "rl"], default="lr")
    ap.add_argument("--first-eye", choices=["right", "left"], default="right")
    ap.add_argument("--frames", type=int, default=81,
                    help="interleaved frames per clip (WAN wants 4k+1)")
    ap.add_argument("--source-stride", type=int, default=1,
                    help="take every k-th SOURCE frame as the next instant. k=1 puts only "
                         "1/24 s of action in each instant (the model then reproduces 0.34 px "
                         "of motion per instant and can satisfy the same-instant pair by "
                         "freezing); k=3 gives 1/8 s per instant = natural motion at 8 stereo "
                         "fps playback. Pairs stay exactly same-instant either way.")
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--crf", type=int, default=12)
    ap.add_argument("--pad", type=float, default=0.5,
                    help="skip this many seconds at the start of each shot (transitions)")
    ap.add_argument("--exclude", default="", help="comma-separated candidate indices to skip")
    ap.add_argument("--auto-align", action="store_true",
                    help="measure each clip's vertical misalignment and crop the two eyes into "
                         "alignment (the source is only approx. vertically rectified)")
    ap.add_argument("--name-prefix", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    W, H, _ = probe(args.movie)
    hw = W // 2
    src_needed = args.frames // 2 + 1          # 81 interleaved <- 41 source frames

    cands = json.load(open(args.candidates))
    excluded = {int(x) for x in args.exclude.split(",") if x.strip() != ""}

    vdir = os.path.join(args.out, "videos")
    if not args.dry_run:
        os.makedirs(vdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.movie))[0]
    prefix = (args.name_prefix + "_") if args.name_prefix else ""

    manifest, rows = [], []
    for k, c in enumerate(cands):
        if k in excluded:
            print("skip #%d (excluded)" % k, flush=True)
            continue
        t0 = c["t0"] + args.pad
        if t0 + src_needed / FPS > c["t1"]:
            t0 = max(c["t0"], c["t1"] - src_needed / FPS)
        if args.dry_run:
            print("#%2d  t=%.2f  (shot %.1f-%.1f, %.1f s)" % (k, t0, c["t0"], c["t1"], c["dur"]))
            continue

        # ---- optional pre-pass: measure this shot's vertical misalignment
        shift, toe_in = 0, False
        if args.auto_align:
            raw1 = subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-ss", "%.3f" % t0, "-i", args.movie,
                 "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                stdout=subprocess.PIPE, check=True).stdout
            if len(raw1) >= W * H * 3:
                f1 = np.frombuffer(raw1[:W * H * 3], np.uint8).reshape(H, W, 3)
                a1 = cv2.resize(f1[:, :hw], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                b1 = cv2.resize(f1[:, hw:], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                fl = cv2.calcOpticalFlowFarneback(cv2.cvtColor(a1, cv2.COLOR_BGR2GRAY),
                                                  cv2.cvtColor(b1, cv2.COLOR_BGR2GRAY),
                                                  None, 0.5, 3, 21, 3, 5, 1.1, 0)
                dyy = fl[..., 1] * 2
                g1 = cv2.cvtColor(a1, cv2.COLOR_BGR2GRAY)
                tx = np.sqrt(cv2.Sobel(g1, cv2.CV_32F, 1, 0, 3) ** 2 + cv2.Sobel(g1, cv2.CV_32F, 0, 1, 3) ** 2)
                mm = tx > np.percentile(tx, 70)
                med_signed = float(np.median(dyy[mm]))
                med_abs = float(np.median(np.abs(dyy[mm])))
                # A constant vertical shift is only a valid model when the residual
                # is dominated by a constant offset. On a toe-in (converged) rig the
                # vertical component changes sign across the frame (keystone) and the
                # signed median collapses to ~0 -- shifting then makes things worse.
                if med_abs > 2.0 and abs(med_signed) >= 0.6 * med_abs:
                    shift = int(round(med_signed))
                    shift = max(-24, min(24, shift))
                    toe_in = False
                else:
                    shift = 0
                    toe_in = med_abs > 2.0

        Hout = H - abs(shift)
        if Hout % 2:
            Hout -= 1                        # yuv420p needs even dimensions
        ry0, ry1 = (0, Hout) if shift >= 0 else (-shift, -shift + Hout)
        sy0, sy1 = (shift, shift + Hout) if shift >= 0 else (0, Hout)

        name = "%s%03d_t%05d" % (prefix, k, int(t0))
        out_mp4 = os.path.join(vdir, name + ".mp4")

        # ---- decode only this segment, stream frames straight to the encoder
        dec_cmd = ["ffmpeg", "-nostdin", "-v", "error", "-ss", "%.3f" % t0, "-i", args.movie]
        if args.source_stride > 1:
            dec_cmd += ["-vf", "select='not(mod(n\\,%d))'" % args.source_stride, "-fps_mode", "passthrough"]
        dec_cmd += ["-frames:v", str(src_needed), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        dec = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE, bufsize=10 ** 8)
        enc = subprocess.Popen(
            ["ffmpeg", "-nostdin", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", "%dx%d" % (hw, Hout), "-framerate", str(args.fps), "-i", "-",
             "-c:v", "libx264", "-crf", str(args.crf), "-pix_fmt", "yuv420p", out_mp4, "-y"],
            stdin=subprocess.PIPE, bufsize=10 ** 8)

        fsz = W * H * 3
        got, emitted, qc_src = 0, 0, []
        while True:
            raw = dec.stdout.read(fsz * 4)
            if not raw or len(raw) < fsz:
                break
            n = len(raw) // fsz
            for i in range(n):
                f = np.frombuffer(raw[i * fsz:(i + 1) * fsz], np.uint8).reshape(H, W, 3)
                hA, hB = f[ry0:ry1, :hw], f[sy0:sy1, hw:]
                L, R = (hA, hB) if args.source_layout == "lr" else (hB, hA)
                first, second = (R, L) if args.first_eye == "right" else (L, R)
                enc.stdin.write(np.ascontiguousarray(first).tobytes())
                emitted += 1
                if emitted >= args.frames:
                    break
                enc.stdin.write(np.ascontiguousarray(second).tobytes())
                emitted += 1
                if emitted >= args.frames:
                    break
                if len(qc_src) < 41:
                    qc_src.append(f)
                got += 1
            if emitted >= args.frames:
                break
        dec.stdout.close()
        dec.wait()
        enc.stdin.close()
        enc.wait()

        # ---- QC on this segment (residual dy after alignment)
        if len(qc_src) >= 4:
            a = [cv2.resize(x[ry0:ry1, :hw], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) for x in qc_src[:9]]
            b = [cv2.resize(x[sy0:sy1, hw:], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) for x in qc_src[:9]]
            gr = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            f01 = cv2.calcOpticalFlowFarneback(gr(a[0]), gr(b[0]), None, 0.5, 3, 21, 3, 5, 1.1, 0)
            dx = f01[..., 0] * 2
            dy = f01[..., 1] * 2
            g = gr(a[0])
            tex = np.sqrt(cv2.Sobel(g, cv2.CV_32F, 1, 0, 3) ** 2 + cv2.Sobel(g, cv2.CV_32F, 0, 1, 3) ** 2)
            m = tex > np.percentile(tex, 70)
            spread = float(np.percentile(dx[m], 97) - np.percentile(dx[m], 3))
            dy_abs = float(np.median(np.abs(dy[m])))
            mags = []
            for i in range(len(a) - 1):
                fa = cv2.calcOpticalFlowFarneback(gr(a[i]), gr(a[i + 1]), None, 0.5, 3, 21, 3, 5, 1.1, 0)
                mags.append(float(np.median(np.sqrt(fa[..., 0] ** 2 + fa[..., 1] ** 2)[m]) * 2))
            motion = float(np.median(mags))
            flat = spread < 2.5
        else:
            spread = dy_abs = motion = 0.0
            flat = False

        row = dict(k=k, name=name, t0=round(t0, 2), src_frames=got, out_frames=emitted,
                   first_eye=args.first_eye, source_layout=args.source_layout,
                   source_stride=args.source_stride,
                   dy_shift=shift, toe_in=toe_in, height=Hout,
                   spread=round(spread, 1), dy_abs=round(dy_abs, 2),
                   motion=round(motion, 3), flat=flat, lum=round(float(qc_src[0].mean()), 1) if qc_src else 0.0)
        rows.append(row)
        with open(os.path.join(vdir, name + ".txt"), "w", encoding="utf-8") as fh:
            fh.write(args.caption + "\n")
        print(json.dumps(row), flush=True)

    if not args.dry_run:
        json.dump(rows, open(os.path.join(args.out, "clips_manifest.json"), "w"), indent=1)
        nf = [r for r in rows if r["flat"]]
        print("\n%d clips written to %s" % (len(rows), vdir))
        if nf:
            print("FLAT (mono insert, exclude from training): %s" % [r["name"] for r in nf])
        print("emit order index0=%s  (source layout %s)" % (args.first_eye, args.source_layout))


if __name__ == "__main__":
    main()