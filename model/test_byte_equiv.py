"""S1/S2 smoke test: byte-equivalence between our port and the .tflite runtime.

Feeds the same 256x256x3 input image through:
  (a) ai-edge-litert Interpreter on the original pose_landmarks_detector.tflite
  (b) BlazePosePort (our pure-PyTorch walker)
Compares each named output tensor.  Fails if max abs diff exceeds tolerance.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from port import load_task

import ai_edge_litert.interpreter as litert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", type=Path)
    ap.add_argument("--atol", type=float, default=1e-3,
                    help="Absolute tolerance per output.")
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    img = rng.random((256, 256, 3)).astype(np.float32)   # in [0, 1]

    # ---- Reference: ai-edge-litert ----
    port, tflite_path = load_task(args.task)
    interp = litert.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    interp.set_tensor(in_det["index"], img[None].astype(np.float32))
    interp.invoke()
    ref = {}
    for d in interp.get_output_details():
        ref[d["name"]] = interp.get_tensor(d["index"])
    print(f"[litert] outputs: {list(ref.keys())}")

    # ---- Port forward ----
    port.eval()
    x = torch.from_numpy(img).permute(2, 0, 1)[None]    # (1, 3, 256, 256) NCHW
    with torch.no_grad():
        out = port(x)
    print(f"[port  ] outputs: {list(out.keys())}")

    # ---- Compare ----
    rc = 0
    for name, ref_arr in ref.items():
        if name not in out:
            print(f"[FAIL] missing port output: {name}")
            rc = 2
            continue
        port_arr = out[name].cpu().numpy()
        if port_arr.shape != ref_arr.shape:
            print(f"[FAIL] {name} shape: ref={ref_arr.shape}, port={port_arr.shape}")
            rc = 2
            continue
        diff = np.abs(port_arr - ref_arr)
        ok = np.allclose(port_arr, ref_arr, atol=args.atol, rtol=args.rtol)
        print(f"  {name:14s} shape={ref_arr.shape}  "
              f"max|diff|={diff.max():.5g}  mean|diff|={diff.mean():.5g}  "
              f"OK={ok}")
        if not ok:
            rc = 1
    if rc == 0:
        print("\n[PASS] all outputs within tolerance")
    else:
        print(f"\n[FAIL] exit code {rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
