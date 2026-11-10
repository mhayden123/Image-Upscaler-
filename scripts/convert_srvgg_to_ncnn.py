#!/usr/bin/env python3
"""Convert a Real-ESRGAN SRVGGNetCompact checkpoint (e.g. realesr-general-x4v3.pth)
into NCNN format (.param + .bin) using pnnx.

The network definition below mirrors SRVGGNetCompact from the Real-ESRGAN project
(BSD-3-Clause) so we don't need to install basicsr/realesrgan.

Usage:
  python convert_srvgg_to_ncnn.py --pth realesr-general-x4v3.pth --out-dir out --name realesr-general-x4v3
  python convert_srvgg_to_ncnn.py --random --out-dir out --name test   # self-test without weights
"""
import argparse
import os
import shutil
import sys
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out + base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", help="Path to the .pth checkpoint")
    ap.add_argument("--random", action="store_true", help="Use random weights (self-test)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", required=True, help="Output base name (writes NAME.param / NAME.bin)")
    ap.add_argument("--num-conv", type=int, default=32)
    ap.add_argument("--num-feat", type=int, default=64)
    args = ap.parse_args()

    torch.manual_seed(0)
    model = SRVGGNetCompact(num_feat=args.num_feat, num_conv=args.num_conv)
    if not args.random:
        if not args.pth:
            sys.exit("--pth is required unless --random is given")
        ckpt = torch.load(args.pth, map_location="cpu", weights_only=True)
        key = "params_ema" if "params_ema" in ckpt else "params"
        model.load_state_dict(ckpt[key], strict=True)
        print(f"loaded weights from {args.pth} (key '{key}')")
    model.eval()

    import pnnx  # imported late so --help works without it

    os.makedirs(args.out_dir, exist_ok=True)
    work = tempfile.mkdtemp(prefix="pnnx_")
    try:
        pt_path = os.path.join(work, "model.pt")
        # Two different input shapes tell pnnx the spatial size is dynamic.
        x1 = torch.rand(1, 3, 64, 64)
        x2 = torch.rand(1, 3, 48, 80)
        with torch.no_grad():
            pnnx.export(model, pt_path, x1, x2, fp16=True)
        shutil.copy(os.path.join(work, "model.ncnn.param"), os.path.join(args.out_dir, args.name + ".param"))
        shutil.copy(os.path.join(work, "model.ncnn.bin"), os.path.join(args.out_dir, args.name + ".bin"))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"wrote {args.out_dir}/{args.name}.param and .bin")

    if args.random:
        torch.save(model.state_dict(), os.path.join(args.out_dir, args.name + ".state.pt"))


if __name__ == "__main__":
    main()
