#!/usr/bin/env python3
"""Convert PyTorch super-resolution checkpoints to NCNN format (.param + .bin) via pnnx.

Supported architectures (definitions mirror BasicSR / Real-ESRGAN, BSD-3-Clause, so we
don't need to install those packages):
  srvgg  - SRVGGNetCompact (realesr-general-x4v3), converted with pnnx
  (rrdb models are written to NCNN format directly: pnnx needs >13 GB RAM for them)
  rrdb   - RRDBNet / "ESRGAN" (RealESRGAN_x4plus, 4xNomosWebPhoto_esrgan, 4x-UltraSharp, ...)
           Both the modern BasicSR key names (conv_first, body.N.rdb1...) and the original
           ESRGAN names (model.0, model.1.sub.N.RDB1...) are accepted.

Usage:
  convert_to_ncnn.py --arch srvgg --weights realesr-general-x4v3.pth --out-dir out --name realesr-general-x4v3
  convert_to_ncnn.py --arch rrdb  --weights 4x-UltraSharp.pth       --out-dir out --name 4x-UltraSharp
  convert_to_ncnn.py --arch rrdb  --random --num-block 2 --out-dir out --name test   # self-test
"""
import argparse
import os
import re
import shutil
import sys
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- SRVGGNetCompact
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


# ---------------------------------------------------------------- RRDBNet (ESRGAN)
class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """4x RRDBNet: two nearest-neighbour 2x upsampling stages."""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


OLD_ESRGAN_TOP = {"model.0": "conv_first", "model.3": "conv_up1", "model.6": "conv_up2",
                  "model.8": "conv_hr", "model.10": "conv_last"}


def convert_old_esrgan_keys(sd):
    """Map original ESRGAN key names (model.1.sub.N.RDB1.conv1.0.weight) to BasicSR names."""
    if not any(k.startswith("model.") for k in sd):
        return sd
    blocks = [int(m.group(1)) for k in sd if (m := re.match(r"model\.1\.sub\.(\d+)\.", k))]
    trunk = max(blocks)  # the last "sub" entry is the trunk conv (conv_body)
    out = {}
    for k, v in sd.items():
        name, _, param = k.rpartition(".")
        if name in OLD_ESRGAN_TOP:
            out[f"{OLD_ESRGAN_TOP[name]}.{param}"] = v
            continue
        m = re.match(r"model\.1\.sub\.(\d+)(?:\.RDB(\d)\.conv(\d)\.0)?$", name)
        if not m:
            raise SystemExit(f"Unrecognised ESRGAN key: {k}")
        i = int(m.group(1))
        if m.group(2) is None:
            if i != trunk:
                raise SystemExit(f"Unexpected key: {k}")
            out[f"conv_body.{param}"] = v
        else:
            out[f"body.{i}.rdb{m.group(2)}.conv{m.group(3)}.{param}"] = v
    return out


def load_state_dict(path):
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(path)
    else:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("params_ema", "params", "state_dict", "model"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
            break
    return {k.removeprefix("module."): v for k, v in sd.items()}


def build_rrdb(sd):
    """Creates an RRDBNet matching the checkpoint's block count / widths."""
    sd = convert_old_esrgan_keys(sd)
    num_block = 1 + max(int(m.group(1)) for k in sd if (m := re.match(r"body\.(\d+)\.", k)))
    num_feat = sd["conv_first.weight"].shape[0]
    num_grow = sd["body.0.rdb1.conv1.weight"].shape[0]
    in_ch = sd["conv_first.weight"].shape[1]
    if in_ch != 3:
        raise SystemExit(f"Only 4x RGB RRDB models are supported (conv_first has {in_ch} input channels)")
    model = RRDBNet(num_feat=num_feat, num_block=num_block, num_grow_ch=num_grow)
    model.load_state_dict(sd, strict=True)
    print(f"RRDBNet: {num_block} blocks, {num_feat} features, growth {num_grow}")
    return model


# ---------------------------------------------------------------- direct NCNN writer for RRDBNet
# pnnx needs >13 GB of RAM for a 23-block RRDBNet, so for this fixed, well-known architecture we
# write the NCNN graph (.param) and weights (.bin) ourselves. Weights are stored as fp16, biases fp32.
FP16_TAG = 0x01306B47


class NcnnWriter:
    def __init__(self):
        self.layers = []  # [type, name, inputs, outputs, params]
        self.weights = []  # bytes, in layer order
        self.n = 0

    def blob(self, prefix="b"):
        self.n += 1
        return f"{prefix}{self.n}"

    def add(self, typ, inputs, params="", out=None):
        out = out or self.blob()
        self.layers.append([typ, f"{typ.lower()}_{len(self.layers)}", list(inputs), [out], params])
        return out

    def conv(self, x, conv: nn.Conv2d, leaky=False, out=None):
        import numpy as np
        w = conv.weight.detach().float().numpy()
        b = conv.bias.detach().float().numpy()
        o, i, kh, kw = w.shape
        params = f"0={o} 1={kw} 11={kh} 2=1 3=1 4={kh // 2} 5=1 6={w.size}"
        if leaky:
            params += " 9=2 -23310=1,2.000000e-01"  # fused LeakyReLU(0.2)
        wb = np.array([FP16_TAG], dtype="<u4").tobytes() + w.astype("<f2").tobytes()
        wb += b"\0" * ((-len(wb)) % 4)  # 4-byte alignment
        self.weights.append(wb + b.astype("<f4").tobytes())
        return self.add("Convolution", [x], params, out)

    def concat(self, xs):
        return self.add("Concat", xs, "0=0")

    def axpy(self, a, ca, b, cb):
        """ca*a + cb*b"""
        return self.add("Eltwise", [a, b], f"0=1 -23301=2,{ca:e},{cb:e}")

    def upsample2(self, x):
        return self.add("Interp", [x], "0=1 1=2.000000e+00 2=2.000000e+00")

    def write(self, param_path, bin_path, input_name):
        # NCNN requires a Split layer when one blob feeds several layers.
        uses = {}
        for L in self.layers:
            for i in L[2]:
                uses[i] = uses.get(i, 0) + 1
        counters, layers = {}, [["Input", "input", [], [input_name], "0=0 1=0 2=3"]]
        for L in [layers[0]] + self.layers:
            if L is not layers[0]:
                new_inputs = []
                for i in L[2]:
                    if uses.get(i, 0) > 1:
                        k = counters.get(i, 0); counters[i] = k + 1
                        new_inputs.append(f"{i}_s{k}")
                    else:
                        new_inputs.append(i)
                L[2] = new_inputs
                layers.append(L)
            o = L[3][0]
            if uses.get(o, 0) > 1:
                layers.append(["Split", f"split_{o}", [o], [f"{o}_s{k}" for k in range(uses[o])], ""])
        blobs = {b for L in layers for b in L[3]}
        with open(param_path, "w") as f:
            f.write(f"7767517\n{len(layers)} {len(blobs)}\n")
            for typ, name, ins, outs, params in layers:
                f.write(f"{typ:<24} {name:<24} {len(ins)} {len(outs)} {' '.join(ins + outs)} {params}".rstrip() + "\n")
        with open(bin_path, "wb") as f:
            for wb in self.weights:
                f.write(wb)


def write_rrdb_ncnn(model: "RRDBNet", param_path, bin_path):
    g = NcnnWriter()
    x = "in0"

    def rdb(x, m: ResidualDenseBlock):
        x1 = g.conv(x, m.conv1, leaky=True)
        x2 = g.conv(g.concat([x, x1]), m.conv2, leaky=True)
        x3 = g.conv(g.concat([x, x1, x2]), m.conv3, leaky=True)
        x4 = g.conv(g.concat([x, x1, x2, x3]), m.conv4, leaky=True)
        x5 = g.conv(g.concat([x, x1, x2, x3, x4]), m.conv5)
        return g.axpy(x5, 0.2, x, 1.0)

    feat = g.conv(x, model.conv_first)
    h = feat
    for block in model.body:
        y = rdb(rdb(rdb(h, block.rdb1), block.rdb2), block.rdb3)
        h = g.axpy(y, 0.2, h, 1.0)
    feat = g.axpy(g.conv(h, model.conv_body), 1.0, feat, 1.0)
    feat = g.conv(g.upsample2(feat), model.conv_up1, leaky=True)
    feat = g.conv(g.upsample2(feat), model.conv_up2, leaky=True)
    feat = g.conv(feat, model.conv_hr, leaky=True)
    g.conv(feat, model.conv_last, out="out0")
    g.write(param_path, bin_path, "in0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["srvgg", "rrdb"], required=True)
    ap.add_argument("--weights", help="Path to the .pth / .safetensors checkpoint")
    ap.add_argument("--random", action="store_true", help="Use random weights (self-test)")
    ap.add_argument("--num-block", type=int, default=23, help="RRDB blocks for --random")
    ap.add_argument("--old-keys", action="store_true", help="With --random: also save a checkpoint using old ESRGAN key names")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", required=True, help="Output base name (writes NAME.param / NAME.bin)")
    args = ap.parse_args()

    torch.manual_seed(0)
    if args.random:
        model = SRVGGNetCompact() if args.arch == "srvgg" else RRDBNet(num_block=args.num_block)
    elif not args.weights:
        sys.exit("--weights is required unless --random is given")
    elif args.arch == "srvgg":
        model = SRVGGNetCompact()
        model.load_state_dict(load_state_dict(args.weights), strict=True)
    else:
        model = build_rrdb(load_state_dict(args.weights))
    model.eval()
    if args.weights:
        print(f"loaded weights from {args.weights}")

    os.makedirs(args.out_dir, exist_ok=True)
    if args.arch == "rrdb":
        write_rrdb_ncnn(model, os.path.join(args.out_dir, args.name + ".param"), os.path.join(args.out_dir, args.name + ".bin"))
        print(f"wrote {args.out_dir}/{args.name}.param and .bin (direct writer)")
        if args.random:
            torch.save(model.state_dict(), os.path.join(args.out_dir, args.name + ".state.pt"))
        return

    import pnnx  # imported late so --help works without it

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
