#!/usr/bin/env python3
"""Export big-lama's TorchScript generator weights as a `.safetensors` file.

The Rust engine reads this instead of the 205 MB TorchScript zip, so no pickle
parsing and no torch dependency is needed at runtime.  The container is the
standard `.safetensors` one, written by `safetensors_light.write_safetensors`
(no `safetensors` package needed) and read by `lightgpu::safetensors` on the
Rust side - the same reader the other engines in the family use.

Tensor names are the keys of the TorchScript generator's `state_dict`, which is
also the layer order the engine walks, so nothing else is recorded.

Run once:

  python3 export_weights.py models/big-lama.pt models/big-lama.safetensors
"""

import argparse
import sys

import torch

from safetensors_light import write_safetensors


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('checkpoint')
    ap.add_argument('out')
    args = ap.parse_args()

    torch.set_num_threads(1)
    blob = torch.jit.load(args.checkpoint, map_location='cpu')
    sd = blob.generator.state_dict()
    del blob

    tensors = []
    total = 0
    for name, t in sd.items():
        t = t.detach().to(torch.float32).contiguous().cpu()
        raw = t.numpy().tobytes()
        total += len(raw)
        tensors.append((name, 'F32', list(t.shape), raw))

    written = write_safetensors(args.out, tensors)
    print(f'wrote {args.out} ({written} bytes, {total} tensor bytes, '
          f'{len(tensors)} tensors)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
