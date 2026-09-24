"""Write a standard `.safetensors` container with no dependencies.

The `.safetensors` layout is a little-endian u64 header length, that many bytes
of JSON, and then the tensor payloads.  This writes it directly so a conversion
script needs no `safetensors` package; the reader on the Rust side is
`lightgpu::safetensors`, which is the same contract.

Payloads are written in the order given, each padded so the next one starts on
an `align`-byte boundary.  A reader that mmaps the file and reinterprets an FP32
payload as `&[f32]` needs that; padding between payloads is legal, because
`data_offsets` is what locates a tensor, not adjacency.  (Packing tensors with
no padding at all is legal too, but only suits readers that copy each payload
out by hand.)
"""

import io
import json
import struct


def write_safetensors(path, tensors, metadata=None, align=8):
    """Write `tensors` (an iterable of `(name, dtype, shape, raw_bytes)`) to
    `path` as a safetensors container.  Returns the number of payload bytes.

    `metadata` is a dict of str -> str stored under `__metadata__`, which is
    where these engines keep their architecture constants.

    `data_offsets` are relative to the start of the payload section, as the
    format specifies; the header's own length is not known until it is built, so
    the offsets are computed against the payload buffer and the file is
    assembled at the end.
    """
    payload = io.BytesIO()
    header = {}
    for name, dtype, shape, raw in tensors:
        if payload.tell() % align:
            payload.write(b"\x00" * (align - payload.tell() % align))
        start = payload.tell()
        payload.write(raw)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, payload.tell()],
        }

    if metadata:
        # Written first so the metadata key leads the header, as usual.
        header = {"__metadata__": {str(k): str(v) for k, v in metadata.items()}, **header}

    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # The header sits between the 8-byte length and the first payload, so its own
    # size decides where every payload lands in the file.  Padding it to `align`
    # is what keeps the first tensor aligned - and there are no escapes in a
    # header of tensor names, so the padding spaces are valid JSON whitespace and
    # do not change its length when serialised again.
    pad = (-(8 + len(blob))) % align
    if pad:
        blob += b" " * pad
    assert (8 + len(blob)) % align == 0, "header must leave the data section aligned"

    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.write(payload.getvalue())
    return payload.tell()
