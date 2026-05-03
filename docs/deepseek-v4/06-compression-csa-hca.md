# KV compression: CSA and HCA paths

## Purpose

The compression node reduces the sequence axis of KV memory. It projects hidden states into KV candidates and gate scores, pools every `R` tokens into one compressed KV entry, normalizes it, applies RoPE to the RoPE slice, then writes it into the compressed KV cache.

## Compressor shape flow

```text
x: [B,S,D]
kv_raw = wkv(x): [B,S,Coff*Hd]
score = wgate(x): [B,S,Coff*Hd]
```

`Coff = 1 + overlap`. Overlap is true only when `R=4`, so:

- `R=4`: `Coff=2`, raw projection shape `[B,S,1024]`.
- `R=128`: `Coff=1`, raw projection shape `[B,S,512]`.

For prefill:

```text
cutoff = S - (S % R)
kv_raw -> [B,cutoff/R,R,Coff*Hd]
score -> [B,cutoff/R,R,Coff*Hd]
score += learned ape
pooled = sum(kv_raw * softmax(score over R))
```

For `R=4`, overlap transform creates a `2R` pooling window where half the dims carry overlap information and half carry normal chunk information.

## Output

```text
compressed kv: [B,floor(S/R),Hd]
compressed cache: [B,max_seq_len/R,Hd]
```

The last `Rd=64` dims receive RoPE. The non-RoPE dims are quantized; with indexer rotation enabled, the code uses FP4 activation quantization for the indexer's compressor.

## Ratio 4 path

Use this as the UI label:

```text
CSA-like layer: R=4
local recent KV + compressed historical KV + learned top-k indexer
```

Details:

- Compression is relatively light.
- An `Indexer` selects top-k compressed positions per query.
- Pro `index_topk=1024`, Flash `index_topk=512`.
- Because the search space is compressed by 4x, the indexer chooses block positions instead of raw token positions.

## Ratio 128 path

Use this as the UI label:

```text
HCA-like layer: R=128
local recent KV + all valid heavily compressed historical KV blocks
```

Details:

- Compression is heavy.
- No learned indexer module is attached in the official code.
- The compressed sequence is short enough that the implementation can use generated valid compressed indices.

## Layer schedules

For the visualization, model the schedules as data arrays from config:

```ts
pro.compressRatios = [
  128, 128, 4, 128, 4, 128, "...", 4, 0
]

flash.compressRatios = [
  0, 0, 4, 128, 4, 128, "...", 4, 0
]
```

The final `0` corresponds to the MTP block. For graph display, do not draw all 61 or 43 blocks by default; show a vertical layer strip with repeated `R=4` and `R=128` blocks, and expand only the selected layer.

## Graph node fields

- Node id: `kv-compressor`
- Title: `KV Compressor`
- Input shape: `[B,S,D]`
- Output shape: `[B,floor(S/R),512]`
- Important controls: ratio switch `4 / 128`, overlap toggle for ratio `4`.

