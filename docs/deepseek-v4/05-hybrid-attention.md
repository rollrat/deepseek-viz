# Hybrid attention

## Purpose

V4 attention node receives a single hidden stream `[B,S,D]` from `hc_pre`, computes a low-rank query, shared KV vector, sliding-window retrieval, optional compressed retrieval, sparse attention, and grouped output projection.

## Main projections

```text
x: [B,S,D]

qr = q_norm(wq_a(x)): [B,S,Qr]
q = wq_b(qr): [B,S,H*Hd]
q = unflatten heads: [B,S,H,Hd]
q[..., -Rd:] gets RoPE

kv = kv_norm(wkv(x)): [B,S,Hd]
kv[..., -Rd:] gets RoPE
kv[..., :Nd] is FP8-quantized in-place for QAT parity
```

Important: `num_key_value_heads = 1` and `wkv` emits only `[B,S,Hd]`, not `[B,S,H,Hd]`. The sparse attention kernel combines multi-head queries with a shared KV representation.

## Sliding window branch

Every attention layer maintains a local recent-token branch.

```text
window KV cache: [B,W,Hd], W=128
window topk indices: [B,S,<=W]
```

For prefill, current `kv` is written into a circular window cache. For decode, the current token writes to `start_pos % W`.

## Compressed branch

If `compress_ratio != 0`, attention also uses compressed KV positions.

```text
compressed cache: [B,floor(T/R),Hd]
R = 4 or 128
compressed topk ids: [B,S,C]
topk_idxs = concat(window ids, compressed ids)
```

For `R=4`, compressed ids are selected by the learned `Indexer`. For `R=128`, compressed ids are generated as all valid compressed positions, which makes it a heavily compressed dense-over-compressed-stream path.

## Sparse attention output

```text
o = sparse_attn(q, selected_kv, attn_sink, topk_idxs)
o: [B,S,H,Hd]
```

After sparse attention, the RoPE sub-dimension of output is inverse-rotated.

```text
apply_rotary_emb(o[..., -Rd:], inverse=True)
```

## Grouped output projection

The attention output is grouped before final projection.

```text
o: [B,S,H,Hd]
o.view(B,S,G,(H/G)*Hd)
wo_a: [G,Or,(H/G)*Hd]
einsum -> [B,S,G,Or]
flatten -> [B,S,G*Or]
wo_b -> [B,S,D]
```

Pro: `H=128`, `G=16`, so each group receives `8*512=4096` dims and outputs `Or=1024`. Flattened grouped output before `wo_b` is `16*1024=16384`.

Flash: `H=64`, `G=8`, so each group also receives `8*512=4096` dims and outputs `Or=1024`. Flattened grouped output before `wo_b` is `8*1024=8192`.

## Graph node fields

- Node id: `attention`
- Title: `Hybrid Attention`
- Input shape: `[B,S,D]`
- Output shape: `[B,S,D]`
- Child nodes: `q-proj`, `kv-proj`, `window-cache`, `compressor`, `indexer`, `sparse-attn`, `grouped-o-proj`
- Visual mode: show the current token attending to `W=128` recent KV slots plus compressed historical slots.

