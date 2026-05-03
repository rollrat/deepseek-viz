# Indexer

## Purpose

The Indexer exists only for `compress_ratio = 4` attention layers. It scores compressed KV blocks and returns top-k compressed positions for each query token. In the graph, this should be a child node under the CSA-like attention path.

## Inputs

```text
x: [B,S,D]
qr: [B,S,Qr]
start_pos: scalar
offset: scalar
```

`qr` is the low-rank query produced by the main attention path before expansion to full heads.

## Query path

```text
q = wq_b(qr): [B,S,index_heads*index_head_dim]
q = unflatten: [B,S,index_heads,index_head_dim]
q[..., -Rd:] gets RoPE
q = randomized Hadamard rotation
q = FP4 activation quantized
```

Constants:

- `index_heads = 64`
- `index_head_dim = 128`
- `Rd = 64`

## Index KV path

The indexer owns a compressor with `head_dim = index_head_dim = 128` and rotation enabled.

```text
index kv cache: [B,max_seq_len/4,128]
```

This cache is independent from the main attention compressed KV cache, which uses `Hd=512`.

## Scoring

```text
index_score_raw = einsum(q, index_kv_cache)
index_score_raw: [B,S,index_heads,T/4]

weights = weights_proj(x): [B,S,index_heads]
weighted score = sum(relu(index_score_raw) * weights over heads)
index_score: [B,S,T/4]

topk_idxs = topk(index_score, index_topk)
```

Pro uses `index_topk = 1024`. Flash uses `index_topk = 512`.

## Output

```text
topk compressed positions: [B,S,min(index_topk,T/4)]
```

The output is offset so that top-k compressed positions line up with the concatenated KV tensor:

```text
selected KV = concat(window KV, compressed KV)
topk_idxs = concat(window_idxs, compressed_topk_idxs)
```

## Graph node fields

- Node id: `indexer`
- Title: `Compressed Block Indexer`
- Input shape: `x [B,S,D]`, `qr [B,S,Qr]`
- Output shape: `[B,S,topK]`
- Visual: show `T/4` compressed blocks as a long strip and highlight top-k selected blocks.

