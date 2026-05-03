# Output head and MTP

## LM head path

After all transformer blocks, hidden state is still a hyper-connection stream:

```text
h: [B,S,4,D]
```

The head collapses it with `hc_head`, then applies RMSNorm and a vocab projection.

```text
hc_head: [B,S,4,D] -> [B,S,D]
norm: [B,S,D]
last token: [B,D]
lm_head: [V,D]
logits: [B,V]
```

The official inference implementation computes logits only for `x[:, -1]`, so the decode graph should show `[B,V]` rather than `[B,S,V]` by default. If the UI later supports full prefill logits, mark `[B,S,V]` as conceptual, not official inference path.

## Parallel head

For tensor parallel inference:

```text
local head weight: [V/world_size,D]
local logits: [B,V/world_size]
all_gather -> [B,V]
```

Single-rank visualization can hide this by default and expose it in an "distributed execution" accordion.

## MTP block

The model config includes `num_nextn_predict_layers = 1`. Official inference code constructs one `MTPBlock` after the main transformer layers.

MTP forward path:

```text
x: [B,S,4,D]              # main hidden stream
input_ids: [B,S]
e = embed(input_ids): [B,S,D]
e = enorm(e): [B,S,D]
x = hnorm(x): [B,S,4,D]
combined = e_proj(e).unsqueeze(2) + h_proj(x): [B,S,4,D]
combined -> Block.forward -> [B,S,4,D]
head -> logits [B,V]
```

The MTP block's layer id is `N`, after normal transformer block ids `0..N-1`. The final entry in `compress_ratios` is `0`, so its attention path is sliding-window-only in the official config.

## Graph node fields

- Node id: `head`
- Title: `HC Head + LM Head`
- Input shape: `[B,S,4,D]`
- Output shape: `[B,V]`
- Child nodes: `hc-head-collapse`, `rmsnorm`, `vocab-projection`

- Node id: `mtp`
- Title: `MTP Block`
- Input shape: hidden `[B,S,4,D]` and `input_ids [B,S]`
- Output shape: `[B,V]`
- Visual: optional side branch from final hidden stream into next-token prediction head.

