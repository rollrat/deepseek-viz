# MoE router and experts

## Purpose

Every transformer block has a MoE FFN. It receives `[B,S,D]`, flattens tokens, routes each token to `K=6` routed experts, adds one shared expert output, and returns `[B,S,D]`.

## Gate

```text
x: [B,S,D]
flat x: [B*S,D]
gate weight: [E,D]
scores = linear(x, gate_weight): [B*S,E]
scores = sqrt(softplus(scores))
```

For the first `n_hash_layers = 3`, the selected expert ids come from a token-id lookup table:

```text
tid2eid: [V,K]
indices = tid2eid[input_ids]: [B*S,K]
```

For later layers:

```text
indices = topk(scores, K): [B*S,K]
```

Weights are gathered from the original scores, normalized across the selected `K`, then scaled by `routed_scaling_factor`.

```text
weights: [B*S,K]
```

## Expert constants

| Variant | Routed experts `E` | Experts/token `K` | Shared experts | Intermediate `I` | Route scale |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pro | 384 | 6 | 1 | 3072 | 2.5 |
| Flash | 256 | 6 | 1 | 2048 | 1.5 |

## Expert FFN

Each expert is a SwiGLU FFN:

```text
expert input: [N_e,D]
w1: D -> I
w3: D -> I
gate = w1(x): [N_e,I]
up = w3(x): [N_e,I]
hidden = silu(gate) * up: [N_e,I]
weighted hidden: [N_e,I]
w2: I -> D
expert output: [N_e,D]
```

The routed expert outputs are accumulated into a zero tensor:

```text
y: [B*S,D]
```

Then the shared expert is added for every token:

```text
y += shared_expert(x): [B*S,D]
output view: [B,S,D]
```

## Precision

In Instruct checkpoints, routed MoE expert parameters use FP4. The shared expert is constructed without `expert_dtype` in the official inference code, so document it separately from routed experts.

## Graph node fields

- Node id: `moe`
- Title: `MoE Router + Experts`
- Input shape: `[B,S,D]`
- Output shape: `[B,S,D]`
- Child nodes: `gate`, `topk-experts`, `routed-experts`, `shared-expert`, `combine`
- Visual: selected token fans out to 6 expert tiles plus 1 shared expert tile.

