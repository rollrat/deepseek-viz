# Hyper-connections

## Purpose

V4 block은 단순 residual stream `[B,S,D]` 하나를 계속 더하는 구조가 아니다. block 사이 hidden state는 `[B,S,M,D]`이고, 각 sublayer 앞뒤에서 hyper-connection mixing을 한다. 공개 code에서는 이를 `hc_pre`와 `hc_post`로 구현한다.

## Constants

```text
M = hc_mult = 4
mix_hc = (2 + M) * M = 24
hc_dim = M * D
```

Pro에서는 `hc_dim = 4 * 7168 = 28672`, Flash에서는 `hc_dim = 4 * 4096 = 16384`다.

## hc_pre

`hc_pre`는 4개 hidden lanes를 하나의 sublayer input으로 접는다.

```text
x: [B,S,M,D]
flatten hidden lanes: [B,S,M*D]
mixes = linear(x, hc_fn): [B,S,24]
hc_split_sinkhorn(mixes) -> pre, post, comb
pre: [B,S,M]
post: [B,S,M]
comb: [B,S,M,M]
y = sum(pre * x over M): [B,S,D]
```

`pre`는 attention 또는 MoE에 들어갈 단일 hidden vector를 만든다. `post`와 `comb`는 sublayer output을 다시 4개 lane으로 복원할 때 사용된다.

## hc_post

```text
sublayer_out: [B,S,D]
residual: [B,S,M,D]
post: [B,S,M]
comb: [B,S,M,M]
y = post * sublayer_out + comb * residual
output: [B,S,M,D]
```

시각화에서는 `post`를 sublayer output이 각 lane으로 얼마나 들어가는지, `comb`를 기존 lane끼리 어떻게 섞이는지 보여주면 된다.

## Block-level flow

```text
hidden [B,S,4,D]
  -> hc_pre(attn) [B,S,D]
  -> attn_norm
  -> attention [B,S,D]
  -> hc_post [B,S,4,D]
  -> hc_pre(ffn) [B,S,D]
  -> ffn_norm
  -> MoE [B,S,D]
  -> hc_post [B,S,4,D]
```

## Graph node fields

- Node id: `m-hc`
- Title: `Manifold Hyper-Connections`
- Input shape: `[B,S,4,D]`
- Intermediate shapes: `[B,S,4D]`, `[B,S,24]`, `[B,S,D]`
- Output shape: `[B,S,4,D]`
- Expand mode: show `pre`, `post`, `comb` as three subnodes.

