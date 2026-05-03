# Hyper-connections

## Purpose

DeepSeek V4 block은 단일 residual stream `[B,S,D]`만 계속 더하는 구조가 아니다.
Block 사이 hidden state는 residual lane 축을 가진 `[B,S,M,D]`이고, 공개 코드 기준 `M = hc_mult = 4`다.
각 attention / MoE sublayer 앞뒤에서 mHC controller가 `pre`, `post`, `comb`를 만들고, 이 중 `comb`가 residual lane mixing의 핵심이다.

## Constants

```text
M = hc_mult = 4
mix_hc = (2 + M) * M = 24
hc_dim = M * D
```

Pro에서는 `hc_dim = 4 * 7168 = 28672`, Flash에서는 `hc_dim = 4 * 4096 = 16384`.

## Controller

```text
x: [B,S,M,D]
flatten hidden lanes: [B,S,M*D]
mixes = linear(x, hc_fn): [B,S,24]
hc_split_sinkhorn(mixes) -> pre, post, comb
pre:  [B,S,M]
post: [B,S,M]
comb: [B,S,M,M]
```

`pre`는 residual lanes를 sublayer input `[B,S,D]`로 읽는 가중치다.
`post`는 sublayer output `[B,S,D]`를 다시 4개 lane에 주입하는 가중치다.
`comb`는 기존 residual lanes `[B,S,4,D]`를 lane 축에서 서로 섞는 residual lane mixing matrix다.

## Residual Lane Mixing

```text
residual: [B,S,M,D]
comb:     [B,S,M,M]
mixed_residual = comb @ residual
mixed_residual: [B,S,M,D]
```

이 부분이 사용자가 그래프에서 직접 보고 싶어 하는 `residual lane mixing`이다.
Attention writeback과 MoE writeback 양쪽에 같은 형태가 있고, 각각 별도의 mHC controller weights를 쓴다.

## Sublayer Output Injection

```text
sublayer_out: [B,S,D]
post:         [B,S,M]
injected = post * sublayer_out
injected: [B,S,M,D]
```

`post`는 sublayer output을 각 residual lane에 얼마나 주입할지 결정한다.
이건 residual lane mixing과 별도의 항이다.

## HC Writeback

```text
output = mixed_residual + injected
output: [B,S,M,D]
```

따라서 writeback은 다음 두 항의 합이다.

```text
comb * residual lanes
+ post * sublayer_out
```

## Block-Level Flow

```text
hidden [B,S,4,D]
  -> mHC controller(attn) -> pre/post/comb
  -> pre read -> attention input [B,S,D]
  -> attention [B,S,D]
  -> attention residual lane mixing: comb @ residual
  -> attention output injection: post * attention
  -> attention HC writeback [B,S,4,D]
  -> mHC controller(ffn) -> pre/post/comb
  -> pre read -> MoE input [B,S,D]
  -> MoE [B,S,D]
  -> MoE residual lane mixing: comb @ residual
  -> MoE output injection: post * MoE
  -> MoE HC writeback [B,S,4,D]
```

## Graph Nodes

The overview graph should expose these as first-class nodes:

- `Attention Residual Lane Mixing`: `comb [B,S,4,4] @ residual [B,S,4,D]`
- `Attention Output Injection`: `post [B,S,4] * attention [B,S,D]`
- `Attention HC Writeback`: sum of mixed residual and injected attention
- `MoE Residual Lane Mixing`: `comb [B,S,4,4] @ residual [B,S,4,D]`
- `MoE Output Injection`: `post [B,S,4] * MoE [B,S,D]`
- `MoE HC Writeback`: sum of mixed residual and injected MoE
