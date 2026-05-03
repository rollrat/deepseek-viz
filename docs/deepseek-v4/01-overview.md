# Overview

DeepSeek V4는 text generation용 MoE causal language model 패밀리다. 공개 checkpoint는 Flash/Pro, Base/Instruct 조합으로 나뉜다. 시각화의 기본 대상은 Instruct 모델이지만, 내부 tensor flow는 Base와 대체로 같은 decoder-only 흐름으로 잡는다.

## Model family

| Variant | Total params | Activated params | Context | Precision |
| --- | ---: | ---: | ---: | --- |
| DeepSeek-V4-Flash | 284B | 13B | 1,048,576 tokens | FP4 experts + FP8 mixed |
| DeepSeek-V4-Pro | 1.6T | 49B | 1,048,576 tokens | FP4 experts + FP8 mixed |

Base variants는 FP8 mixed로 배포되고, Instruct variants는 MoE expert parameters에 FP4를 쓴다. 모델 카드 기준으로 두 Instruct 모델 모두 Non-think, Think High, Think Max reasoning mode를 지원한다.

## Global forward path

V4의 그래프는 아래 흐름으로 잡으면 된다.

```text
input_ids [B,S]
  -> token embedding [B,S,D]
  -> HC expand [B,S,M,D]
  -> repeat N transformer blocks
       -> HC pre for attention [B,S,D]
       -> RMSNorm [B,S,D]
       -> Hybrid Attention [B,S,D]
       -> HC post [B,S,M,D]
       -> HC pre for MoE [B,S,D]
       -> RMSNorm [B,S,D]
       -> MoE FFN [B,S,D]
       -> HC post [B,S,M,D]
  -> HC head collapse [B,S,D]
  -> RMSNorm [B,S,D]
  -> LM head on last token [B,V]
```

`M = hc_mult = 4`다. 일반적인 residual stream 하나가 아니라, hidden state가 4개 copy 축을 가진 `[B,S,4,D]` 상태로 블록 사이를 이동한다. 각 attention/FFN sublayer에 들어갈 때 `hc_pre`가 `[B,S,4,D] -> [B,S,D]`로 접고, sublayer output을 `hc_post`가 다시 `[B,S,4,D]`로 확장한다.

## Layer schedule

각 block의 attention은 `compress_ratios[layer_id]`에 따라 달라진다.

| Ratio | Meaning for visualization |
| ---: | --- |
| `0` | sliding-window-only attention. MTP 마지막 계층에서 사용된다. |
| `4` | CSA-like path. compressed KV 4x + learned indexer top-k + local sliding window. |
| `128` | HCA-like path. heavily compressed KV 128x + dense compressed selection + local sliding window. |

Pro는 61 transformer blocks + 1 MTP block이다. Flash는 43 transformer blocks + 1 MTP block이다. 두 모델 모두 MTP block의 compression ratio는 `0`으로 공개 config에 들어 있다.

## What the page should emphasize

사용자는 V4 내부를 알고 싶어 한다. 그래서 UI copy는 비교 서사 대신 다음 질문에 답해야 한다.

- 지금 tensor가 어떤 shape인지.
- 이 노드가 hidden stream, attention cache, routing score 중 무엇을 조작하는지.
- 어떤 축이 압축되는지.
- 어떤 축이 expert 선택/top-k 선택에 쓰이는지.
- 어떤 값이 cache로 남고 다음 token decode에서 재사용되는지.
