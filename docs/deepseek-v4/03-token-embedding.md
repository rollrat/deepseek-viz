# Token embedding and initial stream

## Purpose

이 노드는 token ids를 dense hidden vectors로 바꾸고, V4의 hyper-connection stream을 시작한다. 그래프에서는 `Input IDs -> ParallelEmbedding -> HC Expand`를 별도 노드로 두는 게 좋다.

## Shape flow

```text
input_ids: [B,S]
embedding table: [V,D]
h = embed(input_ids): [B,S,D]
h.unsqueeze(2).repeat(M): [B,S,M,D]
```

`V = 129280`, `M = 4`다. `D`는 Pro 7168, Flash 4096이다.

## Parallel embedding

공식 inference code의 `ParallelEmbedding`은 vocab 축을 tensor parallel rank별로 나눌 수 있게 설계되어 있다.

```text
local embedding table: [V/world_size,D]
local y: [B,S,D]
all_reduce across ranks -> [B,S,D]
```

단일 rank 시각화에서는 sharding을 접고 `input_ids -> [B,S,D]`로 보여주면 된다. 상세 패널에는 분산 실행에서는 vocab partition mask와 all-reduce가 들어간다고 표시한다.

## Graph node fields

- Node id: `embedding`
- Title: `Token Embedding`
- Input shape: `[B,S]`
- Output shape: `[B,S,D]`
- Main parameters: `vocab_size=129280`, `hidden_size=D`
- Interaction: token id 하나를 클릭하면 embedding row lookup으로 강조.

## HC expand node

- Node id: `hc-expand`
- Title: `HC Expand`
- Input shape: `[B,S,D]`
- Output shape: `[B,S,4,D]`
- Role: block 사이의 hidden stream을 4개 manifold channel로 확장.
- Visual: hidden stream 하나가 4개 parallel lanes로 갈라지는 애니메이션.

