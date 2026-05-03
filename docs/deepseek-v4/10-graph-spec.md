# Graph spec for the web page

## Layout

The main screen should be a large centered graph, not a document page. Use the documentation files as node content, but keep the first viewport visual.

```text
                [Layer Schedule Strip]

[Input IDs] -> [Embedding] -> [HC Expand] -> [Transformer Block Stack] -> [HC Head] -> [Logits]
                                      |
                         expand selected block
                                      v
         [HC Pre] -> [Attention] -> [HC Post] -> [HC Pre] -> [MoE] -> [HC Post]
                         |
             expand selected attention layer
                         v
 [Q Path] [KV Path] [Window Cache] [Compressor] [Indexer if R=4] [Sparse Attention] [Grouped O Projection]
```

## Node list

| Node id | Label | Default shape |
| --- | --- | --- |
| `input-ids` | Input IDs | `[B,S]` |
| `embedding` | Token Embedding | `[B,S] -> [B,S,D]` |
| `hc-expand` | HC Expand | `[B,S,D] -> [B,S,4,D]` |
| `block-stack` | Transformer Blocks | Pro 61 / Flash 43 |
| `layer-strip` | Layer Schedule | `R=4/128/0` |
| `hc-pre-attn` | HC Pre: Attention | `[B,S,4,D] -> [B,S,D]` |
| `attention` | Hybrid Attention | `[B,S,D] -> [B,S,D]` |
| `q-proj` | Query Projection | `[B,S,D] -> [B,S,H,512]` |
| `kv-proj` | Shared KV Projection | `[B,S,D] -> [B,S,512]` |
| `window-cache` | Sliding Window Cache | `[B,128,512]` |
| `kv-compressor` | KV Compressor | `[B,S,D] -> [B,floor(S/R),512]` |
| `indexer` | Indexer | `[B,S,Qr] -> [B,S,topK]` |
| `sparse-attn` | Sparse Attention | selected KV -> `[B,S,H,512]` |
| `grouped-o-proj` | Grouped Output Projection | `[B,S,H,512] -> [B,S,D]` |
| `attn-residual-mix` | Attention Residual Lane Mixing | `comb [B,S,4,4] @ residual [B,S,4,D] -> [B,S,4,D]` |
| `attn-post-inject` | Attention Output Injection | `post [B,S,4] * attention [B,S,D] -> [B,S,4,D]` |
| `hc-write` | Attention HC Writeback | `mixed residual + injected attention -> [B,S,4,D]` |
| `hc-pre-moe` | HC Pre: MoE | `[B,S,4,D] -> [B,S,D]` |
| `moe` | MoE Router + Experts | `[B,S,D] -> [B,S,D]` |
| `gate` | Router Gate | `[B*S,D] -> [B*S,E]` |
| `routed-experts` | 6 Routed Experts | `[N_e,D] -> [N_e,D]` |
| `shared-expert` | Shared Expert | `[B*S,D] -> [B*S,D]` |
| `ffn-residual-mix` | MoE Residual Lane Mixing | `comb [B,S,4,4] @ residual [B,S,4,D] -> [B,S,4,D]` |
| `ffn-post-inject` | MoE Output Injection | `post [B,S,4] * MoE [B,S,D] -> [B,S,4,D]` |
| `hc-post-moe` | MoE HC Writeback | `mixed residual + injected MoE -> [B,S,4,D]` |
| `head` | HC Head + LM Head | `[B,S,4,D] -> [B,V]` |
| `mtp` | MTP Block | optional `[B,S,4,D] -> [B,V]` |

## Detail panel schema

Each node can be represented as:

```ts
type V4NodeDoc = {
  id: string;
  title: string;
  category: "stream" | "attention" | "cache" | "routing" | "expert" | "output";
  summary: string;
  inputShapes: string[];
  outputShapes: string[];
  keyParams: Record<string, string | number>;
  shapeNotes: string[];
  interactions: string[];
  source: "official-config" | "official-code" | "official-card" | "explainer" | "derived";
  docPath: string;
};
```

## Interaction behavior

- Single click: select node and open right-side detail panel.
- Double click or zoom button: center and expand node into child graph.
- Layer strip click: choose Pro/Flash and layer id, then update `R`, indexer visibility, cache size, and top-k display.
- Token hover: highlight the same token path through embedding, q, kv, selected experts, and logits.
- Cache hover: show whether data is in window cache `[B,128,512]` or compressed cache `[B,T/R,512]`.
- Expert hover: show selected expert ids `[B*S,6]` and weights `[B*S,6]`.

## Model switch behavior

The graph structure stays the same between Pro and Flash. Only constants change.

| Field | Pro | Flash |
| --- | ---: | ---: |
| `D` | 7168 | 4096 |
| `N` | 61 | 43 |
| `H` | 128 | 64 |
| `G` | 16 | 8 |
| `Qr` | 1536 | 1024 |
| `E` | 384 | 256 |
| `I` | 3072 | 2048 |
| `index_topk` | 1024 | 512 |

## Avoid in UI copy

- Do not center the page around historical model comparison.
- Do not show benchmark tables in the first viewport.
- Do not explain "what is a transformer" unless the user opens a beginner overlay.
- Do not flatten all layers into dozens of visible repeated nodes by default. Use a layer strip and expand one selected block.
