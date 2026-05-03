# Shape glossary

## Symbols

| Symbol | Meaning |
| --- | --- |
| `B` | batch size |
| `S` | current sequence length in the current forward chunk |
| `T` | cache length already available, usually `start_pos + S` |
| `V` | vocabulary size, `129280` |
| `D` | hidden size |
| `M` | hyper-connection multiplicity, `4` |
| `H` | attention heads |
| `G` | output projection groups |
| `Hd` | full attention head dimension, `512` |
| `Rd` | RoPE sub-dimension per head, `64` |
| `Nd` | non-RoPE sub-dimension, `Hd - Rd = 448` |
| `Qr` | low-rank query dimension |
| `Or` | low-rank output dimension, `1024` |
| `E` | number of routed experts |
| `K` | experts per token, `6` |
| `I` | MoE intermediate size per expert |
| `R` | KV compression ratio, one of `0`, `4`, `128` |
| `W` | sliding window size, `128` |

## Pro constants

| Name | Value |
| --- | ---: |
| `D` | 7168 |
| `N` transformer blocks | 61 |
| `MTP` blocks | 1 |
| `H` | 128 |
| `G` | 16 |
| `Hd` | 512 |
| `Rd` | 64 |
| `Nd` | 448 |
| `Qr` | 1536 |
| `Or` | 1024 |
| `E` routed experts | 384 |
| `K` experts/token | 6 |
| `I` MoE intermediate | 3072 |
| `index_heads` | 64 |
| `index_head_dim` | 128 |
| `index_topk` | 1024 |
| `route_scale` | 2.5 |
| `max_position_embeddings` | 1048576 |

## Flash constants

| Name | Value |
| --- | ---: |
| `D` | 4096 |
| `N` transformer blocks | 43 |
| `MTP` blocks | 1 |
| `H` | 64 |
| `G` | 8 |
| `Hd` | 512 |
| `Rd` | 64 |
| `Nd` | 448 |
| `Qr` | 1024 |
| `Or` | 1024 |
| `E` routed experts | 256 |
| `K` experts/token | 6 |
| `I` MoE intermediate | 2048 |
| `index_heads` | 64 |
| `index_head_dim` | 128 |
| `index_topk` | 512 |
| `route_scale` | 1.5 |
| `max_position_embeddings` | 1048576 |

## Common constants

| Name | Value |
| --- | ---: |
| `V` | 129280 |
| `M` | 4 |
| `num_key_value_heads` | 1 |
| `sliding_window` | 128 |
| `qk_rope_head_dim` | 64 |
| `head_dim` | 512 |
| `n_shared_experts` | 1 |
| `n_hash_layers` | 3 |
| `rms_norm_eps` | 1e-6 |
| `hc_sinkhorn_iters` | 20 |
| `hc_eps` | 1e-6 |
| `rope_scaling` | YaRN, factor 16, original 65536 |
| `compress_rope_theta` | 160000 |

## Derived shapes

| Tensor | Shape |
| --- | --- |
| `input_ids` | `[B,S]` |
| token embeddings | `[B,S,D]` |
| hyper hidden stream | `[B,S,M,D]` |
| attention/FFN sublayer input after `hc_pre` | `[B,S,D]` |
| query low-rank `qr` | `[B,S,Qr]` |
| query heads `q` | `[B,S,H,Hd]` before TP sharding |
| shared KV vector `kv` | `[B,S,Hd]` |
| window KV cache | `[B,W,Hd]` |
| compressed KV cache | `[B,floor(T/R),Hd]` |
| sparse attention output before grouped projection | `[B,S,H,Hd]` |
| grouped attention output | `[B,S,G,(H/G)*Hd]` |
| MoE gate scores | `[B*S,E]` |
| MoE selected expert ids | `[B*S,K]` |
| MoE selected weights | `[B*S,K]` |
| expert hidden | `[tokens_for_expert,I]` |
| logits | `[B,V]` in decode path, because official head reads `x[:, -1]` |

