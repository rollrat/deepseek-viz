# DeepSeek V4 Node Deep Dive Spec

Status: working spec for the interactive graph.
Scope: DeepSeek-V4-Pro first, with Flash differences called out only when they affect tensor shapes or graph behavior.

## Source Priority

1. Official Hugging Face config and inference code:
   - Pro config: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json
   - Pro inference config: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/config.json
   - Pro model code: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/model.py
   - Pro kernel code: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/kernel.py
2. Official/near-official architecture explainers:
   - Hugging Face DeepSeek V4 blog: https://huggingface.co/blog/deepseekv4
   - DeepSeek V4 technical report file: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf
3. Runtime implementation notes:
   - vLLM DeepSeek V4 blog: https://vllm.ai/blog/deepseek-v4

## Prompt-Derived Investigation Targets

These are not treated as "claims" from the user. They are the kinds of detailed explanations the graph should eventually surface per node: why the node exists, what runtime state it touches, what hidden implementation constraint it represents, and what remains unknown.

| Target detail | Current grounding | Notes for node annotations |
|---|---|---|
| There is always exactly one shared expert. | Confirmed for current official code/config. | Pro config has `n_shared_experts=1`; code asserts this and adds `self.shared_experts(x)` after routed experts. |
| Layers 1-3 use input-id expert matching instead of score top-k. | Mostly confirmed, with indexing wording caveat. | Official config uses `num_hash_layers=3`; code uses `layer_id < n_hash_layers`, so zero-based layers `0,1,2` are hash/token-id routed. User-facing "1st-3rd layer" is fine. |
| Early token-id routing may cause expert skew for frequent tokens. | Plausible concern, not resolved by public inference code alone. | Inference code shows fixed `tid2eid[input_ids]`, but does not explain how `tid2eid` was trained/assigned or load-balanced. Need technical report/training details or weight/stat analysis to answer. |
| mHC creates multi-lane residual streams and read/write projections around attention/MoE. | Confirmed. | Code expands hidden to `[B,S,hc_mult,D]`, with `hc_mult=4`; `hc_pre` reads lanes to `[B,S,D]`; `hc_post` writes sublayer output back into lanes. |
| mHC mixing uses a doubly stochastic matrix for stable gradients. | Confirmed structurally. | Kernel Sinkhorn-normalizes `comb [B,S,4,4]`; public explainers connect this to spectral/stability constraints. |
| Controller path -> data path dependency. | Correct direction. | The controller path computes `pre/post/comb`; the data path applies them. If phrased as "data path makes coefficients and controller uses them", that is reversed. |
| Attention name is CSA/HSA. | Rename HSA to HCA. | Public V4 materials use `CSA` and `HCA` (Heavily Compressed Attention). |
| CSA has Lightning indexer + compressor + SWA. | Confirmed. | `compress_ratio==4` constructs `Indexer`; all attention modes keep SWA window ids. |
| C4A is window 8, stride 4. | Confirmed by vLLM and code mechanics. | Official HF blog says CSA compresses 4x; vLLM clarifies c4a uses weighted sum over 8 native tokens with stride 4. Code implements overlap for `ratio==4`. |
| Compressor tail state stores partial tokens. | Confirmed. | Code has `kv_state` and `score_state`; vLLM treats compressor state like sliding-window KV runtime state. |
| Compressed entries use anchor positions like `0,4,8,...`. | Mostly confirmed, with exact boundary caveats. | Code applies compressed RoPE at `freqs_cis[:cutoff:ratio]` in prefill and `start_pos + 1 - ratio` in decode. vLLM calls these anchor positions and documents causality ranges. |
| SWA needs uncompressed rolling KV cache. | Confirmed. | Code keeps first `window_size=128` slots as circular window cache and writes decode token at `start_pos % win`. Runtime is responsible for mapping this efficiently. |
| c4a attention has `n//4 + 128` possible main entries, then indexer top-k reduces compute. | Mostly confirmed. | Logical main attention candidates are compressed entries plus 128 SWA entries. Pro `index_topk=1024`; Flash `index_topk=512`. A `512+128` statement fits Flash/current vLLM defaults, not Pro config. |
| Compressed KV and SWA KV have compatible shape and can share/concat. | Confirmed in prefill path. | Both are shared 512-dim KV entries; prefill code concatenates `kv` and `kv_compress`, decode uses one `kv_cache` with window prefix plus compressed suffix. |
| KV sharing requires inverse RoPE. | Confirmed. | The model has one shared 512-dim KV head. Code applies inverse RoPE to attention output before grouped output projection. |

## Global Shape Glossary

| Symbol | Meaning | Pro value |
|---|---|---:|
| `B` | batch size | runtime |
| `S` | current prefill/decode chunk length | runtime |
| `T` | context length already materialized in cache | runtime |
| `D` | hidden size | `7168` |
| `V` | vocab size | `129280` |
| `H` | query heads | `128` |
| `Hd` | head dimension / shared KV dimension | `512` |
| `Rd` | RoPE subdimension | `64` |
| `Qr` | query LoRA rank | `1536` |
| `Or` | output LoRA rank | `1024` |
| `G` | output projection groups | `16` |
| `E` | routed experts | `384` |
| `K` | routed experts per token | `6` |
| `I` | MoE intermediate size | `3072` |
| `W` | SWA window | `128` |
| `R` | compression ratio | `4`, `128`, or `0` |
| `HC` | residual lanes | `4` |
| `mix_hc` | pre/post/comb logits per token | `(2+HC)*HC = 24` |

## Architecture Flow

The graph should be read as:

```text
input ids
  -> token embedding
  -> HC expand
  -> repeated decoder stack, represented by one selected layer mode
     -> mHC read for attention
     -> hybrid attention path
     -> mHC attention writeback
     -> mHC read for MoE
     -> MoE routing + experts
     -> mHC MoE writeback
  -> final stack state
  -> HC head + LM head / MTP
```

`input-ids`, `embedding`, `hc-expand`, `head`, `mtp`, and `logits` are not per-layer operations in the selected representative layer. They are stack entry/exit boundaries.

## How Node Details Should Read In UI

Each graph node should eventually expose four layers of information:

- **Tensor event:** exact input/output shape and the axis that changes.
- **Why this node exists:** the architectural pressure it solves, e.g. KV memory, expert routing, residual stability, cache locality, or quantized runtime cost.
- **Runtime state:** whether the node owns cache/state, depends on `start_pos`, has prefill/decode branches, or is a pure projection.
- **Open issue:** any detail not directly derivable from public code/config, such as `tid2eid` load balancing or exact checkpoint tensor distribution.

## Node Specs

### Model Entry

#### `input-ids` - Input IDs (model entry)
- Shape: raw prompt -> `[B,S]`.
- Role: tokenizer output. Used once at model entry and also passed to MoE gate for hash-routed early layers.
- Detail: for early hash-routing layers, `input_ids.flatten()` indexes `tid2eid`.
- UI intent: make clear that this is not re-created inside each decoder layer.

#### `embedding` - Token Embedding (once)
- Shape: `[B,S] -> [B,S,D]`.
- Role: maps token ids to hidden vectors.
- Detail: in tensor-parallel mode the embedding table is vocab-sharded and all-reduced.
- UI intent: stack entry only.

#### `hc-expand` - HC Expand
- Shape: `[B,S,D] -> [B,S,4,D]`.
- Role: duplicates/expands the hidden state into four mHC residual lanes.
- Detail: official forward creates `h.unsqueeze(2).repeat(1,1,hc_mult,1)`.

#### `stack-entry` - Decoder Stack Entry
- Shape: `[B,S,4,D] -> repeated decoder state`.
- Role: visual boundary before the repeated decoder stack.
- Detail: the graph's layer selector chooses which attention/cache mode the representative block uses.

### mHC Attention Read/Write

#### `mhc-attn` - mHC Pre/Post: Attention
- Shape: `[B,S,4,D] -> [B,S,4,D]`.
- Role: umbrella node for attention-side mHC controller and data paths.
- Detail: `hc_attn_fn`, `hc_attn_scale`, and `hc_attn_base` are separate from FFN mHC parameters.

#### `hc-flatten` - Flatten HC Lanes
- Shape: `[B,S,4,D] -> [B,S,4D]`.
- Role: controller-path input construction.
- Detail: lane axis and hidden axis are flattened before linear projection.

#### `hc-controller` - Controller Linear
- Shape: `[B,S,4D] -> [B,S,24]`.
- Role: produces mHC logits for `pre`, `post`, and `comb`.
- Detail: code RMS-normalizes the flattened vector by `rsqrt(mean(x^2)+eps)` before/with the controller linear output.

#### `hc-sinkhorn` - Split + Sinkhorn
- Shape: `[B,S,24] -> pre [B,S,4], post [B,S,4], comb [B,S,4,4]`.
- Role: turns controller logits into read weights, write injection weights, and residual mixing matrix.
- Detail: `comb` is repeatedly row/column normalized in `hc_split_sinkhorn`, approximating a doubly stochastic matrix.
- Intent: stable lane-to-lane residual transport; public explainers connect this to bounding signal/gradient amplification.

#### `hc-read` - Read Data Path
- Shape: `pre [B,S,4]`, residual lanes `[B,S,4,D] -> [B,S,D]`.
- Role: creates the single hidden stream consumed by attention.
- Formula: `sum(pre[..., lane] * X[..., lane, :])`.

#### `attn-residual-mix` - Attention Residual Lane Mixing
- Shape: `comb [B,S,4,4]`, residual `[B,S,4,D] -> [B,S,4,D]`.
- Role: mixes old residual lanes before adding new attention output.
- Formula: `sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)`.

#### `attn-post-inject` - Attention Output Injection
- Shape: `post [B,S,4]`, attention output `[B,S,D] -> [B,S,4,D]`.
- Role: writes one attention output stream back into all residual lanes with different coefficients.

#### `hc-write` - Attention HC Writeback
- Shape: mixed residual + injected attention -> `[B,S,4,D]`.
- Role: completes the attention sublayer writeback.
- Formula: `post * attn_out + comb @ residual`.

### Attention Q Path

#### `attention` - Hybrid Attention
- Shape: `[B,S,D] -> [B,S,D]`.
- Role: umbrella node for Q path, shared KV path, SWA, compressed cache, indexer, sparse attention, and grouped output projection.
- Mode: `R=4` CSA, `R=128` HCA, `R=0` SWA-only/MTP.

#### `q-path` - Q LoRA Path
- Shape: `[B,S,D] -> [B,S,H,512]`.
- Role: compact overview node for query construction.

#### `q-wqa` - `wq_a`
- Shape: `[B,S,D] -> [B,S,Qr]`.
- Role: low-rank query projection A.
- Pro: `Qr=1536`.

#### `q-norm` - `q_norm`
- Shape: `[B,S,Qr] -> [B,S,Qr]`.
- Role: RMSNorm in the low-rank query space.
- Detail: output is reused by both main Q expansion and the CSA Lightning indexer.

#### `q-wqb` - `wq_b`
- Shape: `[B,S,Qr] -> [B,S,H*512]`.
- Role: expands low-rank query to per-head query vectors.

#### `q-reshape` - Head Reshape + q Renorm
- Shape: `[B,S,H*512] -> [B,S,H,512]`.
- Role: separates heads and applies per-head RMS re-normalization.

#### `q-rope` - Q RoPE Slice
- Shape: `[B,S,H,512] -> [B,S,H,512]`.
- Role: applies RoPE to the last `64` dims only.
- Detail: non-RoPE dims stay as content dimensions.

### Shared KV + SWA

#### `kv-path` - Shared KV Path
- Shape: `[B,S,D] -> [B,S,512]`.
- Role: compact overview node for shared KV construction.
- Important: V4 has one shared KV vector, not per-head K/V.

#### `kv-wkv` - `wkv`
- Shape: `[B,S,D] -> [B,S,512]`.
- Role: projects hidden state into shared KV space.

#### `kv-norm` - `kv_norm`
- Shape: `[B,S,512] -> [B,S,512]`.
- Role: normalizes shared KV.

#### `kv-rope-quant` - KV RoPE + FP8 Sim
- Shape: `[B,S,512] -> [B,S,512]`.
- Role: applies RoPE to the final 64 dims and FP8-simulates non-RoPE dims.
- Detail: code keeps RoPE dims BF16 for positional precision and quantizes non-RoPE dims.

#### `kv-cache` - KV Cache
- Shape: `[B,S,512] -> [B,128+T/R,512]`.
- Role: logical cache containing an uncompressed SWA window prefix plus compressed suffix.
- Runtime note: decode writes the current uncompressed KV to `start_pos % 128`.

#### `window-topk` - SWA Window IDs
- Shape: `start_pos, S -> [B,S,<=128]`.
- Role: selects local uncompressed KV entries.
- Detail: present in CSA, HCA, and SWA-only modes.

#### `attn-selected` - Attention KV Set
- Shape: SWA ids + active compressed ids -> selected KV positions.
- Role: merges local SWA ids with CSA top-k compressed ids or HCA all-compressed ids.
- Detail: in pure SWA mode, the compressed side is absent.

### Compression: CSA/HCA

#### `compressor` - KV Compressor
- Shape: `[B,S,D] -> [B,floor(S/R),512]`.
- Role: compact node for compressed KV entry generation.
- Modes: active when `R=4` or `R=128`.

#### `comp-wkv` - Compressor `wkv`
- Shape: `[B,S,D] -> [B,S,Coff*512]`.
- Role: candidate value/key vectors for compression.
- Detail: `Coff=2` for CSA/c4a overlap; `Coff=1` for HCA/c128a.

#### `comp-wgate` - Compressor `wgate`
- Shape: `[B,S,D] -> [B,S,Coff*512]`.
- Role: produces per-token scores for softmax-gated pooling.
- Detail: learned `ape [R,Coff*512]` is added to scores.

#### `tail-state` - Compressed Tail State
- Shape: remainder tokens -> `kv_state / score_state`.
- Role: buffers partial blocks during prefill/decode.
- CSA/c4a: vLLM describes the runtime state as an 8-token sliding compressor state.
- HCA/c128a: runtime state is 128 tokens.

#### `overlap-transform` - CSA Overlap Transform
- Shape: `[B,blocks,R,2*512] -> [B,blocks,2R,512]`.
- Role: c4a overlap construction.
- Detail: for `R=4`, one compressed entry pools over up to 8 native tokens with stride 4. Previous block half and current block half are combined before gated pooling.

#### `gated-pool` - Softmax-Gated Pool
- Shape: `kv, score+ape -> [B,blocks,512]`.
- Role: weighted sum that creates the compressed KV entry.
- Detail: score softmax is over the token/window axis, not over feature dim.

#### `comp-norm-rope` - Norm + Compressed RoPE
- Shape: `[B,blocks,512] -> [B,blocks,512]`.
- Role: normalizes compressed KV and applies compressed-position RoPE.
- Anchor detail: prefill uses positions spaced by `R`; decode uses the completed block anchor `start_pos + 1 - R`.

#### `comp-cache-write` - Compressed Cache Write
- Shape: `[B,blocks,512] -> kv_cache[:,128:]`.
- Role: writes compressed entries after the live SWA window region.
- Runtime note: allocator/page layout is runtime-owned; model code exposes logical cache shape and write indices.

#### `hca-all-compressed` - HCA All Compressed Blocks
- Shape: compressed cache `[B,T/128,512] -> [B,S,T/128]`.
- Role: HCA/c128a path attends densely to all valid compressed blocks.
- Detail: no Lightning indexer is constructed when `compress_ratio != 4`.

### Lightning Indexer: CSA-only

#### `indexer` - Lightning Indexer
- Shape: `x [B,S,D]`, `qr [B,S,Qr] -> [B,S,topK]`.
- Role: compact node for CSA sparse block selection.
- Pro: `topK=1024`. Flash: `topK=512`.

#### `idx-q` - Indexer Q
- Shape: `qr [B,S,Qr] -> [B,S,64,128]`.
- Role: projects query latent into 64 index heads.

#### `idx-rotate` - RoPE + Hadamard + FP4
- Shape: `[B,S,64,128] -> [B,S,64,128]`.
- Role: applies RoPE, Hadamard rotation, and FP4 activation quantization.
- Intent: low-cost index scoring path.

#### `idx-cache` - Index KV Cache
- Shape: `x [B,S,D] -> [B,T/4,128]`.
- Role: indexer-specific compressed KV cache, separate from main 512-dim attention cache.
- Detail: created through an indexer-owned compressor with `head_dim=128` and rotation enabled.

#### `idx-einsum` - Lightning Scores
- Shape: index query + index KV -> `[B,S,64,T/4]`.
- Role: computes multi-head dot-product scores over compressed blocks.
- Detail: code applies ReLU to scores.

#### `idx-weight` - `weights_proj` + Head Sum
- Shape: scores + weights `[B,S,64] -> [B,S,T/4]`.
- Role: combines per-index-head scores into one score per compressed block.

#### `idx-topk` - TopK + Offset
- Shape: `[B,S,T/4] -> [B,S,topK]`.
- Role: causal mask, top-k selection, and offset into the shared attention KV index space.
- Detail: offset distinguishes SWA window slots from compressed suffix slots.

### Hybrid Attention Kernel and Output Projection

#### `sparse-attn` - Hybrid Attention Kernel
- Shape: `q [B,S,H,512]`, `kv [B,N,512]`, `topk_idxs [B,S,Kv] -> [B,S,H,512]`.
- Role: gathers selected shared KV entries and runs sparse multi-head attention.
- Detail: includes `attn_sink [H]`.

#### `o-proj` - Grouped O Projection
- Shape: `[B,S,H,512] -> [B,S,D]`.
- Role: grouped low-rank output projection.
- Detail: code applies inverse RoPE to the output's RoPE slice before `wo_a/wo_b`, because shared KV would otherwise leak absolute position into the value-like output.

### mHC MoE Read/Write

#### `mhc-ffn` - mHC Pre/Post: FFN
- Shape: `[B,S,4,D] -> [B,S,4,D]`.
- Role: umbrella node for MoE-side mHC controller/data path.

#### `ffn-hc-flatten` - FFN Flatten HC Lanes
- Shape: `[B,S,4,D] -> [B,S,4D]`.
- Role: controller input for MoE sublayer.

#### `ffn-hc-controller` - FFN Controller Linear
- Shape: `[B,S,4D] -> [B,S,24]`.
- Role: produces MoE-side `pre/post/comb` logits.

#### `ffn-hc-sinkhorn` - FFN Split + Sinkhorn
- Shape: `[B,S,24] -> pre, post, comb`.
- Role: same split/Sinkhorn mechanism as attention, but with separate FFN parameters.

#### `hc-pre-moe` - MoE Read Data Path
- Shape: `pre [B,S,4]`, residual `[B,S,4,D] -> [B,S,D]`.
- Role: creates the single hidden stream for MoE routing and experts.

#### `ffn-residual-mix` - MoE Residual Lane Mixing
- Shape: `comb [B,S,4,4]`, residual `[B,S,4,D] -> [B,S,4,D]`.
- Role: mixes residual lanes for MoE writeback.

#### `ffn-post-inject` - MoE Output Injection
- Shape: `post [B,S,4]`, MoE output `[B,S,D] -> [B,S,4,D]`.
- Role: injects MoE output into residual lanes.

#### `hc-post-moe` - MoE HC Writeback
- Shape: mixed residual + injected MoE -> `[B,S,4,D]`.
- Role: final block output and next block input.

### MoE Router and Experts

#### `moe` - MoE Router + Experts
- Shape: `[B,S,D] -> [B,S,D]`.
- Role: umbrella node for DeepSeekMoE routing, routed experts, shared expert, and combine.
- Detail: all decoder blocks use MoE FFN in the official implementation.

#### `gate` - Router Gate
- Shape: `[B*S,D] -> ids/weights [B*S,6]`.
- Role: compact gate node.
- Detail: computes scores but early hash layers ignore score top-k for index selection.

#### `gate-score` - Gate Scores
- Shape: `[B*S,D] -> [B*S,E]`.
- Role: linear score projection over routed experts.
- Pro: `E=384`.
- Detail: official score function is `sqrtsoftplus`.

#### `hash-route` - Hash Route
- Shape: `input_ids -> [B*S,6]`.
- Role: first `num_hash_layers=3` layers use token-id table lookup.
- Detail: code uses `tid2eid[input_ids]`; no `scores.topk` for indices in those layers.
- Open question: public inference code does not explain how `tid2eid` avoids frequent-token expert skew.

#### `topk-route` - TopK Route
- Shape: `scores + bias -> [B*S,6]`.
- Role: later layers choose top-6 routed experts.
- Detail: bias affects selection only; weights gather from original unbiased scores.

#### `route-weights` - Normalize Weights
- Shape: selected scores -> `[B*S,6]`.
- Role: gathers selected scores, normalizes when score function is not softmax, then applies route scale.
- Pro: `route_scale=2.5`.

#### `routed-experts` - Routed Experts
- Shape: `[N_e,D] -> [N_e,D]`.
- Role: compact node for selected per-expert SwiGLU FFNs.
- Detail: expert weights are FP4 in Pro config.

#### `expert-dispatch` - Expert Dispatch
- Shape: ids, weights -> per-expert token batches.
- Role: groups token rows by selected expert id.
- Detail: code loops through local expert ids and uses `torch.where(indices == i)`.

#### `expert-w1w3` - `w1 / w3`
- Shape: `[N_e,D] -> gate/up [N_e,I]`.
- Role: SwiGLU gate and up projections.

#### `swiglu` - SwiGLU + Clamp
- Shape: gate/up -> `[N_e,I]`.
- Role: applies clamp and `silu(gate) * up`.
- Pro: `swiglu_limit=10.0`.

#### `expert-w2` - `w2` Down Projection
- Shape: `[N_e,I] -> [N_e,D]`.
- Role: returns expert output to hidden size.

#### `shared-expert` - Shared Expert
- Shape: `[B*S,D] -> [B*S,D]`.
- Role: one always-on SwiGLU expert added for every token.
- Detail: not routed; it is computed and added to routed expert accumulation.

#### `expert-combine` - Routed + Shared Combine
- Shape: routed output + shared output -> `[B*S,D]`.
- Role: accumulates routed outputs, all-reduces when tensor-parallel, and adds the shared expert.

### Final Output

#### `stack-exit` - Final Stack State
- Shape: after decoder layer 60 -> `[B,S,4,D]`.
- Role: visual boundary after all decoder layers finish.

#### `head` - Final HC Head + LM Head
- Shape: `[B,S,4,D] -> [B,V]`.
- Role: collapses HC lanes, normalizes, and projects only the last token to vocab logits.
- Detail: official `get_logits` uses `x[:, -1]`.

#### `mtp` - Final MTP Block
- Shape: hidden `[B,S,4,D]`, ids `[B,S] -> [B,V]`.
- Role: auxiliary next-token prediction branch.
- Detail: MTP has its own block path and reuses embedding/head modules.

#### `logits` - Final Logits
- Shape: `[B,D] -> [B,129280]`.
- Role: final token vocabulary scores.
- Detail: sampling is outside the model graph.

## Expanded Annotation Cards

This section is the richer source material for the graph UI. Each card is written as if the node were clicked and the right panel had to explain not only the tensor shape, but also the reason this component exists.

### Model Entry Cards

#### `input-ids`
- Tensor event: prompt/tokenizer output enters as integer ids `[B,S]`.
- Why it exists: this is the only sequence-level symbolic input the model receives. Later numeric activations are all derived from it.
- Hidden dependency: MoE hash-routing layers reuse the original token ids through `tid2eid[input_ids]`; this means input ids remain relevant beyond embedding for layers `0,1,2`.
- UI detail: show two outgoing conceptual uses: embedding lookup and early-layer hash routing metadata.
- Open issue: `tid2eid` construction/load balancing is not visible from inference code.

#### `embedding`
- Tensor event: `[B,S] -> [B,S,D]`.
- Why it exists: converts discrete ids into the continuous hidden state that all later controller, attention, and MoE paths consume.
- Runtime detail: vocab sharding masks out-of-range ids and all-reduces embedding outputs under tensor parallelism.
- UI detail: mark as "once before decoder stack", not a per-layer embedding.

#### `hc-expand`
- Tensor event: `[B,S,D] -> [B,S,4,D]`.
- Why it exists: V4 does not keep one residual stream; it creates four residual lanes so later sublayers can read and write mixtures of lanes.
- Stability angle: the extra lane axis is the substrate mHC operates on; without it, `pre/post/comb` would have nowhere to act.
- UI detail: show the lane axis as a real dimension, not as four separate models.

#### `stack-entry`
- Tensor event: `[B,S,4,D] -> decoder layer stack`.
- Why it exists: separates model entry from the representative selected layer drawn in the graph.
- UI detail: explain that selecting `CSA/HCA/SWA` changes the layer-mode view, not the existence of input/embedding nodes.

### mHC Attention Cards

#### `mhc-attn`
- Tensor event: wraps attention-side transformation from residual lanes to sublayer input and back to residual lanes.
- Why it exists: replaces ordinary residual `x + attention(x)` with a controller-generated read/write over four lanes.
- Runtime detail: no cache; all coefficients are recomputed from the current hidden state during inference.
- UI detail: label as "controller + data path around attention", not attention itself.

#### `hc-flatten`
- Tensor event: `[B,S,4,D] -> [B,S,4D]`.
- Why it exists: controller must see all residual lanes jointly to produce lane coefficients.
- Detail: this is a controller-path tensor, not the tensor consumed by attention.
- UI detail: draw this as side/control flow into coefficient generation.

#### `hc-controller`
- Tensor event: `[B,S,4D] -> [B,S,24]`.
- Why it exists: generates the 24 logits needed for `pre[4]`, `post[4]`, and `comb[4,4]`.
- Math detail: `mix_hc=(2+HC)*HC=(2+4)*4=24`.
- Stability detail: flattened activations are RMS-normalized before/with the controller linear output, reducing scale sensitivity.
- UI detail: show a split into `pre`, `post`, and `comb`.

#### `hc-sinkhorn`
- Tensor event: `[B,S,24] -> pre [B,S,4], post [B,S,4], comb [B,S,4,4]`.
- Why it exists: converts unconstrained controller logits into usable read/write coefficients.
- mHC point: `comb` is projected by Sinkhorn-style row/column normalization toward a doubly stochastic matrix.
- Stability point: the doubly stochastic constraint keeps lane mixing from arbitrarily amplifying the residual stream; this is the "manifold-constrained" part that should be visually emphasized.
- UI detail: show `pre/post` as vectors and `comb` as a 4x4 heatmap.

#### `hc-read`
- Tensor event: `pre [B,S,4]` reads `[B,S,4,D] -> [B,S,D]`.
- Why it exists: attention still expects a single hidden stream, so mHC must collapse four residual lanes before attention.
- Formula: `x_attn = sum_l pre_l * lane_l`.
- UI detail: this is the first data-path use of controller coefficients.

#### `attn-residual-mix`
- Tensor event: `comb [B,S,4,4]` mixes old residual lanes into new residual lanes.
- Why it exists: carries residual information across layers without a simple identity addition.
- Important distinction: this is residual lane-to-lane transport; it is separate from injecting the attention output.
- UI detail: render as a matrix applied to four lanes.

#### `attn-post-inject`
- Tensor event: `post [B,S,4]`, attention output `[B,S,D] -> [B,S,4,D]`.
- Why it exists: distributes the newly computed attention output back into the lane space.
- Detail: every lane can receive a different amount of the same attention output.
- UI detail: show one stream fan-out into four lanes.

#### `hc-write`
- Tensor event: mixed residual + injected attention -> `[B,S,4,D]`.
- Why it exists: finalizes the attention sublayer state.
- Formula: `new_lanes = comb @ residual_lanes + post * attention_out`.
- UI detail: this is the attention-side replacement for residual add.

### Attention Q Cards

#### `attention`
- Tensor event: `[B,S,D] -> [B,S,D]`.
- Why it exists: represents the inner attention computation after mHC has read a single hidden stream.
- Mode detail: the drawn internals differ by selected layer mode: `CSA/c4a`, `HCA/c128a`, or `SWA-only`.
- UI detail: avoid showing CSA and HCA as simultaneous if the layer mode says only one active compressed path exists.

#### `q-path`
- Tensor event: compact summary of `wq_a -> q_norm -> wq_b -> reshape -> RoPE`.
- Why it exists: V4 keeps query construction low-rank before expanding to many heads.
- UI detail: useful in overview, but expanded graph should show the subnodes below.

#### `q-wqa`
- Tensor event: `[B,S,D] -> [B,S,Qr]`.
- Why it exists: low-rank query projection lowers parameter/runtime cost before head expansion.
- Pro detail: `Qr=1536`; Flash has lower rank.
- UI detail: mark as LoRA-like low-rank A projection.

#### `q-norm`
- Tensor event: `[B,S,Qr] -> [B,S,Qr]`.
- Why it exists: stabilizes the low-rank query latent before it branches.
- Branch detail: output feeds both main query expansion and CSA Lightning indexer query projection.
- UI detail: show it as a fork point.

#### `q-wqb`
- Tensor event: `[B,S,Qr] -> [B,S,H*512]`.
- Why it exists: expands compact query latent into per-head 512-dim queries.
- Pro detail: `H=128`, so logical output last dim is `65536`.
- UI detail: highlight large fan-out from low-rank to full head space.

#### `q-reshape`
- Tensor event: `[B,S,H*512] -> [B,S,H,512]`.
- Why it exists: exposes the head axis needed by sparse attention.
- Detail: code also renormalizes per head after reshape.
- UI detail: show axis split, not a learned operation.

#### `q-rope`
- Tensor event: `[B,S,H,512] -> [B,S,H,512]`.
- Why it exists: injects absolute/relative position information into the query's RoPE slice.
- Detail: only final `64` dims rotate; the other `448` dims remain non-RoPE content dims.
- UI detail: show `448 no-RoPE + 64 RoPE` inside each head.

### Shared KV, SWA, And KV Sharing Cards

#### `kv-path`
- Tensor event: compact summary of shared KV creation.
- Why it exists: DeepSeek V4 uses one shared 512-dim KV stream instead of per-head KV, which saves cache memory.
- Consequence: query is multi-head, KV is shared; output must later undo positional rotation with inverse RoPE.
- UI detail: explicitly label as "shared KV head".

#### `kv-wkv`
- Tensor event: `[B,S,D] -> [B,S,512]`.
- Why it exists: creates the single shared KV vector used as both key-like and value-like cache entry.
- Design tension: K wants RoPE for positional attention, while V normally should not carry absolute position. V4 resolves this with inverse RoPE after attention.
- UI detail: this node is the root of the KV sharing story.

#### `kv-norm`
- Tensor event: `[B,S,512] -> [B,S,512]`.
- Why it exists: keeps shared KV scale stable before RoPE, quantization, compression, and cache write.
- UI detail: pure normalization, but important because the same shared tensor goes to several branches.

#### `kv-rope-quant`
- Tensor event: applies RoPE to last 64 dims and FP8 simulation to non-RoPE dims.
- Why it exists: keys need positional information, and cache memory needs aggressive quantization.
- Detail: RoPE dims stay BF16 for positional precision; non-RoPE dims use FP8-style quantization.
- UI detail: split KV into `448 fp8-ish content dims` and `64 bf16 RoPE dims`.

#### `kv-cache`
- Tensor event: logical cache `[B,128 + T/R,512]`.
- Why it exists: one cache abstraction holds both recent uncompressed SWA entries and older compressed entries.
- Runtime detail: SWA lives in the first 128 slots; compressed entries live after that.
- UI detail: draw cache as two regions: `window prefix` and `compressed suffix`.

#### `window-topk`
- Tensor event: `start_pos,S -> [B,S,<=128]` indices.
- Why it exists: preserves local information that compression cannot safely represent, especially near causality boundaries.
- Runtime detail: decode uses a ring-buffer-like current-token write at `start_pos % 128`.
- UI detail: call it SWA/local window, not top-k over scores.

#### `attn-selected`
- Tensor event: concatenates/merges index lists for the attention kernel.
- Why it exists: the kernel expects selected KV positions; those positions can come from SWA, CSA indexer, or HCA dense compressed ids.
- Mode detail: CSA = SWA ids + indexer topK; HCA = SWA ids + all valid c128a ids; SWA-only = SWA ids only.
- UI detail: show this as an index-set union node.

### Compression Cards

#### `compressor`
- Tensor event: `[B,S,D] -> [B,floor(S/R),512]`.
- Why it exists: old tokens cannot remain uncompressed at 1M context; compression creates cheaper cache entries.
- Mode detail: `R=4` CSA/c4a and `R=128` HCA/c128a use the same conceptual compressor with different ratio/overlap behavior.
- UI detail: show `c4a` and `c128a` sublabels depending on selected mode.

#### `comp-wkv`
- Tensor event: `[B,S,D] -> [B,S,Coff*512]`.
- Why it exists: builds candidate vectors that will be pooled into compressed KV entries.
- c4a detail: `Coff=2` because overlap stores previous/current halves.
- c128a detail: `Coff=1` because it pools non-overlapping 128-token chunks.

#### `comp-wgate`
- Tensor event: `[B,S,D] -> [B,S,Coff*512]`.
- Why it exists: compression is not a plain average; it learns per-token/per-dim pooling scores.
- Detail: `ape` is added before softmax to bias token positions inside the compression window.
- UI detail: show this as "where the weighted sum weights come from".

#### `tail-state`
- Tensor event: incomplete native tokens -> `kv_state` and `score_state`.
- Why it exists: during decode, compression boundaries do not occur every token. The model must retain partial windows until enough tokens arrive.
- c4a detail: runtime state is effectively an 8-token overlapped sliding compressor state.
- c128a detail: runtime state is 128 tokens.
- UI detail: mark this as persistent per-request state, unlike projections.

#### `overlap-transform`
- Tensor event: `[B,blocks,R,2*512] -> [B,blocks,2R,512]`.
- Why it exists: c4a compresses by 4x but each compressed entry can see an 8-token window with stride 4.
- Detail: previous block half and current block half are arranged into one pooling window; missing positions use zero or `-inf` score padding.
- UI detail: show native token spans like `[j*4-4, ..., j*4+3]` conceptually, while warning exact boundary handling follows code.

#### `gated-pool`
- Tensor event: weighted sum over compression-window tokens -> `[B,blocks,512]`.
- Why it exists: creates one cache entry that keeps salient information from several native tokens.
- Math detail: `kv * softmax(score + ape)` summed over the token/window axis.
- UI detail: show this as "8-to-1" for c4a and "128-to-1" for c128a.

#### `comp-norm-rope`
- Tensor event: compressed entry normalized, then RoPE applied at an anchor position.
- Why it exists: compressed entries still need positional phase for attention scoring.
- Anchor detail: c4a anchors advance by stride 4; c128a anchors advance by 128. Code uses `freqs_cis[:cutoff:ratio]` for prefill and `start_pos + 1 - ratio` at decode completion.
- UI detail: expose anchor position separately from the native token span.

#### `comp-cache-write`
- Tensor event: compressed KV entry -> compressed suffix of cache.
- Why it exists: stores the compressed long-context memory that replaces most ordinary KV cache entries.
- Runtime detail: in model code, compressed cache index is roughly `start_pos // ratio`; serving engines may remap this to pages/blocks.
- UI detail: distinguish logical model index from runtime allocator page.

#### `hca-all-compressed`
- Tensor event: all valid c128a compressed entries become attention candidates.
- Why it exists: with 128x compression, the compressed sequence is small enough for dense attention over compressed blocks.
- Detail: no Lightning indexer path in HCA; `self.indexer=None` when ratio is 128.
- UI detail: show HCA as dense over compressed memory plus SWA, not sparse-selected top-k.

### Lightning Indexer Cards

#### `indexer`
- Tensor event: `x, qr, start_pos, offset -> [B,S,topK]`.
- Why it exists: c4a still has too many compressed blocks at 1M context, so CSA needs sparse selection over compressed entries.
- Pro/Flash detail: Pro config topK is 1024; Flash topK is 512.
- UI detail: show it only in CSA/c4a mode.

#### `idx-q`
- Tensor event: `qr [B,S,Qr] -> [B,S,64,128]`.
- Why it exists: builds cheaper index-query heads for scoring compressed blocks.
- Detail: indexer query derives from already-normalized query latent, not from final main Q heads.

#### `idx-rotate`
- Tensor event: index query -> RoPE + Hadamard-rotated + FP4-simulated query.
- Why it exists: the indexer must be cheap enough to run as a side path; FP4 and Hadamard reduce runtime/memory pressure.
- UI detail: label this as approximate/quantized retrieval scoring, not the main attention.

#### `idx-cache`
- Tensor event: `x [B,S,D] -> [B,T/4,128]`.
- Why it exists: indexer needs its own smaller compressed keys for block scoring.
- Detail: it uses a separate compressor with `head_dim=128` and rotation enabled; it is not the same tensor as main compressed KV `[B,T/4,512]`.

#### `idx-einsum`
- Tensor event: `[B,S,64,128] x [B,T/4,128] -> [B,S,64,T/4]`.
- Why it exists: computes candidate compressed-block relevance for each query token.
- Detail: ReLU is applied before weighted head sum; negative evidence is discarded.

#### `idx-weight`
- Tensor event: per-head scores + `weights_proj(x) [B,S,64] -> [B,S,T/4]`.
- Why it exists: the indexer has many scoring heads, but attention needs one compressed-block rank list.
- Detail: score head weights depend on the current hidden state.

#### `idx-topk`
- Tensor event: `[B,S,T/4] -> [B,S,topK]`.
- Why it exists: bounds attention compute by selecting only top compressed blocks.
- Causality detail: masks compressed blocks that would include future native tokens.
- Offset detail: adds cache offset so selected compressed ids line up after SWA window slots.

### Attention Kernel And Output Cards

#### `sparse-attn`
- Tensor event: selected shared KV entries + multi-head Q -> `[B,S,H,512]`.
- Why it exists: unifies SWA, CSA, and HCA into one gather-based attention kernel.
- Detail: even HCA can be represented as sparse attention with "all valid compressed ids" plus SWA ids.
- UI detail: show actual candidate count for selected mode: `<=128`, `topK+128`, or `T/128+128`.

#### `o-proj`
- Tensor event: `[B,S,H,512] -> [B,S,D]`.
- Why it exists: collapses the multi-head attention output back to hidden size using grouped low-rank projection.
- KV sharing detail: inverse RoPE is applied to the output RoPE slice before projection so value-like output does not carry absolute position.
- UI detail: this is where the shared-KV design is repaired for value semantics.

### mHC MoE Cards

#### `mhc-ffn`
- Tensor event: wraps MoE-side mHC read/write around MoE.
- Why it exists: MoE, like attention, consumes one stream and writes back into four residual lanes.
- Detail: uses a separate FFN mHC parameter set from attention mHC.

#### `ffn-hc-flatten`
- Tensor event: `[B,S,4,D] -> [B,S,4D]`.
- Why it exists: controller must observe all lanes before deciding MoE read/write coefficients.

#### `ffn-hc-controller`
- Tensor event: `[B,S,4D] -> [B,S,24]`.
- Why it exists: produces MoE-side `pre/post/comb`.
- UI detail: visually mirror attention controller but label parameter set as independent.

#### `ffn-hc-sinkhorn`
- Tensor event: MoE mHC logits -> `pre/post/comb`.
- Why it exists: imposes the same manifold constraint on MoE residual lane mixing.
- UI detail: re-use the 4x4 matrix visual from attention.

#### `hc-pre-moe`
- Tensor event: `pre` reads four lanes into `[B,S,D]`.
- Why it exists: router and experts operate on a single hidden vector per token.

#### `ffn-residual-mix`
- Tensor event: `comb @ residual_lanes -> [B,S,4,D]`.
- Why it exists: carries previous lane information across the MoE sublayer.

#### `ffn-post-inject`
- Tensor event: `post * moe_out -> [B,S,4,D]`.
- Why it exists: distributes MoE output into residual lanes.

#### `hc-post-moe`
- Tensor event: residual mix + MoE injection -> next `[B,S,4,D]`.
- Why it exists: creates the next decoder block's residual lane state.

### MoE Cards

#### `moe`
- Tensor event: `[B,S,D] -> [B,S,D]`.
- Why it exists: FFN capacity is sparse-routed across 384 routed experts plus one shared expert.
- Architectural note: all decoder layers use MoE in the public implementation.
- UI detail: emphasize that shared expert is always-on and routed experts are conditional.

#### `gate`
- Tensor event: hidden rows -> expert ids and route weights.
- Why it exists: decides sparse FFN computation per token.
- Dual behavior: hash-routed early layers choose ids from token id; later layers choose ids by score top-k.

#### `gate-score`
- Tensor event: `[B*S,D] -> [B*S,E]`.
- Why it exists: computes expert affinity scores.
- Detail: public code computes these even in hash layers, because weights are gathered from scores after ids are chosen.
- UI detail: avoid implying hash layers do no scoring at all; they skip score top-k selection, not necessarily score computation.

#### `hash-route`
- Tensor event: `input_ids -> [B*S,6]` expert ids.
- Why it exists: early token-level routing uses a fixed token-id to expert-id table.
- Efficiency hypothesis: this avoids expensive score-based top-k for early layers and can encode lexical/token priors.
- Open issue: public code does not show how frequent-token expert skew is mitigated; likely answer requires training report or checkpoint distribution analysis.

#### `topk-route`
- Tensor event: biased scores -> top-6 expert ids.
- Why it exists: later layers use activation-dependent routing once hidden representations are richer.
- Detail: bias changes selection only, while route weights gather original scores.

#### `route-weights`
- Tensor event: selected original scores -> normalized route weights.
- Why it exists: scales expert outputs after dispatch.
- Detail: with `sqrtsoftplus`, weights are normalized by selected-score sum and multiplied by `route_scale=2.5`.

#### `routed-experts`
- Tensor event: selected token rows -> selected expert FFNs.
- Why it exists: gives model high parameter count without activating all experts per token.
- Detail: Pro has 384 routed experts, 6 activated per token.

#### `expert-dispatch`
- Tensor event: ids/weights -> per-expert mini-batches.
- Why it exists: groups sparse token work so each expert processes only assigned rows.
- Runtime detail: distributed runs all-reduce routed outputs across expert/tensor parallel ranks.

#### `expert-w1w3`
- Tensor event: `[N_e,D] -> gate/up [N_e,I]`.
- Why it exists: SwiGLU FFN uses separate gate and up projections.
- Detail: expert weights can be FP4 in Pro.

#### `swiglu`
- Tensor event: gate/up -> `[N_e,I]`.
- Why it exists: non-linear expert transformation.
- Detail: `up` and `gate` are clamped when `swiglu_limit > 0`; Pro uses `10.0`.

#### `expert-w2`
- Tensor event: `[N_e,I] -> [N_e,D]`.
- Why it exists: returns expert intermediate back to hidden size.

#### `shared-expert`
- Tensor event: `[B*S,D] -> [B*S,D]`.
- Why it exists: always-on dense-ish expert path gives every token a common FFN component independent of sparse routing.
- MoE balancing intuition: shared expert can absorb common transformations, reducing pressure on routed experts to handle universal token patterns. This is plausible but not proven as the sole answer to hash-routing skew.
- UI detail: draw as parallel to routed experts and always active.

#### `expert-combine`
- Tensor event: routed weighted sum + shared expert -> `[B*S,D]`.
- Why it exists: merges conditional expert computation and universal shared expert computation.
- Formula: routed accumulation is initialized to zeros, selected experts add weighted outputs, then shared expert is added.

### Final Output Cards

#### `stack-exit`
- Tensor event: final decoder layer output `[B,S,4,D]`.
- Why it exists: separates repeated decoder layers from output-only heads.
- UI detail: prevents the false reading that LM head runs every layer.

#### `head`
- Tensor event: `[B,S,4,D] -> [B,V]`.
- Why it exists: collapses HC lanes to a single hidden stream, normalizes, and projects to vocabulary.
- Detail: code computes logits only for `x[:, -1]`, so it is final-token generation output, not per-layer logits.

#### `mtp`
- Tensor event: final hidden + ids -> auxiliary logits.
- Why it exists: multi-token prediction branch, reused from earlier DeepSeek design.
- Detail: MTP is represented as final branch; selected graph's `R=0` mode is useful for showing its SWA-only attention.

#### `logits`
- Tensor event: `[B,D] -> [B,129280]`.
- Why it exists: vocabulary score vector for generation.
- UI detail: sampling/top-p/tool decoding are outside this architecture graph.

## Runtime Responsibilities To Represent Separately

These are not just model-weight nodes; they are serving/runtime responsibilities:

1. SWA ring-buffer management:
   - Maintain 128 uncompressed KV entries per attention layer.
   - Decode write index: `start_pos % 128`.
2. Compressed cache suffix:
   - Maintain `T/4` c4a entries or `T/128` c128a entries after the SWA prefix.
   - Respect causality masks for compressed blocks.
3. Compressor tail state:
   - Keep partial c4a/c128a blocks until compression boundary.
   - vLLM treats this as sliding-window-like state for prefix caching/disaggregated prefill.
4. Heterogeneous layer modes:
   - Pro: layers 0-1 HCA; layers 2-60 alternate CSA/HCA; MTP SWA-only.
   - Flash: starts with two SWA-only layers, then CSA/HCA alternation, then MTP SWA-only.
5. Indexer cache:
   - CSA-only cache `[B,T/4,128]`.
   - Pro topK differs from Flash.
6. Quantization/runtime kernels:
   - Main attention cache uses FP8-style storage/simulation except RoPE dims.
   - Indexer path uses FP4 activation quantization.
   - Expert weights are FP4 in Pro.

## Open Questions

1. Early hash-routing load balance:
   - Confirmed mechanism: `tid2eid[input_ids]`.
   - Missing: how `tid2eid` was constructed/trained to avoid frequent-token expert concentration.
   - Possible investigation: inspect `tid2eid` tensor distribution in checkpoint shards or find the relevant section in `DeepSeek_V4.pdf`.
2. Exact paper terminology:
   - Code uses `compress_ratio`; public runtime writing uses `c4a/c128a`; official blog uses `CSA/HCA`.
   - UI should show both: `CSA / c4a` and `HCA / c128a`.
3. Causality ranges for anchors:
   - vLLM documents exact native-token ranges.
   - The graph should eventually show compressed entry `j` spans and anchor position, not only `R`.
4. Prefill vs decode branches:
   - Current graph merges both.
   - Future detail view should show separate prefill and decode mini-flows for SWA write, compressor tail, and cache insertion.
