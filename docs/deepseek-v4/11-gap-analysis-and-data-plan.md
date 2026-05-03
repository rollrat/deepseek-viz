# DeepSeek V4 graph gap analysis and data plan

이 문서는 현재 시각화에서 빠진 내부 요소를 공식 inference code 기준으로 다시 쪼개기 위한 체크리스트다.

## Current gaps

### mHC

- [ ] `hc_pre` controller path: `[B,S,4,D] -> [B,S,4D] -> F.linear -> mixes [B,S,24]`
- [ ] `rsqrt` normalization applied to controller output
- [ ] `hc_split_sinkhorn` kernel
- [ ] `pre [B,S,4]`
- [ ] `post [B,S,4]`
- [ ] `comb [B,S,4,4]`
- [ ] data path: `sum(pre * residual lanes) -> [B,S,D]`
- [ ] post path: `post * sublayer_out + comb * residual`
- [ ] attention mHC and FFN mHC as separate parameter sets
- [ ] head `hc_head` collapse

### Attention

- [ ] q LoRA path: `wq_a -> q_norm -> wq_b`
- [ ] query head reshape: `[B,S,H,512]`
- [ ] query RMS re-normalization
- [ ] RoPE slice `Rd=64`, non-RoPE slice `448`
- [ ] `wkv -> kv_norm`
- [ ] non-RoPE FP8 simulation
- [ ] `attn_sink`
- [ ] sparse attention selected indices
- [ ] inverse RoPE on output
- [ ] grouped output projection: `wo_a`, `wo_b`

### KV cache and compressor

- [ ] `kv_cache = window cache + compressed cache`
- [ ] prefill write path
- [ ] decode circular write path
- [ ] `get_window_topk_idxs`
- [ ] `get_compress_topk_idxs`
- [ ] compressor `wkv`, `wgate`
- [ ] learned positional gate bias `ape`
- [ ] `kv_state` / `score_state`
- [ ] compressed tail state handling
- [ ] cutoff / remainder split
- [ ] `overlap_transform` for R=4
- [ ] softmax-gated pooling
- [ ] compressor norm
- [ ] compressed RoPE
- [ ] compressed cache write

### Lightning indexer

- [ ] index query projection from `qr`
- [ ] index RoPE
- [ ] Hadamard rotation
- [ ] FP4 activation quantization
- [ ] index compressor cache `[B,T/4,128]`
- [ ] `weights_proj`
- [ ] `einsum("bshd,btd->bsht")`
- [ ] ReLU score
- [ ] head-weighted sum
- [ ] causal mask
- [ ] top-k compressed block ids
- [ ] offset adjustment

### MoE

- [ ] `Gate`
- [ ] hash routing for first 3 layers
- [ ] score routing after hash layers
- [ ] `sqrtsoftplus`
- [ ] router bias
- [ ] top-6 ids and weights
- [ ] normalize selected weights
- [ ] route scale
- [ ] expert dispatch / counts
- [ ] expert `w1`, `w3`, `w2`
- [ ] SwiGLU and clamp
- [ ] routed expert FP4 weights
- [ ] shared expert
- [ ] all-reduce combine

## Data model

Graph data should not live inside renderer code. Each node should carry:

- `id`
- `title`
- `category`
- `input`
- `output`
- `summary`
- `params`
- `notes`
- `scene`
- `drill`
- `visibleWhen`
- `sources`

Each scene should carry:

- `id`
- `title`
- `subtitle`
- `nodes`
- `edges`
- `rankdir`
- optional `conditionNotes`

## Interaction model

- Single click: select node and update right panel.
- Double click: drill into node scene if it has one.
- Right panel button: drill into node scene if it has one.
- Breadcrumb: return to overview or parent scene.
- D3 zoom/pan stays active in all scenes.

