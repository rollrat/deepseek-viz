window.DSV4_GRAPH = (() => {
  const sources = {
    card: { label: "official-card", url: "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro" },
    proConfig: { label: "official-config: Pro", url: "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json" },
    flashConfig: { label: "official-config: Flash", url: "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json" },
    code: { label: "official-code", url: "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/inference/model.py" },
    kernel: { label: "official-kernel", url: "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/inference/kernel.py" },
    blog: { label: "explainer", url: "https://huggingface.co/blog/deepseekv4" },
  };

  const models = {
    pro: {
      label: "DeepSeek-V4-Pro",
      total: "1.6T",
      active: "49B",
      context: "1,048,576",
      D: 7168,
      layers: 61,
      H: 128,
      G: 16,
      Qr: 1536,
      Or: 1024,
      E: 384,
      I: 3072,
      indexTopK: 1024,
      routeScale: 2.5,
      schedule: [
        128, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
        128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
        128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
        128, 4, 128, 4, 128, 4, 0,
      ],
    },
    flash: {
      label: "DeepSeek-V4-Flash",
      total: "284B",
      active: "13B",
      context: "1,048,576",
      D: 4096,
      layers: 43,
      H: 64,
      G: 8,
      Qr: 1024,
      Or: 1024,
      E: 256,
      I: 2048,
      indexTopK: 512,
      routeScale: 1.5,
      schedule: [
        0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
        128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
        128, 4, 128, 4, 128, 4, 0,
      ],
    },
  };

  const common = {
    src: [sources.code, sources.proConfig, sources.flashConfig],
  };

  const nodes = {
    "input-ids": n("input-ids", "Input IDs", "stream", "prompt_text [string]", "ids [B,S]", "", { vocab: "129280" }, [], [sources.card, sources.proConfig, sources.flashConfig]),
    embedding: n("embedding", "Token Embedding", "stream", "[B,S]", "[B,S,D]", "", { V: "129280", D: "$D" }, [], common.src),
    "hc-expand": n("hc-expand", "HC Expand", "hc", "[B,S,D]", "[B,S,4,D]", "", { hc_mult: 4 }, [], common.src, "mhc"),
    "mhc-attn": n("mhc-attn", "mHC Pre/Post: Attention", "hc", "[B,S,4,D]", "[B,S,4,D]", "", { mix_hc: 24, hc_dim: "4D" }, [], [sources.code, sources.kernel], "mhc"),
    attention: n("attention", "Hybrid Attention", "attention", "[B,S,D]", "[B,S,D]", "", { H: "$H", R: "$R", Hd: 512 }, [], [sources.code, sources.blog], "attention"),
    "q-path": n("q-path", "Q LoRA Path", "attention", "[B,S,D]", "[B,S,H,512]", "", { q_lora_rank: "$Qr", rope_dim: 64 }, [], common.src, "attention"),
    "kv-path": n("kv-path", "Shared KV Path", "attention", "[B,S,D]", "[B,S,512]", "", { kv_heads: 1, nope_dim: 448, rope_dim: 64 }, [], common.src, "attention"),
    "kv-cache": n("kv-cache", "KV Cache", "cache", "[B,S,512]", "[B,128+T/R,512]", "", { window: 128, compressed: "max_seq_len/R" }, [], [sources.code], "compression"),
    compressor: n("compressor", "KV Compressor", "cache", "[B,S,D]", "[B,floor(S/R),512]", "", { R: "$R", ape: "[R,Coff*512]" }, [], [sources.code], "compression"),
    indexer: n("indexer", "Lightning Indexer", "attention", "x [B,S,D], qr [B,S,Qr]", "[B,S,topK]", "", { index_heads: 64, index_dim: 128, topK: "$indexTopK" }, [], [sources.code], "indexer", { ratio: 4 }),
    "sparse-attn": n("sparse-attn", "Sparse Attention", "attention", "q [B,S,H,512], kv_selected [B,S,N,512]", "heads [B,S,H,512]", "", { attn_sink: "[H]" }, [], [sources.code]),
    "o-proj": n("o-proj", "Grouped O Projection", "attention", "[B,S,H,512]", "[B,S,D]", "", { groups: "$G", o_lora_rank: "$Or" }, [], common.src),
    "mhc-ffn": n("mhc-ffn", "mHC Pre/Post: FFN", "hc", "[B,S,4,D]", "[B,S,4,D]", "", { mix_hc: 24 }, [], [sources.code, sources.kernel], "mhc"),
    moe: n("moe", "MoE Router + Experts", "routing", "[B,S,D]", "[B,S,D]", "", { E: "$E", K: 6, I: "$I" }, [], common.src, "moe"),
    gate: n("gate", "Router Gate", "routing", "tokens [B*S,D]", "expert_ids [B*S,6], weights [B*S,6]", "", { route_scale: "$routeScale", scoring: "sqrtsoftplus" }, [], [sources.code], "moe"),
    "routed-experts": n("routed-experts", "Routed Experts", "expert", "[N_e,D]", "[N_e,D]", "", { w1: "D->I", w3: "D->I", w2: "I->D" }, [], [sources.code, sources.card], "moe"),
    "shared-expert": n("shared-expert", "Shared Expert", "expert", "[B*S,D]", "[B*S,D]", "", { shared: 1, I: "$I" }, [], [sources.code], "moe"),
    "hc-post-moe": n("hc-post-moe", "MoE HC Writeback", "hc", "mixed_lanes [B,S,4,D], moe_inject [B,S,4,D]", "next_lanes [B,S,4,D]", "", { output: "[B,S,4,D]" }, [], [sources.code]),
    head: n("head", "HC Head + LM Head", "output", "[B,S,4,D]", "[B,V]", "", { V: 129280, D: "$D" }, [], common.src, "output"),
    mtp: n("mtp", "MTP Block", "output", "hidden [B,S,4,D], ids [B,S]", "mtp_logits [B,V]", "", { num_nextn: 1, R: 0 }, [], common.src, "output"),
    logits: n("logits", "Logits", "output", "[B,D]", "[B,129280]", "", { vocab: 129280 }, [], [sources.card, sources.code]),

    "hc-flatten": n("hc-flatten", "Flatten HC Lanes", "hc", "[B,S,4,D]", "[B,S,4D]", "", { hc_dim: "4D" }, [], [sources.code]),
    "hc-controller": n("hc-controller", "Controller Linear", "hc", "[B,S,4D]", "[B,S,24]", "", { weight: "[24,4D]" }, [], [sources.code]),
    "hc-split": n("hc-split", "Split Controller Mixes", "hc", "mixes [B,S,24]", "pre_logits [B,S,4], post_logits [B,S,4], comb_logits [B,S,4,4]", "", { hc: 4, mix_hc: 24 }, [], [sources.code, sources.kernel]),
    "hc-pre-sigmoid": n("hc-pre-sigmoid", "Pre: sigmoid + eps", "hc", "pre_logits [B,S,4]", "pre [B,S,4]", "", { scale: "hc_scale[0]", base: "hc_base[0:4]", eps: "1e-6" }, [], [sources.kernel]),
    "hc-post-sigmoid": n("hc-post-sigmoid", "Post: 2 * sigmoid", "hc", "post_logits [B,S,4]", "post [B,S,4]", "", { scale: "hc_scale[1]", base: "hc_base[4:8]" }, [], [sources.kernel]),
    "hc-comb-softmax": n("hc-comb-softmax", "Comb Row Softmax", "hc", "comb_logits [B,S,4,4]", "comb_row [B,S,4,4]", "", { scale: "hc_scale[2]", base: "hc_base[8:24]" }, [], [sources.kernel]),
    "hc-comb-sinkhorn": n("hc-comb-sinkhorn", "Comb Sinkhorn Normalize", "hc", "comb_row [B,S,4,4]", "comb [B,S,4,4]", "", { iters: 20, eps: "1e-6" }, [], [sources.kernel]),
    "hc-sinkhorn": n("hc-sinkhorn", "Split + Sinkhorn", "hc", "mixes [B,S,24]", "pre [B,S,4], post [B,S,4], comb [B,S,4,4]", "", { iters: 20, eps: "1e-6" }, [], [sources.kernel]),
    "hc-read": n("hc-read", "Read Data Path", "hc", "pre [B,S,4], X [B,S,4,D]", "[B,S,D]", "", {}, [], [sources.code]),
    "attn-residual-mix": n("attn-residual-mix", "Attention Residual Lane Mixing", "hc", "comb [B,S,4,4], residual [B,S,4,D]", "[B,S,4,D]", "", { comb: "[B,S,4,4]", lanes: 4 }, [], [sources.code, sources.kernel]),
    "attn-post-inject": n("attn-post-inject", "Attention Output Injection", "hc", "post [B,S,4], y [B,S,D]", "[B,S,4,D]", "", { post: "[B,S,4]" }, [], [sources.code]),
    "hc-write": n("hc-write", "Attention HC Writeback", "hc", "mixed_lanes [B,S,4,D], attn_inject [B,S,4,D]", "next_lanes [B,S,4,D]", "", {}, [], [sources.code]),
    "ffn-hc-flatten": n("ffn-hc-flatten", "FFN Flatten HC Lanes", "hc", "[B,S,4,D]", "[B,S,4D]", "", { hc_dim: "4D" }, [], [sources.code]),
    "ffn-hc-controller": n("ffn-hc-controller", "FFN Controller Linear", "hc", "[B,S,4D]", "[B,S,24]", "", { weight: "[24,4D]", mix_hc: 24 }, [], [sources.code]),
    "ffn-hc-split": n("ffn-hc-split", "FFN Split Controller Mixes", "hc", "mixes [B,S,24]", "pre_logits [B,S,4], post_logits [B,S,4], comb_logits [B,S,4,4]", "", { hc: 4, mix_hc: 24 }, [], [sources.code, sources.kernel]),
    "ffn-hc-pre-sigmoid": n("ffn-hc-pre-sigmoid", "FFN Pre: sigmoid + eps", "hc", "pre_logits [B,S,4]", "pre [B,S,4]", "", { scale: "hc_scale[0]", eps: "1e-6" }, [], [sources.kernel]),
    "ffn-hc-post-sigmoid": n("ffn-hc-post-sigmoid", "FFN Post: 2 * sigmoid", "hc", "post_logits [B,S,4]", "post [B,S,4]", "", { scale: "hc_scale[1]" }, [], [sources.kernel]),
    "ffn-hc-comb-softmax": n("ffn-hc-comb-softmax", "FFN Comb Row Softmax", "hc", "comb_logits [B,S,4,4]", "comb_row [B,S,4,4]", "", { scale: "hc_scale[2]" }, [], [sources.kernel]),
    "ffn-hc-comb-sinkhorn": n("ffn-hc-comb-sinkhorn", "FFN Comb Sinkhorn Normalize", "hc", "comb_row [B,S,4,4]", "comb [B,S,4,4]", "", { iters: 20, eps: "1e-6" }, [], [sources.kernel]),
    "ffn-hc-sinkhorn": n("ffn-hc-sinkhorn", "FFN Split + Sinkhorn", "hc", "mixes [B,S,24]", "pre [B,S,4], post [B,S,4], comb [B,S,4,4]", "", { comb: "[B,S,4,4]" }, [], [sources.kernel]),
    "hc-pre-moe": n("hc-pre-moe", "MoE Read Data Path", "hc", "pre [B,S,4], residual [B,S,4,D]", "[B,S,D]", "", { pre: "[B,S,4]" }, [], [sources.code, sources.kernel]),
    "ffn-residual-mix": n("ffn-residual-mix", "MoE Residual Lane Mixing", "hc", "comb [B,S,4,4], residual [B,S,4,D]", "[B,S,4,D]", "", { comb: "[B,S,4,4]", lanes: 4 }, [], [sources.code, sources.kernel]),
    "ffn-post-inject": n("ffn-post-inject", "MoE Output Injection", "hc", "post [B,S,4], moe [B,S,D]", "[B,S,4,D]", "", { post: "[B,S,4]" }, [], [sources.code]),

    "q-wqa": n("q-wqa", "wq_a", "attention", "[B,S,D]", "[B,S,Qr]", "", { Qr: "$Qr" }, [], [sources.code]),
    "q-norm": n("q-norm", "q_norm", "attention", "[B,S,Qr]", "[B,S,Qr]", "", { eps: "1e-6" }, [], [sources.code]),
    "q-wqb": n("q-wqb", "wq_b", "attention", "[B,S,Qr]", "[B,S,H*512]", "", { H: "$H" }, [], [sources.code]),
    "q-reshape": n("q-reshape", "Head Reshape + q Renorm", "attention", "[B,S,H*512]", "[B,S,H,512]", "", {}, [], [sources.code]),
    "q-rope": n("q-rope", "Q RoPE Slice", "attention", "[B,S,H,512]", "[B,S,H,512]", "", { rope_dim: 64 }, [], [sources.code]),
    "kv-wkv": n("kv-wkv", "wkv", "attention", "[B,S,D]", "[B,S,512]", "", { kv_heads: 1 }, [], [sources.code]),
    "kv-norm": n("kv-norm", "kv_norm", "attention", "[B,S,512]", "[B,S,512]", "", {}, [], [sources.code]),
    "kv-rope-quant": n("kv-rope-quant", "KV RoPE + FP8 Sim", "attention", "[B,S,512]", "[B,S,512]", "", { rope: 64, nope: 448 }, [], [sources.code, sources.kernel]),
    "kv-slice": n("kv-slice", "KV Slice Split", "attention", "kv_norm [B,S,512]", "kv_nope [B,S,448], kv_rope [B,S,64]", "", { nope: 448, rope: 64 }, [], [sources.code]),
    "window-topk": n("window-topk", "Window TopK IDs", "cache", "start_pos [scalar], query_len [S]", "window_ids [B,S,W<=128]", "", { window: 128 }, [], [sources.code]),
    "swa-prefill-write": n("swa-prefill-write", "SWA Prefill Write", "cache", "kv [B,S,512]", "window_cache [B,min(S,128),512]", "", { window: 128 }, [], [sources.code]),
    "swa-decode-write": n("swa-decode-write", "SWA Decode Ring Write", "cache", "kv_t [B,1,512], start_pos [scalar]", "window_cache[:, start_pos % 128] [B,512]", "", { slot: "start_pos % 128" }, [], [sources.code]),
    "cache-layout": n("cache-layout", "Logical Cache Layout", "cache", "swa_cache [B,128,512], c_cache [B,T/R,512]", "kv_cache [B,128+T/R,512]", "", { prefix: 128, suffix: "T/R" }, [], [sources.code]),
    "attn-selected": n("attn-selected", "Selected KV IDs", "attention", "window_ids [B,S,W], compressed_ids [B,S,C]", "selected_ids [B,S,W+C]", "", {}, [], [sources.code]),

    "comp-wkv": n("comp-wkv", "Compressor wkv", "cache", "[B,S,D]", "[B,S,Coff*512]", "", { Coff: "1 or 2" }, [], [sources.code]),
    "comp-wgate": n("comp-wgate", "Compressor wgate", "cache", "[B,S,D]", "[B,S,Coff*512]", "", { ape: "[R,Coff*512]" }, [], [sources.code]),
    "comp-ape": n("comp-ape", "Compressor APE Add", "cache", "score_proj [B,S,Coff*512], ape [R,Coff*512]", "score_with_ape [B,S,Coff*512]", "", { ape: "[R,Coff*512]" }, [], [sources.code]),
    "comp-cutoff": n("comp-cutoff", "Cutoff / Remainder Split", "cache", "proj [B,S,Coff*512] + tail_state", "full_blocks [B,N_full,R,Coff*512], remainder [B,T_tail,Coff*512]", "", { T_tail: "< R" }, [], [sources.code]),
    "tail-append": n("tail-append", "Tail Append + Trim", "cache", "remainder [B,T_tail,Coff*512]", "tail_state' [B,<R,Coff*512]", "", { persistent: "per request" }, [], [sources.code]),
    "tail-state": n("tail-state", "Compressed Tail State", "cache", "kv_tail [B,T_tail,Coff,512], score_tail [B,T_tail,Coff,512]", "kv_state [B,Coff*R,Coff*512], score_state [B,Coff*R,Coff*512]", "", { T_tail: "< R", kv_state: "[B,Coff*R,Coff*512]", score_state: "[B,Coff*R,Coff*512]" }, [], [sources.code]),
    "comp-block-view": n("comp-block-view", "Block View", "cache", "full_blocks [B,N_full,R,Coff*512]", "kv_block [B,N_full,span,512], gate_block [B,N_full,span,512]", "", { span: "R or 2R" }, [], [sources.code]),
    "overlap-transform": n("overlap-transform", "Overlap Transform", "cache", "[B,blocks,R,2*512]", "[B,blocks,2R,512]", "", { active: "R=4 only" }, [], [sources.code]),
    "gated-pool": n("gated-pool", "Softmax-Gated Pool", "cache", "kv_block [B,blocks,R,512], gate_block [B,blocks,R,512]", "compressed_kv [B,blocks,512]", "", {}, [], [sources.code]),
    "comp-anchor": n("comp-anchor", "Anchor Positions", "cache", "block_ids [B,N_full]", "anchor_ids [B,N_full]", "", { c4a: "0,4,8,...", c128a: "0,128,256,..." }, [], [sources.code]),
    "comp-norm-rope": n("comp-norm-rope", "Norm + Compressed RoPE", "cache", "[B,blocks,512]", "[B,blocks,512]", "", { theta: 160000 }, [], [sources.code]),
    "comp-cache-slot": n("comp-cache-slot", "Compressed Slot Map", "cache", "anchor_ids [B,N], R [scalar]", "cache_slots [B,N]", "", { slot: "128 + block_id" }, [], [sources.code]),
    "comp-cache-write": n("comp-cache-write", "Compressed Cache Write", "cache", "compressed_kv [B,blocks,512]", "kv_cache_compressed [B,T/R,512]", "", {}, [], [sources.code]),

    "idx-q": n("idx-q", "Indexer Q", "attention", "qr [B,S,Qr]", "[B,S,64,128]", "", { heads: 64, dim: 128 }, [], [sources.code]),
    "idx-rope": n("idx-rope", "Indexer RoPE", "attention", "idx_q [B,S,64,128]", "idx_q_rope [B,S,64,128]", "", { dim: 128 }, [], [sources.code]),
    "idx-hadamard": n("idx-hadamard", "Hadamard Rotate", "attention", "idx_q_rope [B,S,64,128]", "idx_q_rot [B,S,64,128]", "", {}, [], [sources.kernel]),
    "idx-fp4": n("idx-fp4", "FP4 Activation Quant", "attention", "idx_q_rot [B,S,64,128]", "idx_q_fp4 [B,S,64,128]", "", {}, [], [sources.kernel]),
    "idx-rotate": n("idx-rotate", "RoPE + Hadamard + FP4", "attention", "[B,S,64,128]", "[B,S,64,128]", "", {}, [], [sources.code, sources.kernel]),
    "idx-cache-compress": n("idx-cache-compress", "Index Cache Compress", "cache", "x [B,S,D]", "idx_entries [B,T/4,128]", "", { R: 4, dim: 128 }, [], [sources.code]),
    "idx-cache-write": n("idx-cache-write", "Index Cache Write", "cache", "idx_entries [B,T/4,128]", "idx_cache [B,T/4,128]", "", { dim: 128 }, [], [sources.code]),
    "idx-cache": n("idx-cache", "Index KV Cache", "cache", "x [B,S,D]", "[B,T/4,128]", "", { R: 4, dim: 128 }, [], [sources.code]),
    "idx-einsum": n("idx-einsum", "Lightning Scores", "attention", "idx_q [B,S,64,128], idx_cache [B,T/4,128]", "scores [B,S,64,T/4]", "", {}, [], [sources.code]),
    "idx-weight": n("idx-weight", "weights_proj + Head Sum", "attention", "scores [B,S,64,T/4], weights [B,S,64]", "block_scores [B,S,T/4]", "", {}, [], [sources.code]),
    "idx-mask": n("idx-mask", "Compressed Causal Mask", "attention", "block_scores [B,S,T/4], query_pos [S]", "masked_scores [B,S,T/4]", "", {}, [], [sources.code]),
    "idx-topk": n("idx-topk", "Compressed TopK", "attention", "masked_scores [B,S,T/4]", "block_ids [B,S,topK]", "", { topK: "$indexTopK" }, [], [sources.code]),
    "idx-offset": n("idx-offset", "Cache Offset Map", "attention", "block_ids [B,S,topK]", "compressed_ids [B,S,topK]", "", { offset: 128 }, [], [sources.code]),

    "gate-score": n("gate-score", "Gate Scores", "routing", "[B*S,D]", "[B*S,E]", "", { E: "$E" }, [], [sources.code]),
    "hash-route": n("hash-route", "Hash Route", "routing", "input_ids_flat [B*S]", "expert_ids [B*S,6]", "", { layers: 3 }, [], [sources.code]),
    "route-bias": n("route-bias", "Selection Bias Add", "routing", "scores [B*S,E], bias [E]", "selection_scores [B*S,E]", "", {}, [], [sources.code]),
    "topk-route": n("topk-route", "TopK Route", "routing", "selection_scores [B*S,E]", "expert_ids [B*S,6]", "", {}, [], [sources.code]),
    "route-score-gather": n("route-score-gather", "Original Score Gather", "routing", "scores [B*S,E], expert_ids [B*S,6]", "selected_scores [B*S,6]", "", {}, [], [sources.code]),
    "route-weights": n("route-weights", "Normalize Weights", "routing", "selected_scores [B*S,6]", "route_weights [B*S,6]", "", { scale: "$routeScale" }, [], [sources.code]),
    "expert-counts": n("expert-counts", "Expert Counts", "routing", "expert_ids [B*S,6]", "counts [E]", "", { E: "$E" }, [], [sources.code]),
    "expert-dispatch": n("expert-dispatch", "Expert Dispatch", "expert", "tokens [B*S,D], expert_ids [B*S,6], weights [B*S,6]", "expert_batches [N_e,D]", "", {}, [], [sources.code]),
    "expert-w1w3": n("expert-w1w3", "w1 / w3", "expert", "[N_e,D]", "gate/up [N_e,I]", "", { I: "$I" }, [], [sources.code]),
    swiglu: n("swiglu", "SwiGLU + Clamp", "expert", "gate [N_e,I], up [N_e,I]", "activation [N_e,I]", "", { limit: 10.0 }, [], [sources.code]),
    "expert-w2": n("expert-w2", "w2 Down Projection", "expert", "[N_e,I]", "[N_e,D]", "", {}, [], [sources.code]),
    "routed-accum": n("routed-accum", "Weighted Routed Accum", "expert", "expert_y [N_e,D], route_weights [B*S,6]", "routed_y [B*S,D]", "", {}, [], [sources.code]),
    "shared-w1w3": n("shared-w1w3", "Shared w1 / w3", "expert", "tokens [B*S,D]", "shared_gate/up [B*S,I]", "", { I: "$I" }, [], [sources.code]),
    "shared-swiglu": n("shared-swiglu", "Shared SwiGLU", "expert", "shared_gate/up [B*S,I]", "shared_act [B*S,I]", "", { limit: 10.0 }, [], [sources.code]),
    "shared-w2": n("shared-w2", "Shared w2", "expert", "shared_act [B*S,I]", "shared_y [B*S,D]", "", {}, [], [sources.code]),
    "expert-combine": n("expert-combine", "Routed + Shared Combine", "expert", "routed_y [B*S,D], shared_y [B*S,D]", "moe_y [B*S,D]", "", {}, [], [sources.code]),
    "moe-allreduce": n("moe-allreduce", "MoE TP All-Reduce", "expert", "moe_y_shard [B*S,D]", "moe_y [B*S,D]", "", {}, [], [sources.code]),

    "attn-gather": n("attn-gather", "Gather Selected KV", "attention", "cache [B,128+T/R,512], selected_ids [B,S,N]", "kv_selected [B,S,N,512]", "", {}, [], [sources.code]),
    "attn-score": n("attn-score", "QK Score", "attention", "q [B,S,H,512], k_selected [B,S,N,512]", "scores [B,S,H,N]", "", { scale: "1/sqrt(512)" }, [], [sources.code]),
    "attn-mask-sink": n("attn-mask-sink", "Mask + Attention Sink", "attention", "scores [B,S,H,N]", "biased_scores [B,S,H,N]", "", { attn_sink: "[H]" }, [], [sources.code]),
    "attn-softmax": n("attn-softmax", "Online Softmax", "attention", "biased_scores [B,S,H,N]", "prob [B,S,H,N]", "", {}, [], [sources.code]),
    "attn-value-sum": n("attn-value-sum", "Value Weighted Sum", "attention", "prob [B,S,H,N], v_selected [B,S,N,512]", "heads [B,S,H,512]", "", {}, [], [sources.code]),
    "attn-inv-rope": n("attn-inv-rope", "Inverse RoPE Value Fix", "attention", "heads [B,S,H,512]", "heads_value [B,S,H,512]", "", { rope_dim: 64 }, [], [sources.code]),
    "o-woa": n("o-woa", "wo_a Group Projection", "attention", "heads_value [B,S,H,512]", "o_latent [B,S,G,Or]", "", { groups: "$G", Or: "$Or" }, [], [sources.code]),
    "o-wob": n("o-wob", "wo_b Output Projection", "attention", "o_latent [B,S,G,Or]", "attn_y [B,S,D]", "", { D: "$D" }, [], [sources.code]),

    "hc-head-collapse": n("hc-head-collapse", "HC Head Collapse", "output", "final_lanes [B,S,4,D]", "hidden [B,S,D]", "", { lanes: 4 }, [], [sources.code]),
    "final-rmsnorm": n("final-rmsnorm", "Final RMSNorm", "output", "hidden [B,S,D]", "hidden_norm [B,S,D]", "", { eps: "1e-6" }, [], [sources.code]),
    "last-token": n("last-token", "Last Token Slice", "output", "hidden_norm [B,S,D]", "hidden_last [B,D]", "", {}, [], [sources.code]),
    "lm-project": n("lm-project", "Vocab Projection", "output", "hidden_last [B,D]", "logits [B,129280]", "", { vocab: 129280 }, [], common.src),
    "mtp-embed": n("mtp-embed", "MTP Token Embedding", "output", "ids [B,S]", "mtp_embed [B,S,D]", "", {}, [], [sources.code]),
    "mtp-hidden-proj": n("mtp-hidden-proj", "MTP Hidden Projection", "output", "final_lanes [B,S,4,D]", "mtp_hidden [B,S,D]", "", {}, [], [sources.code]),
    "mtp-combine": n("mtp-combine", "MTP Combine", "output", "mtp_embed [B,S,D], mtp_hidden [B,S,D]", "mtp_x [B,S,D]", "", {}, [], [sources.code]),
    "mtp-block": n("mtp-block", "MTP Decoder Block", "output", "mtp_x [B,S,D]", "mtp_y [B,S,D]", "", { R: 0 }, [], [sources.code]),
    "mtp-head": n("mtp-head", "MTP Head", "output", "mtp_y [B,S,D]", "mtp_logits [B,129280]", "", { vocab: 129280 }, [], [sources.code]),
  };

  Object.assign(nodes["window-topk"], {
    title: "SWA Window IDs",
    summary: "",
  });
  Object.assign(nodes["attn-selected"], {
    title: "Attention KV Set",
    input: "window_ids [B,S,W], compressed_ids [B,S,C]",
    output: "selected_ids [B,S,W+C]",
    summary: "",
  });
  Object.assign(nodes["sparse-attn"], {
    title: "Hybrid Attention Kernel",
    summary: "",
  });
  Object.assign(nodes["input-ids"], {
    title: "Input IDs (model entry)",
    summary: "",
  });
  Object.assign(nodes["embedding"], {
    title: "Token Embedding (once)",
    summary: "",
  });
  nodes["stack-entry"] = n("stack-entry", "Decoder Stack Entry", "stream", "initial_lanes [B,S,4,D]", "layer_state [B,S,4,D]", "", { decoder_layers: "$layers" }, [], common.src, );
  nodes["stack-exit"] = n("stack-exit", "Final Stack State", "output", "layer_state_after_L [B,S,4,D]", "final_lanes [B,S,4,D]", "", { lm_head: "final token only" }, [], common.src, );
  Object.assign(nodes["head"], {
    title: "Final HC Head + LM Head",
    summary: "",
  });
  Object.assign(nodes["mtp"], {
    title: "Final MTP Block",
    summary: "",
  });
  Object.assign(nodes["logits"], {
    title: "Final Logits",
    summary: "",
  });

  ["comp-wkv", "comp-wgate", "comp-ape", "comp-cutoff", "tail-append", "comp-block-view", "comp-anchor", "comp-cache-slot"].forEach((id) => {
    nodes[id].visibleWhen = { mode: ["csa", "hca"] };
  });
  Object.assign(nodes["tail-state"], {
    summary: "",
    visibleWhen: { mode: ["csa", "hca"] },
  });
  Object.assign(nodes["overlap-transform"], {
    title: "CSA Overlap Transform",
    summary: "",
    params: { active: "CSA / R=4 only" },
    visibleWhen: { mode: "csa" },
  });
  Object.assign(nodes["gated-pool"], {
    summary: "",
    visibleWhen: { mode: ["csa", "hca"] },
  });
  Object.assign(nodes["comp-norm-rope"], { visibleWhen: { mode: ["csa", "hca"] } });
  Object.assign(nodes["comp-cache-write"], {
    summary: "",
    visibleWhen: { mode: ["csa", "hca"] },
  });
  nodes["hca-all-compressed"] = n("hca-all-compressed", "HCA All Compressed Blocks", "attention", "compressed cache [B,T/128,512]", "[B,S,T/128]", "", { R: 128 }, [], [sources.code], null, { mode: "hca" }, );

  ["idx-q", "idx-rope", "idx-hadamard", "idx-fp4", "idx-rotate", "idx-cache-compress", "idx-cache-write", "idx-cache", "idx-einsum", "idx-weight", "idx-mask", "idx-topk", "idx-offset"].forEach((id) => {
    nodes[id].visibleWhen = { mode: "csa" };
  });
  Object.assign(nodes["idx-q"], { summary: "" });
  Object.assign(nodes["idx-cache"], { summary: "" });
  Object.assign(nodes["idx-topk"], { summary: "" });

  Object.entries({
    "input-ids": [{"latex":"\\mathrm{ids}\\in\\mathbb{N}^{B\\times S}"}],
    "embedding": [{"latex":"x_{b,s}=E[\\mathrm{ids}_{b,s}],\\qquad x\\in\\mathbb{R}^{B\\times S\\times D}"}],
    "hc-expand": [{"latex":"X_{b,s,l,d}=x_{b,s,d},\\qquad l\\in\\{1,\\dots,4\\}"}],
    "mhc-attn": [{"latex":"X'=\\operatorname{mHCWrite}(X,\\operatorname{Attention}(\\operatorname{mHCRead}(X)))"}],
    "attention": [{"latex":"y=\\operatorname{Attn}(Q(x),K_{\\mathcal{I}},V_{\\mathcal{I}})W_o"}],
    "q-path": [{"latex":"Q=\\operatorname{RoPE}(\\operatorname{reshape}(\\operatorname{RMSNorm}(xW_{q,a})W_{q,b}))"}],
    "kv-path": [{"latex":"k\\!v=\\operatorname{RMSNorm}(xW_{kv}^{\\top}),\\qquad k\\!v\\in\\mathbb{R}^{B\\times S\\times512}"}],
    "kv-cache": [{"latex":"\\mathrm{cache}=[\\mathrm{SWA}_{0:128}\\;||\\;\\mathrm{Compressed}_{0:\\lfloor T/R\\rfloor}]"}],
    "compressor": [{"latex":"c_j=\\operatorname{Pool}_{t\\in\\mathrm{block}(j)}(W_{kv}x_t,W_gx_t,\\mathrm{APE}_t)"}],
    "indexer": [{"latex":"\\mathcal{I}_{\\mathrm{csa}}=\\operatorname{TopK}(\\operatorname{Score}(q_{\\mathrm{idx}},C_{\\mathrm{idx}}),K)"}],
    "sparse-attn": [{"latex":"a_{h,t}=\\frac{\\langle q_h,k_{h,t}\\rangle}{\\sqrt{512}}+\\mathrm{mask}_t+\\mathrm{sink}_h"},{"latex":"y_h=\\sum_{t\\in\\mathcal{I}}\\operatorname{softmax}(a_h)_t\\,v_t"}],
    "o-proj": [{"latex":"y=\\operatorname{GroupProj}_b(\\operatorname{GroupProj}_a(\\operatorname{concat}_h y_h))"}],
    "mhc-ffn": [{"latex":"X'=\\operatorname{mHCWrite}_{ffn}(X,\\operatorname{MoE}(\\operatorname{mHCRead}_{ffn}(X)))"}],
    "moe": [{"latex":"y_{\\mathrm{moe}}=\\sum_{i\\in\\mathcal{E}(x)}w_iE_i(x)+E_{\\mathrm{shared}}(x)"}],
    "gate": [{"latex":"\\mathcal{E}(x)=\\begin{cases}\\operatorname{tid2eid}(\\mathrm{ids}),&\\ell<3\\\\\\operatorname{TopK}(\\operatorname{score}(x),6),&\\ell\\ge3\\end{cases}"}],
    "routed-experts": [{"latex":"E_i(x)=W_{2,i}\\left(\\operatorname{SiLU}(W_{1,i}x)\\odot W_{3,i}x\\right)"}],
    "shared-expert": [{"latex":"y_{\\mathrm{shared}}=W_{2,s}\\left(\\operatorname{SiLU}(W_{1,s}x)\\odot W_{3,s}x\\right)"}],
    "hc-post-moe": [{"latex":"X^{next}_l=\\sum_j\\tilde C^{ffn}_{l,j}X_j+p^{ffn}_{\\mathrm{write},l}y_{\\mathrm{moe}}"}],
    "head": [{"latex":"h=\\operatorname{HCHead}(X^{(L)}_{:,-1,:,:}),\\qquad \\mathrm{logits}=hW_{\\mathrm{lm}}^\\top"}],
    "mtp": [{"latex":"y_{\\mathrm{mtp}}=\\operatorname{MTP}(X^{(L)},\\mathrm{ids})"}],
    "logits": [{"latex":"p(v\\mid x)=\\operatorname{softmax}(\\mathrm{logits})_v"}],
    "hc-flatten": [{"latex":"z_{b,s}=\\operatorname{concat}(X_{b,s,1,:},\\dots,X_{b,s,4,:})\\in\\mathbb{R}^{4D}"}],
    "hc-controller": [{"latex":"m=zW_{\\mathrm{hc}}^\\top,\\qquad m\\in\\mathbb{R}^{24}"}],
    "hc-split": [{"latex":"m_{0:4}\\to p_{\\mathrm{pre}},\\quad m_{4:8}\\to p_{\\mathrm{post}},\\quad m_{8:24}\\to C_{\\mathrm{raw}}\\in\\mathbb{R}^{4\\times4}"}],
    "hc-pre-sigmoid": [{"latex":"pre_j=\\sigma(m_j\\,s_0+b_j)+\\epsilon"}],
    "hc-post-sigmoid": [{"latex":"post_j=2\\,\\sigma(m_{j+4}\\,s_1+b_{j+4})"}],
    "hc-comb-softmax": [{"latex":"C_{j,k}^{(0)}=\\frac{\\exp(m_{8+4j+k}s_2+b_{8+4j+k})}{\\sum_{k'}\\exp(m_{8+4j+k'}s_2+b_{8+4j+k'})}+\\epsilon"}],
    "hc-comb-sinkhorn": [{"latex":"C\\leftarrow C / \\operatorname{sum}_{row}(C),\\qquad C\\leftarrow C / \\operatorname{sum}_{col}(C)"}],
    "hc-sinkhorn": [{"latex":"m\\rightarrow (p_{\\mathrm{read}}\\in\\mathbb{R}^{4},\\;p_{\\mathrm{write}}\\in\\mathbb{R}^{4},\\;C\\in\\mathbb{R}^{4\\times4})"},{"latex":"\\tilde C=\\operatorname{Sinkhorn}(C),\\qquad \\sum_i \\tilde C_{ij}\\approx 1,\\quad \\sum_j \\tilde C_{ij}\\approx 1"}],
    "hc-read": [{"latex":"x_{\\mathrm{attn}}=\\sum_{l=1}^{4}p_{\\mathrm{read},l}\\,X_l"}],
    "attn-residual-mix": [{"latex":"M_l=\\sum_{j=1}^{4}\\tilde C_{l,j}X_j"}],
    "attn-post-inject": [{"latex":"I_l=p_{\\mathrm{write},l}\\,y_{\\mathrm{attn}}"}],
    "hc-write": [{"latex":"X'_l=\\sum_{j=1}^{4}\\tilde C_{l,j}X_j+p_{\\mathrm{write},l}\\,y_{\\mathrm{attn}}"}],
    "ffn-hc-flatten": [{"latex":"z^{ffn}_{b,s}=\\operatorname{concat}_{l=1}^{4}X_{b,s,l,:}"}],
    "ffn-hc-controller": [{"latex":"m^{ffn}=z^{ffn}W_{\\mathrm{hc},ffn}^{\\top},\\qquad m^{ffn}\\in\\mathbb{R}^{24}"}],
    "ffn-hc-split": [{"latex":"m^{ffn}_{0:4}\\to pre,\\quad m^{ffn}_{4:8}\\to post,\\quad m^{ffn}_{8:24}\\to comb_{\\mathrm{raw}}"}],
    "ffn-hc-pre-sigmoid": [{"latex":"pre^{ffn}_j=\\sigma(m^{ffn}_j\\,s_0+b_j)+\\epsilon"}],
    "ffn-hc-post-sigmoid": [{"latex":"post^{ffn}_j=2\\,\\sigma(m^{ffn}_{j+4}\\,s_1+b_{j+4})"}],
    "ffn-hc-comb-softmax": [{"latex":"C^{ffn,(0)}=\\operatorname{softmax}_{row}(C^{ffn}_{raw})+\\epsilon"}],
    "ffn-hc-comb-sinkhorn": [{"latex":"C^{ffn}\\leftarrow \\operatorname{Sinkhorn}(C^{ffn,(0)})"}],
    "ffn-hc-sinkhorn": [{"latex":"m^{ffn}\\rightarrow(p_{\\mathrm{read}},p_{\\mathrm{write}},\\tilde C),\\qquad \\tilde C=\\operatorname{Sinkhorn}(C)"}],
    "hc-pre-moe": [{"latex":"x_{\\mathrm{moe}}=\\sum_{l=1}^{4}p^{ffn}_{\\mathrm{read},l}X_l"}],
    "ffn-residual-mix": [{"latex":"M^{ffn}_l=\\sum_{j=1}^{4}\\tilde C^{ffn}_{l,j}X_j"}],
    "ffn-post-inject": [{"latex":"I^{ffn}_l=p^{ffn}_{\\mathrm{write},l}y_{\\mathrm{moe}}"}],
    "q-wqa": [{"latex":"q_a=xW_{q,a}^{\\top},\\qquad q_a\\in\\mathbb{R}^{B\\times S\\times Q_r}"}],
    "q-norm": [{"latex":"\\operatorname{RMSNorm}(u)=\\frac{u}{\\sqrt{\\frac{1}{n}\\sum_i u_i^2+\\epsilon}}\\odot w"}],
    "q-wqb": [{"latex":"q_b=q_a^{\\mathrm{norm}}W_{q,b}^{\\top},\\qquad q_b\\in\\mathbb{R}^{B\\times S\\times (H\\cdot512)}"}],
    "q-reshape": [{"latex":"Q=\\operatorname{reshape}(q_b,[B,S,H,512]),\\qquad Q_h\\leftarrow \\operatorname{RMSNorm}(Q_h)"}],
    "q-rope": [{"latex":"\\operatorname{RoPE}(q_{2i},q_{2i+1},p)=\\begin{bmatrix}q_{2i}\\cos\\theta_{p,i}-q_{2i+1}\\sin\\theta_{p,i}\\\\q_{2i}\\sin\\theta_{p,i}+q_{2i+1}\\cos\\theta_{p,i}\\end{bmatrix}"}],
    "kv-wkv": [{"latex":"u=xW_{kv}^{\\top},\\qquad u\\in\\mathbb{R}^{B\\times S\\times512}"}],
    "kv-norm": [{"latex":"k\\!v=\\frac{u}{\\sqrt{\\operatorname{mean}(u^2)+\\epsilon}}\\odot w_{kv}"}],
    "kv-rope-quant": [{"latex":"k=[\\operatorname{FP8Sim}(k_{\\mathrm{nope}}),\\operatorname{RoPE}(k_{\\mathrm{rope}},p)]"}],
    "kv-slice": [{"latex":"k\\!v=[k\\!v_{\\mathrm{nope}}\\in\\mathbb{R}^{448}\\;||\\;k\\!v_{\\mathrm{rope}}\\in\\mathbb{R}^{64}]"}],
    "window-topk": [{"latex":"\\mathcal{I}_{\\mathrm{swa}}=\\{t\\mid \\max(0,n-127)\\le t\\le n\\}"}],
    "swa-prefill-write": [{"latex":"\\mathrm{SWA}\\leftarrow K\\!V_{\\max(0,S-128):S}"}],
    "swa-decode-write": [{"latex":"\\mathrm{SWA}_{n\\bmod128}\\leftarrow K\\!V_n"}],
    "cache-layout": [{"latex":"\\mathcal{C}=[\\mathcal{C}_{\\mathrm{swa}}\\;||\\;\\mathcal{C}_{\\mathrm{comp}}]"}],
    "attn-selected": [{"latex":"\\mathcal{I}=\\mathcal{I}_{\\mathrm{swa}}\\cup\\mathcal{I}_{\\mathrm{compressed}}"}],
    "comp-wkv": [{"latex":"u_t=x_tW_{\\mathrm{comp},kv}^{\\top}"}],
    "comp-wgate": [{"latex":"g_t=x_tW_{\\mathrm{comp},g}^{\\top}+\\mathrm{APE}_t"}],
    "comp-ape": [{"latex":"\\tilde g_{j,r}=g_{j,r}+a_r,\\qquad a_r\\in\\mathbb{R}^{C_{\\mathrm{off}}\\cdot512}"}],
    "comp-cutoff": [{"latex":"[u_{\\mathrm{full}},u_{\\mathrm{tail}}]=\\operatorname{split}_{R}([\\mathrm{tail};u_{\\mathrm{new}}])"}],
    "tail-append": [{"latex":"\\mathrm{tail}'=u_{\\mathrm{tail}},\\qquad |\\mathrm{tail}'|<R"}],
    "tail-state": [{"latex":"\\mathrm{tail}_{n+1}=\\operatorname{append\\_and\\_trim}(\\mathrm{tail}_{n},u_n,g_n;R)"}],
    "comp-block-view": [{"latex":"u_{\\mathrm{full}}\\rightarrow U\\in\\mathbb{R}^{B\\times N_{\\mathrm{full}}\\times \\mathrm{span}\\times512}"}],
    "overlap-transform": [{"latex":"\\mathrm{span}_j=[x_{4j-4},\\dots,x_{4j+3}]"}],
    "gated-pool": [{"latex":"c_j=\\sum_{t\\in\\mathrm{block}(j)}u_t\\cdot\\operatorname{softmax}(g)_t"}],
    "comp-anchor": [{"latex":"a_j=jR,\\qquad R\\in\\{4,128\\}"}],
    "comp-norm-rope": [{"latex":"\\hat c_j=\\operatorname{RoPE}(\\operatorname{RMSNorm}(c_j),a_j),\\qquad a_j\\in\\{0,R,2R,\\dots\\}"}],
    "comp-cache-slot": [{"latex":"\\mathrm{slot}_j=128+j"}],
    "comp-cache-write": [{"latex":"\\mathrm{cache}_{128+j}\\leftarrow \\hat c_j,\\qquad j=\\left\\lfloor\\frac{n}{R}\\right\\rfloor"}],
    "idx-q": [{"latex":"q_{\\mathrm{idx}}=\\operatorname{reshape}(q_{\\mathrm{norm}}W_{\\mathrm{idx},q}^{\\top},[B,S,64,128])"}],
    "idx-rope": [{"latex":"q^{r}_{\\mathrm{idx}}=\\operatorname{RoPE}(q_{\\mathrm{idx}},p)"}],
    "idx-hadamard": [{"latex":"q^{h}_{\\mathrm{idx}}=H_{128}q^{r}_{\\mathrm{idx}}"}],
    "idx-fp4": [{"latex":"\\tilde q_{\\mathrm{idx}}=\\operatorname{Quant}_{\\mathrm{FP4}}(q^{h}_{\\mathrm{idx}})"}],
    "idx-rotate": [{"latex":"\\tilde q=\\operatorname{FP4}(\\operatorname{Hadamard}(\\operatorname{RoPE}(q_{\\mathrm{idx}})))"}],
    "idx-cache-compress": [{"latex":"z_j=\\operatorname{Compress}_{128}(x_{4j:4j+3})"}],
    "idx-cache-write": [{"latex":"C_{\\mathrm{idx},j}\\leftarrow z_j"}],
    "idx-cache": [{"latex":"C_{\\mathrm{idx}}\\in\\mathbb{R}^{B\\times \\lfloor T/4\\rfloor\\times128}"}],
    "idx-einsum": [{"latex":"s_{b,s,h,j}=\\operatorname{ReLU}(\\langle \\tilde q_{b,s,h},C_{\\mathrm{idx},b,j}\\rangle)"}],
    "idx-weight": [{"latex":"S_{b,s,j}=\\sum_{h=1}^{64}\\alpha_{b,s,h}\\,s_{b,s,h,j}"}],
    "idx-mask": [{"latex":"\\tilde S_{b,s,j}=S_{b,s,j}+M(j\\le\\lfloor p_s/4\\rfloor)"}],
    "idx-topk": [{"latex":"\\mathcal{I}_{\\mathrm{topk}}=\\operatorname{TopK}(S+\\mathrm{causal\\_mask},K)"}],
    "idx-offset": [{"latex":"\\mathcal{I}_{\\mathrm{comp}}=128+\\mathcal{I}_{\\mathrm{topk}}"}],
    "gate-score": [{"latex":"r=\\sqrt{\\operatorname{softplus}(xW_g^\\top)},\\qquad r\\in\\mathbb{R}^{B S\\times E}"}],
    "hash-route": [{"latex":"\\mathcal{E}_{b,s}=\\mathrm{tid2eid}[\\mathrm{ids}_{b,s}]"}],
    "route-bias": [{"latex":"r^{sel}=r+b_{\\mathrm{route}}"}],
    "topk-route": [{"latex":"\\mathcal{E}_{b,s}=\\operatorname{TopK}(r_{b,s}+b_{\\mathrm{route}},6)"}],
    "route-score-gather": [{"latex":"r_{\\mathcal{E}}=\\operatorname{gather}(r,\\mathcal{E})"}],
    "route-weights": [{"latex":"w_i=\\frac{r_i}{\\sum_{j\\in\\mathcal{E}}r_j}\\cdot \\mathrm{route\\_scale}"}],
    "expert-counts": [{"latex":"n_i=\\sum_t \\mathbf{1}[i\\in\\mathcal{E}(x_t)]"}],
    "expert-dispatch": [{"latex":"X_i=\\{x_n\\mid i\\in\\mathcal{E}(x_n)\\}"}],
    "expert-w1w3": [{"latex":"g=xW_{1,i}^{\\top},\\qquad u=xW_{3,i}^{\\top}"}],
    "swiglu": [{"latex":"h=\\operatorname{SiLU}(\\operatorname{clip}(g))\\odot \\operatorname{clip}(u)"}],
    "expert-w2": [{"latex":"y_i=hW_{2,i}^{\\top},\\qquad y_i\\in\\mathbb{R}^{D}"}],
    "routed-accum": [{"latex":"y_{\\mathrm{routed},t}=\\sum_{i\\in\\mathcal{E}(x_t)}w_{t,i}E_i(x_t)"}],
    "shared-w1w3": [{"latex":"g_s=xW_{1,s}^{\\top},\\qquad u_s=xW_{3,s}^{\\top}"}],
    "shared-swiglu": [{"latex":"h_s=\\operatorname{SiLU}(\\operatorname{clip}(g_s))\\odot\\operatorname{clip}(u_s)"}],
    "shared-w2": [{"latex":"y_s=h_sW_{2,s}^{\\top}"}],
    "expert-combine": [{"latex":"y=\\sum_{i\\in\\mathcal{E}(x)}w_i\\,E_i(x)+E_{\\mathrm{shared}}(x)"}],
    "moe-allreduce": [{"latex":"y_{\\mathrm{moe}}=\\operatorname{AllReduce}(y_{\\mathrm{moe}}^{\\mathrm{shard}})"}],
    "attn-gather": [{"latex":"K\\!V_{\\mathcal{I}}=\\operatorname{gather}(\\mathcal{C},\\mathcal{I})"}],
    "attn-score": [{"latex":"A_{b,s,h,t}=\\langle Q_{b,s,h},K_{\\mathcal{I}_{b,s,t}}\\rangle/\\sqrt{512}"}],
    "attn-mask-sink": [{"latex":"\\tilde A=A+M_{\\mathrm{causal/window}}+\\beta_h"}],
    "attn-softmax": [{"latex":"P_{b,s,h,:}=\\operatorname{softmax}(\\tilde A_{b,s,h,:})"}],
    "attn-value-sum": [{"latex":"Y_{b,s,h}=\\sum_t P_{b,s,h,t}V_{\\mathcal{I}_{b,s,t}}"}],
    "attn-inv-rope": [{"latex":"Y_{\\mathrm{value}}\\leftarrow\\operatorname{RoPE}^{-1}(Y_{\\mathrm{shared}})"}],
    "o-woa": [{"latex":"o_a=\\operatorname{GroupLinear}_a(\\operatorname{concat}_h Y_h),\\qquad o_a\\in\\mathbb{R}^{B\\times S\\times G\\times O_r}"}],
    "o-wob": [{"latex":"y_{\\mathrm{attn}}=o_aW_{o,b}^{\\top},\\qquad y_{\\mathrm{attn}}\\in\\mathbb{R}^{B\\times S\\times D}"}],
    "hc-head-collapse": [{"latex":"h=\\operatorname{HCHead}(X^{(L)})"}],
    "final-rmsnorm": [{"latex":"\\hat h=\\operatorname{RMSNorm}(h)"}],
    "last-token": [{"latex":"h_{\\mathrm{last}}=\\hat h_{:,-1,:}"}],
    "lm-project": [{"latex":"\\mathrm{logits}=h_{\\mathrm{last}}W_{\\mathrm{lm}}^\\top"}],
    "mtp-embed": [{"latex":"e_{\\mathrm{mtp}}=E(\\mathrm{ids})"}],
    "mtp-hidden-proj": [{"latex":"h_{\\mathrm{mtp}}=W_h\\,\\operatorname{HCHead}(X^{(L)})"}],
    "mtp-combine": [{"latex":"x_{\\mathrm{mtp}}=\\operatorname{Combine}(e_{\\mathrm{mtp}},h_{\\mathrm{mtp}})"}],
    "mtp-block": [{"latex":"y_{\\mathrm{mtp}}=F_{\\mathrm{mtp}}(x_{\\mathrm{mtp}};\\,R=0)"}],
    "mtp-head": [{"latex":"\\mathrm{logits}_{\\mathrm{mtp}}=y_{\\mathrm{mtp}}W_{\\mathrm{lm}}^\\top"}],
    "stack-entry": [{"latex":"X^{(0)}\\in\\mathbb{R}^{B\\times S\\times 4\\times D},\\qquad X^{(n+1)}=F_n(X^{(n)})"}],
    "stack-exit": [{"latex":"X^{(L)}=F_{L-1}\\circ\\cdots\\circ F_0(X^{(0)})"}],
    "hca-all-compressed": [{"latex":"\\mathcal{I}_{\\mathrm{compressed}}=\\{0,\\dots,\\lfloor T/128\\rfloor-1\\}"}],
  }).forEach(([id, formulas]) => {
    if (nodes[id]) nodes[id].details = { formula: formulas };
  });

  const scenes = {
    overview: scene("overview", "Full V4 internal graph", "All major controller paths, cache paths, routing paths, and expert internals are expanded in one graph.", [
      "input-ids", "embedding", "hc-expand", "stack-entry",
      "hc-flatten", "hc-controller", "hc-split", "hc-pre-sigmoid", "hc-post-sigmoid", "hc-comb-softmax", "hc-comb-sinkhorn", "hc-read",
      "q-wqa", "q-norm", "q-wqb", "q-reshape", "q-rope",
      "kv-wkv", "kv-norm", "kv-slice", "kv-rope-quant", "swa-prefill-write", "swa-decode-write", "window-topk", "cache-layout",
      "comp-wkv", "comp-wgate", "comp-ape", "tail-state", "comp-cutoff", "tail-append", "comp-block-view", "overlap-transform", "gated-pool", "comp-anchor", "comp-norm-rope", "comp-cache-slot", "comp-cache-write",
      "idx-q", "idx-rope", "idx-hadamard", "idx-fp4", "idx-cache-compress", "idx-cache-write", "idx-cache", "idx-einsum", "idx-weight", "idx-mask", "idx-topk", "idx-offset", "hca-all-compressed",
      "attn-selected", "attn-gather", "attn-score", "attn-mask-sink", "attn-softmax", "attn-value-sum", "attn-inv-rope", "o-woa", "o-wob", "attn-residual-mix", "attn-post-inject", "hc-write",
      "ffn-hc-flatten", "ffn-hc-controller", "ffn-hc-split", "ffn-hc-pre-sigmoid", "ffn-hc-post-sigmoid", "ffn-hc-comb-softmax", "ffn-hc-comb-sinkhorn", "hc-pre-moe", "gate-score", "hash-route", "route-bias", "topk-route", "route-score-gather", "route-weights", "expert-counts", "expert-dispatch",
      "expert-w1w3", "swiglu", "expert-w2", "routed-accum", "shared-w1w3", "shared-swiglu", "shared-w2", "expert-combine", "moe-allreduce", "ffn-residual-mix", "ffn-post-inject", "hc-post-moe",
      "stack-exit", "hc-head-collapse", "final-rmsnorm", "last-token", "lm-project", "mtp-embed", "mtp-hidden-proj", "mtp-combine", "mtp-block", "mtp-head", "logits",
    ], [
      e("input-ids", "embedding"), e("embedding", "hc-expand"),
      e("hc-expand", "stack-entry"), e("stack-entry", "hc-flatten"), e("hc-flatten", "hc-controller"), e("hc-controller", "hc-split"), e("hc-split", "hc-pre-sigmoid", "main", null, "pre_logits [B,S,4]"), e("hc-pre-sigmoid", "hc-read", "main", null, "pre [B,S,4]"),
      e("hc-read", "q-wqa"), e("q-wqa", "q-norm"), e("q-norm", "q-wqb"), e("q-wqb", "q-reshape"), e("q-reshape", "q-rope"),
      e("hc-read", "kv-wkv", "branch"), e("kv-wkv", "kv-norm"), e("kv-norm", "kv-slice"), e("kv-slice", "kv-rope-quant"), e("kv-rope-quant", "swa-prefill-write", "branch"), e("kv-rope-quant", "swa-decode-write", "branch"), e("swa-prefill-write", "cache-layout"), e("swa-decode-write", "cache-layout"), e("cache-layout", "window-topk"),
      e("kv-rope-quant", "comp-wkv", "branch"), e("kv-rope-quant", "comp-wgate", "branch"), e("comp-wgate", "comp-ape"), e("comp-wkv", "tail-state", "main", null, "kv_tail [B,S,Coff*512]"), e("comp-ape", "tail-state", "main", null, "score_tail [B,S,Coff*512]"), e("tail-state", "comp-cutoff"), e("comp-cutoff", "tail-append", "branch", null, "remainder [B,<R,Coff*512]"), e("comp-cutoff", "comp-block-view", "main", null, "full_blocks [B,N,R,Coff*512]"), e("comp-block-view", "overlap-transform", "branch", { mode: "csa" }, "block [B,N,4,2*512]"), e("overlap-transform", "gated-pool", "branch", { mode: "csa" }, "kv_block [B,N,8,512]"), e("comp-block-view", "gated-pool", "branch", { mode: "hca" }, "kv_block [B,N,128,512]"), e("gated-pool", "comp-anchor"), e("comp-anchor", "comp-norm-rope", "main", null, "anchor_ids [B,N]"), e("comp-norm-rope", "comp-cache-slot"), e("comp-cache-slot", "comp-cache-write"), e("comp-cache-write", "cache-layout", "branch"),
      e("q-norm", "idx-q", "branch", { mode: "csa" }), e("idx-q", "idx-rope"), e("idx-rope", "idx-hadamard"), e("idx-hadamard", "idx-fp4"), e("kv-rope-quant", "idx-cache-compress", "branch", { mode: "csa" }), e("idx-cache-compress", "idx-cache-write"), e("idx-cache-write", "idx-cache"), e("idx-fp4", "idx-einsum"), e("idx-cache", "idx-einsum"), e("idx-einsum", "idx-weight"), e("idx-weight", "idx-mask"), e("idx-mask", "idx-topk"), e("idx-topk", "idx-offset"),
      e("window-topk", "attn-selected"), e("comp-cache-write", "hca-all-compressed", "branch", { mode: "hca" }), e("hca-all-compressed", "attn-selected", "branch", { mode: "hca" }), e("idx-offset", "attn-selected", "branch", { mode: "csa" }), e("attn-selected", "attn-gather"), e("cache-layout", "attn-gather", "branch"), e("q-rope", "attn-score"), e("attn-gather", "attn-score"), e("attn-score", "attn-mask-sink"), e("attn-mask-sink", "attn-softmax"), e("attn-softmax", "attn-value-sum"), e("attn-gather", "attn-value-sum", "branch"), e("attn-value-sum", "attn-inv-rope"), e("attn-inv-rope", "o-woa"), e("o-woa", "o-wob"),
      e("hc-split", "hc-post-sigmoid", "branch", null, "post_logits [B,S,4]"), e("hc-split", "hc-comb-softmax", "branch", null, "comb_logits [B,S,4,4]"), e("hc-comb-softmax", "hc-comb-sinkhorn"), e("hc-comb-sinkhorn", "attn-residual-mix", "branch", null, "comb [B,S,4,4]"), e("hc-expand", "attn-residual-mix", "branch"), e("o-wob", "attn-post-inject"), e("hc-post-sigmoid", "attn-post-inject", "branch", null, "post [B,S,4]"), e("attn-residual-mix", "hc-write", "main", null, "mixed_lanes [B,S,4,D]"), e("attn-post-inject", "hc-write", "main", null, "attn_inject [B,S,4,D]"),
      e("hc-write", "ffn-hc-flatten"), e("ffn-hc-flatten", "ffn-hc-controller"), e("ffn-hc-controller", "ffn-hc-split"), e("ffn-hc-split", "ffn-hc-pre-sigmoid", "main", null, "pre_logits [B,S,4]"), e("ffn-hc-pre-sigmoid", "hc-pre-moe", "main", null, "pre [B,S,4]"), e("hc-write", "ffn-residual-mix", "branch"), e("ffn-hc-split", "ffn-hc-comb-softmax", "branch", null, "comb_logits [B,S,4,4]"), e("ffn-hc-comb-softmax", "ffn-hc-comb-sinkhorn"), e("ffn-hc-comb-sinkhorn", "ffn-residual-mix", "branch", null, "comb [B,S,4,4]"),
      e("hc-pre-moe", "gate-score"), e("gate-score", "hash-route", "branch"), e("gate-score", "route-bias", "branch"), e("route-bias", "topk-route"), e("hash-route", "route-score-gather", "main", null, "expert_ids [B*S,6]"), e("topk-route", "route-score-gather", "main", null, "expert_ids [B*S,6]"), e("gate-score", "route-score-gather", "branch", null, "scores [B*S,E]"), e("route-score-gather", "route-weights"), e("route-score-gather", "expert-counts", "branch"), e("route-weights", "expert-dispatch"), e("expert-counts", "expert-dispatch", "branch"), e("expert-dispatch", "expert-w1w3"), e("expert-w1w3", "swiglu", "main", null, "gate [N_e,I], up [N_e,I]"), e("swiglu", "expert-w2"), e("expert-w2", "routed-accum"), e("route-weights", "routed-accum", "branch"), e("hc-pre-moe", "shared-w1w3", "branch"), e("shared-w1w3", "shared-swiglu"), e("shared-swiglu", "shared-w2"), e("routed-accum", "expert-combine", "main", null, "routed_y [B*S,D]"), e("shared-w2", "expert-combine", "main", null, "shared_y [B*S,D]"), e("expert-combine", "moe-allreduce"), e("moe-allreduce", "ffn-post-inject"), e("ffn-hc-split", "ffn-hc-post-sigmoid", "branch", null, "post_logits [B,S,4]"), e("ffn-hc-post-sigmoid", "ffn-post-inject", "branch", null, "post [B,S,4]"), e("ffn-residual-mix", "hc-post-moe", "main", null, "mixed_lanes [B,S,4,D]"), e("ffn-post-inject", "hc-post-moe", "main", null, "moe_inject [B,S,4,D]"),
      e("hc-post-moe", "stack-exit"), e("stack-exit", "hc-head-collapse"), e("hc-head-collapse", "final-rmsnorm"), e("final-rmsnorm", "last-token"), e("last-token", "lm-project"), e("lm-project", "logits"), e("stack-exit", "mtp-hidden-proj", "branch"), e("input-ids", "mtp-embed", "branch"), e("mtp-hidden-proj", "mtp-combine"), e("mtp-embed", "mtp-combine"), e("mtp-combine", "mtp-block"), e("mtp-block", "mtp-head"),
    ], [
      group("Model entry (once)", ["input-ids", "embedding", "hc-expand", "stack-entry"], "stream"),
      group("mHC controller + read path", ["hc-flatten", "hc-controller", "hc-split", "hc-pre-sigmoid", "hc-post-sigmoid", "hc-comb-softmax", "hc-comb-sinkhorn", "hc-read"], "hc"),
      group("Attention Q/KV paths", ["q-wqa", "q-norm", "q-wqb", "q-reshape", "q-rope", "kv-wkv", "kv-norm", "kv-slice", "kv-rope-quant", "window-topk", "cache-layout", "hca-all-compressed", "attn-selected", "attn-gather", "attn-score", "attn-mask-sink", "attn-softmax", "attn-value-sum", "attn-inv-rope", "o-woa", "o-wob"], "attention"),
      group("SWA cache write path", ["swa-prefill-write", "swa-decode-write", "cache-layout", "window-topk"], "cache"),
      group("KV compressor + tail state", ["comp-wkv", "comp-wgate", "comp-ape", "tail-state", "comp-cutoff", "tail-append", "comp-block-view", "overlap-transform", "gated-pool", "comp-anchor", "comp-norm-rope", "comp-cache-slot", "comp-cache-write"], "cache"),
      group("Lightning indexer", ["idx-q", "idx-rope", "idx-hadamard", "idx-fp4", "idx-cache-compress", "idx-cache-write", "idx-cache", "idx-einsum", "idx-weight", "idx-mask", "idx-topk", "idx-offset"], "attention"),
      group("mHC attention residual mixing", ["attn-residual-mix", "attn-post-inject", "hc-write"], "hc"),
      group("mHC MoE controller + read path", ["ffn-hc-flatten", "ffn-hc-controller", "ffn-hc-split", "ffn-hc-pre-sigmoid", "ffn-hc-post-sigmoid", "ffn-hc-comb-softmax", "ffn-hc-comb-sinkhorn", "hc-pre-moe"], "hc"),
      group("MoE routing + SwiGLU experts", ["gate-score", "hash-route", "route-bias", "topk-route", "route-score-gather", "route-weights", "expert-counts", "expert-dispatch", "expert-w1w3", "swiglu", "expert-w2", "routed-accum", "shared-w1w3", "shared-swiglu", "shared-w2", "expert-combine", "moe-allreduce"], "expert"),
      group("mHC MoE residual mixing", ["ffn-residual-mix", "ffn-post-inject", "hc-post-moe"], "hc"),
      group("Final output + MTP", ["stack-exit", "hc-head-collapse", "final-rmsnorm", "last-token", "lm-project", "mtp-embed", "mtp-hidden-proj", "mtp-combine", "mtp-block", "mtp-head", "logits"], "output"),
    ]),
    mhc: scene("mhc", "mHC controller/data path", "pre/post/comb generation plus read/write data path.", [
      "hc-expand",
      "hc-flatten", "hc-controller", "hc-split", "hc-pre-sigmoid", "hc-post-sigmoid", "hc-comb-softmax", "hc-comb-sinkhorn", "hc-read",
      "attention", "attn-residual-mix", "attn-post-inject", "hc-write",
      "ffn-hc-flatten", "ffn-hc-controller", "ffn-hc-split", "ffn-hc-pre-sigmoid", "ffn-hc-post-sigmoid", "ffn-hc-comb-softmax", "ffn-hc-comb-sinkhorn", "hc-pre-moe",
      "moe", "ffn-residual-mix", "ffn-post-inject", "hc-post-moe",
    ], [
      e("hc-expand", "hc-flatten"), e("hc-flatten", "hc-controller"), e("hc-controller", "hc-split"), e("hc-split", "hc-pre-sigmoid", "main", null, "pre_logits [B,S,4]"), e("hc-pre-sigmoid", "hc-read", "main", null, "pre [B,S,4]"), e("hc-read", "attention"),
      e("hc-split", "hc-comb-softmax", "branch", null, "comb_logits [B,S,4,4]"), e("hc-comb-softmax", "hc-comb-sinkhorn"), e("hc-comb-sinkhorn", "attn-residual-mix", "branch", null, "comb [B,S,4,4]"), e("hc-expand", "attn-residual-mix", "branch"), e("attention", "attn-post-inject"), e("hc-split", "hc-post-sigmoid", "branch", null, "post_logits [B,S,4]"), e("hc-post-sigmoid", "attn-post-inject", "branch", null, "post [B,S,4]"), e("attn-residual-mix", "hc-write", "main", null, "mixed_lanes [B,S,4,D]"), e("attn-post-inject", "hc-write", "main", null, "attn_inject [B,S,4,D]"),
      e("hc-write", "ffn-hc-flatten"), e("ffn-hc-flatten", "ffn-hc-controller"), e("ffn-hc-controller", "ffn-hc-split"), e("ffn-hc-split", "ffn-hc-pre-sigmoid", "main", null, "pre_logits [B,S,4]"), e("ffn-hc-pre-sigmoid", "hc-pre-moe", "main", null, "pre [B,S,4]"), e("hc-pre-moe", "moe"),
      e("hc-write", "ffn-residual-mix", "branch"), e("ffn-hc-split", "ffn-hc-comb-softmax", "branch", null, "comb_logits [B,S,4,4]"), e("ffn-hc-comb-softmax", "ffn-hc-comb-sinkhorn"), e("ffn-hc-comb-sinkhorn", "ffn-residual-mix", "branch", null, "comb [B,S,4,4]"), e("moe", "ffn-post-inject"), e("ffn-hc-split", "ffn-hc-post-sigmoid", "branch", null, "post_logits [B,S,4]"), e("ffn-hc-post-sigmoid", "ffn-post-inject", "branch", null, "post [B,S,4]"), e("ffn-residual-mix", "hc-post-moe", "main", null, "mixed_lanes [B,S,4,D]"), e("ffn-post-inject", "hc-post-moe", "main", null, "moe_inject [B,S,4,D]"),
    ], [
      group("mHC attention controller + read path", ["hc-flatten", "hc-controller", "hc-split", "hc-pre-sigmoid", "hc-post-sigmoid", "hc-comb-softmax", "hc-comb-sinkhorn", "hc-read"], "hc"),
      group("mHC attention residual mixing", ["attn-residual-mix", "attn-post-inject", "hc-write"], "hc"),
      group("mHC MoE controller + read path", ["ffn-hc-flatten", "ffn-hc-controller", "ffn-hc-split", "ffn-hc-pre-sigmoid", "ffn-hc-post-sigmoid", "ffn-hc-comb-softmax", "ffn-hc-comb-sinkhorn", "hc-pre-moe"], "hc"),
      group("mHC MoE residual mixing", ["ffn-residual-mix", "ffn-post-inject", "hc-post-moe"], "hc"),
    ]),
    attention: scene("attention", "Attention internals", "Q LoRA, shared KV, cache IDs, sparse attention, grouped output projection.", [
      "mhc-attn", "q-wqa", "q-norm", "q-wqb", "q-reshape", "q-rope",
      "kv-wkv", "kv-norm", "kv-slice", "kv-rope-quant", "cache-layout", "window-topk",
      "compressor", "indexer", "idx-offset", "hca-all-compressed", "attn-selected",
      "attn-gather", "attn-score", "attn-mask-sink", "attn-softmax", "attn-value-sum", "attn-inv-rope", "o-woa", "o-wob", "hc-write",
    ], [
      e("mhc-attn", "q-wqa"), e("q-wqa", "q-norm"), e("q-norm", "q-wqb"), e("q-wqb", "q-reshape"), e("q-reshape", "q-rope"),
      e("mhc-attn", "kv-wkv", "branch"), e("kv-wkv", "kv-norm"), e("kv-norm", "kv-slice"), e("kv-slice", "kv-rope-quant"), e("kv-rope-quant", "cache-layout"), e("cache-layout", "window-topk"),
      e("kv-rope-quant", "compressor", "branch"), e("q-norm", "indexer", "branch"), e("indexer", "idx-offset", "branch", { mode: "csa" }), e("compressor", "hca-all-compressed", "branch", { mode: "hca" }), e("window-topk", "attn-selected"), e("idx-offset", "attn-selected", "branch", { mode: "csa" }), e("hca-all-compressed", "attn-selected", "branch", { mode: "hca" }),
      e("cache-layout", "attn-gather", "branch"), e("attn-selected", "attn-gather"), e("q-rope", "attn-score"), e("attn-gather", "attn-score"), e("attn-score", "attn-mask-sink"), e("attn-mask-sink", "attn-softmax"), e("attn-softmax", "attn-value-sum"), e("attn-gather", "attn-value-sum", "branch"), e("attn-value-sum", "attn-inv-rope"), e("attn-inv-rope", "o-woa"), e("o-woa", "o-wob"), e("o-wob", "hc-write"),
    ], [
      group("mHC attention entry/exit", ["mhc-attn", "hc-write"], "hc"),
      group("Query LoRA + RoPE", ["q-wqa", "q-norm", "q-wqb", "q-reshape", "q-rope"], "attention"),
      group("Shared KV + SWA", ["kv-wkv", "kv-norm", "kv-slice", "kv-rope-quant", "cache-layout", "window-topk"], "cache"),
      group("Compressed selection", ["compressor", "indexer", "idx-offset", "hca-all-compressed", "attn-selected"], "attention"),
      group("Core attention kernel", ["attn-gather", "attn-score", "attn-mask-sink", "attn-softmax", "attn-value-sum"], "attention"),
      group("KV sharing output fix", ["attn-inv-rope", "o-woa", "o-wob"], "attention"),
    ]),
    compression: scene("compression", "KV cache and compressor", "Window cache, compressed cache, tail state, overlap pooling, and cache writes.", [
      "kv-path", "kv-cache", "swa-prefill-write", "swa-decode-write", "cache-layout", "window-topk",
      "comp-wkv", "comp-wgate", "comp-ape", "tail-state", "comp-cutoff", "tail-append", "comp-block-view", "overlap-transform", "gated-pool", "comp-anchor", "comp-norm-rope", "comp-cache-slot", "comp-cache-write", "attn-gather",
    ], [
      e("kv-path", "kv-cache"), e("kv-cache", "swa-prefill-write"), e("kv-cache", "swa-decode-write", "branch"), e("swa-prefill-write", "cache-layout"), e("swa-decode-write", "cache-layout"), e("cache-layout", "window-topk"),
      e("kv-path", "comp-wkv", "branch"), e("kv-path", "comp-wgate", "branch"), e("comp-wgate", "comp-ape"), e("comp-wkv", "tail-state", "main", null, "kv_tail [B,S,Coff*512]"), e("comp-ape", "tail-state", "main", null, "score_tail [B,S,Coff*512]"), e("tail-state", "comp-cutoff"), e("comp-cutoff", "tail-append", "branch", null, "remainder [B,<R,Coff*512]"), e("comp-cutoff", "comp-block-view", "main", null, "full_blocks [B,N,R,Coff*512]"), e("comp-block-view", "overlap-transform", "branch", { mode: "csa" }), e("overlap-transform", "gated-pool", "branch", { mode: "csa" }), e("comp-block-view", "gated-pool", "branch", { mode: "hca" }), e("gated-pool", "comp-anchor"), e("comp-anchor", "comp-norm-rope"), e("comp-norm-rope", "comp-cache-slot"), e("comp-cache-slot", "comp-cache-write"), e("comp-cache-write", "cache-layout", "branch"), e("cache-layout", "attn-gather"),
    ], [
      group("SWA window cache", ["kv-path", "kv-cache", "swa-prefill-write", "swa-decode-write", "cache-layout", "window-topk"], "cache"),
      group("Compressor projections", ["comp-wkv", "comp-wgate", "comp-ape"], "cache"),
      group("Tail / cutoff runtime state", ["tail-state", "comp-cutoff", "tail-append"], "cache"),
      group("Block pooling", ["comp-block-view", "overlap-transform", "gated-pool"], "cache"),
      group("Compressed entry write", ["comp-anchor", "comp-norm-rope", "comp-cache-slot", "comp-cache-write"], "cache"),
      group("Attention consumer", ["attn-gather"], "attention"),
    ]),
    indexer: scene("indexer", "Lightning indexer", "R=4 compressed block selector.", [
      "q-norm", "kv-path", "idx-q", "idx-rope", "idx-hadamard", "idx-fp4",
      "idx-cache-compress", "idx-cache-write", "idx-cache",
      "idx-einsum", "idx-weight", "idx-mask", "idx-topk", "idx-offset", "attn-selected",
    ], [
      e("q-norm", "idx-q"), e("idx-q", "idx-rope"), e("idx-rope", "idx-hadamard"), e("idx-hadamard", "idx-fp4"),
      e("kv-path", "idx-cache-compress", "branch"), e("idx-cache-compress", "idx-cache-write"), e("idx-cache-write", "idx-cache"),
      e("idx-fp4", "idx-einsum"), e("idx-cache", "idx-einsum"), e("idx-einsum", "idx-weight"), e("idx-weight", "idx-mask"), e("idx-mask", "idx-topk"), e("idx-topk", "idx-offset"), e("idx-offset", "attn-selected"),
    ], [
      group("Indexer query path", ["q-norm", "idx-q", "idx-rope", "idx-hadamard", "idx-fp4"], "attention"),
      group("Indexer compressed KV cache", ["kv-path", "idx-cache-compress", "idx-cache-write", "idx-cache"], "cache"),
      group("Score + head weighting", ["idx-einsum", "idx-weight"], "attention"),
      group("Masked TopK selected blocks", ["idx-mask", "idx-topk", "idx-offset", "attn-selected"], "attention"),
    ]),
    moe: scene("moe", "MoE and SwiGLU experts", "Routing, expert dispatch, FP4 SwiGLU experts, shared expert, and combine.", [
      "mhc-ffn", "gate-score", "hash-route", "route-bias", "topk-route", "route-score-gather", "route-weights", "expert-counts", "expert-dispatch",
      "expert-w1w3", "swiglu", "expert-w2", "routed-accum",
      "shared-w1w3", "shared-swiglu", "shared-w2", "expert-combine", "moe-allreduce", "hc-post-moe",
    ], [
      e("mhc-ffn", "gate-score"), e("gate-score", "hash-route", "branch"), e("gate-score", "route-bias", "branch"), e("route-bias", "topk-route"), e("hash-route", "route-score-gather", "main", null, "expert_ids [B*S,6]"), e("topk-route", "route-score-gather", "main", null, "expert_ids [B*S,6]"), e("gate-score", "route-score-gather", "branch", null, "scores [B*S,E]"), e("route-score-gather", "route-weights"), e("route-score-gather", "expert-counts", "branch"), e("route-weights", "expert-dispatch"), e("expert-counts", "expert-dispatch", "branch"), e("expert-dispatch", "expert-w1w3"), e("expert-w1w3", "swiglu", "main", null, "gate [N_e,I], up [N_e,I]"), e("swiglu", "expert-w2"), e("expert-w2", "routed-accum"), e("route-weights", "routed-accum", "branch"), e("mhc-ffn", "shared-w1w3", "branch"), e("shared-w1w3", "shared-swiglu"), e("shared-swiglu", "shared-w2"), e("routed-accum", "expert-combine", "main", null, "routed_y [B*S,D]"), e("shared-w2", "expert-combine", "main", null, "shared_y [B*S,D]"), e("expert-combine", "moe-allreduce"), e("moe-allreduce", "hc-post-moe"),
    ], [
      group("mHC MoE entry/exit", ["mhc-ffn", "hc-post-moe"], "hc"),
      group("Router scores + ids", ["gate-score", "hash-route", "route-bias", "topk-route", "route-score-gather", "route-weights"], "routing"),
      group("Routed expert dispatch", ["expert-counts", "expert-dispatch"], "routing"),
      group("Routed SwiGLU internals", ["expert-w1w3", "swiglu", "expert-w2", "routed-accum"], "expert"),
      group("Shared expert + combine", ["shared-w1w3", "shared-swiglu", "shared-w2", "expert-combine", "moe-allreduce"], "expert"),
    ]),
    output: scene("output", "Output and MTP", "Final HC head collapse, LM head, and MTP branch.", [
      "input-ids", "hc-post-moe", "stack-exit", "hc-head-collapse", "final-rmsnorm", "last-token", "lm-project", "logits",
      "mtp-embed", "mtp-hidden-proj", "mtp-combine", "mtp-block", "mtp-head",
    ], [
      e("hc-post-moe", "stack-exit"), e("stack-exit", "hc-head-collapse"), e("hc-head-collapse", "final-rmsnorm"), e("final-rmsnorm", "last-token"), e("last-token", "lm-project"), e("lm-project", "logits"),
      e("stack-exit", "mtp-hidden-proj", "branch"), e("input-ids", "mtp-embed", "branch"), e("mtp-hidden-proj", "mtp-combine"), e("mtp-embed", "mtp-combine"), e("mtp-combine", "mtp-block"), e("mtp-block", "mtp-head"),
    ], [
      group("Final stack state", ["input-ids", "hc-post-moe", "stack-exit"], "hc"),
      group("LM head path", ["hc-head-collapse", "final-rmsnorm", "last-token", "lm-project", "logits"], "output"),
      group("MTP branch", ["mtp-embed", "mtp-hidden-proj", "mtp-combine", "mtp-block", "mtp-head"], "output"),
    ]),
  };

  // Simple mode is intentionally a separate topology per scene. It reuses the
  // same node docs, but owns its own nodes/edges/groups so summary edges do not
  // depend on detailed-mode intermediate nodes being hidden correctly.
  const simpleViews = {
    overview: simpleView([
      "input-ids", "embedding", "hc-expand", "stack-entry",
      "hc-controller", "hc-read", "attention", "compressor", "indexer", "hc-write",
      "hc-pre-moe", "gate", "routed-experts", "shared-expert", "moe", "hc-post-moe",
      "stack-exit", "head", "mtp", "logits",
    ], [
      e("input-ids", "embedding"), e("embedding", "hc-expand"), e("hc-expand", "stack-entry"),
      e("stack-entry", "hc-controller"), e("hc-controller", "hc-read"), e("hc-read", "attention"),
      e("hc-read", "compressor", "branch", { mode: ["csa", "hca"] }), e("compressor", "attention", "branch", { mode: ["csa", "hca"] }),
      e("hc-read", "indexer", "branch", { mode: "csa" }), e("indexer", "attention", "branch", { mode: "csa" }),
      e("attention", "hc-write"), e("hc-write", "hc-pre-moe"), e("hc-pre-moe", "gate"),
      e("gate", "routed-experts"), e("hc-pre-moe", "shared-expert", "branch"), e("routed-experts", "moe"), e("shared-expert", "moe", "branch"),
      e("moe", "hc-post-moe"), e("hc-post-moe", "stack-exit"), e("stack-exit", "head"), e("head", "logits"), e("stack-exit", "mtp", "branch"),
    ], [
      group("Model entry", ["input-ids", "embedding", "hc-expand", "stack-entry"], "stream"),
      group("mHC wrapper", ["hc-controller", "hc-read", "hc-write", "hc-pre-moe", "hc-post-moe"], "hc"),
      group("Attention + memory", ["attention", "compressor", "indexer"], "attention"),
      group("MoE summary", ["gate", "routed-experts", "shared-expert", "moe"], "expert"),
      group("Final output", ["stack-exit", "head", "mtp", "logits"], "output"),
    ]),
    mhc: simpleView([
      "hc-expand", "hc-controller", "hc-read", "attention", "attn-residual-mix", "attn-post-inject", "hc-write",
      "ffn-hc-controller", "hc-pre-moe", "moe", "ffn-residual-mix", "ffn-post-inject", "hc-post-moe",
    ], [
      e("hc-expand", "hc-controller"), e("hc-controller", "hc-read"), e("hc-read", "attention"),
      e("hc-expand", "attn-residual-mix", "branch"), e("attention", "attn-post-inject"), e("attn-residual-mix", "hc-write"), e("attn-post-inject", "hc-write"),
      e("hc-write", "ffn-hc-controller"), e("ffn-hc-controller", "hc-pre-moe"), e("hc-pre-moe", "moe"),
      e("hc-write", "ffn-residual-mix", "branch"), e("moe", "ffn-post-inject"), e("ffn-residual-mix", "hc-post-moe"), e("ffn-post-inject", "hc-post-moe"),
    ], [
      group("Attention mHC", ["hc-expand", "hc-controller", "hc-read", "attention", "attn-residual-mix", "attn-post-inject", "hc-write"], "hc"),
      group("MoE mHC", ["ffn-hc-controller", "hc-pre-moe", "moe", "ffn-residual-mix", "ffn-post-inject", "hc-post-moe"], "hc"),
    ]),
    attention: simpleView([
      "mhc-attn", "q-path", "kv-path", "cache-layout", "window-topk", "compressor", "indexer", "attn-selected", "sparse-attn", "o-proj", "hc-write",
    ], [
      e("mhc-attn", "q-path"), e("mhc-attn", "kv-path", "branch"), e("kv-path", "cache-layout"), e("cache-layout", "window-topk"),
      e("kv-path", "compressor", "branch", { mode: ["csa", "hca"] }), e("q-path", "indexer", "branch", { mode: "csa" }),
      e("window-topk", "attn-selected"), e("compressor", "attn-selected", "branch", { mode: "hca" }), e("indexer", "attn-selected", "branch", { mode: "csa" }),
      e("q-path", "sparse-attn"), e("attn-selected", "sparse-attn"), e("sparse-attn", "o-proj"), e("o-proj", "hc-write"),
    ], [
      group("Query/KV paths", ["q-path", "kv-path"], "attention"),
      group("Cache selection", ["cache-layout", "window-topk", "compressor", "indexer", "attn-selected"], "cache"),
      group("Attention output", ["sparse-attn", "o-proj", "hc-write"], "attention"),
    ]),
    compression: simpleView([
      "kv-path", "kv-cache", "swa-prefill-write", "swa-decode-write", "cache-layout", "window-topk",
      "compressor", "tail-state", "gated-pool", "comp-norm-rope", "comp-cache-write", "attn-gather",
    ], [
      e("kv-path", "kv-cache"), e("kv-cache", "swa-prefill-write"), e("kv-cache", "swa-decode-write", "branch"),
      e("swa-prefill-write", "cache-layout"), e("swa-decode-write", "cache-layout"), e("cache-layout", "window-topk"),
      e("kv-path", "compressor", "branch", { mode: ["csa", "hca"] }), e("compressor", "tail-state"), e("tail-state", "gated-pool"), e("gated-pool", "comp-norm-rope"), e("comp-norm-rope", "comp-cache-write"), e("comp-cache-write", "cache-layout", "branch"), e("cache-layout", "attn-gather"),
    ], [
      group("SWA cache", ["kv-path", "kv-cache", "swa-prefill-write", "swa-decode-write", "cache-layout", "window-topk"], "cache"),
      group("Compressed cache", ["compressor", "tail-state", "gated-pool", "comp-norm-rope", "comp-cache-write"], "cache"),
      group("Consumer", ["attn-gather"], "attention"),
    ]),
    indexer: simpleView([
      "q-norm", "idx-q", "idx-cache", "idx-einsum", "idx-weight", "idx-topk", "idx-offset", "attn-selected",
    ], [
      e("q-norm", "idx-q"), e("idx-q", "idx-einsum"), e("idx-cache", "idx-einsum"), e("idx-einsum", "idx-weight"), e("idx-weight", "idx-topk"), e("idx-topk", "idx-offset"), e("idx-offset", "attn-selected"),
    ], [
      group("Indexer query", ["q-norm", "idx-q"], "attention"),
      group("Index cache", ["idx-cache"], "cache"),
      group("Block selection", ["idx-einsum", "idx-weight", "idx-topk", "idx-offset", "attn-selected"], "attention"),
    ]),
    moe: simpleView([
      "mhc-ffn", "gate-score", "hash-route", "topk-route", "route-weights", "expert-dispatch", "routed-experts", "shared-expert", "expert-combine", "hc-post-moe",
    ], [
      e("mhc-ffn", "gate-score"), e("gate-score", "hash-route", "branch"), e("gate-score", "topk-route", "branch"),
      e("hash-route", "route-weights"), e("topk-route", "route-weights"), e("route-weights", "expert-dispatch"),
      e("expert-dispatch", "routed-experts"), e("mhc-ffn", "shared-expert", "branch"), e("routed-experts", "expert-combine"), e("shared-expert", "expert-combine", "branch"), e("expert-combine", "hc-post-moe"),
    ], [
      group("Routing", ["gate-score", "hash-route", "topk-route", "route-weights"], "routing"),
      group("Experts", ["expert-dispatch", "routed-experts", "shared-expert", "expert-combine"], "expert"),
    ]),
    output: simpleView([
      "input-ids", "hc-post-moe", "stack-exit", "head", "mtp", "logits",
    ], [
      e("hc-post-moe", "stack-exit"), e("stack-exit", "head"), e("head", "logits"), e("input-ids", "mtp", "branch"), e("stack-exit", "mtp", "branch"),
    ], [
      group("Final state", ["hc-post-moe", "stack-exit"], "hc"),
      group("LM head", ["head", "logits"], "output"),
      group("MTP summary", ["input-ids", "mtp"], "output"),
    ]),
  };

  Object.entries(scenes).forEach(([id, item]) => {
    item.views = {
      detailed: { nodeIds: item.nodeIds, edges: item.edges, groups: item.groups },
      simple: simpleViews[id] || { nodeIds: item.nodeIds, edges: item.edges, groups: item.groups },
    };
  });

  function n(id, title, category, input, output, summary, params, notes, sources, drill = null, visibleWhen = null) {
    return { id, title, category, input, output, summary, params, notes, sources, drill, visibleWhen };
  }


  function e(from, to, type = "main", visibleWhen = null, label = null) {
    return { from, to, type, visibleWhen, label };
  }

  function scene(id, title, subtitle, nodeIds, edges, groups = []) {
    return { id, title, subtitle, nodeIds, edges, groups };
  }

  function group(label, nodeIds, category) {
    return { label, nodeIds, category };
  }

  function simpleView(nodeIds, edges, groups = []) {
    return { nodeIds, edges, groups };
  }

  return { models, sources, nodes, scenes };
})();
