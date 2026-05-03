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
    "input-ids": n("input-ids", "Input IDs", "stream", "raw prompt", "[B,S]", "Tokenizer가 만든 token id matrix.", { vocab: "129280" }, ["B는 batch size, S는 현재 forward chunk length."], [sources.card, sources.proConfig, sources.flashConfig]),
    embedding: n("embedding", "Token Embedding", "stream", "[B,S]", "[B,S,D]", "Token id를 dense hidden vector로 lookup한다.", { V: "129280", D: "$D" }, ["TP에서는 vocab shard 후 all-reduce."], common.src),
    "hc-expand": n("hc-expand", "HC Expand", "hc", "[B,S,D]", "[B,S,4,D]", "mHC residual lanes를 4개로 확장한다.", { hc_mult: 4 }, ["Block 사이 hidden state는 [B,S,4,D]."], common.src, "mhc"),
    "mhc-attn": n("mhc-attn", "mHC Pre/Post: Attention", "hc", "[B,S,4,D]", "[B,S,4,D]", "Attention sublayer 앞뒤의 controller/data path.", { mix_hc: 24, hc_dim: "4D" }, ["pre/post/comb를 생성하고 data path를 섞는다."], [sources.code, sources.kernel], "mhc"),
    attention: n("attention", "Hybrid Attention", "attention", "[B,S,D]", "[B,S,D]", "Q path, KV path, cache, compressor/indexer, sparse attention을 결합한다.", { H: "$H", R: "$R", Hd: 512 }, ["R=4는 Lightning Indexer, R=128은 compressed dense path."], [sources.code, sources.blog], "attention"),
    "q-path": n("q-path", "Q LoRA Path", "attention", "[B,S,D]", "[B,S,H,512]", "wq_a, q_norm, wq_b, q renorm, RoPE.", { q_lora_rank: "$Qr", rope_dim: 64 }, ["q 마지막 64 dims에 RoPE."], common.src, "attention"),
    "kv-path": n("kv-path", "Shared KV Path", "attention", "[B,S,D]", "[B,S,512]", "wkv, kv_norm, RoPE, non-RoPE FP8 simulation.", { kv_heads: 1, nope_dim: 448, rope_dim: 64 }, ["KV는 shared [B,S,512]."], common.src, "attention"),
    "kv-cache": n("kv-cache", "KV Cache", "cache", "[B,S,512]", "[B,128+T/R,512]", "Window cache와 compressed cache를 하나의 buffer로 관리한다.", { window: 128, compressed: "max_seq_len/R" }, ["prefill과 decode write path가 다르다."], [sources.code], "compression"),
    compressor: n("compressor", "KV Compressor", "cache", "[B,S,D]", "[B,floor(S/R),512]", "wkv/wgate/ape/tail state로 compressed KV를 만든다.", { R: "$R", ape: "[R,Coff*512]" }, ["R=4는 overlap, R=128은 non-overlap."], [sources.code], "compression"),
    indexer: n("indexer", "Lightning Indexer", "attention", "x [B,S,D], qr [B,S,Qr]", "[B,S,topK]", "R=4 layer에서 compressed blocks top-k를 선택한다.", { index_heads: 64, index_dim: 128, topK: "$indexTopK" }, ["Hadamard rotation, FP4 quant, weighted ReLU score."], [sources.code], "indexer", { ratio: 4 }),
    "sparse-attn": n("sparse-attn", "Sparse Attention", "attention", "q + selected KV", "[B,S,H,512]", "window ids와 compressed ids를 합쳐 attention을 계산한다.", { attn_sink: "[H]" }, ["output RoPE slice는 inverse RoPE."], [sources.code]),
    "o-proj": n("o-proj", "Grouped O Projection", "attention", "[B,S,H,512]", "[B,S,D]", "wo_a group low-rank projection 후 wo_b로 D차원 복원.", { groups: "$G", o_lora_rank: "$Or" }, ["각 group은 8 heads * 512 dims."], common.src),
    "mhc-ffn": n("mhc-ffn", "mHC Pre/Post: FFN", "hc", "[B,S,4,D]", "[B,S,4,D]", "MoE FFN 앞뒤의 mHC controller/data path.", { mix_hc: 24 }, ["Attention mHC와 별도 파라미터 세트."], [sources.code, sources.kernel], "mhc"),
    moe: n("moe", "MoE Router + Experts", "routing", "[B,S,D]", "[B,S,D]", "Gate가 top-6 experts를 고르고 routed/shared experts를 결합한다.", { E: "$E", K: 6, I: "$I" }, ["First 3 layers는 hash routing."], common.src, "moe"),
    gate: n("gate", "Router Gate", "routing", "[B*S,D]", "ids/weights [B*S,6]", "sqrtsoftplus score, hash/top-k selection, weight normalize.", { route_scale: "$routeScale", scoring: "sqrtsoftplus" }, ["bias는 selection score에만 적용."], [sources.code], "moe"),
    "routed-experts": n("routed-experts", "Routed Experts", "expert", "[N_e,D]", "[N_e,D]", "선택된 expert별 FP4 SwiGLU FFN.", { w1: "D->I", w3: "D->I", w2: "I->D" }, ["silu(w1(x)) * w3(x) 후 w2."], [sources.code, sources.card], "moe"),
    "shared-expert": n("shared-expert", "Shared Expert", "expert", "[B*S,D]", "[B*S,D]", "모든 token에 항상 더해지는 shared SwiGLU expert.", { shared: 1, I: "$I" }, ["routed expert output에 더해진다."], [sources.code], "moe"),
    "hc-post-moe": n("hc-post-moe", "HC Post MoE", "hc", "moe [B,S,D], residual [B,S,4,D]", "[B,S,4,D]", "MoE output을 residual lanes에 post/comb로 write-back.", { post: "[B,S,4]", comb: "[B,S,4,4]" }, ["다음 block input이 된다."], [sources.code]),
    head: n("head", "HC Head + LM Head", "output", "[B,S,4,D]", "[B,V]", "hc_head collapse 후 last token vocab projection.", { V: 129280, D: "$D" }, ["공식 path는 x[:, -1]만 logits 계산."], common.src, "output"),
    mtp: n("mtp", "MTP Block", "output", "hidden [B,S,4,D], ids [B,S]", "[B,V]", "추가 next-token prediction block.", { num_nextn: 1, R: 0 }, ["embedding path와 hidden path를 결합."], common.src, "output"),
    logits: n("logits", "Logits", "output", "[B,D]", "[B,129280]", "마지막 token vocabulary scores.", { vocab: 129280 }, ["sampling은 이 그래프 범위 밖."], [sources.card, sources.code]),

    "hc-flatten": n("hc-flatten", "Flatten HC Lanes", "hc", "[B,S,4,D]", "[B,S,4D]", "controller path용으로 lane 축과 hidden 축을 flatten.", { hc_dim: "4D" }, [], [sources.code]),
    "hc-controller": n("hc-controller", "Controller Linear", "hc", "[B,S,4D]", "[B,S,24]", "hc_fn linear로 mHC controller logits를 생성한다.", { weight: "[24,4D]" }, ["rsqrt normalization factor를 곱한다."], [sources.code]),
    "hc-sinkhorn": n("hc-sinkhorn", "Split + Sinkhorn", "hc", "mixes [B,S,24]", "pre, post, comb", "TileLang kernel이 pre/post/comb를 나누고 comb를 Sinkhorn normalize.", { iters: 20, eps: "1e-6" }, ["comb는 [B,S,4,4]."], [sources.kernel]),
    "hc-read": n("hc-read", "Read Data Path", "hc", "pre [B,S,4], X [B,S,4,D]", "[B,S,D]", "pre 가중합으로 sublayer input을 만든다.", {}, ["sum(pre * X) over lane axis."], [sources.code]),
    "hc-write": n("hc-write", "Write Data Path", "hc", "y [B,S,D], residual [B,S,4,D]", "[B,S,4,D]", "post * y + comb * residual.", {}, ["기존 lane mixing과 sublayer write를 동시에 수행."], [sources.code]),
    "hc-pre-moe": n("hc-pre-moe", "MoE Read Data Path", "hc", "residual [B,S,4,D]", "[B,S,D]", "MoE FFN에 들어갈 hidden state를 mHC pre weights로 lane 축에서 읽는다.", { pre: "[B,S,4]" }, ["attention write 이후 별도의 FFN controller/read path가 적용된다."], [sources.code, sources.kernel]),

    "q-wqa": n("q-wqa", "wq_a", "attention", "[B,S,D]", "[B,S,Qr]", "Query low-rank A projection.", { Qr: "$Qr" }, [], [sources.code]),
    "q-norm": n("q-norm", "q_norm", "attention", "[B,S,Qr]", "[B,S,Qr]", "Low-rank query RMSNorm.", { eps: "1e-6" }, [], [sources.code]),
    "q-wqb": n("q-wqb", "wq_b", "attention", "[B,S,Qr]", "[B,S,H*512]", "Query head expansion.", { H: "$H" }, [], [sources.code]),
    "q-reshape": n("q-reshape", "Head Reshape + q Renorm", "attention", "[B,S,H*512]", "[B,S,H,512]", "Head reshape 후 per-head RMS re-normalization.", {}, [], [sources.code]),
    "q-rope": n("q-rope", "Q RoPE Slice", "attention", "[B,S,H,512]", "[B,S,H,512]", "마지막 64 dims에 RoPE 적용.", { rope_dim: 64 }, [], [sources.code]),
    "kv-wkv": n("kv-wkv", "wkv", "attention", "[B,S,D]", "[B,S,512]", "Shared KV projection.", { kv_heads: 1 }, [], [sources.code]),
    "kv-norm": n("kv-norm", "kv_norm", "attention", "[B,S,512]", "[B,S,512]", "KV RMSNorm.", {}, [], [sources.code]),
    "kv-rope-quant": n("kv-rope-quant", "KV RoPE + FP8 Sim", "attention", "[B,S,512]", "[B,S,512]", "RoPE dims는 BF16, non-RoPE dims는 FP8 simulation.", { rope: 64, nope: 448 }, [], [sources.code, sources.kernel]),
    "window-topk": n("window-topk", "Window TopK IDs", "cache", "start_pos, S", "[B,S,<=128]", "최근 128 token window indices.", { window: 128 }, [], [sources.code]),
    "attn-selected": n("attn-selected", "Selected KV IDs", "attention", "window ids + compressed ids", "topk_idxs int", "sparse attention이 읽을 KV positions.", {}, [], [sources.code]),

    "comp-wkv": n("comp-wkv", "Compressor wkv", "cache", "[B,S,D]", "[B,S,Coff*512]", "Compression candidate KV projection.", { Coff: "1 or 2" }, [], [sources.code]),
    "comp-wgate": n("comp-wgate", "Compressor wgate", "cache", "[B,S,D]", "[B,S,Coff*512]", "Softmax pooling score projection.", { ape: "[R,Coff*512]" }, [], [sources.code]),
    "tail-state": n("tail-state", "Compressed Tail State", "cache", "remainder tokens", "kv_state / score_state", "아직 R개가 안 찬 tail tokens를 buffer에 보관한다.", { kv_state: "[B,Coff*R,Coff*512]", score_state: "same" }, [], [sources.code]),
    "overlap-transform": n("overlap-transform", "Overlap Transform", "cache", "[B,blocks,R,2*512]", "[B,blocks,2R,512]", "R=4에서 이전 chunk와 현재 chunk를 겹쳐 pooling.", { active: "R=4 only" }, [], [sources.code]),
    "gated-pool": n("gated-pool", "Softmax-Gated Pool", "cache", "kv, score+ape", "[B,blocks,512]", "R tokens를 softmax(score)로 가중합.", {}, [], [sources.code]),
    "comp-norm-rope": n("comp-norm-rope", "Norm + Compressed RoPE", "cache", "[B,blocks,512]", "[B,blocks,512]", "compressed KV norm 후 compressed position RoPE.", { theta: 160000 }, [], [sources.code]),
    "comp-cache-write": n("comp-cache-write", "Compressed Cache Write", "cache", "[B,blocks,512]", "kv_cache[:,128:]", "window 영역 뒤 compressed cache에 저장.", {}, [], [sources.code]),

    "idx-q": n("idx-q", "Indexer Q", "attention", "qr [B,S,Qr]", "[B,S,64,128]", "indexer wq_b projection.", { heads: 64, dim: 128 }, [], [sources.code]),
    "idx-rotate": n("idx-rotate", "RoPE + Hadamard + FP4", "attention", "[B,S,64,128]", "[B,S,64,128]", "index query rotation and FP4 activation quant.", {}, [], [sources.code, sources.kernel]),
    "idx-cache": n("idx-cache", "Index KV Cache", "cache", "x [B,S,D]", "[B,T/4,128]", "indexer 전용 compressor cache.", { R: 4, dim: 128 }, [], [sources.code]),
    "idx-einsum": n("idx-einsum", "Lightning Scores", "attention", "q, index KV", "[B,S,64,T/4]", "ReLU dot product scores.", {}, [], [sources.code]),
    "idx-weight": n("idx-weight", "weights_proj + Head Sum", "attention", "scores, weights [B,S,64]", "[B,S,T/4]", "head별 score를 weighted sum.", {}, [], [sources.code]),
    "idx-topk": n("idx-topk", "TopK + Offset", "attention", "[B,S,T/4]", "[B,S,topK]", "causal mask 후 top-k compressed block ids.", { topK: "$indexTopK" }, [], [sources.code]),

    "gate-score": n("gate-score", "Gate Scores", "routing", "[B*S,D]", "[B*S,E]", "linear + sqrtsoftplus expert scores.", { E: "$E" }, [], [sources.code]),
    "hash-route": n("hash-route", "Hash Route", "routing", "input_ids", "[B*S,6]", "first 3 layers use tid2eid lookup.", { layers: 3 }, [], [sources.code]),
    "topk-route": n("topk-route", "TopK Route", "routing", "scores + bias", "[B*S,6]", "later layers choose top-6 experts.", {}, [], [sources.code]),
    "route-weights": n("route-weights", "Normalize Weights", "routing", "selected scores", "[B*S,6]", "gather original scores, normalize, apply route scale.", { scale: "$routeScale" }, [], [sources.code]),
    "expert-dispatch": n("expert-dispatch", "Expert Dispatch", "expert", "ids, weights", "per-expert token batches", "torch.where(indices == expert_id)로 token dispatch.", {}, [], [sources.code]),
    "expert-w1w3": n("expert-w1w3", "w1 / w3", "expert", "[N_e,D]", "gate/up [N_e,I]", "SwiGLU의 gate와 up projection.", { I: "$I" }, [], [sources.code]),
    swiglu: n("swiglu", "SwiGLU + Clamp", "expert", "gate, up", "[N_e,I]", "clamp 후 silu(gate) * up.", { limit: 10.0 }, [], [sources.code]),
    "expert-w2": n("expert-w2", "w2 Down Projection", "expert", "[N_e,I]", "[N_e,D]", "expert output projection.", {}, [], [sources.code]),
    "expert-combine": n("expert-combine", "Routed + Shared Combine", "expert", "routed y, shared y", "[B*S,D]", "routed outputs accumulate, shared expert added.", {}, [], [sources.code]),
  };

  const scenes = {
    overview: scene("overview", "Full V4 internal graph", "All major controller paths, cache paths, routing paths, and expert internals are expanded in one graph.", [
      "input-ids", "embedding", "hc-expand",
      "hc-flatten", "hc-controller", "hc-sinkhorn", "hc-read",
      "q-wqa", "q-norm", "q-wqb", "q-reshape", "q-rope",
      "kv-wkv", "kv-norm", "kv-rope-quant", "window-topk",
      "comp-wkv", "comp-wgate", "tail-state", "overlap-transform", "gated-pool", "comp-norm-rope", "comp-cache-write",
      "idx-q", "idx-rotate", "idx-cache", "idx-einsum", "idx-weight", "idx-topk",
      "attn-selected", "sparse-attn", "o-proj", "hc-write",
      "hc-pre-moe", "gate-score", "hash-route", "topk-route", "route-weights", "expert-dispatch",
      "expert-w1w3", "swiglu", "expert-w2", "shared-expert", "expert-combine", "hc-post-moe",
      "head", "mtp", "logits",
    ], [
      e("input-ids", "embedding"), e("embedding", "hc-expand"),
      e("hc-expand", "hc-flatten"), e("hc-flatten", "hc-controller"), e("hc-controller", "hc-sinkhorn"), e("hc-sinkhorn", "hc-read"),
      e("hc-read", "q-wqa"), e("q-wqa", "q-norm"), e("q-norm", "q-wqb"), e("q-wqb", "q-reshape"), e("q-reshape", "q-rope"),
      e("hc-read", "kv-wkv", "branch"), e("kv-wkv", "kv-norm"), e("kv-norm", "kv-rope-quant"), e("kv-rope-quant", "window-topk", "branch"),
      e("kv-rope-quant", "comp-wkv", "branch"), e("kv-rope-quant", "comp-wgate", "branch"), e("comp-wkv", "tail-state"), e("comp-wgate", "tail-state"), e("tail-state", "overlap-transform"), e("overlap-transform", "gated-pool"), e("comp-wgate", "gated-pool", "branch"), e("gated-pool", "comp-norm-rope"), e("comp-norm-rope", "comp-cache-write"),
      e("q-norm", "idx-q", "branch"), e("idx-q", "idx-rotate"), e("kv-rope-quant", "idx-cache", "branch"), e("idx-rotate", "idx-einsum"), e("idx-cache", "idx-einsum"), e("idx-einsum", "idx-weight"), e("idx-weight", "idx-topk"),
      e("window-topk", "attn-selected"), e("comp-cache-write", "attn-selected", "branch"), e("idx-topk", "attn-selected", "branch"), e("q-rope", "sparse-attn"), e("attn-selected", "sparse-attn"), e("sparse-attn", "o-proj"),
      e("o-proj", "hc-write"), e("hc-sinkhorn", "hc-write", "branch"), e("hc-expand", "hc-write", "branch"),
      e("hc-write", "hc-pre-moe"), e("hc-pre-moe", "gate-score"), e("gate-score", "hash-route", "branch"), e("gate-score", "topk-route", "branch"), e("hash-route", "route-weights"), e("topk-route", "route-weights"), e("route-weights", "expert-dispatch"), e("expert-dispatch", "expert-w1w3"), e("expert-w1w3", "swiglu"), e("swiglu", "expert-w2"), e("hc-pre-moe", "shared-expert", "branch"), e("expert-w2", "expert-combine"), e("shared-expert", "expert-combine"), e("expert-combine", "hc-post-moe"),
      e("hc-post-moe", "head"), e("hc-post-moe", "mtp", "branch"), e("head", "logits"),
    ], [
      group("Input stream", ["input-ids", "embedding", "hc-expand"], "stream"),
      group("mHC controller + read path", ["hc-flatten", "hc-controller", "hc-sinkhorn", "hc-read"], "hc"),
      group("Attention Q/KV paths", ["q-wqa", "q-norm", "q-wqb", "q-reshape", "q-rope", "kv-wkv", "kv-norm", "kv-rope-quant", "window-topk", "attn-selected", "sparse-attn", "o-proj"], "attention"),
      group("KV compressor + tail state", ["comp-wkv", "comp-wgate", "tail-state", "overlap-transform", "gated-pool", "comp-norm-rope", "comp-cache-write"], "cache"),
      group("Lightning indexer", ["idx-q", "idx-rotate", "idx-cache", "idx-einsum", "idx-weight", "idx-topk"], "attention"),
      group("mHC attention writeback", ["hc-write", "hc-pre-moe"], "hc"),
      group("MoE routing + SwiGLU experts", ["gate-score", "hash-route", "topk-route", "route-weights", "expert-dispatch", "expert-w1w3", "swiglu", "expert-w2", "shared-expert", "expert-combine"], "expert"),
      group("mHC MoE writeback", ["hc-post-moe"], "hc"),
      group("Output heads", ["head", "mtp", "logits"], "output"),
    ]),
    mhc: scene("mhc", "mHC controller/data path", "pre/post/comb generation plus read/write data path.", [
      "hc-expand", "hc-flatten", "hc-controller", "hc-sinkhorn", "hc-read", "attention", "moe", "hc-write", "hc-post-moe",
    ], [
      e("hc-expand", "hc-flatten"), e("hc-flatten", "hc-controller"), e("hc-controller", "hc-sinkhorn"), e("hc-sinkhorn", "hc-read"), e("hc-read", "attention"), e("hc-read", "moe"), e("attention", "hc-write"), e("moe", "hc-write"), e("hc-sinkhorn", "hc-write", "branch"), e("hc-expand", "hc-write", "branch"), e("hc-write", "hc-post-moe"),
    ]),
    attention: scene("attention", "Attention internals", "Q LoRA, shared KV, cache IDs, sparse attention, grouped output projection.", [
      "mhc-attn", "q-wqa", "q-norm", "q-wqb", "q-reshape", "q-rope", "kv-wkv", "kv-norm", "kv-rope-quant", "window-topk", "compressor", "indexer", "attn-selected", "sparse-attn", "o-proj", "hc-post-attn",
    ], [
      e("mhc-attn", "q-wqa"), e("q-wqa", "q-norm"), e("q-norm", "q-wqb"), e("q-wqb", "q-reshape"), e("q-reshape", "q-rope"), e("mhc-attn", "kv-wkv", "branch"), e("kv-wkv", "kv-norm"), e("kv-norm", "kv-rope-quant"), e("kv-rope-quant", "window-topk", "branch"), e("kv-rope-quant", "compressor", "branch"), e("q-norm", "indexer", "branch"), e("window-topk", "attn-selected"), e("compressor", "attn-selected", "branch"), e("indexer", "attn-selected", "branch"), e("q-rope", "sparse-attn"), e("attn-selected", "sparse-attn"), e("sparse-attn", "o-proj"), e("o-proj", "hc-post-attn"),
    ]),
    compression: scene("compression", "KV cache and compressor", "Window cache, compressed cache, tail state, overlap pooling, and cache writes.", [
      "kv-path", "kv-cache", "window-topk", "comp-wkv", "comp-wgate", "tail-state", "overlap-transform", "gated-pool", "comp-norm-rope", "comp-cache-write", "sparse-attn",
    ], [
      e("kv-path", "kv-cache"), e("kv-cache", "window-topk"), e("kv-path", "comp-wkv", "branch"), e("kv-path", "comp-wgate", "branch"), e("comp-wkv", "tail-state"), e("comp-wgate", "tail-state"), e("tail-state", "overlap-transform"), e("overlap-transform", "gated-pool"), e("comp-wgate", "gated-pool", "branch"), e("gated-pool", "comp-norm-rope"), e("comp-norm-rope", "comp-cache-write"), e("comp-cache-write", "sparse-attn"),
    ]),
    indexer: scene("indexer", "Lightning indexer", "R=4 compressed block selector.", [
      "q-norm", "idx-q", "idx-rotate", "idx-cache", "idx-einsum", "idx-weight", "idx-topk", "attn-selected",
    ], [
      e("q-norm", "idx-q"), e("idx-q", "idx-rotate"), e("kv-path", "idx-cache", "branch"), e("idx-rotate", "idx-einsum"), e("idx-cache", "idx-einsum"), e("idx-einsum", "idx-weight"), e("idx-weight", "idx-topk"), e("idx-topk", "attn-selected"),
    ]),
    moe: scene("moe", "MoE and SwiGLU experts", "Routing, expert dispatch, FP4 SwiGLU experts, shared expert, and combine.", [
      "mhc-ffn", "gate-score", "hash-route", "topk-route", "route-weights", "expert-dispatch", "expert-w1w3", "swiglu", "expert-w2", "shared-expert", "expert-combine", "hc-post-moe",
    ], [
      e("mhc-ffn", "gate-score"), e("gate-score", "hash-route", "branch"), e("gate-score", "topk-route", "branch"), e("hash-route", "route-weights"), e("topk-route", "route-weights"), e("route-weights", "expert-dispatch"), e("expert-dispatch", "expert-w1w3"), e("expert-w1w3", "swiglu"), e("swiglu", "expert-w2"), e("mhc-ffn", "shared-expert", "branch"), e("expert-w2", "expert-combine"), e("shared-expert", "expert-combine"), e("expert-combine", "hc-post-moe"),
    ]),
    output: scene("output", "Output and MTP", "Final HC head collapse, LM head, and MTP branch.", [
      "hc-post-moe", "head", "mtp", "logits",
    ], [
      e("hc-post-moe", "head"), e("head", "logits"), e("hc-post-moe", "mtp", "branch"),
    ]),
  };

  function n(id, title, category, input, output, summary, params, notes, sources, drill = null, visibleWhen = null) {
    return { id, title, category, input, output, summary, params, notes, sources, drill, visibleWhen };
  }

  function e(from, to, type = "main") {
    return { from, to, type };
  }

  function scene(id, title, subtitle, nodeIds, edges, groups = []) {
    return { id, title, subtitle, nodeIds, edges, groups };
  }

  function group(label, nodeIds, category) {
    return { label, nodeIds, category };
  }

  return { models, sources, nodes, scenes };
})();
