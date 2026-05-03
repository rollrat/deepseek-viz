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
    "hc-post-moe": n("hc-post-moe", "MoE HC Writeback", "hc", "mixed residual, injected MoE", "[B,S,4,D]", "MoE residual lane mixing 결과와 MoE output injection을 합쳐 다음 block residual lanes를 만든다.", { output: "[B,S,4,D]" }, ["다음 block input이 된다."], [sources.code]),
    head: n("head", "HC Head + LM Head", "output", "[B,S,4,D]", "[B,V]", "hc_head collapse 후 last token vocab projection.", { V: 129280, D: "$D" }, ["공식 path는 x[:, -1]만 logits 계산."], common.src, "output"),
    mtp: n("mtp", "MTP Block", "output", "hidden [B,S,4,D], ids [B,S]", "[B,V]", "추가 next-token prediction block.", { num_nextn: 1, R: 0 }, ["embedding path와 hidden path를 결합."], common.src, "output"),
    logits: n("logits", "Logits", "output", "[B,D]", "[B,129280]", "마지막 token vocabulary scores.", { vocab: 129280 }, ["sampling은 이 그래프 범위 밖."], [sources.card, sources.code]),

    "hc-flatten": n("hc-flatten", "Flatten HC Lanes", "hc", "[B,S,4,D]", "[B,S,4D]", "controller path용으로 lane 축과 hidden 축을 flatten.", { hc_dim: "4D" }, [], [sources.code]),
    "hc-controller": n("hc-controller", "Controller Linear", "hc", "[B,S,4D]", "[B,S,24]", "hc_fn linear로 mHC controller logits를 생성한다.", { weight: "[24,4D]" }, ["rsqrt normalization factor를 곱한다."], [sources.code]),
    "hc-sinkhorn": n("hc-sinkhorn", "Split + Sinkhorn", "hc", "mixes [B,S,24]", "pre, post, comb", "TileLang kernel이 pre/post/comb를 나누고 comb를 Sinkhorn normalize.", { iters: 20, eps: "1e-6" }, ["comb는 [B,S,4,4]."], [sources.kernel]),
    "hc-read": n("hc-read", "Read Data Path", "hc", "pre [B,S,4], X [B,S,4,D]", "[B,S,D]", "pre 가중합으로 sublayer input을 만든다.", {}, ["sum(pre * X) over lane axis."], [sources.code]),
    "attn-residual-mix": n("attn-residual-mix", "Attention Residual Lane Mixing", "hc", "comb [B,S,4,4], residual [B,S,4,D]", "[B,S,4,D]", "comb matrix가 기존 4개 residual lane을 token별로 서로 섞는다.", { comb: "[B,S,4,4]", lanes: 4 }, ["이 노드가 attention writeback의 핵심 residual lane mixing이다."], [sources.code, sources.kernel]),
    "attn-post-inject": n("attn-post-inject", "Attention Output Injection", "hc", "post [B,S,4], y [B,S,D]", "[B,S,4,D]", "attention output을 post weights로 4개 lane에 주입한다.", { post: "[B,S,4]" }, ["residual lane mixing과 별도 항으로 더해진다."], [sources.code]),
    "hc-write": n("hc-write", "Attention HC Writeback", "hc", "mixed residual, injected attention", "[B,S,4,D]", "comb * residual과 post * attention output을 합쳐 다음 residual lanes를 만든다.", {}, ["writeback = residual lane mixing + sublayer output injection."], [sources.code]),
    "ffn-hc-flatten": n("ffn-hc-flatten", "FFN Flatten HC Lanes", "hc", "[B,S,4,D]", "[B,S,4D]", "MoE/FFN용 mHC controller 입력을 만들기 위해 residual lanes를 flatten한다.", { hc_dim: "4D" }, ["attention mHC와 별도의 controller path."], [sources.code]),
    "ffn-hc-controller": n("ffn-hc-controller", "FFN Controller Linear", "hc", "[B,S,4D]", "[B,S,24]", "MoE/FFN 앞뒤에서 쓸 pre/post/comb logits를 생성한다.", { weight: "[24,4D]", mix_hc: 24 }, ["attention mHC parameter set과 분리된다."], [sources.code]),
    "ffn-hc-sinkhorn": n("ffn-hc-sinkhorn", "FFN Split + Sinkhorn", "hc", "mixes [B,S,24]", "pre, post, comb", "FFN용 pre/post/comb를 나누고 comb를 Sinkhorn normalize한다.", { comb: "[B,S,4,4]" }, ["여기서 MoE writeback residual lane mixing weights가 나온다."], [sources.kernel]),
    "hc-pre-moe": n("hc-pre-moe", "MoE Read Data Path", "hc", "pre [B,S,4], residual [B,S,4,D]", "[B,S,D]", "MoE FFN에 들어갈 hidden state를 mHC pre weights로 lane 축에서 읽는다.", { pre: "[B,S,4]" }, ["sum(pre * residual) over lane axis."], [sources.code, sources.kernel]),
    "ffn-residual-mix": n("ffn-residual-mix", "MoE Residual Lane Mixing", "hc", "comb [B,S,4,4], residual [B,S,4,D]", "[B,S,4,D]", "MoE writeback에서 기존 residual lanes를 comb matrix로 다시 섞는다.", { comb: "[B,S,4,4]", lanes: 4 }, ["attention writeback과 같은 residual lane mixing 구조지만 별도 mHC weights를 쓴다."], [sources.code, sources.kernel]),
    "ffn-post-inject": n("ffn-post-inject", "MoE Output Injection", "hc", "post [B,S,4], moe [B,S,D]", "[B,S,4,D]", "MoE output을 post weights로 4개 residual lane에 주입한다.", { post: "[B,S,4]" }, [], [sources.code]),

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

  Object.assign(nodes["window-topk"], {
    title: "SWA Window IDs",
    summary: "Sliding-window branch over the most recent 128 tokens; active in every attention mode.",
  });
  Object.assign(nodes["attn-selected"], {
    title: "Attention KV Set",
    input: "SWA ids + active compressed ids",
    output: "selected KV positions",
    summary: "The final KV set merges the local SWA window with the active compressed path: CSA top-k blocks, HCA all compressed blocks, or no compressed blocks for MTP.",
  });
  Object.assign(nodes["sparse-attn"], {
    title: "Hybrid Attention Kernel",
    summary: "Runs attention over Q and the selected KV set produced by SWA, CSA, or HCA mode.",
  });
  Object.assign(nodes["input-ids"], {
    title: "Input IDs (model entry)",
    summary: "Tokenizer output enters the model once before the decoder layer stack.",
  });
  Object.assign(nodes["embedding"], {
    title: "Token Embedding (once)",
    summary: "Looks up token vectors once before the decoder layer stack begins.",
  });
  nodes["stack-entry"] = n(
    "stack-entry",
    "Decoder Stack Entry",
    "stream",
    "[B,S,4,D]",
    "repeated decoder state",
    "The expanded graph below is one representative decoder layer selected by the layer-mode control; this is not a second input stream inside every layer.",
    { decoder_layers: "$layers" },
    ["Input and embedding are outside the repeated decoder block."],
    common.src,
  );
  nodes["stack-exit"] = n(
    "stack-exit",
    "Final Stack State",
    "output",
    "after decoder layer 60",
    "[B,S,4,D]",
    "Only after the decoder stack finishes does the graph branch to HC head / LM head and MTP.",
    { lm_head: "final token only" },
    ["The official path computes logits from x[:, -1], not from every layer."],
    common.src,
  );
  Object.assign(nodes["head"], {
    title: "Final HC Head + LM Head",
    summary: "Runs after the full decoder stack, collapsing HC lanes and projecting only the last token to vocabulary logits.",
  });
  Object.assign(nodes["mtp"], {
    title: "Final MTP Block",
    summary: "Auxiliary next-token prediction branch after the final stack state.",
  });
  Object.assign(nodes["logits"], {
    title: "Final Logits",
    summary: "Vocabulary scores for the final-token path; sampling is outside this graph.",
  });

  Object.assign(nodes["comp-wkv"], { visibleWhen: { mode: ["csa", "hca"] } });
  Object.assign(nodes["comp-wgate"], { visibleWhen: { mode: ["csa", "hca"] } });
  Object.assign(nodes["tail-state"], {
    summary: "Buffers tail tokens until a full compressed block can be formed.",
    visibleWhen: { mode: ["csa", "hca"] },
  });
  Object.assign(nodes["overlap-transform"], {
    title: "CSA Overlap Transform",
    summary: "R=4 CSA path overlaps previous and current chunks before gated pooling.",
    params: { active: "CSA / R=4 only" },
    visibleWhen: { mode: "csa" },
  });
  Object.assign(nodes["gated-pool"], {
    summary: "Pools R tokens into a compressed KV block using softmax weights.",
    visibleWhen: { mode: ["csa", "hca"] },
  });
  Object.assign(nodes["comp-norm-rope"], { visibleWhen: { mode: ["csa", "hca"] } });
  Object.assign(nodes["comp-cache-write"], {
    summary: "Writes compressed KV after the live sliding-window cache region.",
    visibleWhen: { mode: ["csa", "hca"] },
  });
  nodes["hca-all-compressed"] = n(
    "hca-all-compressed",
    "HCA All Compressed Blocks",
    "attention",
    "compressed cache [B,T/128,512]",
    "[B,S,T/128]",
    "R=128 HCA layers read all valid heavily-compressed blocks instead of running the Lightning indexer.",
    { R: 128 },
    ["No Lightning indexer is used in HCA mode."],
    [sources.code],
    null,
    { mode: "hca" },
  );

  ["idx-q", "idx-rotate", "idx-cache", "idx-einsum", "idx-weight", "idx-topk"].forEach((id) => {
    nodes[id].visibleWhen = { mode: "csa" };
  });
  Object.assign(nodes["idx-q"], { summary: "CSA-only indexer query projection from q_norm." });
  Object.assign(nodes["idx-cache"], { summary: "CSA-only compressed index cache used by the Lightning indexer." });
  Object.assign(nodes["idx-topk"], { summary: "Applies causal masking and selects top-k compressed block ids for CSA." });

  Object.entries({
    "input-ids": { why: "모델이 받는 유일한 이산 시퀀스 입력입니다. 임베딩으로 한 번 들어가고, 초반 MoE hash routing에서도 token id 그대로 다시 쓰입니다.", runtime: "hash routing 레이어는 score top-k로 expert id를 고르지 않고 input_ids.flatten()으로 tid2eid를 조회합니다.", ui: "임베딩 입력과 초반 라우팅 메타데이터라는 두 용도를 같이 보여주는 게 좋습니다.", open: "tid2eid가 학습/배정될 때 자주 등장하는 토큰의 expert 쏠림을 어떻게 막았는지는 공개 inference 코드만으로는 확인되지 않습니다." },
    embedding: { why: "token id를 mHC, attention, MoE가 처리할 연속 hidden vector로 바꿉니다.", runtime: "tensor parallel에서는 vocab shard 밖 id를 mask하고 partial embedding을 all-reduce합니다.", ui: "decoder stack 안에서 매번 실행되는 노드가 아니라 stack 진입 전 1회 노드로 표시합니다." },
    "hc-expand": { why: "mHC가 읽고 쓸 4개의 residual lane을 만듭니다.", runtime: "공식 forward는 [B,S,D]를 [B,S,hc_mult,D]로 repeat합니다.", ui: "lane 축을 실제 tensor 차원으로 보여주고, 모델 복사본 4개처럼 보이지 않게 합니다." },
    "stack-entry": { why: "1회성 입력 처리와 아래에 그려진 대표 decoder layer를 분리하는 경계입니다.", ui: "레이어 모드 버튼은 대표 layer 내부 경로만 바꾸며 input node 자체를 바꾸는 것이 아닙니다." },
    "mhc-attn": { why: "일반 residual add 대신 attention 앞뒤를 mHC read/write로 감쌉니다.", runtime: "controller coefficient는 inference에서도 현재 residual lane에서 매번 다시 계산됩니다.", ui: "attention 자체가 아니라 attention 주변의 controller + data path로 표시합니다." },
    "hc-flatten": { why: "controller가 4개 lane을 동시에 보고 coefficient를 만들기 위해 lane과 hidden 축을 합칩니다.", ui: "실제 attention data path가 아니라 coefficient 생성으로 들어가는 control flow로 그립니다." },
    "hc-controller": { why: "pre[4], post[4], comb[4,4]에 해당하는 총 24개 logit을 만듭니다.", runtime: "flatten된 lane은 controller linear 출력 전후로 RMS scale 정규화가 들어갑니다.", ui: "pre, post, comb 세 갈래로 split되는 지점으로 보여줍니다." },
    "hc-sinkhorn": { why: "제약 없는 controller logit을 read weight, write weight, 4x4 lane mixing matrix로 바꿉니다.", runtime: "comb는 hc_split_sinkhorn kernel에서 row/column normalization을 반복해 doubly stochastic에 가깝게 만듭니다.", ui: "pre/post는 vector, comb는 4x4 heatmap처럼 보여주면 좋습니다.", open: "mHC constraint가 학습 안정성에 기여하는 정확한 분석은 technical report와 같이 대조해야 합니다." },
    "hc-read": { why: "attention은 하나의 hidden stream을 받으므로 4개 residual lane을 [B,S,D]로 읽어야 합니다.", formula: "x_attn = sum_l pre_l * lane_l.", ui: "controller coefficient가 data path에 처음 적용되는 지점입니다." },
    "attn-residual-mix": { why: "기존 residual lane 정보를 identity add가 아니라 학습된 4x4 lane transport로 다음 상태에 넘깁니다.", formula: "mixed = comb @ residual_lanes.", ui: "attention output injection과 분리해서 보여줘야 합니다." },
    "attn-post-inject": { why: "새로 계산된 attention output을 4개 residual lane에 다시 주입합니다.", formula: "inject_l = post_l * attention_out.", ui: "하나의 attention stream이 4개 lane으로 fan-out되는 구조입니다." },
    "hc-write": { why: "attention 쪽 mHC writeback을 완성합니다.", formula: "new_lanes = comb @ residual_lanes + post * attention_out." },
    attention: { why: "mHC가 단일 hidden stream으로 읽어낸 뒤 token mixing을 수행합니다.", runtime: "내부 cache path는 layer ratio에 따라 CSA/c4a R=4, HCA/c128a R=128, SWA-only R=0으로 달라집니다.", ui: "선택된 한 layer에서 CSA와 HCA가 동시에 켜진 것처럼 보이지 않게 합니다." },
    "q-path": { why: "multi-head로 확장하기 전 low-rank query 생성 과정을 요약합니다.", ui: "compact scene에서만 쓰고, overview에서는 하위 노드들을 직접 보여주는 편이 좋습니다." },
    "q-wqa": { why: "128개 query head로 확장하기 전에 query 계산을 low-rank 공간으로 줄입니다.", runtime: "Pro의 q_lora_rank는 1536입니다.", ui: "LoRA식 low-rank A projection으로 표시합니다." },
    "q-norm": { why: "분기되기 전 low-rank query latent의 scale을 안정화합니다.", runtime: "main query expansion과 CSA Lightning indexer query path가 둘 다 이 출력을 씁니다.", ui: "fork 지점으로 보여줍니다." },
    "q-wqb": { why: "low-rank query latent를 per-head query vector로 확장합니다.", runtime: "Pro 기준 논리 출력은 H*512 = 128*512 = 65536 channel입니다." },
    "q-reshape": { why: "sparse attention이 사용할 query head 축을 드러냅니다.", runtime: "코드에서는 reshape 후 per-head RMS re-normalization도 수행합니다.", ui: "학습 파라미터가 있는 layer라기보다 axis split으로 보여줍니다." },
    "q-rope": { why: "attention score 계산을 위해 query vector에 위치 위상을 넣습니다.", runtime: "마지막 64 dim만 RoPE가 적용되고 나머지 448 dim은 content dim으로 남습니다.", ui: "각 head를 448 no-RoPE dim + 64 RoPE dim으로 나눠 보여줍니다." },
    "kv-path": { why: "V4는 per-head KV 대신 하나의 shared 512-dim KV stream을 써서 cache memory를 줄입니다.", runtime: "동일한 shared vector가 logit 계산에서는 key처럼, output 계산에서는 value처럼 쓰입니다.", ui: "KV sharing과 inverse RoPE 이야기가 시작되는 핵심 노드입니다." },
    "kv-wkv": { why: "hidden state를 단일 shared KV cache vector로 투영합니다.", runtime: "K에는 RoPE가 필요하지만 V에는 절대 위치가 섞이면 이상하므로 뒤에서 inverse RoPE로 보정합니다." },
    "kv-norm": { why: "RoPE, quantization, compression, cache write 전에 shared KV의 scale을 안정화합니다." },
    "kv-rope-quant": { why: "key 역할에 필요한 위치 정보를 넣으면서 cache memory는 낮게 유지합니다.", runtime: "RoPE dim은 BF16으로 유지하고 non-RoPE dim은 FP8 simulation을 적용합니다.", ui: "448개 quantized content dim과 64개 BF16 RoPE dim으로 나눠 보여줍니다." },
    "kv-cache": { why: "최근 uncompressed SWA entry와 오래된 compressed entry를 하나의 논리 cache로 관리합니다.", runtime: "앞 128 slot은 SWA, 그 뒤 suffix는 compressed entry입니다. decode에서는 SWA를 start_pos % 128 위치에 씁니다.", ui: "window prefix + compressed suffix 구조로 그립니다." },
    "window-topk": { why: "compressed block이 causality 근처의 local 정보를 안전하게 표현하지 못하므로 최근 token을 uncompressed로 보존합니다.", runtime: "CSA, HCA, SWA-only 모든 모드에서 존재합니다.", ui: "score 기반 top-k가 아니라 sliding-window index set입니다." },
    "attn-selected": { why: "attention kernel이 읽을 최종 KV index set을 만듭니다.", runtime: "CSA = SWA ids + indexer topK, HCA = SWA ids + 모든 valid c128a ids, SWA-only = SWA ids만 사용합니다.", ui: "index-set union 노드로 보여줍니다." },
    compressor: { why: "1M context에서 오래된 token을 모두 일반 KV로 유지할 수 없으므로 더 싼 memory entry로 압축합니다.", runtime: "R=4는 c4a overlap, R=128은 c128a non-overlap 동작입니다.", ui: "선택된 layer mode에 따라 c4a/c128a label을 보여줍니다." },
    "comp-wkv": { why: "여러 native token을 pooling해서 compressed KV entry로 만들 후보 vector를 생성합니다.", runtime: "c4a overlap에서는 Coff=2, c128a에서는 Coff=1입니다." },
    "comp-wgate": { why: "압축은 단순 평균이 아니라 learned softmax-gated pooling입니다.", runtime: "softmax 전에 learned ape가 score에 더해집니다.", ui: "weighted sum의 weight가 어디서 나오는지 보여주는 노드입니다." },
    "tail-state": { why: "decode는 token이 하나씩 들어오므로 compression boundary에 도달할 때까지 partial window를 보관해야 합니다.", runtime: "c4a는 overlap된 8-token 스타일 compressor state, c128a는 128-token state를 유지합니다.", ui: "projection이 아니라 request별 persistent state로 표시합니다." },
    "overlap-transform": { why: "c4a는 stride 4로 압축하지만 pooling은 8-token overlap span을 봅니다.", runtime: "이전 block half와 현재 block half를 gated pooling 전에 재배치하며 boundary는 0 또는 -inf padding으로 처리됩니다.", ui: "native span과 anchor position을 분리해서 보여줍니다." },
    "gated-pool": { why: "여러 native token에서 중요한 정보를 골라 하나의 compressed KV entry를 만듭니다.", formula: "compressed = sum_t kv_t * softmax(score_t + ape_t).", ui: "c4a는 8-to-1, c128a는 128-to-1 pooling처럼 보여줍니다." },
    "comp-norm-rope": { why: "compressed entry도 attention score 계산을 위해 위치 위상이 필요합니다.", runtime: "prefill anchor는 R 간격 위치를 쓰고, decode에서는 block 완성 시 start_pos + 1 - R을 anchor로 씁니다.", ui: "anchor position이라는 용어를 명시적으로 보여줍니다." },
    "comp-cache-write": { why: "압축된 long-context memory를 live SWA window 뒤쪽에 저장합니다.", runtime: "논리 compressed index는 대략 start_pos // R이며 serving runtime은 이를 page로 다시 매핑할 수 있습니다." },
    "hca-all-compressed": { why: "128x compression이면 compressed block 수가 충분히 작아져 모든 block에 dense하게 attend할 수 있습니다.", runtime: "HCA에는 Lightning indexer가 없고 R=128에서는 Attention.indexer가 None입니다.", ui: "compressed memory 전체 + SWA에 attend하는 구조로 표시합니다." },
    indexer: { why: "c4a는 1M context에서 여전히 compressed block이 너무 많으므로 CSA는 sparse block retrieval이 필요합니다.", runtime: "Pro topK는 1024, Flash topK는 512입니다.", ui: "CSA/c4a 모드에서만 보여줍니다." },
    "idx-q": { why: "retrieval scoring을 위한 더 싼 index-query head를 만듭니다.", runtime: "최종 main Q head가 아니라 q_norm latent에서 파생됩니다." },
    "idx-rotate": { why: "indexer side path를 충분히 싸게 만들기 위한 변환입니다.", runtime: "RoPE, Hadamard rotation, FP4 activation quantization을 적용합니다.", ui: "main attention이 아니라 approximate retrieval scoring으로 표시합니다." },
    "idx-cache": { why: "indexer는 main 512-dim KV와 별도의 작은 128-dim compressed cache가 필요합니다.", runtime: "head_dim=128, rotate=true인 indexer 전용 compressor를 씁니다." },
    "idx-einsum": { why: "각 query token에 대해 candidate compressed block 점수를 계산합니다.", runtime: "dot product score는 weighted head sum 전에 ReLU를 통과합니다." },
    "idx-weight": { why: "64개 index head 점수를 block별 rank score 하나로 합칩니다.", runtime: "weights_proj(x)가 query-dependent index-head weight를 만듭니다." },
    "idx-topk": { why: "상위 compressed block만 골라 c4a attention compute를 제한합니다.", runtime: "causality mask를 적용하고 compressed id를 SWA window slot 뒤쪽 offset으로 맞춥니다." },
    "sparse-attn": { why: "SWA, CSA, HCA를 selected shared KV entry에 대한 gather-based attention으로 통합합니다.", runtime: "per-head attn_sink와 online-softmax 스타일 kernel 동작을 포함합니다.", ui: "선택 모드별 후보 수를 <=128, topK+128, T/128+128처럼 보여줍니다." },
    "o-proj": { why: "multi-head attention output을 다시 hidden size로 접습니다.", runtime: "KV가 shared라서 grouped low-rank output projection 전에 inverse RoPE를 적용합니다.", ui: "shared-KV attention 이후 value semantics를 보정하는 지점입니다." },
    "mhc-ffn": { why: "MoE도 attention과 같은 mHC read/write 패턴으로 감쌉니다.", runtime: "attention mHC와 별도의 FFN mHC parameter set을 씁니다." },
    "ffn-hc-flatten": { why: "MoE controller도 read/write coefficient를 만들기 전에 4개 lane을 모두 봐야 합니다." },
    "ffn-hc-controller": { why: "MoE-side pre/post/comb logit을 생성합니다.", ui: "attention controller와 같은 구조지만 독립 parameter라고 표시합니다." },
    "ffn-hc-sinkhorn": { why: "MoE sublayer에도 같은 manifold-constrained lane mixing을 적용합니다.", ui: "attention 쪽과 동일한 4x4 comb matrix 시각화를 재사용합니다." },
    "hc-pre-moe": { why: "router와 expert는 token당 하나의 hidden vector를 처리하므로 lane을 [B,S,D]로 읽어야 합니다." },
    "ffn-residual-mix": { why: "MoE sublayer를 지나며 residual lane 정보를 운반합니다." },
    "ffn-post-inject": { why: "MoE output을 residual lane들에 다시 분배합니다." },
    "hc-post-moe": { why: "다음 decoder block으로 넘어갈 4-lane residual state를 만듭니다." },
    moe: { why: "384개 routed expert 중 token당 6개만 활성화하고, 여기에 shared expert 1개를 더해 sparse FFN capacity를 확보합니다.", runtime: "공개 구현에서는 모든 decoder block이 MoE FFN을 씁니다.", ui: "shared expert는 항상 active, routed expert는 conditional로 그립니다." },
    gate: { why: "각 token을 어떤 sparse FFN expert가 처리할지 결정합니다.", runtime: "초반 hash layer는 token id로 expert id를 고르고, 이후 layer는 score top-k로 고릅니다." },
    "gate-score": { why: "routed expert affinity score를 계산합니다.", runtime: "hash layer에서도 route weight를 원래 score에서 gather하기 때문에 score 계산 자체는 남아 있습니다.", ui: "hash layer가 모든 scoring을 생략한다고 오해하지 않게 합니다. 생략되는 것은 score top-k selection입니다." },
    "hash-route": { why: "처음 3개 decoder layer에서 고정 token-id -> expert-id table을 사용합니다.", runtime: "tid2eid[input_ids]가 token당 6개 expert id를 반환합니다.", ui: "lexical/token-prior routing으로 보여줍니다.", open: "자주 등장하는 token의 expert 쏠림을 어떻게 완화했는지는 checkpoint나 report 분석이 필요합니다." },
    "topk-route": { why: "후반 layer는 더 풍부한 hidden representation을 바탕으로 activation-dependent routing을 사용합니다.", runtime: "bias는 selection에만 영향을 주고, route weight는 bias 없는 original score에서 gather합니다." },
    "route-weights": { why: "선택된 expert output을 누적하기 전에 scale을 정합니다.", runtime: "sqrtsoftplus score를 선택 expert들 사이에서 normalize하고 route_scale을 곱합니다." },
    "routed-experts": { why: "거대한 FFN parameter pool 중 일부만 token별로 활성화합니다.", runtime: "Pro는 384 routed expert, token당 6 activated expert를 사용합니다." },
    "expert-dispatch": { why: "expert id별로 token row를 묶어 각 expert가 배정된 row만 처리하게 합니다.", runtime: "코드는 torch.where(indices == expert_id)를 쓰고 parallel 환경에서는 routed output을 all-reduce합니다." },
    "expert-w1w3": { why: "SwiGLU는 gate projection과 up projection이 따로 필요합니다.", runtime: "Pro에서는 expert weight가 FP4일 수 있습니다." },
    swiglu: { why: "expert 내부의 비선형 변환입니다.", runtime: "Pro는 swiglu_limit=10.0으로 gate/up을 clamp한 뒤 silu(gate) * up을 계산합니다." },
    "expert-w2": { why: "expert intermediate activation을 다시 hidden size로 내립니다." },
    "shared-expert": { why: "sparse routing과 무관하게 모든 token에 공통 FFN 경로를 제공합니다.", runtime: "모든 token에 대해 계산되고 routed expert 누적값 뒤에 더해집니다.", ui: "routing gate 뒤가 아니라 routed expert와 병렬인 always-on path로 그립니다.", open: "공통 변환을 흡수해 routed expert 부담을 줄일 수 있다는 해석은 가능하지만, 공개 inference 코드만으로 load-balance 메커니즘이라고 단정할 수는 없습니다." },
    "expert-combine": { why: "조건부 routed expert 계산과 universal shared expert 계산을 합칩니다.", formula: "y = sum_selected expert_i(x) * weight_i + shared_expert(x)." },
    "stack-exit": { why: "최종 decoder output과 output-only head를 분리합니다.", ui: "LM head가 매 layer마다 실행되는 것처럼 보이는 오해를 막습니다." },
    head: { why: "HC lane을 하나로 접고 final hidden state를 vocabulary logits로 projection합니다.", runtime: "공식 get_logits는 x[:, -1]만 사용하므로 마지막 token만 projection합니다." },
    mtp: { why: "final stack state 뒤에 붙는 auxiliary multi-token prediction branch입니다.", runtime: "embedding/head module을 재사용하고 SWA-only attention mode를 가집니다." },
    logits: { why: "generation에 쓰이는 최종 vocabulary score vector입니다.", ui: "sampling, top-p, tool decoding은 이 architecture graph 밖의 단계입니다." },
  }).forEach(([id, details]) => {
    if (nodes[id]) nodes[id].details = details;
  });

  const scenes = {
    overview: scene("overview", "Full V4 internal graph", "All major controller paths, cache paths, routing paths, and expert internals are expanded in one graph.", [
      "input-ids", "embedding", "hc-expand", "stack-entry",
      "hc-flatten", "hc-controller", "hc-sinkhorn", "hc-read",
      "q-wqa", "q-norm", "q-wqb", "q-reshape", "q-rope",
      "kv-wkv", "kv-norm", "kv-rope-quant", "window-topk",
      "comp-wkv", "comp-wgate", "tail-state", "overlap-transform", "gated-pool", "comp-norm-rope", "comp-cache-write",
      "idx-q", "idx-rotate", "idx-cache", "idx-einsum", "idx-weight", "idx-topk", "hca-all-compressed",
      "attn-selected", "sparse-attn", "o-proj", "attn-residual-mix", "attn-post-inject", "hc-write",
      "ffn-hc-flatten", "ffn-hc-controller", "ffn-hc-sinkhorn", "hc-pre-moe", "gate-score", "hash-route", "topk-route", "route-weights", "expert-dispatch",
      "expert-w1w3", "swiglu", "expert-w2", "shared-expert", "expert-combine", "ffn-residual-mix", "ffn-post-inject", "hc-post-moe",
      "stack-exit", "head", "mtp", "logits",
    ], [
      e("input-ids", "embedding"), e("embedding", "hc-expand"),
      e("hc-expand", "stack-entry"), e("stack-entry", "hc-flatten"), e("hc-flatten", "hc-controller"), e("hc-controller", "hc-sinkhorn"), e("hc-sinkhorn", "hc-read"),
      e("hc-read", "q-wqa"), e("q-wqa", "q-norm"), e("q-norm", "q-wqb"), e("q-wqb", "q-reshape"), e("q-reshape", "q-rope"),
      e("hc-read", "kv-wkv", "branch"), e("kv-wkv", "kv-norm"), e("kv-norm", "kv-rope-quant"), e("kv-rope-quant", "window-topk", "branch"),
      e("kv-rope-quant", "comp-wkv", "branch"), e("kv-rope-quant", "comp-wgate", "branch"), e("comp-wkv", "tail-state"), e("comp-wgate", "tail-state"), e("tail-state", "overlap-transform", "branch", { mode: "csa" }), e("overlap-transform", "gated-pool", "branch", { mode: "csa" }), e("tail-state", "gated-pool", "branch", { mode: "hca" }), e("comp-wgate", "gated-pool", "branch"), e("gated-pool", "comp-norm-rope"), e("comp-norm-rope", "comp-cache-write"),
      e("q-norm", "idx-q", "branch", { mode: "csa" }), e("idx-q", "idx-rotate"), e("kv-rope-quant", "idx-cache", "branch", { mode: "csa" }), e("idx-rotate", "idx-einsum"), e("idx-cache", "idx-einsum"), e("idx-einsum", "idx-weight"), e("idx-weight", "idx-topk"),
      e("window-topk", "attn-selected"), e("comp-cache-write", "hca-all-compressed", "branch", { mode: "hca" }), e("hca-all-compressed", "attn-selected", "branch", { mode: "hca" }), e("idx-topk", "attn-selected", "branch", { mode: "csa" }), e("q-rope", "sparse-attn"), e("attn-selected", "sparse-attn"), e("sparse-attn", "o-proj"),
      e("hc-sinkhorn", "attn-residual-mix", "branch"), e("hc-expand", "attn-residual-mix", "branch"), e("o-proj", "attn-post-inject"), e("hc-sinkhorn", "attn-post-inject", "branch"), e("attn-residual-mix", "hc-write"), e("attn-post-inject", "hc-write"),
      e("hc-write", "ffn-hc-flatten"), e("ffn-hc-flatten", "ffn-hc-controller"), e("ffn-hc-controller", "ffn-hc-sinkhorn"), e("ffn-hc-sinkhorn", "hc-pre-moe"), e("hc-write", "ffn-residual-mix", "branch"), e("ffn-hc-sinkhorn", "ffn-residual-mix", "branch"),
      e("hc-pre-moe", "gate-score"), e("gate-score", "hash-route", "branch"), e("gate-score", "topk-route", "branch"), e("hash-route", "route-weights"), e("topk-route", "route-weights"), e("route-weights", "expert-dispatch"), e("expert-dispatch", "expert-w1w3"), e("expert-w1w3", "swiglu"), e("swiglu", "expert-w2"), e("hc-pre-moe", "shared-expert", "branch"), e("expert-w2", "expert-combine"), e("shared-expert", "expert-combine"), e("expert-combine", "ffn-post-inject"), e("ffn-hc-sinkhorn", "ffn-post-inject", "branch"), e("ffn-residual-mix", "hc-post-moe"), e("ffn-post-inject", "hc-post-moe"),
      e("hc-post-moe", "stack-exit"), e("stack-exit", "head"), e("stack-exit", "mtp", "branch"), e("head", "logits"),
    ], [
      group("Model entry (once)", ["input-ids", "embedding", "hc-expand", "stack-entry"], "stream"),
      group("mHC controller + read path", ["hc-flatten", "hc-controller", "hc-sinkhorn", "hc-read"], "hc"),
      group("Attention Q/KV paths", ["q-wqa", "q-norm", "q-wqb", "q-reshape", "q-rope", "kv-wkv", "kv-norm", "kv-rope-quant", "window-topk", "hca-all-compressed", "attn-selected", "sparse-attn", "o-proj"], "attention"),
      group("KV compressor + tail state", ["comp-wkv", "comp-wgate", "tail-state", "overlap-transform", "gated-pool", "comp-norm-rope", "comp-cache-write"], "cache"),
      group("Lightning indexer", ["idx-q", "idx-rotate", "idx-cache", "idx-einsum", "idx-weight", "idx-topk"], "attention"),
      group("mHC attention residual mixing", ["attn-residual-mix", "attn-post-inject", "hc-write"], "hc"),
      group("mHC MoE controller + read path", ["ffn-hc-flatten", "ffn-hc-controller", "ffn-hc-sinkhorn", "hc-pre-moe"], "hc"),
      group("MoE routing + SwiGLU experts", ["gate-score", "hash-route", "topk-route", "route-weights", "expert-dispatch", "expert-w1w3", "swiglu", "expert-w2", "shared-expert", "expert-combine"], "expert"),
      group("mHC MoE residual mixing", ["ffn-residual-mix", "ffn-post-inject", "hc-post-moe"], "hc"),
      group("Final output only", ["stack-exit", "head", "mtp", "logits"], "output"),
    ]),
    mhc: scene("mhc", "mHC controller/data path", "pre/post/comb generation plus read/write data path.", [
      "hc-expand",
      "hc-flatten", "hc-controller", "hc-sinkhorn", "hc-read",
      "attention", "attn-residual-mix", "attn-post-inject", "hc-write",
      "ffn-hc-flatten", "ffn-hc-controller", "ffn-hc-sinkhorn", "hc-pre-moe",
      "moe", "ffn-residual-mix", "ffn-post-inject", "hc-post-moe",
    ], [
      e("hc-expand", "hc-flatten"), e("hc-flatten", "hc-controller"), e("hc-controller", "hc-sinkhorn"), e("hc-sinkhorn", "hc-read"), e("hc-read", "attention"),
      e("hc-sinkhorn", "attn-residual-mix", "branch"), e("hc-expand", "attn-residual-mix", "branch"), e("attention", "attn-post-inject"), e("hc-sinkhorn", "attn-post-inject", "branch"), e("attn-residual-mix", "hc-write"), e("attn-post-inject", "hc-write"),
      e("hc-write", "ffn-hc-flatten"), e("ffn-hc-flatten", "ffn-hc-controller"), e("ffn-hc-controller", "ffn-hc-sinkhorn"), e("ffn-hc-sinkhorn", "hc-pre-moe"), e("hc-pre-moe", "moe"),
      e("hc-write", "ffn-residual-mix", "branch"), e("ffn-hc-sinkhorn", "ffn-residual-mix", "branch"), e("moe", "ffn-post-inject"), e("ffn-hc-sinkhorn", "ffn-post-inject", "branch"), e("ffn-residual-mix", "hc-post-moe"), e("ffn-post-inject", "hc-post-moe"),
    ], [
      group("mHC attention controller + read path", ["hc-flatten", "hc-controller", "hc-sinkhorn", "hc-read"], "hc"),
      group("mHC attention residual mixing", ["attn-residual-mix", "attn-post-inject", "hc-write"], "hc"),
      group("mHC MoE controller + read path", ["ffn-hc-flatten", "ffn-hc-controller", "ffn-hc-sinkhorn", "hc-pre-moe"], "hc"),
      group("mHC MoE residual mixing", ["ffn-residual-mix", "ffn-post-inject", "hc-post-moe"], "hc"),
    ]),
    attention: scene("attention", "Attention internals", "Q LoRA, shared KV, cache IDs, sparse attention, grouped output projection.", [
      "mhc-attn", "q-wqa", "q-norm", "q-wqb", "q-reshape", "q-rope", "kv-wkv", "kv-norm", "kv-rope-quant", "window-topk", "compressor", "indexer", "attn-selected", "sparse-attn", "o-proj", "hc-write",
    ], [
      e("mhc-attn", "q-wqa"), e("q-wqa", "q-norm"), e("q-norm", "q-wqb"), e("q-wqb", "q-reshape"), e("q-reshape", "q-rope"), e("mhc-attn", "kv-wkv", "branch"), e("kv-wkv", "kv-norm"), e("kv-norm", "kv-rope-quant"), e("kv-rope-quant", "window-topk", "branch"), e("kv-rope-quant", "compressor", "branch"), e("q-norm", "indexer", "branch"), e("window-topk", "attn-selected"), e("compressor", "attn-selected", "branch"), e("indexer", "attn-selected", "branch"), e("q-rope", "sparse-attn"), e("attn-selected", "sparse-attn"), e("sparse-attn", "o-proj"), e("o-proj", "hc-write"),
    ], [
      group("mHC attention entry/exit", ["mhc-attn", "hc-write"], "hc"),
      group("Query LoRA + RoPE", ["q-wqa", "q-norm", "q-wqb", "q-reshape", "q-rope"], "attention"),
      group("Shared KV + SWA", ["kv-wkv", "kv-norm", "kv-rope-quant", "window-topk"], "cache"),
      group("Compressed selection", ["compressor", "indexer", "attn-selected"], "attention"),
      group("Attention output projection", ["sparse-attn", "o-proj"], "attention"),
    ]),
    compression: scene("compression", "KV cache and compressor", "Window cache, compressed cache, tail state, overlap pooling, and cache writes.", [
      "kv-path", "kv-cache", "window-topk", "comp-wkv", "comp-wgate", "tail-state", "overlap-transform", "gated-pool", "comp-norm-rope", "comp-cache-write", "sparse-attn",
    ], [
      e("kv-path", "kv-cache"), e("kv-cache", "window-topk"), e("kv-path", "comp-wkv", "branch"), e("kv-path", "comp-wgate", "branch"), e("comp-wkv", "tail-state"), e("comp-wgate", "tail-state"), e("tail-state", "overlap-transform"), e("overlap-transform", "gated-pool"), e("comp-wgate", "gated-pool", "branch"), e("gated-pool", "comp-norm-rope"), e("comp-norm-rope", "comp-cache-write"), e("comp-cache-write", "sparse-attn"),
    ], [
      group("SWA window cache", ["kv-path", "kv-cache", "window-topk"], "cache"),
      group("Compressor projections", ["comp-wkv", "comp-wgate"], "cache"),
      group("Compressor tail + overlap", ["tail-state", "overlap-transform"], "cache"),
      group("Compressed entry write", ["gated-pool", "comp-norm-rope", "comp-cache-write"], "cache"),
      group("Attention consumer", ["sparse-attn"], "attention"),
    ]),
    indexer: scene("indexer", "Lightning indexer", "R=4 compressed block selector.", [
      "q-norm", "idx-q", "idx-rotate", "idx-cache", "idx-einsum", "idx-weight", "idx-topk", "attn-selected",
    ], [
      e("q-norm", "idx-q"), e("idx-q", "idx-rotate"), e("kv-path", "idx-cache", "branch"), e("idx-rotate", "idx-einsum"), e("idx-cache", "idx-einsum"), e("idx-einsum", "idx-weight"), e("idx-weight", "idx-topk"), e("idx-topk", "attn-selected"),
    ], [
      group("Indexer query path", ["q-norm", "idx-q", "idx-rotate"], "attention"),
      group("Indexer compressed KV cache", ["idx-cache"], "cache"),
      group("Score + head weighting", ["idx-einsum", "idx-weight"], "attention"),
      group("TopK selected blocks", ["idx-topk", "attn-selected"], "attention"),
    ]),
    moe: scene("moe", "MoE and SwiGLU experts", "Routing, expert dispatch, FP4 SwiGLU experts, shared expert, and combine.", [
      "mhc-ffn", "gate-score", "hash-route", "topk-route", "route-weights", "expert-dispatch", "expert-w1w3", "swiglu", "expert-w2", "shared-expert", "expert-combine", "hc-post-moe",
    ], [
      e("mhc-ffn", "gate-score"), e("gate-score", "hash-route", "branch"), e("gate-score", "topk-route", "branch"), e("hash-route", "route-weights"), e("topk-route", "route-weights"), e("route-weights", "expert-dispatch"), e("expert-dispatch", "expert-w1w3"), e("expert-w1w3", "swiglu"), e("swiglu", "expert-w2"), e("mhc-ffn", "shared-expert", "branch"), e("expert-w2", "expert-combine"), e("shared-expert", "expert-combine"), e("expert-combine", "hc-post-moe"),
    ], [
      group("mHC MoE entry/exit", ["mhc-ffn", "hc-post-moe"], "hc"),
      group("Router scores + ids", ["gate-score", "hash-route", "topk-route", "route-weights"], "routing"),
      group("Routed expert dispatch", ["expert-dispatch"], "routing"),
      group("SwiGLU expert internals", ["expert-w1w3", "swiglu", "expert-w2"], "expert"),
      group("Shared expert + combine", ["shared-expert", "expert-combine"], "expert"),
    ]),
    output: scene("output", "Output and MTP", "Final HC head collapse, LM head, and MTP branch.", [
      "hc-post-moe", "head", "mtp", "logits",
    ], [
      e("hc-post-moe", "head"), e("head", "logits"), e("hc-post-moe", "mtp", "branch"),
    ], [
      group("Final stack state", ["hc-post-moe"], "hc"),
      group("LM head path", ["head", "logits"], "output"),
      group("MTP branch", ["mtp"], "output"),
    ]),
  };

  function n(id, title, category, input, output, summary, params, notes, sources, drill = null, visibleWhen = null) {
    return { id, title, category, input, output, summary, params, notes, sources, drill, visibleWhen };
  }

  function e(from, to, type = "main", visibleWhen = null) {
    return { from, to, type, visibleWhen };
  }

  function scene(id, title, subtitle, nodeIds, edges, groups = []) {
    return { id, title, subtitle, nodeIds, edges, groups };
  }

  function group(label, nodeIds, category) {
    return { label, nodeIds, category };
  }

  return { models, sources, nodes, scenes };
})();
