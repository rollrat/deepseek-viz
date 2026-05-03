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
    "input-ids": n("input-ids", "Input IDs", "stream", "prompt_text [string]", "ids [B,S]", "Tokenizer가 만든 token id matrix.", { vocab: "129280" }, ["B는 batch size, S는 현재 forward chunk length."], [sources.card, sources.proConfig, sources.flashConfig]),
    embedding: n("embedding", "Token Embedding", "stream", "[B,S]", "[B,S,D]", "Token id를 dense hidden vector로 lookup한다.", { V: "129280", D: "$D" }, ["TP에서는 vocab shard 후 all-reduce."], common.src),
    "hc-expand": n("hc-expand", "HC Expand", "hc", "[B,S,D]", "[B,S,4,D]", "mHC residual lanes를 4개로 확장한다.", { hc_mult: 4 }, ["Block 사이 hidden state는 [B,S,4,D]."], common.src, "mhc"),
    "mhc-attn": n("mhc-attn", "mHC Pre/Post: Attention", "hc", "[B,S,4,D]", "[B,S,4,D]", "Attention sublayer 앞뒤의 controller/data path.", { mix_hc: 24, hc_dim: "4D" }, ["pre/post/comb를 생성하고 data path를 섞는다."], [sources.code, sources.kernel], "mhc"),
    attention: n("attention", "Hybrid Attention", "attention", "[B,S,D]", "[B,S,D]", "Q path, KV path, cache, compressor/indexer, sparse attention을 결합한다.", { H: "$H", R: "$R", Hd: 512 }, ["R=4는 Lightning Indexer, R=128은 compressed dense path."], [sources.code, sources.blog], "attention"),
    "q-path": n("q-path", "Q LoRA Path", "attention", "[B,S,D]", "[B,S,H,512]", "wq_a, q_norm, wq_b, q renorm, RoPE.", { q_lora_rank: "$Qr", rope_dim: 64 }, ["q 마지막 64 dims에 RoPE."], common.src, "attention"),
    "kv-path": n("kv-path", "Shared KV Path", "attention", "[B,S,D]", "[B,S,512]", "wkv, kv_norm, RoPE, non-RoPE FP8 simulation.", { kv_heads: 1, nope_dim: 448, rope_dim: 64 }, ["KV는 shared [B,S,512]."], common.src, "attention"),
    "kv-cache": n("kv-cache", "KV Cache", "cache", "[B,S,512]", "[B,128+T/R,512]", "Window cache와 compressed cache를 하나의 buffer로 관리한다.", { window: 128, compressed: "max_seq_len/R" }, ["prefill과 decode write path가 다르다."], [sources.code], "compression"),
    compressor: n("compressor", "KV Compressor", "cache", "[B,S,D]", "[B,floor(S/R),512]", "wkv/wgate/ape/tail state로 compressed KV를 만든다.", { R: "$R", ape: "[R,Coff*512]" }, ["R=4는 overlap, R=128은 non-overlap."], [sources.code], "compression"),
    indexer: n("indexer", "Lightning Indexer", "attention", "x [B,S,D], qr [B,S,Qr]", "[B,S,topK]", "R=4 layer에서 compressed blocks top-k를 선택한다.", { index_heads: 64, index_dim: 128, topK: "$indexTopK" }, ["Hadamard rotation, FP4 quant, weighted ReLU score."], [sources.code], "indexer", { ratio: 4 }),
    "sparse-attn": n("sparse-attn", "Sparse Attention", "attention", "q [B,S,H,512], kv_selected [B,S,N,512]", "heads [B,S,H,512]", "window ids와 compressed ids를 합쳐 attention을 계산한다.", { attn_sink: "[H]" }, ["N은 선택된 KV entry 수이며 mode에 따라 <=128, 128+topK, 128+T/R로 달라진다."], [sources.code]),
    "o-proj": n("o-proj", "Grouped O Projection", "attention", "[B,S,H,512]", "[B,S,D]", "wo_a group low-rank projection 후 wo_b로 D차원 복원.", { groups: "$G", o_lora_rank: "$Or" }, ["각 group은 8 heads * 512 dims."], common.src),
    "mhc-ffn": n("mhc-ffn", "mHC Pre/Post: FFN", "hc", "[B,S,4,D]", "[B,S,4,D]", "MoE FFN 앞뒤의 mHC controller/data path.", { mix_hc: 24 }, ["Attention mHC와 별도 파라미터 세트."], [sources.code, sources.kernel], "mhc"),
    moe: n("moe", "MoE Router + Experts", "routing", "[B,S,D]", "[B,S,D]", "Gate가 top-6 experts를 고르고 routed/shared experts를 결합한다.", { E: "$E", K: 6, I: "$I" }, ["First 3 layers는 hash routing."], common.src, "moe"),
    gate: n("gate", "Router Gate", "routing", "tokens [B*S,D]", "expert_ids [B*S,6], weights [B*S,6]", "sqrtsoftplus score, hash/top-k selection, weight normalize.", { route_scale: "$routeScale", scoring: "sqrtsoftplus" }, ["bias는 selection score에만 적용."], [sources.code], "moe"),
    "routed-experts": n("routed-experts", "Routed Experts", "expert", "[N_e,D]", "[N_e,D]", "선택된 expert별 FP4 SwiGLU FFN.", { w1: "D->I", w3: "D->I", w2: "I->D" }, ["silu(w1(x)) * w3(x) 후 w2."], [sources.code, sources.card], "moe"),
    "shared-expert": n("shared-expert", "Shared Expert", "expert", "[B*S,D]", "[B*S,D]", "모든 token에 항상 더해지는 shared SwiGLU expert.", { shared: 1, I: "$I" }, ["routed expert output에 더해진다."], [sources.code], "moe"),
    "hc-post-moe": n("hc-post-moe", "MoE HC Writeback", "hc", "mixed_lanes [B,S,4,D], moe_inject [B,S,4,D]", "next_lanes [B,S,4,D]", "MoE residual lane mixing 결과와 MoE output injection을 합쳐 다음 block residual lanes를 만든다.", { output: "[B,S,4,D]" }, ["다음 block input이 된다."], [sources.code]),
    head: n("head", "HC Head + LM Head", "output", "[B,S,4,D]", "[B,V]", "hc_head collapse 후 last token vocab projection.", { V: 129280, D: "$D" }, ["공식 path는 x[:, -1]만 logits 계산."], common.src, "output"),
    mtp: n("mtp", "MTP Block", "output", "hidden [B,S,4,D], ids [B,S]", "mtp_logits [B,V]", "추가 next-token prediction block.", { num_nextn: 1, R: 0 }, ["embedding path와 hidden path를 결합."], common.src, "output"),
    logits: n("logits", "Logits", "output", "[B,D]", "[B,129280]", "마지막 token vocabulary scores.", { vocab: 129280 }, ["sampling은 이 그래프 범위 밖."], [sources.card, sources.code]),

    "hc-flatten": n("hc-flatten", "Flatten HC Lanes", "hc", "[B,S,4,D]", "[B,S,4D]", "controller path용으로 lane 축과 hidden 축을 flatten.", { hc_dim: "4D" }, [], [sources.code]),
    "hc-controller": n("hc-controller", "Controller Linear", "hc", "[B,S,4D]", "[B,S,24]", "hc_fn linear로 mHC controller logits를 생성한다.", { weight: "[24,4D]" }, ["rsqrt normalization factor를 곱한다."], [sources.code]),
    "hc-split": n("hc-split", "Split Controller Mixes", "hc", "mixes [B,S,24]", "pre_logits [B,S,4], post_logits [B,S,4], comb_logits [B,S,4,4]", "mixes의 24 channel을 pre 4, post 4, comb 16으로 나눈다.", { hc: 4, mix_hc: 24 }, ["공식 kernel index: 0:4 pre, 4:8 post, 8:24 comb."], [sources.code, sources.kernel]),
    "hc-pre-sigmoid": n("hc-pre-sigmoid", "Pre: sigmoid + eps", "hc", "pre_logits [B,S,4]", "pre [B,S,4]", "read weight를 scaled sigmoid와 eps로 만든다.", { scale: "hc_scale[0]", base: "hc_base[0:4]", eps: "1e-6" }, ["pre는 hc_read에서 lane weighted sum에 쓰인다."], [sources.kernel]),
    "hc-post-sigmoid": n("hc-post-sigmoid", "Post: 2 * sigmoid", "hc", "post_logits [B,S,4]", "post [B,S,4]", "sublayer output을 4개 lane에 주입할 post weight를 만든다.", { scale: "hc_scale[1]", base: "hc_base[4:8]" }, ["post에는 pre와 달리 +eps가 없고 2배 sigmoid를 쓴다."], [sources.kernel]),
    "hc-comb-softmax": n("hc-comb-softmax", "Comb Row Softmax", "hc", "comb_logits [B,S,4,4]", "comb_row [B,S,4,4]", "comb logits를 4x4 matrix로 보고 row softmax + eps를 적용한다.", { scale: "hc_scale[2]", base: "hc_base[8:24]" }, ["Sinkhorn 반복 전 초기 row-normalized matrix."], [sources.kernel]),
    "hc-comb-sinkhorn": n("hc-comb-sinkhorn", "Comb Sinkhorn Normalize", "hc", "comb_row [B,S,4,4]", "comb [B,S,4,4]", "row/column normalization을 반복해 comb를 doubly stochastic에 가깝게 만든다.", { iters: 20, eps: "1e-6" }, ["residual lane mixing에 들어가는 최종 4x4 matrix."], [sources.kernel]),
    "hc-sinkhorn": n("hc-sinkhorn", "Split + Sinkhorn", "hc", "mixes [B,S,24]", "pre [B,S,4], post [B,S,4], comb [B,S,4,4]", "TileLang kernel이 pre/post/comb를 나누고 comb를 Sinkhorn normalize.", { iters: 20, eps: "1e-6" }, ["comb는 [B,S,4,4]."], [sources.kernel]),
    "hc-read": n("hc-read", "Read Data Path", "hc", "pre [B,S,4], X [B,S,4,D]", "[B,S,D]", "pre 가중합으로 sublayer input을 만든다.", {}, ["sum(pre * X) over lane axis."], [sources.code]),
    "attn-residual-mix": n("attn-residual-mix", "Attention Residual Lane Mixing", "hc", "comb [B,S,4,4], residual [B,S,4,D]", "[B,S,4,D]", "comb matrix가 기존 4개 residual lane을 token별로 서로 섞는다.", { comb: "[B,S,4,4]", lanes: 4 }, ["이 노드가 attention writeback의 핵심 residual lane mixing이다."], [sources.code, sources.kernel]),
    "attn-post-inject": n("attn-post-inject", "Attention Output Injection", "hc", "post [B,S,4], y [B,S,D]", "[B,S,4,D]", "attention output을 post weights로 4개 lane에 주입한다.", { post: "[B,S,4]" }, ["residual lane mixing과 별도 항으로 더해진다."], [sources.code]),
    "hc-write": n("hc-write", "Attention HC Writeback", "hc", "mixed_lanes [B,S,4,D], attn_inject [B,S,4,D]", "next_lanes [B,S,4,D]", "comb * residual과 post * attention output을 합쳐 다음 residual lanes를 만든다.", {}, ["writeback = residual lane mixing + sublayer output injection."], [sources.code]),
    "ffn-hc-flatten": n("ffn-hc-flatten", "FFN Flatten HC Lanes", "hc", "[B,S,4,D]", "[B,S,4D]", "MoE/FFN용 mHC controller 입력을 만들기 위해 residual lanes를 flatten한다.", { hc_dim: "4D" }, ["attention mHC와 별도의 controller path."], [sources.code]),
    "ffn-hc-controller": n("ffn-hc-controller", "FFN Controller Linear", "hc", "[B,S,4D]", "[B,S,24]", "MoE/FFN 앞뒤에서 쓸 pre/post/comb logits를 생성한다.", { weight: "[24,4D]", mix_hc: 24 }, ["attention mHC parameter set과 분리된다."], [sources.code]),
    "ffn-hc-split": n("ffn-hc-split", "FFN Split Controller Mixes", "hc", "mixes [B,S,24]", "pre_logits [B,S,4], post_logits [B,S,4], comb_logits [B,S,4,4]", "MoE-side mixes를 pre/post/comb logits로 나눈다.", { hc: 4, mix_hc: 24 }, ["attention-side와 같은 kernel 구조지만 별도 parameter set을 쓴다."], [sources.code, sources.kernel]),
    "ffn-hc-pre-sigmoid": n("ffn-hc-pre-sigmoid", "FFN Pre: sigmoid + eps", "hc", "pre_logits [B,S,4]", "pre [B,S,4]", "MoE read weight를 scaled sigmoid와 eps로 만든다.", { scale: "hc_scale[0]", eps: "1e-6" }, [], [sources.kernel]),
    "ffn-hc-post-sigmoid": n("ffn-hc-post-sigmoid", "FFN Post: 2 * sigmoid", "hc", "post_logits [B,S,4]", "post [B,S,4]", "MoE output injection weight를 만든다.", { scale: "hc_scale[1]" }, [], [sources.kernel]),
    "ffn-hc-comb-softmax": n("ffn-hc-comb-softmax", "FFN Comb Row Softmax", "hc", "comb_logits [B,S,4,4]", "comb_row [B,S,4,4]", "MoE-side comb logits에 row softmax + eps를 적용한다.", { scale: "hc_scale[2]" }, [], [sources.kernel]),
    "ffn-hc-comb-sinkhorn": n("ffn-hc-comb-sinkhorn", "FFN Comb Sinkhorn Normalize", "hc", "comb_row [B,S,4,4]", "comb [B,S,4,4]", "MoE-side residual lane mixing matrix를 Sinkhorn normalize한다.", { iters: 20, eps: "1e-6" }, [], [sources.kernel]),
    "ffn-hc-sinkhorn": n("ffn-hc-sinkhorn", "FFN Split + Sinkhorn", "hc", "mixes [B,S,24]", "pre [B,S,4], post [B,S,4], comb [B,S,4,4]", "FFN용 pre/post/comb를 나누고 comb를 Sinkhorn normalize한다.", { comb: "[B,S,4,4]" }, ["여기서 MoE writeback residual lane mixing weights가 나온다."], [sources.kernel]),
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
    "kv-slice": n("kv-slice", "KV Slice Split", "attention", "kv_norm [B,S,512]", "kv_nope [B,S,448], kv_rope [B,S,64]", "shared KV를 content slice와 RoPE slice로 나눈다.", { nope: 448, rope: 64 }, ["K score에는 RoPE slice가 필요하고 V sum에는 position phase를 제거한 shared value가 필요하다."], [sources.code]),
    "window-topk": n("window-topk", "Window TopK IDs", "cache", "start_pos [scalar], query_len [S]", "window_ids [B,S,W<=128]", "최근 128 token window indices.", { window: 128 }, [], [sources.code]),
    "swa-prefill-write": n("swa-prefill-write", "SWA Prefill Write", "cache", "kv [B,S,512]", "window_cache [B,min(S,128),512]", "prefill chunk에서 최근 128개 uncompressed KV를 window cache에 채운다.", { window: 128 }, ["긴 prompt에서는 window 밖 token은 compressed path로만 남는다."], [sources.code]),
    "swa-decode-write": n("swa-decode-write", "SWA Decode Ring Write", "cache", "kv_t [B,1,512], start_pos [scalar]", "window_cache[:, start_pos % 128] [B,512]", "decode에서는 최근 KV를 128-slot ring buffer에 rolling write한다.", { slot: "start_pos % 128" }, ["runtime cache manager가 오래된 uncompressed entry를 덮어쓴다."], [sources.code]),
    "cache-layout": n("cache-layout", "Logical Cache Layout", "cache", "swa_cache [B,128,512], c_cache [B,T/R,512]", "kv_cache [B,128+T/R,512]", "SWA prefix와 compressed suffix를 attention kernel이 읽는 하나의 논리 cache로 붙인다.", { prefix: 128, suffix: "T/R" }, [], [sources.code]),
    "attn-selected": n("attn-selected", "Selected KV IDs", "attention", "window_ids [B,S,W], compressed_ids [B,S,C]", "selected_ids [B,S,W+C]", "sparse attention이 읽을 KV positions.", {}, [], [sources.code]),

    "comp-wkv": n("comp-wkv", "Compressor wkv", "cache", "[B,S,D]", "[B,S,Coff*512]", "Compression candidate KV projection.", { Coff: "1 or 2" }, [], [sources.code]),
    "comp-wgate": n("comp-wgate", "Compressor wgate", "cache", "[B,S,D]", "[B,S,Coff*512]", "Softmax pooling score projection.", { ape: "[R,Coff*512]" }, [], [sources.code]),
    "comp-ape": n("comp-ape", "Compressor APE Add", "cache", "score_proj [B,S,Coff*512], ape [R,Coff*512]", "score_with_ape [B,S,Coff*512]", "pooling score에 compressor-local absolute position embedding을 더한다.", { ape: "[R,Coff*512]" }, ["block 내부 상대 위치를 softmax gate에 알려주는 path다."], [sources.code]),
    "comp-cutoff": n("comp-cutoff", "Cutoff / Remainder Split", "cache", "proj [B,S,Coff*512] + tail_state", "full_blocks [B,N_full,R,Coff*512], remainder [B,T_tail,Coff*512]", "현재 chunk와 이전 tail을 합쳐 완성 block과 남은 remainder를 나눈다.", { T_tail: "< R" }, ["decode에서는 대부분 remainder가 누적되다가 block boundary에서만 compressed entry가 생긴다."], [sources.code]),
    "tail-append": n("tail-append", "Tail Append + Trim", "cache", "remainder [B,T_tail,Coff*512]", "tail_state' [B,<R,Coff*512]", "다음 decode step에서 이어 쓸 미완성 compressor state를 갱신한다.", { persistent: "per request" }, [], [sources.code]),
    "tail-state": n("tail-state", "Compressed Tail State", "cache", "kv_tail [B,T_tail,Coff,512], score_tail [B,T_tail,Coff,512]", "kv_state [B,Coff*R,Coff*512], score_state [B,Coff*R,Coff*512]", "아직 R개가 안 찬 tail tokens를 buffer에 보관한다.", { T_tail: "< R", kv_state: "[B,Coff*R,Coff*512]", score_state: "[B,Coff*R,Coff*512]" }, [], [sources.code]),
    "comp-block-view": n("comp-block-view", "Block View", "cache", "full_blocks [B,N_full,R,Coff*512]", "kv_block [B,N_full,span,512], gate_block [B,N_full,span,512]", "projection channel을 pooling이 볼 block/token/value 축으로 재배치한다.", { span: "R or 2R" }, ["CSA는 overlap 후 span=8, HCA는 span=128로 해석한다."], [sources.code]),
    "overlap-transform": n("overlap-transform", "Overlap Transform", "cache", "[B,blocks,R,2*512]", "[B,blocks,2R,512]", "R=4에서 이전 chunk와 현재 chunk를 겹쳐 pooling.", { active: "R=4 only" }, [], [sources.code]),
    "gated-pool": n("gated-pool", "Softmax-Gated Pool", "cache", "kv_block [B,blocks,R,512], gate_block [B,blocks,R,512]", "compressed_kv [B,blocks,512]", "R tokens를 softmax(score)로 가중합.", {}, [], [sources.code]),
    "comp-anchor": n("comp-anchor", "Anchor Positions", "cache", "block_ids [B,N_full]", "anchor_ids [B,N_full]", "compressed entry에 적용할 RoPE anchor position을 만든다.", { c4a: "0,4,8,...", c128a: "0,128,256,..." }, ["block 내부 token마다 position을 따로 주지 않고 대표 anchor를 쓴다."], [sources.code]),
    "comp-norm-rope": n("comp-norm-rope", "Norm + Compressed RoPE", "cache", "[B,blocks,512]", "[B,blocks,512]", "compressed KV norm 후 compressed position RoPE.", { theta: 160000 }, [], [sources.code]),
    "comp-cache-slot": n("comp-cache-slot", "Compressed Slot Map", "cache", "anchor_ids [B,N], R [scalar]", "cache_slots [B,N]", "compressed block id를 SWA prefix 뒤 cache slot으로 매핑한다.", { slot: "128 + block_id" }, [], [sources.code]),
    "comp-cache-write": n("comp-cache-write", "Compressed Cache Write", "cache", "compressed_kv [B,blocks,512]", "kv_cache_compressed [B,T/R,512]", "window 영역 뒤 compressed cache에 저장.", {}, [], [sources.code]),

    "idx-q": n("idx-q", "Indexer Q", "attention", "qr [B,S,Qr]", "[B,S,64,128]", "indexer wq_b projection.", { heads: 64, dim: 128 }, [], [sources.code]),
    "idx-rope": n("idx-rope", "Indexer RoPE", "attention", "idx_q [B,S,64,128]", "idx_q_rope [B,S,64,128]", "Lightning index query에 position phase를 넣는다.", { dim: 128 }, [], [sources.code]),
    "idx-hadamard": n("idx-hadamard", "Hadamard Rotate", "attention", "idx_q_rope [B,S,64,128]", "idx_q_rot [B,S,64,128]", "retrieval score용 query를 cheap orthogonal rotation으로 섞는다.", {}, [], [sources.kernel]),
    "idx-fp4": n("idx-fp4", "FP4 Activation Quant", "attention", "idx_q_rot [B,S,64,128]", "idx_q_fp4 [B,S,64,128]", "indexer score path의 activation을 FP4 형태로 양자화한다.", {}, [], [sources.kernel]),
    "idx-rotate": n("idx-rotate", "RoPE + Hadamard + FP4", "attention", "[B,S,64,128]", "[B,S,64,128]", "index query rotation and FP4 activation quant.", {}, [], [sources.code, sources.kernel]),
    "idx-cache-compress": n("idx-cache-compress", "Index Cache Compress", "cache", "x [B,S,D]", "idx_entries [B,T/4,128]", "main KV와 별도의 128-dim compressor로 indexer cache entry를 만든다.", { R: 4, dim: 128 }, [], [sources.code]),
    "idx-cache-write": n("idx-cache-write", "Index Cache Write", "cache", "idx_entries [B,T/4,128]", "idx_cache [B,T/4,128]", "Lightning score가 조회할 compressed index cache에 기록한다.", { dim: 128 }, [], [sources.code]),
    "idx-cache": n("idx-cache", "Index KV Cache", "cache", "x [B,S,D]", "[B,T/4,128]", "indexer 전용 compressor cache.", { R: 4, dim: 128 }, [], [sources.code]),
    "idx-einsum": n("idx-einsum", "Lightning Scores", "attention", "idx_q [B,S,64,128], idx_cache [B,T/4,128]", "scores [B,S,64,T/4]", "ReLU dot product scores.", {}, [], [sources.code]),
    "idx-weight": n("idx-weight", "weights_proj + Head Sum", "attention", "scores [B,S,64,T/4], weights [B,S,64]", "block_scores [B,S,T/4]", "head별 score를 weighted sum.", {}, [], [sources.code]),
    "idx-mask": n("idx-mask", "Compressed Causal Mask", "attention", "block_scores [B,S,T/4], query_pos [S]", "masked_scores [B,S,T/4]", "미래 compressed block을 topK 후보에서 제외한다.", {}, [], [sources.code]),
    "idx-topk": n("idx-topk", "Compressed TopK", "attention", "masked_scores [B,S,T/4]", "block_ids [B,S,topK]", "causal mask 후 top-k compressed block ids.", { topK: "$indexTopK" }, [], [sources.code]),
    "idx-offset": n("idx-offset", "Cache Offset Map", "attention", "block_ids [B,S,topK]", "compressed_ids [B,S,topK]", "compressed block id를 128-slot SWA prefix 뒤의 cache id로 바꾼다.", { offset: 128 }, [], [sources.code]),

    "gate-score": n("gate-score", "Gate Scores", "routing", "[B*S,D]", "[B*S,E]", "linear + sqrtsoftplus expert scores.", { E: "$E" }, [], [sources.code]),
    "hash-route": n("hash-route", "Hash Route", "routing", "input_ids_flat [B*S]", "expert_ids [B*S,6]", "first 3 layers use tid2eid lookup.", { layers: 3 }, [], [sources.code]),
    "route-bias": n("route-bias", "Selection Bias Add", "routing", "scores [B*S,E], bias [E]", "selection_scores [B*S,E]", "topK 선택용 score에만 route bias를 더한다.", {}, ["weight 계산에는 bias 없는 original score를 다시 gather한다."], [sources.code]),
    "topk-route": n("topk-route", "TopK Route", "routing", "selection_scores [B*S,E]", "expert_ids [B*S,6]", "later layers choose top-6 experts.", {}, [], [sources.code]),
    "route-score-gather": n("route-score-gather", "Original Score Gather", "routing", "scores [B*S,E], expert_ids [B*S,6]", "selected_scores [B*S,6]", "선택된 expert id에 대해 bias 없는 original score를 모은다.", {}, [], [sources.code]),
    "route-weights": n("route-weights", "Normalize Weights", "routing", "selected_scores [B*S,6]", "route_weights [B*S,6]", "gather original scores, normalize, apply route scale.", { scale: "$routeScale" }, [], [sources.code]),
    "expert-counts": n("expert-counts", "Expert Counts", "routing", "expert_ids [B*S,6]", "counts [E]", "각 expert가 처리할 token row 수를 센다.", { E: "$E" }, ["dispatch kernel sizing과 load 관찰에 필요한 runtime metadata다."], [sources.code]),
    "expert-dispatch": n("expert-dispatch", "Expert Dispatch", "expert", "tokens [B*S,D], expert_ids [B*S,6], weights [B*S,6]", "expert_batches [N_e,D]", "torch.where(indices == expert_id)로 token dispatch.", {}, [], [sources.code]),
    "expert-w1w3": n("expert-w1w3", "w1 / w3", "expert", "[N_e,D]", "gate/up [N_e,I]", "SwiGLU의 gate와 up projection.", { I: "$I" }, [], [sources.code]),
    swiglu: n("swiglu", "SwiGLU + Clamp", "expert", "gate [N_e,I], up [N_e,I]", "activation [N_e,I]", "clamp 후 silu(gate) * up.", { limit: 10.0 }, [], [sources.code]),
    "expert-w2": n("expert-w2", "w2 Down Projection", "expert", "[N_e,I]", "[N_e,D]", "expert output projection.", {}, [], [sources.code]),
    "routed-accum": n("routed-accum", "Weighted Routed Accum", "expert", "expert_y [N_e,D], route_weights [B*S,6]", "routed_y [B*S,D]", "expert output을 원 token row 위치로 scatter-add하고 routing weight를 곱한다.", {}, [], [sources.code]),
    "shared-w1w3": n("shared-w1w3", "Shared w1 / w3", "expert", "tokens [B*S,D]", "shared_gate/up [B*S,I]", "always-on shared expert의 gate/up projection.", { I: "$I" }, [], [sources.code]),
    "shared-swiglu": n("shared-swiglu", "Shared SwiGLU", "expert", "shared_gate/up [B*S,I]", "shared_act [B*S,I]", "shared expert 내부의 clamp + SiLU gate.", { limit: 10.0 }, [], [sources.code]),
    "shared-w2": n("shared-w2", "Shared w2", "expert", "shared_act [B*S,I]", "shared_y [B*S,D]", "shared expert output projection.", {}, [], [sources.code]),
    "expert-combine": n("expert-combine", "Routed + Shared Combine", "expert", "routed_y [B*S,D], shared_y [B*S,D]", "moe_y [B*S,D]", "routed outputs accumulate, shared expert added.", {}, [], [sources.code]),
    "moe-allreduce": n("moe-allreduce", "MoE TP All-Reduce", "expert", "moe_y_shard [B*S,D]", "moe_y [B*S,D]", "tensor/expert parallel shard의 MoE output을 합친다.", {}, ["단일 GPU 개념 그래프에서는 identity처럼 보이지만 distributed inference에서는 명시적 동기화 지점이다."], [sources.code]),

    "attn-gather": n("attn-gather", "Gather Selected KV", "attention", "cache [B,128+T/R,512], selected_ids [B,S,N]", "kv_selected [B,S,N,512]", "선택된 SWA/compressed KV entry만 attention kernel 입력으로 gather한다.", {}, [], [sources.code]),
    "attn-score": n("attn-score", "QK Score", "attention", "q [B,S,H,512], k_selected [B,S,N,512]", "scores [B,S,H,N]", "query와 selected key의 scaled dot product를 계산한다.", { scale: "1/sqrt(512)" }, [], [sources.code]),
    "attn-mask-sink": n("attn-mask-sink", "Mask + Attention Sink", "attention", "scores [B,S,H,N]", "biased_scores [B,S,H,N]", "causal/window mask와 head별 attention sink bias를 더한다.", { attn_sink: "[H]" }, [], [sources.code]),
    "attn-softmax": n("attn-softmax", "Online Softmax", "attention", "biased_scores [B,S,H,N]", "prob [B,S,H,N]", "선택 KV set 위에서 attention probability를 정규화한다.", {}, [], [sources.code]),
    "attn-value-sum": n("attn-value-sum", "Value Weighted Sum", "attention", "prob [B,S,H,N], v_selected [B,S,N,512]", "heads [B,S,H,512]", "shared KV value를 attention probability로 가중합한다.", {}, [], [sources.code]),
    "attn-inv-rope": n("attn-inv-rope", "Inverse RoPE Value Fix", "attention", "heads [B,S,H,512]", "heads_value [B,S,H,512]", "KV sharing에서 value 역할에는 position phase가 남지 않도록 보정한다.", { rope_dim: 64 }, ["K score에는 RoPE가 필요하지만 V sum에서는 위치 phase를 제거한다."], [sources.code]),
    "o-woa": n("o-woa", "wo_a Group Projection", "attention", "heads_value [B,S,H,512]", "o_latent [B,S,G,Or]", "head group별 low-rank output latent를 만든다.", { groups: "$G", Or: "$Or" }, [], [sources.code]),
    "o-wob": n("o-wob", "wo_b Output Projection", "attention", "o_latent [B,S,G,Or]", "attn_y [B,S,D]", "group latent를 hidden size D로 복원한다.", { D: "$D" }, [], [sources.code]),

    "hc-head-collapse": n("hc-head-collapse", "HC Head Collapse", "output", "final_lanes [B,S,4,D]", "hidden [B,S,D]", "최종 4-lane residual state를 단일 hidden stream으로 접는다.", { lanes: 4 }, [], [sources.code]),
    "final-rmsnorm": n("final-rmsnorm", "Final RMSNorm", "output", "hidden [B,S,D]", "hidden_norm [B,S,D]", "LM head 전에 최종 hidden scale을 맞춘다.", { eps: "1e-6" }, [], [sources.code]),
    "last-token": n("last-token", "Last Token Slice", "output", "hidden_norm [B,S,D]", "hidden_last [B,D]", "decode logits는 마지막 token hidden만 vocab projection한다.", {}, ["모든 layer나 모든 token에서 LM head를 매번 계산하는 구조가 아니다."], [sources.code]),
    "lm-project": n("lm-project", "Vocab Projection", "output", "hidden_last [B,D]", "logits [B,129280]", "최종 hidden을 vocabulary shard/output head로 투영한다.", { vocab: 129280 }, [], common.src),
    "mtp-embed": n("mtp-embed", "MTP Token Embedding", "output", "ids [B,S]", "mtp_embed [B,S,D]", "MTP branch가 next-token 보조 학습/추론에 쓸 token embedding path.", {}, [], [sources.code]),
    "mtp-hidden-proj": n("mtp-hidden-proj", "MTP Hidden Projection", "output", "final_lanes [B,S,4,D]", "mtp_hidden [B,S,D]", "최종 hidden/lane state를 MTP block 입력 공간으로 투영한다.", {}, [], [sources.code]),
    "mtp-combine": n("mtp-combine", "MTP Combine", "output", "mtp_embed [B,S,D], mtp_hidden [B,S,D]", "mtp_x [B,S,D]", "embedding path와 hidden projection path를 결합한다.", {}, [], [sources.code]),
    "mtp-block": n("mtp-block", "MTP Decoder Block", "output", "mtp_x [B,S,D]", "mtp_y [B,S,D]", "보조 next-token prediction용 block을 통과한다. attention mode는 SWA-only로 표시한다.", { R: 0 }, [], [sources.code]),
    "mtp-head": n("mtp-head", "MTP Head", "output", "mtp_y [B,S,D]", "mtp_logits [B,129280]", "MTP branch의 auxiliary vocabulary projection.", { vocab: 129280 }, [], [sources.code]),
  };

  Object.assign(nodes["window-topk"], {
    title: "SWA Window IDs",
    summary: "Sliding-window branch over the most recent 128 tokens; active in every attention mode.",
  });
  Object.assign(nodes["attn-selected"], {
    title: "Attention KV Set",
    input: "window_ids [B,S,W], compressed_ids [B,S,C]",
    output: "selected_ids [B,S,W+C]",
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
    "initial_lanes [B,S,4,D]",
    "layer_state [B,S,4,D]",
    "The expanded graph below is one representative decoder layer selected by the layer-mode control; this is not a second input stream inside every layer.",
    { decoder_layers: "$layers" },
    ["Input and embedding are outside the repeated decoder block."],
    common.src,
  );
  nodes["stack-exit"] = n(
    "stack-exit",
    "Final Stack State",
    "output",
    "layer_state_after_L [B,S,4,D]",
    "final_lanes [B,S,4,D]",
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

  ["comp-wkv", "comp-wgate", "comp-ape", "comp-cutoff", "tail-append", "comp-block-view", "comp-anchor", "comp-cache-slot"].forEach((id) => {
    nodes[id].visibleWhen = { mode: ["csa", "hca"] };
  });
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

  ["idx-q", "idx-rope", "idx-hadamard", "idx-fp4", "idx-rotate", "idx-cache-compress", "idx-cache-write", "idx-cache", "idx-einsum", "idx-weight", "idx-mask", "idx-topk", "idx-offset"].forEach((id) => {
    nodes[id].visibleWhen = { mode: "csa" };
  });
  Object.assign(nodes["idx-q"], { summary: "CSA-only indexer query projection from q_norm." });
  Object.assign(nodes["idx-cache"], { summary: "CSA-only compressed index cache used by the Lightning indexer." });
  Object.assign(nodes["idx-topk"], { summary: "Applies causal masking and selects top-k compressed block ids for CSA." });

  Object.entries({
    "input-ids": { why: "토큰 임베딩뿐 아니라 초반 hash routing의 안정적인 lexical prior를 같은 원천에서 공급합니다. input_ids가 tid2eid lookup에 직접 들어가지만, 자주 등장하는 토큰의 expert 쏠림을 어떻게 완화했는지는 공개 코드만으로는 보이지 않습니다.", runtime: "hash routing 레이어는 score top-k로 expert id를 고르지 않고 input_ids.flatten()으로 tid2eid를 조회합니다.", ui: "임베딩 입력과 초반 라우팅 메타데이터라는 두 용도를 같이 보여주는 게 좋습니다.", open: "tid2eid가 학습/배정될 때 자주 등장하는 토큰의 expert 쏠림을 어떻게 막았는지는 공개 inference 코드만으로는 확인되지 않습니다." },
    embedding: { why: "이산 token id를 모든 sublayer가 공유할 연속 표현으로 바꾸는 표준 진입점입니다. vocab-parallel embedding과 all-reduce 흐름을 통해 tensor-parallel 환경에서도 같은 hidden 표현을 만듭니다.", runtime: "tensor parallel에서는 vocab shard 밖 id를 mask하고 partial embedding을 all-reduce합니다.", ui: "decoder stack 안에서 매번 실행되는 노드가 아니라 stack 진입 전 1회 노드로 표시합니다." },
    "hc-expand": { why: "단일 residual stream을 4개 lane으로 복제해 mHC가 layer마다 read/write/mixing을 선택할 수 있는 상태 공간을 만듭니다. mHC는 residual connection을 강화하고 layer 간 signal propagation을 안정화하려는 구조입니다. lane별 역할 분화는 구조상 가능한 해석입니다.", runtime: "공식 forward는 [B,S,D]를 [B,S,hc_mult,D]로 repeat합니다.", ui: "lane 축을 실제 tensor 차원으로 보여주고, 모델 복사본 4개처럼 보이지 않게 합니다." },
    "mhc-attn": { why: "attention sublayer가 residual lane 전체를 직접 받는 대신 read projection으로 필요한 stream을 읽고 write projection으로 다시 lane에 분배하게 합니다. mHC는 residual 전달을 안정화하는 쪽에 초점이 있고, attention 주변에서는 read/write path가 분리됩니다.", runtime: "controller coefficient는 inference에서도 현재 residual lane에서 매번 다시 계산됩니다.", ui: "attention 자체가 아니라 attention 주변의 controller + data path로 표시합니다." },
    attention: { why: "긴 context에서 모든 과거 token을 dense attention으로 보지 않고 SWA, compressed attention, sparse retrieval을 layer별로 조합합니다. 이 조합으로 long-context memory와 compute를 줄입니다.", runtime: "내부 cache path는 layer ratio에 따라 CSA/c4a R=4, HCA/c128a R=128, SWA-only R=0으로 달라집니다.", ui: "선택된 한 layer에서 CSA와 HCA가 동시에 켜진 것처럼 보이지 않게 합니다." },
    "q-path": { why: "query 생성을 low-rank latent 단계와 head expansion 단계로 나눠 큰 H*512 projection을 직접 다루는 부담을 줄입니다. q_lora_rank와 wq_a/wq_b 분리 덕분에 query projection 비용을 낮출 수 있습니다.", ui: "compact scene에서만 쓰고, overview에서는 하위 노드들을 직접 보여주는 편이 좋습니다." },
    "kv-path": { why: "head마다 별도 KV를 저장하지 않고 하나의 512-dim shared KV를 key/value 양쪽에 재사용해 cache memory를 크게 줄입니다. shared KV를 쓰면 cache memory가 줄지만, value로 사용할 때는 inverse RoPE로 position phase를 보정해야 합니다.", runtime: "동일한 shared vector가 logit 계산에서는 key처럼, output 계산에서는 value처럼 쓰입니다.", ui: "KV sharing과 inverse RoPE 이야기가 시작되는 핵심 노드입니다." },
    "kv-cache": { why: "최근 128개 uncompressed SWA entry와 오래된 compressed entry를 같은 attention id 공간에 놓아 kernel을 단순화합니다. 이 배치로 long-context memory를 줄이면서 최근 local detail은 보존합니다.", runtime: "앞 128 slot은 SWA, 그 뒤 suffix는 compressed entry입니다. decode에서는 SWA를 start_pos % 128 위치에 씁니다.", ui: "window prefix + compressed suffix 구조로 그립니다." },
    compressor: { why: "1M context의 오래된 KV를 모두 원본으로 유지하지 않고 compressed entry로 바꿔 long-context memory를 줄입니다.", runtime: "R=4는 c4a overlap, R=128은 c128a non-overlap 동작입니다.", ui: "선택된 layer mode에 따라 c4a/c128a label을 보여줍니다." },
    indexer: { why: "c4a의 T/4 compressed blocks가 여전히 많기 때문에 attention 전에 중요한 compressed block만 고르는 cheap retrieval path를 둡니다. 후보 block을 먼저 좁혀 long-context attention compute를 줄입니다.", runtime: "Pro topK는 1024, Flash topK는 512입니다.", ui: "CSA/c4a 모드에서만 보여줍니다." },
    "sparse-attn": { why: "SWA, CSA, HCA의 후보 선택 차이를 하나의 selected-KV attention kernel로 실행합니다. 같은 kernel 추상화 안에서 long-context compute와 memory를 줄이고 local 정보는 유지합니다.", runtime: "per-head attn_sink와 online-softmax 스타일 kernel 동작을 포함합니다.", ui: "선택 모드별 후보 수를 <=128, topK+128, T/128+128처럼 보여줍니다." },
    "o-proj": { why: "sparse attention head 결과를 residual stream 크기로 되돌리면서 grouped low-rank output projection으로 비용을 낮춥니다. wo_a/wo_b의 group low-rank 구조로 projection 비용을 낮추고, inverse RoPE로 value semantics를 보정합니다.", runtime: "KV가 shared라서 grouped low-rank output projection 전에 inverse RoPE를 적용합니다.", ui: "shared-KV attention 이후 value semantics를 보정하는 지점입니다." },
    "mhc-ffn": { why: "MoE sublayer도 attention과 같은 mHC read/write 구조로 감싸 residual lane 안정성을 유지합니다. FFN 쪽 mHC도 attention 쪽과 같은 residual 안정화 목적을 공유하되 별도 parameter set을 씁니다.", runtime: "attention mHC와 별도의 FFN mHC parameter set을 씁니다." },
    moe: { why: "전체 parameter capacity는 크게 유지하되 token당 활성화 expert 수를 제한해 inference compute를 줄입니다. top-6 routed expert와 shared expert를 결합해 token당 compute는 제한하고 전체 capacity는 키웁니다.", runtime: "공개 구현에서는 모든 decoder block이 MoE FFN을 씁니다.", ui: "shared expert는 항상 active, routed expert는 conditional로 그립니다." },
    gate: { why: "token별로 필요한 expert subset을 선택해 sparse FFN을 구성합니다. 초반에는 token-id 기반 routing으로 싸게 처리하고, 이후에는 hidden representation 기반 top-k routing을 씁니다.", runtime: "초반 hash layer는 token id로 expert id를 고르고, 이후 layer는 score top-k로 고릅니다." },
    "routed-experts": { why: "token별로 일부 FFN expert만 실행해 거대한 expert pool의 capacity를 조건부로 씁니다. token당 compute를 줄이면서 큰 expert pool의 parameter capacity를 유지합니다.", runtime: "Pro는 384 routed expert, token당 6 activated expert를 사용합니다." },
    "shared-expert": { why: "sparse routing과 무관하게 모든 token에 공통 FFN 경로를 제공합니다. routed output에 항상 더해지므로 공통 변환을 흡수하는 역할로 볼 수 있지만, routed expert load를 실제로 얼마나 완화하는지는 공개 코드만으로 단정하기 어렵습니다.", runtime: "모든 token에 대해 계산되고 routed expert 누적값 뒤에 더해집니다.", ui: "routing gate 뒤가 아니라 routed expert와 병렬인 always-on path로 그립니다.", open: "공통 변환을 흡수해 routed expert 부담을 줄일 수 있다는 해석은 가능하지만, 공개 inference 코드만으로 load-balance 메커니즘이라고 단정할 수는 없습니다." },
    "hc-post-moe": { why: "MoE residual mixing과 output injection을 합쳐 다음 decoder block이 받을 4-lane residual state를 만듭니다. writeback 흐름에서 residual state가 sublayer 경계를 안정적으로 넘도록 맞춥니다." },
    head: { why: "mHC lane collapse, final norm, last-token vocab projection을 output-only stage로 묶습니다. decoder stack 반복부와 최종 head를 분리해 보여줍니다.", runtime: "공식 get_logits는 x[:, -1]만 사용하므로 마지막 token만 projection합니다." },
    mtp: { why: "최종 stack state 뒤에서 auxiliary next-token prediction을 수행해 학습/추론 보조 경로를 제공합니다. MTP branch의 정확한 serving 사용 범위는 공개 코드 해석이 더 필요합니다.", runtime: "embedding/head module을 재사용하고 SWA-only attention mode를 가집니다." },
    logits: { why: "최종 token distribution을 만들기 위한 vocabulary score vector를 제공합니다. sampling이나 top-p 같은 decoding policy는 모델 구조 밖 runtime 단계입니다.", ui: "sampling, top-p, tool decoding은 이 architecture graph 밖의 단계입니다." },
    "hc-flatten": { why: "controller path가 4개 lane 전체를 동시에 보고 현재 token의 read/write/mix coefficient를 만들게 합니다. 이 flatten은 coefficient 생성 경로이며, data path를 직접 키우지 않고 lane 선택만 동적으로 바꾸게 해줍니다.", ui: "실제 attention data path가 아니라 coefficient 생성으로 들어가는 control flow로 그립니다." },
    "hc-controller": { why: "token별 residual lane 상태에서 pre, post, comb를 한 번에 예측하는 작은 controller를 둡니다. 24-channel linear 출력에서 attention 계산과 residual mixing 정책을 분리해 안정성과 표현력을 함께 확보합니다.", runtime: "flatten된 lane은 controller linear 출력 전후로 RMS scale 정규화가 들어갑니다.", ui: "pre, post, comb 세 갈래로 split되는 지점으로 보여줍니다." },
    "hc-split": { why: "controller output을 read coefficient, write coefficient, residual lane mixing matrix라는 서로 다른 의미의 텐서로 명확히 분해합니다. 0:4, 4:8, 8:24 index split 덕분에 이후 edge shape이 pre/post/comb로 나뉘고 data dependency가 명확해집니다." },
    "hc-pre-sigmoid": { why: "read weight를 양수 범위로 제한해 lane weighted sum이 무제한 부호/scale로 흔들리지 않게 합니다. scaled sigmoid와 epsilon이 read coefficient를 bounded positive 값으로 잡아 주므로 scale이 덜 흔들립니다." },
    "hc-post-sigmoid": { why: "sublayer output을 4개 lane에 주입할 때 lane별 강도를 bounded coefficient로 제어합니다. 2 * sigmoid 형태로 writeback scale을 제한합니다." },
    "hc-comb-softmax": { why: "4x4 comb raw logits를 먼저 row-wise probability-like matrix로 만들어 Sinkhorn 반복의 출발점을 안정화합니다. row softmax와 epsilon을 먼저 적용해 Sinkhorn 반복이 시작할 matrix를 안정화합니다." },
    "hc-comb-sinkhorn": { why: "comb matrix를 row/column 양쪽에서 정규화해 residual lane transport가 특정 lane으로 붕괴하지 않게 합니다. Sinkhorn 반복으로 comb가 doubly stochastic에 가까워지고, 이 lane transport가 gradient와 signal propagation을 안정화하는 역할을 합니다." },
    "hc-sinkhorn": { why: "controller logit을 실제 data path coefficient로 바꾸면서 comb에는 doubly stochastic 제약을 걸어 residual lane 이동을 안정화합니다. split과 Sinkhorn을 거치면서 read/write/mix 계수가 안정적인 data path coefficient로 바뀝니다.", runtime: "comb는 hc_split_sinkhorn kernel에서 row/column normalization을 반복해 doubly stochastic에 가깝게 만듭니다.", ui: "pre/post는 vector, comb는 4x4 heatmap처럼 보여주면 좋습니다.", open: "mHC constraint가 학습 안정성에 기여하는 정확한 분석은 technical report와 같이 대조해야 합니다." },
    "hc-read": { why: "4-lane residual state를 attention이 처리할 단일 hidden stream으로 읽습니다. pre coefficient로 lane weighted sum을 만들기 때문에 attention compute를 4배로 늘리지 않고 mHC 표현력을 유지합니다.", ui: "controller coefficient가 data path에 처음 적용되는 지점입니다." },
    "attn-residual-mix": { why: "attention output과 별개로 기존 residual lane 자체를 comb matrix로 운반합니다. identity residual보다 lane 간 정보 이동을 허용하므로 안정성과 표현력을 높이는 방향으로 해석할 수 있습니다.", ui: "attention output injection과 분리해서 보여줘야 합니다." },
    "attn-post-inject": { why: "attention output 하나를 post coefficient로 4개 lane에 배분해 sublayer 결과가 어느 lane에 남을지 조절합니다. post write path가 residual update의 scale과 lane placement를 제어합니다.", ui: "하나의 attention stream이 4개 lane으로 fan-out되는 구조입니다." },
    "hc-write": { why: "comb로 운반한 기존 residual과 post로 주입한 attention output을 합쳐 다음 residual lane state를 만듭니다. 이 writeback으로 residual 안정성과 attention update 표현력을 동시에 확보합니다." },
    "ffn-hc-flatten": { why: "MoE controller가 현재 4-lane residual 상태 전체를 보고 read/write/mix coefficient를 만들게 합니다. FFN controller path가 MoE 입력과 residual update를 동적으로 조절합니다." },
    "ffn-hc-controller": { why: "MoE sublayer 전용 pre/post/comb logits를 attention과 독립적으로 생성합니다. attention과 parameter set을 분리해 sublayer별 lane mixing 정책을 따로 둡니다.", ui: "attention controller와 같은 구조지만 독립 parameter라고 표시합니다." },
    "ffn-hc-split": { why: "MoE-side controller output을 read, write, comb 텐서로 분해해 router/expert data path에 각각 다른 coefficient를 공급합니다. MoE path의 dependency가 read/write/comb로 명확히 나뉩니다." },
    "ffn-hc-pre-sigmoid": { why: "MoE가 읽을 hidden stream의 lane weight를 bounded positive coefficient로 만듭니다. bounded positive coefficient가 router와 expert 입력 scale을 안정화합니다." },
    "ffn-hc-post-sigmoid": { why: "MoE output을 4개 residual lane에 주입할 강도를 bounded coefficient로 제어합니다. bounded coefficient로 sparse FFN output scale을 안정화합니다." },
    "ffn-hc-comb-softmax": { why: "MoE residual lane mixing의 raw 4x4 logits를 Sinkhorn 가능한 row-normalized matrix로 초기화합니다. row-normalized 초기값으로 lane mixing의 수치 안정성을 높입니다." },
    "ffn-hc-comb-sinkhorn": { why: "MoE writeback의 comb matrix도 doubly stochastic에 가깝게 만들어 lane 붕괴를 줄입니다. doubly stochastic에 가까운 comb가 residual gradient와 signal을 안정화합니다." },
    "ffn-hc-sinkhorn": { why: "MoE sublayer의 controller output을 data path coefficient로 변환하면서 attention 쪽과 같은 mHC 제약을 적용합니다. MoE에서도 residual lane stability가 유지됩니다.", ui: "attention 쪽과 동일한 4x4 comb matrix 시각화를 재사용합니다." },
    "hc-pre-moe": { why: "router와 experts가 처리할 단일 hidden stream을 4-lane residual에서 읽어냅니다. lane을 먼저 읽어 단일 hidden stream으로 만들기 때문에 MoE compute를 4개 lane마다 반복하지 않습니다." },
    "ffn-residual-mix": { why: "MoE sublayer를 지나도 기존 residual lane 정보를 comb matrix로 계속 운반합니다. sparse expert update와 별개로 residual signal을 안정적으로 보존합니다." },
    "ffn-post-inject": { why: "MoE output을 post coefficient로 lane별 분배해 sparse FFN 결과를 residual lane state에 주입합니다. post coefficient가 expert output scale과 lane placement를 제어합니다." },
    "q-wqa": { why: "hidden state를 먼저 작은 query latent로 투영해 query 계산의 중간 rank를 제한합니다. q_lora_rank로 query 중간 rank를 제한해 projection compute와 parameter 배치를 줄입니다.", runtime: "Pro의 q_lora_rank는 1536입니다.", ui: "LoRA식 low-rank A projection으로 표시합니다." },
    "q-norm": { why: "main Q expansion과 Lightning indexer가 공유하는 low-rank query latent의 scale을 분기 전에 맞춥니다. 두 path가 같은 normalized latent를 쓰므로 attention score와 retrieval score의 scale 민감도를 줄이는 배치로 볼 수 있습니다.", runtime: "main query expansion과 CSA Lightning indexer query path가 둘 다 이 출력을 씁니다.", ui: "fork 지점으로 보여줍니다." },
    "q-wqb": { why: "안정화된 query latent를 실제 attention head별 512-dim query로 확장합니다. wq_b가 low-rank compute 절감 뒤 필요한 head 표현력을 복원합니다.", runtime: "Pro 기준 논리 출력은 H*512 = 128*512 = 65536 channel입니다." },
    "q-reshape": { why: "flat channel을 head 축으로 재해석하고 per-head scale을 다시 맞춰 attention kernel 입력 형태로 만듭니다. reshape 뒤 per-head renorm을 거쳐 head별 score scale을 맞춥니다.", runtime: "코드에서는 reshape 후 per-head RMS re-normalization도 수행합니다.", ui: "학습 파라미터가 있는 layer라기보다 axis split으로 보여줍니다." },
    "q-rope": { why: "query의 일부 차원에만 위치 위상을 넣어 content dimension과 positional dimension을 분리합니다. 마지막 64 RoPE dim만 position phase를 맡아 shared-KV 설계와 맞물립니다.", runtime: "마지막 64 dim만 RoPE가 적용되고 나머지 448 dim은 content dim으로 남습니다.", ui: "각 head를 448 no-RoPE dim + 64 RoPE dim으로 나눠 보여줍니다." },
    "kv-wkv": { why: "hidden stream에서 shared key/value entry 하나를 생성합니다. single KV projection이 per-head KV cache를 피하게 해 memory를 줄입니다.", runtime: "K에는 RoPE가 필요하지만 V에는 절대 위치가 섞이면 이상하므로 뒤에서 inverse RoPE로 보정합니다." },
    "kv-norm": { why: "RoPE, quantization, compression, cache write로 갈라지기 전에 shared KV scale을 맞춥니다. 여러 downstream path가 같은 normalized KV를 쓰기 때문에 RoPE, compression, cache write의 scale 민감도가 줄어듭니다." },
    "kv-rope-quant": { why: "key score에 필요한 RoPE 정보는 보존하면서 non-RoPE content는 저정밀 표현으로 cache 부담을 낮춥니다. BF16 RoPE dim은 score 안정성을 맡고, FP8-sim non-RoPE dim은 memory 부담을 낮춥니다.", runtime: "RoPE dim은 BF16으로 유지하고 non-RoPE dim은 FP8 simulation을 적용합니다.", ui: "448개 quantized content dim과 64개 BF16 RoPE dim으로 나눠 보여줍니다." },
    "kv-slice": { why: "shared KV 안에서 position-aware key slice와 content/value slice를 분리해 K와 V가 서로 다른 의미를 갖게 합니다. 448/64 split으로 RoPE를 필요한 key slice에 집중시킵니다.", runtime: "RoPE는 마지막 64 dim에만 들어가고, 448 dim은 content/value semantics를 유지합니다." },
    "window-topk": { why: "compressed block retrieval과 무관하게 최근 128개 token을 항상 후보로 넣어 local causality와 근접 문맥을 보존합니다. 이 SWA path가 long-context 압축에서 생기는 local detail 손실을 보완합니다.", runtime: "CSA, HCA, SWA-only 모든 모드에서 존재합니다.", ui: "score 기반 top-k가 아니라 sliding-window index set입니다." },
    "swa-prefill-write": { why: "prefill 중에도 최근 local token을 압축하지 않고 보존해 short-range attention 품질을 지킵니다. 128-window cache가 compressed memory의 정보 손실을 local window로 보완합니다.", runtime: "SWA window 밖의 오래된 token은 compressed path가 담당합니다." },
    "swa-decode-write": { why: "decode에서 새 token KV를 고정 크기 128-slot ring buffer에 갱신해 uncompressed local cache memory를 상수로 유지합니다. start_pos modulo write로 memory bounded local attention을 유지합니다.", runtime: "slot은 start_pos % 128이며 오래된 uncompressed KV는 자연스럽게 덮어씁니다." },
    "cache-layout": { why: "SWA prefix와 compressed suffix를 하나의 logical cache로 붙여 selected id gather를 단순하게 만듭니다. 128 offset 구조가 sparse attention kernel의 indexing을 단순화합니다.", runtime: "compressed id는 128 offset 뒤쪽 suffix로 매핑됩니다." },
    "attn-selected": { why: "attention이 실제로 읽을 KV 후보를 mode별로 합성합니다. CSA/HCA/SWA-only 선택 규칙을 한 노드에서 합쳐 layer mode별 compute budget을 명확히 제한합니다.", runtime: "CSA = SWA ids + indexer topK, HCA = SWA ids + 모든 valid c128a ids, SWA-only = SWA ids만 사용합니다.", ui: "index-set union 노드로 보여줍니다." },
    "comp-wkv": { why: "pooling될 token별 KV 후보를 만들어 압축이 raw hidden이 아니라 learned projection 위에서 일어나게 합니다. learned compressor projection을 거쳐 compressed entry 표현력을 보존합니다.", runtime: "c4a overlap에서는 Coff=2, c128a에서는 Coff=1입니다." },
    "comp-wgate": { why: "어떤 token/channel 정보를 compressed entry에 더 반영할지 learned score로 결정합니다. wgate와 softmax pooling으로 단순 평균보다 더 선택적으로 정보를 모읍니다.", runtime: "softmax 전에 learned ape가 score에 더해집니다.", ui: "weighted sum의 weight가 어디서 나오는지 보여주는 노드입니다." },
    "comp-ape": { why: "compressor-local block 내부 위치 정보를 gate score에 더해 pooling이 순서/위치를 완전히 잃지 않게 합니다. APE add가 compressed entry의 block-local 위치 민감도를 보완합니다.", runtime: "attention RoPE와 별개인 compressor-local score path입니다." },
    "comp-cutoff": { why: "현재 chunk와 이전 tail을 합친 뒤 완성된 block만 compressed cache로 보내고 나머지는 tail로 넘깁니다. compressor state 흐름이 chunking과 decode step 크기에 따른 결과 흔들림을 줄입니다.", runtime: "block boundary에 닿지 않으면 compressed cache write가 발생하지 않을 수 있습니다." },
    "tail-append": { why: "미완성 remainder를 다음 forward 호출까지 유지해 compression window가 요청 경계를 넘어서도 이어지게 합니다. stateful compressor 흐름으로 runtime chunking artifact를 줄입니다.", runtime: "decode/prefill chunking에 따라 tail 길이는 0 이상 R 미만입니다." },
    "tail-state": { why: "decode처럼 token이 조금씩 들어오는 상황에서 아직 R개가 차지 않은 projection을 request state로 보관합니다. block boundary에 도달했을 때만 compressed cache write가 발생하므로 streaming decode에서도 compression window가 끊기지 않습니다. tail state는 streaming compression을 위한 request별 runtime state입니다.", runtime: "c4a는 overlap된 8-token 스타일 compressor state, c128a는 128-token state를 유지합니다.", ui: "projection이 아니라 request별 persistent state로 표시합니다." },
    "comp-block-view": { why: "projection 결과를 pooling kernel이 읽기 쉬운 block/token/value 축으로 바꿔 c4a overlap과 c128a block 처리를 같은 추상화에 올립니다. reshape와 overlap 처리가 compressor compute를 block/token/value 축에 맞춥니다.", runtime: "CSA는 overlap 때문에 span이 8처럼 보이고 HCA는 128-token block이 됩니다." },
    "overlap-transform": { why: "c4a에서 stride 4로 cache entry를 만들면서도 pooling span은 8 token을 보게 해 block 경계 정보 손실을 줄입니다. overlap transform은 block boundary 정보 손실을 완화하는 쪽으로 작동합니다.", runtime: "이전 block half와 현재 block half를 gated pooling 전에 재배치하며 boundary는 0 또는 -inf padding으로 처리됩니다.", ui: "native span과 anchor position을 분리해서 보여줍니다." },
    "gated-pool": { why: "여러 native token을 learned softmax weight로 하나의 compressed KV entry에 모읍니다. compressor pooling이 memory를 줄이면서 중요한 token/channel 정보를 더 남깁니다.", ui: "c4a는 8-to-1, c128a는 128-to-1 pooling처럼 보여줍니다." },
    "comp-anchor": { why: "compressed block 전체에 하나의 대표 position을 부여해 RoPE와 causal indexing이 가능한 cache entry로 만듭니다. stride별 anchor id를 쓰기 때문에 block 내부 세부 위치는 일부 손실될 수 있습니다.", runtime: "c4a는 stride 4 anchor, c128a는 stride 128 anchor를 씁니다." },
    "comp-norm-rope": { why: "compressed KV entry도 attention key로 쓰일 수 있게 scale과 position phase를 맞춥니다. norm과 anchor RoPE가 compressed attention score를 안정화합니다.", runtime: "prefill anchor는 R 간격 위치를 쓰고, decode에서는 block 완성 시 start_pos + 1 - R을 anchor로 씁니다.", ui: "anchor position이라는 용어를 명시적으로 보여줍니다." },
    "comp-cache-slot": { why: "compressed block id를 SWA prefix 뒤 logical cache slot으로 변환해 attention gather가 같은 id 체계를 쓰게 합니다. 128 offset 구조가 runtime indexing을 단순화합니다.", runtime: "SWA prefix 128개 뒤에 compressed suffix가 이어집니다." },
    "comp-cache-write": { why: "완성된 compressed KV를 long-context memory 영역에 기록해 오래된 문맥을 싼 entry로 유지합니다. compressed cache write가 long-context memory를 줄입니다.", runtime: "논리 compressed index는 대략 start_pos // R이며 serving runtime은 이를 page로 다시 매핑할 수 있습니다." },
    "idx-q": { why: "main attention Q를 그대로 쓰지 않고 retrieval 전용 64-head 128-dim query를 만들어 index scoring 비용을 낮춥니다. q_norm latent에서 retrieval query를 파생해 compute를 줄입니다.", runtime: "최종 main Q head가 아니라 q_norm latent에서 파생됩니다." },
    "idx-rope": { why: "retrieval query도 현재 token position을 반영해 compressed block과의 시간적 관련성을 평가하게 합니다. indexer RoPE로 retrieval score가 query position을 반영합니다." },
    "idx-hadamard": { why: "저렴한 orthogonal mixing으로 index query 표현을 섞어 FP4 ranking path의 표현력을 보완합니다. Hadamard rotation은 cheap retrieval representation의 channel mixing을 보완합니다.", runtime: "main attention projection이 아니라 approximate retrieval representation입니다." },
    "idx-fp4": { why: "indexer activation을 FP4로 낮춰 top-k scoring 경로의 memory와 compute를 줄입니다. FP4 activation quantization으로 scoring path의 compute와 memory를 줄입니다.", runtime: "정밀한 attention 값이 아니라 topK 후보 ranking용 표현입니다." },
    "idx-rotate": { why: "RoPE, Hadamard, FP4를 묶어 cheap retrieval representation을 만듭니다. 이 경로는 accurate attention 대신 후보 ranking을 싸게 만드는 데 집중합니다.", runtime: "RoPE, Hadamard rotation, FP4 activation quantization을 적용합니다.", ui: "main attention이 아니라 approximate retrieval scoring으로 표시합니다." },
    "idx-cache-compress": { why: "main 512-dim compressed KV와 별도로 top-k ranking 전용 128-dim cache를 만들어 retrieval 비용을 낮춥니다. 전용 index compressor가 memory와 score compute를 줄입니다.", runtime: "score 계산용 cache라 value sum에는 직접 들어가지 않습니다." },
    "idx-cache-write": { why: "새 compressed block에 대응하는 index entry를 별도 retrieval cache에 기록합니다. index cache write가 다음 token들의 top-k 검색을 가능하게 합니다." },
    "idx-cache": { why: "Lightning Indexer가 main KV cache를 직접 스캔하지 않고 작은 128-dim entry만 보게 합니다. 작은 index cache만 읽어 long-context retrieval memory bandwidth를 줄입니다.", runtime: "head_dim=128, rotate=true인 indexer 전용 compressor를 씁니다." },
    "idx-einsum": { why: "query token과 compressed index cache 사이의 후보 block score를 빠르게 계산합니다. ReLU dot score로 attention 전에 후보 공간을 줄입니다.", runtime: "dot product score는 weighted head sum 전에 ReLU를 통과합니다." },
    "idx-weight": { why: "64개 index head score를 query-dependent weight로 합쳐 block rank score 하나로 만듭니다. weights_proj가 retrieval head 중요도를 token별로 조절합니다.", runtime: "weights_proj(x)가 query-dependent index-head weight를 만듭니다." },
    "idx-mask": { why: "아직 생성되지 않았거나 causal boundary를 넘는 compressed block이 top-k에 들어오지 못하게 합니다. mask가 causal correctness를 지킵니다." },
    "idx-topk": { why: "c4a compressed cache 중 일부만 attention 후보로 남겨 sparse attention compute를 제한합니다. topK로 남길 compressed block 수를 제한해 long-context compute를 줄입니다.", runtime: "Pro topK=1024, Flash topK=512입니다." },
    "idx-offset": { why: "top-k block id를 SWA prefix가 붙은 logical cache id로 변환해 gather가 같은 cache space를 쓰게 합니다. offset mapping으로 SWA와 compressed cache의 indexing을 단순화합니다.", runtime: "compressed entry는 SWA 128-slot 뒤에 있으므로 offset이 필요합니다." },
    "gate-score": { why: "expert affinity를 연속 score로 만들어 top-k selection과 route weight 계산의 공통 기반을 제공합니다. hash layer에서도 original score를 gather해 routing weight 계산을 일관되게 유지합니다.", runtime: "hash layer에서도 route weight를 원래 score에서 gather하기 때문에 score 계산 자체는 남아 있습니다.", ui: "hash layer가 모든 scoring을 생략한다고 오해하지 않게 합니다. 생략되는 것은 score top-k selection입니다." },
    "hash-route": { why: "초반 layer에서 아직 hidden representation이 얕을 때 token id 기반 expert prior를 써 routing 결정을 싸게 만듭니다. tid2eid lookup을 쓰지만, frequent token 쏠림 완화 방식은 공개 문서에 남아 있지 않습니다.", runtime: "tid2eid[input_ids]가 token당 6개 expert id를 반환합니다.", ui: "lexical/token-prior routing으로 보여줍니다.", open: "자주 등장하는 token의 expert 쏠림을 어떻게 완화했는지는 checkpoint나 report 분석이 필요합니다." },
    "route-bias": { why: "expert 선택 확률만 조정하고 실제 mixture weight는 원래 affinity score에서 계산해 output magnitude 왜곡을 줄입니다. bias를 top-k selection에만 써서 output weight 왜곡을 줄이는 구조로 해석할 수 있습니다.", runtime: "bias는 topK에만 들어가고 normalize weight는 original score gather로 계산합니다." },
    "topk-route": { why: "후반 layer에서 token representation 기반으로 가장 관련 있는 6개 expert만 활성화합니다. top-k routing이 MoE compute를 줄이면서 conditional capacity를 제공합니다.", runtime: "bias는 selection에만 영향을 주고, route weight는 bias 없는 original score에서 gather합니다." },
    "route-score-gather": { why: "선택된 expert id에 대해 bias 없는 original score를 다시 가져와 mixture weight를 만들게 합니다. original score gather로 selection correction과 output weighting을 분리합니다.", runtime: "초반 hash route에서도 score 계산이 필요한 이유입니다." },
    "route-weights": { why: "선택 expert들의 기여도를 normalize하고 route_scale로 전체 MoE output scale을 맞춥니다. route_scale과 normalization이 routing stability와 activation scale을 제어합니다.", runtime: "sqrtsoftplus score를 선택 expert들 사이에서 normalize하고 route_scale을 곱합니다." },
    "expert-counts": { why: "expert별 token 수를 알아 dispatch/scatter 및 병렬 실행 크기를 정하는 runtime metadata를 만듭니다. expert counts가 routing 실행을 관리하고 load를 관찰할 수 있게 합니다.", runtime: "load imbalance를 관찰할 수 있는 지점이기도 합니다." },
    "expert-dispatch": { why: "선택된 expert별로 token rows를 모아 해당 expert weight로만 처리하게 합니다. expert별 token 묶음으로 sparse expert compute를 실제 batch 연산으로 실행할 수 있습니다.", runtime: "코드는 torch.where(indices == expert_id)를 쓰고 parallel 환경에서는 routed output을 all-reduce합니다." },
    "expert-w1w3": { why: "SwiGLU에 필요한 gate projection과 up projection을 expert별로 계산합니다. gate/up projection이 FFN 표현력을 만듭니다.", runtime: "Pro에서는 expert weight가 FP4일 수 있습니다." },
    swiglu: { why: "gate와 up activation의 곱으로 expert FFN의 비선형 선택성을 높입니다. SiLU gate와 clamp가 비선형성을 주고 activation 폭을 제한합니다.", runtime: "Pro는 swiglu_limit=10.0으로 gate/up을 clamp한 뒤 silu(gate) * up을 계산합니다." },
    "expert-w2": { why: "expert intermediate activation을 residual hidden dimension으로 되돌려 routed accumulation에 합칠 수 있게 합니다. down projection이 expert output shape을 hidden dimension으로 복원합니다." },
    "routed-accum": { why: "expert별로 흩어진 결과를 원 token 위치로 되돌리고 route weight를 곱해 하나의 routed output으로 만듭니다. dispatch/scatter 흐름이 sparse compute 결과를 dense token stream으로 복원합니다." },
    "shared-w1w3": { why: "routing과 무관하게 모든 token이 거치는 shared expert의 gate/up projection을 만듭니다. always-on shared expert가 공통 FFN 변환을 제공합니다.", runtime: "routing 결과와 무관하게 모든 token에 대해 실행됩니다." },
    "shared-swiglu": { why: "shared expert 내부에서도 routed expert와 같은 SwiGLU 비선형성을 유지합니다. shared path도 routed expert와 같은 비선형 표현력을 유지합니다.", runtime: "routed SwiGLU와 동일하게 clamp와 SiLU gate를 거칩니다." },
    "shared-w2": { why: "shared expert activation을 hidden dimension으로 복원해 routed output과 더할 수 있게 합니다. shared output을 routed output과 더할 수 있게 shape을 맞춥니다." },
    "expert-combine": { why: "token별 conditional routed output과 universal shared output을 하나의 MoE output으로 합칩니다. routed specialization과 common transform을 동시에 씁니다." },
    "moe-allreduce": { why: "tensor/expert parallel 환경에서 shard별 MoE 결과를 동일한 hidden stream으로 합칩니다. all-reduce가 분산 shard의 MoE 결과를 맞춥니다.", runtime: "single-device conceptual graph에서는 거의 identity처럼 보일 수 있습니다." },
    "attn-gather": { why: "전체 cache를 dense하게 읽지 않고 선택된 id만 모아 attention kernel 입력으로 만듭니다. selected KV만 gather해 memory bandwidth와 attention compute를 줄입니다.", runtime: "SWA entry와 compressed entry는 같은 512-dim KV shape라 gather 후 concat됩니다." },
    "attn-score": { why: "selected key에 대해서만 QK logit을 계산해 sparse attention의 핵심 compute를 제한하고 dense T 길이 score matrix를 피합니다.", runtime: "KV sharing 때문에 key 역할에서는 RoPE가 들어간 representation을 씁니다." },
    "attn-mask-sink": { why: "sparse 후보 안에서도 causal/window 제약을 지키고 head별 attention sink로 확률 질량을 조절합니다. mask는 causal/window 제약을 지키고, attn_sink는 head별 score bias를 제공합니다.", runtime: "local SWA mask와 compressed causal mask가 최종 score 공간에서 합쳐집니다." },
    "attn-softmax": { why: "selected KV set 안에서만 확률을 정규화해 dense cache softmax를 대체합니다. selected KV set 안에서만 softmax를 계산해 long-context compute와 memory traffic을 줄입니다.", runtime: "긴 context용 kernel은 전체 cache dense softmax가 아니라 selected entry softmax입니다." },
    "attn-value-sum": { why: "sparse softmax 결과로 selected shared value entry만 가중합합니다. 선택된 문맥만 value sum에 쓰므로 attention output 계산이 작아집니다.", runtime: "이 단계 이후에는 key-position semantics보다 value semantics가 중요합니다." },
    "attn-inv-rope": { why: "shared KV가 key로 쓰일 때 들어간 RoPE phase가 value output에 남는 문제를 보정합니다. inverse RoPE path가 KV sharing의 memory 이득을 유지하면서 value phase 부작용을 줄입니다.", runtime: "K에는 RoPE가 필요하지만 V에는 위치 회전이 직접 섞이면 부자연스럽기 때문에 output path에서 보정합니다." },
    "o-woa": { why: "head output을 group별 low-rank latent로 먼저 접어 output projection의 중간 표현을 줄입니다. wo_a group projection이 compute와 parameter 사용을 줄입니다.", runtime: "Pro는 G=16, Flash는 G=8 group 구성을 씁니다." },
    "o-wob": { why: "group low-rank latent를 다시 residual hidden dimension으로 복원합니다. wo_b output projection이 attention output을 mHC writeback 가능한 [B,S,D] stream으로 맞춥니다." },
    "hc-head-collapse": { why: "최종 4-lane residual state를 LM head가 받을 단일 hidden stream으로 접습니다. head collapse가 mHC 내부 표현을 일반 output projection 형태로 바꿉니다." },
    "final-rmsnorm": { why: "vocab projection 직전 hidden scale을 맞춰 logits magnitude를 안정화합니다. final RMSNorm은 vocab projection 직전 logits magnitude를 안정화합니다." },
    "last-token": { why: "autoregressive decode에서 필요한 마지막 token hidden만 vocab projection해 output compute를 줄입니다. x[:, -1]만 projection해 output compute를 줄이고 그래프 의미를 명확히 합니다.", runtime: "이 노드가 'LM head가 매 layer마다 돈다'는 오해를 막는 핵심 경계입니다." },
    "lm-project": { why: "최종 hidden을 vocabulary logit 공간으로 사영해 sampling 가능한 score를 만듭니다. LM head가 architecture 내부 표현을 token distribution으로 변환합니다.", runtime: "TP 환경에서는 vocab shard projection 후 gather/reduce가 붙을 수 있습니다." },
    "mtp-embed": { why: "MTP branch가 token id 기반 정보를 별도 embedding path로 다시 사용하게 합니다. MTP embedding path가 auxiliary prediction 입력을 구성합니다." },
    "mtp-hidden-proj": { why: "최종 decoder hidden/lane state를 MTP block이 받을 공간으로 맞춥니다. MTP hidden path가 main stack representation을 auxiliary branch에 연결합니다." },
    "mtp-combine": { why: "token embedding 정보와 final hidden 정보를 합쳐 MTP block 입력을 만듭니다. 두 입력을 합쳐 auxiliary next-token objective에 더 풍부한 입력을 줍니다." },
    "mtp-block": { why: "main output 뒤에 보조 prediction block을 붙여 multi-token prediction 신호를 제공합니다. 그래프의 R=0 표시는 MTP block의 SWA-only 흐름을 반영합니다.", runtime: "그래프에서는 R=0 SWA-only attention mode로 표시합니다." },
    "mtp-head": { why: "MTP branch output을 auxiliary vocabulary logits로 변환합니다. MTP head가 main logits와 별도의 보조 prediction score를 만듭니다." },
    "stack-entry": { why: "입력 처리와 반복 decoder layer 내부를 분리해 LM head가 매 layer마다 붙는 것처럼 보이는 오해를 막습니다. 반복 layer와 최종 logits path의 경계를 분명하게 보여줍니다.", ui: "레이어 모드 버튼은 대표 layer 내부 경로만 바꾸며 input node 자체를 바꾸는 것이 아닙니다." },
    "stack-exit": { why: "모든 decoder layer를 지난 최종 residual lane state와 output head를 분리합니다. logits path가 최종 state 뒤에만 붙는다는 점을 드러내 per-layer LM head처럼 보이는 오해를 줄입니다.", ui: "LM head가 매 layer마다 실행되는 것처럼 보이는 오해를 막습니다." },
    "hca-all-compressed": { why: "R=128처럼 매우 강하게 압축된 layer에서는 indexer 없이 모든 compressed block을 읽어 retrieval overhead를 없앱니다. HCA에서는 indexer를 생략해 retrieval overhead를 없애고 compressed memory를 그대로 활용합니다.", runtime: "HCA에는 Lightning indexer가 없고 R=128에서는 Attention.indexer가 None입니다.", ui: "compressed memory 전체 + SWA에 attend하는 구조로 표시합니다." },
  }).forEach(([id, details]) => {
    if (nodes[id]) nodes[id].details = details;
  });

  Object.entries({
    "input-ids": [{ title: "입력 텐서", latex: String.raw`\mathrm{ids}\in\mathbb{N}^{B\times S}`, note: "토크나이저 결과가 batch와 sequence 축을 가진 정수 matrix로 들어옵니다." }],
    embedding: [{ title: "Embedding lookup", latex: String.raw`x_{b,s}=E[\mathrm{ids}_{b,s}],\qquad x\in\mathbb{R}^{B\times S\times D}`, note: "TP에서는 vocab shard별 lookup 결과를 합쳐 동일한 hidden vector를 만듭니다." }],
    "hc-expand": [{ title: "Lane repeat", latex: String.raw`X_{b,s,l,d}=x_{b,s,d},\qquad l\in\{1,\dots,4\}`, note: "초기 hidden stream을 4개의 residual lane으로 복제합니다." }],
    "stack-entry": [{ title: "반복 layer state", latex: String.raw`X^{(0)}\in\mathbb{R}^{B\times S\times 4\times D},\qquad X^{(n+1)}=F_n(X^{(n)})`, note: "아래 그래프는 선택된 대표 decoder layer의 내부 전개입니다." }],

    "mhc-attn": [{ title: "Attention mHC wrapper", latex: String.raw`X'=\operatorname{mHCWrite}(X,\operatorname{Attention}(\operatorname{mHCRead}(X)))`, note: "attention 연산 자체보다 read/write lane projection을 감싸는 구조입니다." }],
    "hc-flatten": [{ title: "Controller flatten", latex: String.raw`z_{b,s}=\operatorname{concat}(X_{b,s,1,:},\dots,X_{b,s,4,:})\in\mathbb{R}^{4D}`, note: "controller path가 4개 lane을 한 번에 보도록 lane 축을 hidden 축으로 합칩니다." }],
    "hc-controller": [{ title: "Controller linear", latex: String.raw`m=zW_{\mathrm{hc}}^\top,\qquad m\in\mathbb{R}^{24}`, note: "24개 출력은 pre 4개, post 4개, comb 16개로 split됩니다." }],
    "hc-split": [{ title: "Split indices", latex: String.raw`m_{0:4}\to p_{\mathrm{pre}},\quad m_{4:8}\to p_{\mathrm{post}},\quad m_{8:24}\to C_{\mathrm{raw}}\in\mathbb{R}^{4\times4}` }],
    "hc-pre-sigmoid": [{ title: "Pre weights", latex: String.raw`pre_j=\sigma(m_j\,s_0+b_j)+\epsilon`, note: "read path에서 residual lanes를 하나의 hidden stream으로 읽는 coefficient입니다." }],
    "hc-post-sigmoid": [{ title: "Post weights", latex: String.raw`post_j=2\,\sigma(m_{j+4}\,s_1+b_{j+4})`, note: "sublayer output을 4개 residual lane에 주입하는 coefficient입니다." }],
    "hc-comb-softmax": [{ title: "Comb row softmax", latex: String.raw`C_{j,k}^{(0)}=\frac{\exp(m_{8+4j+k}s_2+b_{8+4j+k})}{\sum_{k'}\exp(m_{8+4j+k'}s_2+b_{8+4j+k'})}+\epsilon` }],
    "hc-comb-sinkhorn": [{ title: "Sinkhorn iterations", latex: String.raw`C\leftarrow C / \operatorname{sum}_{row}(C),\qquad C\leftarrow C / \operatorname{sum}_{col}(C)`, note: "row/column normalization을 반복해 comb를 doubly stochastic에 가깝게 만듭니다." }],
    "hc-sinkhorn": [
      { title: "Split", latex: String.raw`m\rightarrow (p_{\mathrm{read}}\in\mathbb{R}^{4},\;p_{\mathrm{write}}\in\mathbb{R}^{4},\;C\in\mathbb{R}^{4\times4})` },
      { title: "Doubly stochastic mixing", latex: String.raw`\tilde C=\operatorname{Sinkhorn}(C),\qquad \sum_i \tilde C_{ij}\approx 1,\quad \sum_j \tilde C_{ij}\approx 1`, note: "row/column 합을 안정화해 layer 간 residual gradient transport를 덜 흔들리게 만드는 목적입니다." },
    ],
    "hc-read": [{ title: "Read projection", latex: String.raw`x_{\mathrm{attn}}=\sum_{l=1}^{4}p_{\mathrm{read},l}\,X_l`, note: "4-lane residual을 attention이 받을 단일 hidden stream으로 읽습니다." }],
    "attn-residual-mix": [{ title: "Residual lane mixing", latex: String.raw`M_l=\sum_{j=1}^{4}\tilde C_{l,j}X_j`, note: "기존 residual lane을 4x4 transport matrix로 섞습니다." }],
    "attn-post-inject": [{ title: "Attention injection", latex: String.raw`I_l=p_{\mathrm{write},l}\,y_{\mathrm{attn}}`, note: "단일 attention output을 4개 lane으로 다시 분배합니다." }],
    "hc-write": [{ title: "Attention writeback", latex: String.raw`X'_l=\sum_{j=1}^{4}\tilde C_{l,j}X_j+p_{\mathrm{write},l}\,y_{\mathrm{attn}}`, note: "residual lane mixing과 attention output injection을 더해 다음 state를 만듭니다." }],

    attention: [{ title: "Attention summary", latex: String.raw`y=\operatorname{Attn}(Q(x),K_{\mathcal{I}},V_{\mathcal{I}})W_o`, note: "인덱스 집합 I는 SWA, CSA, HCA mode에 따라 달라집니다." }],
    "q-path": [{ title: "Low-rank query path", latex: String.raw`Q=\operatorname{RoPE}(\operatorname{reshape}(\operatorname{RMSNorm}(xW_{q,a})W_{q,b}))`, note: "overview에서는 하위 q-wqa/q-norm/q-wqb/q-rope 노드로 풀어 보여줍니다." }],
    "q-wqa": [{ title: "Query A projection", latex: String.raw`q_a=xW_{q,a}^{\top},\qquad q_a\in\mathbb{R}^{B\times S\times Q_r}` }],
    "q-norm": [{ title: "RMSNorm", latex: String.raw`\operatorname{RMSNorm}(u)=\frac{u}{\sqrt{\frac{1}{n}\sum_i u_i^2+\epsilon}}\odot w`, note: "q_lora_rank 축을 기준으로 scale을 맞춘 뒤 main query와 indexer query가 이 출력을 공유합니다." }],
    "q-wqb": [{ title: "Query B projection", latex: String.raw`q_b=q_a^{\mathrm{norm}}W_{q,b}^{\top},\qquad q_b\in\mathbb{R}^{B\times S\times (H\cdot512)}` }],
    "q-reshape": [{ title: "Head split + renorm", latex: String.raw`Q=\operatorname{reshape}(q_b,[B,S,H,512]),\qquad Q_h\leftarrow \operatorname{RMSNorm}(Q_h)`, note: "projection 출력의 channel 축을 head 축과 head dim 축으로 나눕니다." }],
    "q-rope": [{ title: "RoPE slice", latex: String.raw`\operatorname{RoPE}(q_{2i},q_{2i+1},p)=\begin{bmatrix}q_{2i}\cos\theta_{p,i}-q_{2i+1}\sin\theta_{p,i}\\q_{2i}\sin\theta_{p,i}+q_{2i+1}\cos\theta_{p,i}\end{bmatrix}`, note: "512차원 중 마지막 64차원에 position phase를 넣습니다." }],

    "kv-path": [{ title: "Shared KV path", latex: String.raw`k\!v=\operatorname{RMSNorm}(xW_{kv}^{\top}),\qquad k\!v\in\mathbb{R}^{B\times S\times512}`, note: "동일한 512-dim vector가 key/value cache의 공유 표현으로 쓰입니다." }],
    "kv-wkv": [{ title: "Shared KV projection", latex: String.raw`u=xW_{kv}^{\top},\qquad u\in\mathbb{R}^{B\times S\times512}` }],
    "kv-norm": [{ title: "KV RMSNorm", latex: String.raw`k\!v=\frac{u}{\sqrt{\operatorname{mean}(u^2)+\epsilon}}\odot w_{kv}`, note: "RoPE, compressor, cache write 전에 shared KV scale을 맞춥니다." }],
    "kv-slice": [{ title: "Content / RoPE split", latex: String.raw`k\!v=[k\!v_{\mathrm{nope}}\in\mathbb{R}^{448}\;||\;k\!v_{\mathrm{rope}}\in\mathbb{R}^{64}]`, note: "512-dim shared KV 내부에서 position-aware key slice와 content/value slice를 분리해서 생각합니다." }],
    "kv-rope-quant": [{ title: "RoPE + quantized content", latex: String.raw`k=[\operatorname{FP8Sim}(k_{\mathrm{nope}}),\operatorname{RoPE}(k_{\mathrm{rope}},p)]`, note: "content 448 dim은 low precision simulation, RoPE 64 dim은 position-aware key로 남깁니다." }],
    "kv-cache": [{ title: "Cache layout", latex: String.raw`\mathrm{cache}=[\mathrm{SWA}_{0:128}\;||\;\mathrm{Compressed}_{0:\lfloor T/R\rfloor}]`, note: "앞쪽은 최근 128개 uncompressed entry, 뒤쪽은 compressed entry 영역입니다." }],
    "window-topk": [{ title: "Sliding window ids", latex: String.raw`\mathcal{I}_{\mathrm{swa}}=\{t\mid \max(0,n-127)\le t\le n\}`, note: "score top-k가 아니라 최근 local token index set입니다." }],
    "swa-prefill-write": [{ title: "Prefill window write", latex: String.raw`\mathrm{SWA}\leftarrow K\!V_{\max(0,S-128):S}`, note: "prefill에서는 chunk 끝의 local window만 uncompressed cache에 남깁니다." }],
    "swa-decode-write": [{ title: "Decode ring write", latex: String.raw`\mathrm{SWA}_{n\bmod128}\leftarrow K\!V_n`, note: "decode step마다 128-slot circular buffer의 한 칸을 갱신합니다." }],
    "cache-layout": [{ title: "Logical concat", latex: String.raw`\mathcal{C}=[\mathcal{C}_{\mathrm{swa}}\;||\;\mathcal{C}_{\mathrm{comp}}]`, note: "attention index는 SWA prefix와 compressed suffix를 같은 cache id 공간에서 봅니다." }],
    "attn-selected": [{ title: "KV index union", latex: String.raw`\mathcal{I}=\mathcal{I}_{\mathrm{swa}}\cup\mathcal{I}_{\mathrm{compressed}}`, note: "CSA는 indexer top-k, HCA는 valid compressed block 전체, MTP는 SWA만 사용합니다." }],
    "attn-gather": [{ title: "Selected gather", latex: String.raw`K\!V_{\mathcal{I}}=\operatorname{gather}(\mathcal{C},\mathcal{I})`, note: "attention kernel은 전체 cache가 아니라 선택된 entry만 읽습니다." }],
    "attn-score": [{ title: "QK score", latex: String.raw`A_{b,s,h,t}=\langle Q_{b,s,h},K_{\mathcal{I}_{b,s,t}}\rangle/\sqrt{512}` }],
    "attn-mask-sink": [{ title: "Mask + sink", latex: String.raw`\tilde A=A+M_{\mathrm{causal/window}}+\beta_h`, note: "head별 attention sink와 causal/window mask를 같은 logit 공간에 더합니다." }],
    "attn-softmax": [{ title: "Softmax", latex: String.raw`P_{b,s,h,:}=\operatorname{softmax}(\tilde A_{b,s,h,:})` }],
    "attn-value-sum": [{ title: "Value sum", latex: String.raw`Y_{b,s,h}=\sum_t P_{b,s,h,t}V_{\mathcal{I}_{b,s,t}}` }],
    "attn-inv-rope": [{ title: "Inverse RoPE", latex: String.raw`Y_{\mathrm{value}}\leftarrow\operatorname{RoPE}^{-1}(Y_{\mathrm{shared}})`, note: "shared KV가 value로 쓰일 때 position phase가 남는 문제를 보정하는 path입니다." }],
    "o-woa": [{ title: "Group low-rank A", latex: String.raw`o_a=\operatorname{GroupLinear}_a(\operatorname{concat}_h Y_h),\qquad o_a\in\mathbb{R}^{B\times S\times G\times O_r}` }],
    "o-wob": [{ title: "Output B", latex: String.raw`y_{\mathrm{attn}}=o_aW_{o,b}^{\top},\qquad y_{\mathrm{attn}}\in\mathbb{R}^{B\times S\times D}` }],
    "sparse-attn": [
      { title: "Attention logits", latex: String.raw`a_{h,t}=\frac{\langle q_h,k_{h,t}\rangle}{\sqrt{512}}+\mathrm{mask}_t+\mathrm{sink}_h` },
      { title: "Weighted value sum", latex: String.raw`y_h=\sum_{t\in\mathcal{I}}\operatorname{softmax}(a_h)_t\,v_t`, note: "선택된 KV entry만 gather해 online-softmax kernel에서 계산합니다." },
    ],
    "o-proj": [{ title: "Grouped output projection", latex: String.raw`y=\operatorname{GroupProj}_b(\operatorname{GroupProj}_a(\operatorname{concat}_h y_h))`, note: "head output을 group low-rank projection으로 D차원 hidden stream에 복원합니다." }],

    compressor: [{ title: "Compression summary", latex: String.raw`c_j=\operatorname{Pool}_{t\in\mathrm{block}(j)}(W_{kv}x_t,W_gx_t,\mathrm{APE}_t)`, note: "R=4/128 layer mode에 따라 block span과 overlap 처리가 달라집니다." }],
    "comp-wkv": [{ title: "Compressor KV candidate", latex: String.raw`u_t=x_tW_{\mathrm{comp},kv}^{\top}` }],
    "comp-wgate": [{ title: "Compressor gate score", latex: String.raw`g_t=x_tW_{\mathrm{comp},g}^{\top}+\mathrm{APE}_t`, note: "softmax pooling weight를 만들기 위한 learned score입니다." }],
    "comp-ape": [{ title: "APE score add", latex: String.raw`\tilde g_{j,r}=g_{j,r}+a_r,\qquad a_r\in\mathbb{R}^{C_{\mathrm{off}}\cdot512}` }],
    "comp-cutoff": [{ title: "Full block split", latex: String.raw`[u_{\mathrm{full}},u_{\mathrm{tail}}]=\operatorname{split}_{R}([\mathrm{tail};u_{\mathrm{new}}])`, note: "R개 단위로 완성된 projection만 pooling으로 보내고 remainder는 tail로 남깁니다." }],
    "tail-append": [{ title: "Persistent tail", latex: String.raw`\mathrm{tail}'=u_{\mathrm{tail}},\qquad |\mathrm{tail}'|<R`, note: "request별 cache state로 다음 token/chunk까지 유지됩니다." }],
    "tail-state": [{ title: "Tail state update", latex: String.raw`\mathrm{tail}_{n+1}=\operatorname{append\_and\_trim}(\mathrm{tail}_{n},u_n,g_n;R)`, note: "decode에서 아직 block이 완성되지 않은 token projection을 임시 저장합니다." }],
    "comp-block-view": [{ title: "Block view", latex: String.raw`u_{\mathrm{full}}\rightarrow U\in\mathbb{R}^{B\times N_{\mathrm{full}}\times \mathrm{span}\times512}` }],
    "overlap-transform": [{ title: "CSA overlap span", latex: String.raw`\mathrm{span}_j=[x_{4j-4},\dots,x_{4j+3}]`, note: "c4a는 stride 4이지만 pooling span은 8-token overlap으로 볼 수 있습니다." }],
    "gated-pool": [{ title: "Softmax-gated pooling", latex: String.raw`c_j=\sum_{t\in\mathrm{block}(j)}u_t\cdot\operatorname{softmax}(g)_t`, note: "여러 native token을 하나의 compressed KV entry로 줄입니다." }],
    "comp-anchor": [{ title: "Anchor id", latex: String.raw`a_j=jR,\qquad R\in\{4,128\}`, note: "c4a는 0,4,8,..., c128a는 0,128,256,... anchor를 씁니다." }],
    "comp-norm-rope": [{ title: "Compressed norm + anchor RoPE", latex: String.raw`\hat c_j=\operatorname{RoPE}(\operatorname{RMSNorm}(c_j),a_j),\qquad a_j\in\{0,R,2R,\dots\}`, note: "compressed block 내부 token 위치가 아니라 anchor position을 사용합니다." }],
    "comp-cache-slot": [{ title: "Slot map", latex: String.raw`\mathrm{slot}_j=128+j`, note: "attention cache id 공간에서 compressed entry는 SWA 128-slot 뒤에 배치됩니다." }],
    "comp-cache-write": [{ title: "Compressed cache write", latex: String.raw`\mathrm{cache}_{128+j}\leftarrow \hat c_j,\qquad j=\left\lfloor\frac{n}{R}\right\rfloor`, note: "SWA 영역 뒤쪽 compressed cache slot에 기록합니다." }],
    "hca-all-compressed": [{ title: "HCA compressed set", latex: String.raw`\mathcal{I}_{\mathrm{compressed}}=\{0,\dots,\lfloor T/128\rfloor-1\}`, note: "HCA layer에서는 Lightning indexer 없이 valid c128a block 전체를 사용합니다." }],

    indexer: [{ title: "Indexer summary", latex: String.raw`\mathcal{I}_{\mathrm{csa}}=\operatorname{TopK}(\operatorname{Score}(q_{\mathrm{idx}},C_{\mathrm{idx}}),K)`, note: "R=4 CSA에서 compressed block 후보를 sparse하게 고릅니다." }],
    "idx-q": [{ title: "Indexer query projection", latex: String.raw`q_{\mathrm{idx}}=\operatorname{reshape}(q_{\mathrm{norm}}W_{\mathrm{idx},q}^{\top},[B,S,64,128])` }],
    "idx-rope": [{ title: "Indexer RoPE", latex: String.raw`q^{r}_{\mathrm{idx}}=\operatorname{RoPE}(q_{\mathrm{idx}},p)` }],
    "idx-hadamard": [{ title: "Hadamard rotation", latex: String.raw`q^{h}_{\mathrm{idx}}=H_{128}q^{r}_{\mathrm{idx}}` }],
    "idx-fp4": [{ title: "FP4 query", latex: String.raw`\tilde q_{\mathrm{idx}}=\operatorname{Quant}_{\mathrm{FP4}}(q^{h}_{\mathrm{idx}})` }],
    "idx-rotate": [{ title: "Cheap rotated query", latex: String.raw`\tilde q=\operatorname{FP4}(\operatorname{Hadamard}(\operatorname{RoPE}(q_{\mathrm{idx}})))`, note: "정확한 attention Q가 아니라 retrieval score용 cheap representation입니다." }],
    "idx-cache-compress": [{ title: "Index compressor", latex: String.raw`z_j=\operatorname{Compress}_{128}(x_{4j:4j+3})`, note: "main 512-dim compressed KV와 별개의 128-dim retrieval cache를 만듭니다." }],
    "idx-cache-write": [{ title: "Index cache write", latex: String.raw`C_{\mathrm{idx},j}\leftarrow z_j` }],
    "idx-cache": [{ title: "Index cache", latex: String.raw`C_{\mathrm{idx}}\in\mathbb{R}^{B\times \lfloor T/4\rfloor\times128}`, note: "main 512-dim KV cache와 별도의 128-dim indexer cache입니다." }],
    "idx-einsum": [{ title: "Block score", latex: String.raw`s_{b,s,h,j}=\operatorname{ReLU}(\langle \tilde q_{b,s,h},C_{\mathrm{idx},b,j}\rangle)`, note: "candidate compressed block별 retrieval score를 계산합니다." }],
    "idx-weight": [{ title: "Head weighted sum", latex: String.raw`S_{b,s,j}=\sum_{h=1}^{64}\alpha_{b,s,h}\,s_{b,s,h,j}`, note: "query-dependent head weight로 64개 index head score를 하나로 합칩니다." }],
    "idx-mask": [{ title: "Causal mask", latex: String.raw`\tilde S_{b,s,j}=S_{b,s,j}+M(j\le\lfloor p_s/4\rfloor)` }],
    "idx-topk": [{ title: "Causal TopK", latex: String.raw`\mathcal{I}_{\mathrm{topk}}=\operatorname{TopK}(S+\mathrm{causal\_mask},K)`, note: "Pro는 K=1024, Flash는 K=512로 표시됩니다." }],
    "idx-offset": [{ title: "Cache id offset", latex: String.raw`\mathcal{I}_{\mathrm{comp}}=128+\mathcal{I}_{\mathrm{topk}}`, note: "selected_ids가 SWA prefix와 compressed suffix를 같은 cache id로 참조하게 맞춥니다." }],

    "mhc-ffn": [{ title: "MoE mHC wrapper", latex: String.raw`X'=\operatorname{mHCWrite}_{ffn}(X,\operatorname{MoE}(\operatorname{mHCRead}_{ffn}(X)))` }],
    "ffn-hc-flatten": [{ title: "MoE controller flatten", latex: String.raw`z^{ffn}_{b,s}=\operatorname{concat}_{l=1}^{4}X_{b,s,l,:}` }],
    "ffn-hc-controller": [{ title: "MoE controller linear", latex: String.raw`m^{ffn}=z^{ffn}W_{\mathrm{hc},ffn}^{\top},\qquad m^{ffn}\in\mathbb{R}^{24}` }],
    "ffn-hc-split": [{ title: "MoE split indices", latex: String.raw`m^{ffn}_{0:4}\to pre,\quad m^{ffn}_{4:8}\to post,\quad m^{ffn}_{8:24}\to comb_{\mathrm{raw}}` }],
    "ffn-hc-pre-sigmoid": [{ title: "MoE pre weights", latex: String.raw`pre^{ffn}_j=\sigma(m^{ffn}_j\,s_0+b_j)+\epsilon` }],
    "ffn-hc-post-sigmoid": [{ title: "MoE post weights", latex: String.raw`post^{ffn}_j=2\,\sigma(m^{ffn}_{j+4}\,s_1+b_{j+4})` }],
    "ffn-hc-comb-softmax": [{ title: "MoE comb row softmax", latex: String.raw`C^{ffn,(0)}=\operatorname{softmax}_{row}(C^{ffn}_{raw})+\epsilon` }],
    "ffn-hc-comb-sinkhorn": [{ title: "MoE Sinkhorn", latex: String.raw`C^{ffn}\leftarrow \operatorname{Sinkhorn}(C^{ffn,(0)})` }],
    "ffn-hc-sinkhorn": [{ title: "MoE split + Sinkhorn", latex: String.raw`m^{ffn}\rightarrow(p_{\mathrm{read}},p_{\mathrm{write}},\tilde C),\qquad \tilde C=\operatorname{Sinkhorn}(C)` }],
    "hc-pre-moe": [{ title: "MoE read projection", latex: String.raw`x_{\mathrm{moe}}=\sum_{l=1}^{4}p^{ffn}_{\mathrm{read},l}X_l` }],
    "ffn-residual-mix": [{ title: "MoE residual mixing", latex: String.raw`M^{ffn}_l=\sum_{j=1}^{4}\tilde C^{ffn}_{l,j}X_j` }],
    "ffn-post-inject": [{ title: "MoE output injection", latex: String.raw`I^{ffn}_l=p^{ffn}_{\mathrm{write},l}y_{\mathrm{moe}}` }],
    "hc-post-moe": [{ title: "MoE writeback", latex: String.raw`X^{next}_l=\sum_j\tilde C^{ffn}_{l,j}X_j+p^{ffn}_{\mathrm{write},l}y_{\mathrm{moe}}` }],

    moe: [{ title: "MoE summary", latex: String.raw`y_{\mathrm{moe}}=\sum_{i\in\mathcal{E}(x)}w_iE_i(x)+E_{\mathrm{shared}}(x)`, note: "routed expert top-6과 always-on shared expert를 합칩니다." }],
    gate: [{ title: "Routing abstraction", latex: String.raw`\mathcal{E}(x)=\begin{cases}\operatorname{tid2eid}(\mathrm{ids}),&\ell<3\\\operatorname{TopK}(\operatorname{score}(x),6),&\ell\ge3\end{cases}` }],
    "gate-score": [{ title: "Expert score", latex: String.raw`r=\sqrt{\operatorname{softplus}(xW_g^\top)},\qquad r\in\mathbb{R}^{B S\times E}`, note: "selection bias는 expert 선택에만 영향을 주고 weight는 원 score에서 gather합니다." }],
    "hash-route": [{ title: "Token-id routing", latex: String.raw`\mathcal{E}_{b,s}=\mathrm{tid2eid}[\mathrm{ids}_{b,s}]`, note: "초반 layer에서는 score top-k가 아니라 input id 기반 expert id table을 사용합니다." }],
    "route-bias": [{ title: "Selection score", latex: String.raw`r^{sel}=r+b_{\mathrm{route}}`, note: "bias는 expert id 선택용이며 route weight는 original score에서 다시 gather합니다." }],
    "topk-route": [{ title: "Activation routing", latex: String.raw`\mathcal{E}_{b,s}=\operatorname{TopK}(r_{b,s}+b_{\mathrm{route}},6)` }],
    "route-score-gather": [{ title: "Gather original score", latex: String.raw`r_{\mathcal{E}}=\operatorname{gather}(r,\mathcal{E})` }],
    "route-weights": [{ title: "Normalize route weights", latex: String.raw`w_i=\frac{r_i}{\sum_{j\in\mathcal{E}}r_j}\cdot \mathrm{route\_scale}`, note: "선택된 expert의 original score를 normalize한 뒤 scale을 곱합니다." }],
    "routed-experts": [{ title: "Routed expert FFN", latex: String.raw`E_i(x)=W_{2,i}\left(\operatorname{SiLU}(W_{1,i}x)\odot W_{3,i}x\right)` }],
    "expert-counts": [{ title: "Expert counts", latex: String.raw`n_i=\sum_t \mathbf{1}[i\in\mathcal{E}(x_t)]` }],
    "expert-dispatch": [{ title: "Token dispatch", latex: String.raw`X_i=\{x_n\mid i\in\mathcal{E}(x_n)\}`, note: "expert id별 token row를 모아 해당 expert weight로 처리합니다." }],
    "expert-w1w3": [{ title: "Gate/up projection", latex: String.raw`g=xW_{1,i}^{\top},\qquad u=xW_{3,i}^{\top}` }],
    swiglu: [{ title: "SwiGLU", latex: String.raw`h=\operatorname{SiLU}(\operatorname{clip}(g))\odot \operatorname{clip}(u)`, note: "공개 config의 swiglu_limit을 반영해 gate/up activation을 clamp합니다." }],
    "expert-w2": [{ title: "Down projection", latex: String.raw`y_i=hW_{2,i}^{\top},\qquad y_i\in\mathbb{R}^{D}` }],
    "routed-accum": [{ title: "Scatter weighted sum", latex: String.raw`y_{\mathrm{routed},t}=\sum_{i\in\mathcal{E}(x_t)}w_{t,i}E_i(x_t)` }],
    "shared-w1w3": [{ title: "Shared gate/up", latex: String.raw`g_s=xW_{1,s}^{\top},\qquad u_s=xW_{3,s}^{\top}` }],
    "shared-swiglu": [{ title: "Shared SwiGLU", latex: String.raw`h_s=\operatorname{SiLU}(\operatorname{clip}(g_s))\odot\operatorname{clip}(u_s)` }],
    "shared-w2": [{ title: "Shared down", latex: String.raw`y_s=h_sW_{2,s}^{\top}` }],
    "shared-expert": [{ title: "Shared expert", latex: String.raw`y_{\mathrm{shared}}=W_{2,s}\left(\operatorname{SiLU}(W_{1,s}x)\odot W_{3,s}x\right)`, note: "routing과 무관하게 모든 token에서 계산됩니다." }],
    "expert-combine": [{ title: "Routed + shared combine", latex: String.raw`y=\sum_{i\in\mathcal{E}(x)}w_i\,E_i(x)+E_{\mathrm{shared}}(x)` }],
    "moe-allreduce": [{ title: "Parallel reduce", latex: String.raw`y_{\mathrm{moe}}=\operatorname{AllReduce}(y_{\mathrm{moe}}^{\mathrm{shard}})`, note: "distributed runtime에서 shard별 output을 합치는 단계입니다." }],

    "stack-exit": [{ title: "Final decoder state", latex: String.raw`X^{(L)}=F_{L-1}\circ\cdots\circ F_0(X^{(0)})`, note: "LM head는 각 layer가 아니라 최종 stack state 뒤에 붙습니다." }],
    "hc-head-collapse": [{ title: "HC head", latex: String.raw`h=\operatorname{HCHead}(X^{(L)})`, note: "4-lane 최종 residual을 단일 hidden stream으로 접습니다." }],
    "final-rmsnorm": [{ title: "Final RMSNorm", latex: String.raw`\hat h=\operatorname{RMSNorm}(h)` }],
    "last-token": [{ title: "Last token only", latex: String.raw`h_{\mathrm{last}}=\hat h_{:,-1,:}` }],
    "lm-project": [{ title: "LM projection", latex: String.raw`\mathrm{logits}=h_{\mathrm{last}}W_{\mathrm{lm}}^\top` }],
    head: [{ title: "HC collapse + LM head", latex: String.raw`h=\operatorname{HCHead}(X^{(L)}_{:,-1,:,:}),\qquad \mathrm{logits}=hW_{\mathrm{lm}}^\top`, note: "공식 path는 마지막 token만 vocab projection합니다." }],
    "mtp-embed": [{ title: "MTP embedding path", latex: String.raw`e_{\mathrm{mtp}}=E(\mathrm{ids})` }],
    "mtp-hidden-proj": [{ title: "MTP hidden path", latex: String.raw`h_{\mathrm{mtp}}=W_h\,\operatorname{HCHead}(X^{(L)})` }],
    "mtp-combine": [{ title: "MTP combine", latex: String.raw`x_{\mathrm{mtp}}=\operatorname{Combine}(e_{\mathrm{mtp}},h_{\mathrm{mtp}})` }],
    "mtp-block": [{ title: "MTP block", latex: String.raw`y_{\mathrm{mtp}}=F_{\mathrm{mtp}}(x_{\mathrm{mtp}};\,R=0)` }],
    "mtp-head": [{ title: "MTP logits", latex: String.raw`\mathrm{logits}_{\mathrm{mtp}}=y_{\mathrm{mtp}}W_{\mathrm{lm}}^\top` }],
    mtp: [{ title: "MTP branch", latex: String.raw`y_{\mathrm{mtp}}=\operatorname{MTP}(X^{(L)},\mathrm{ids})`, note: "보조 next-token prediction block이며 attention mode는 SWA-only로 표시합니다." }],
    logits: [{ title: "Vocabulary scores", latex: String.raw`p(v\mid x)=\operatorname{softmax}(\mathrm{logits})_v`, note: "sampling 정책은 모델 구조 그래프 밖의 runtime 단계입니다." }],
  }).forEach(([id, formula]) => {
    if (nodes[id]) nodes[id].details = { ...nodes[id].details, formula };
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
