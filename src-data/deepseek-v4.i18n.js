window.DSV4_I18N = {
  ui: {
    ko: {
      langKo: "KO",
      langEn: "EN",
      language: "Lang",
      description: "설명",
      why: "설명",
      runtime: "런타임 동작",
      ui: "시각화 포인트",
      open: "남은 질문",
      formula: "계산식",
      input: "Input",
      output: "Output",
      nodes: "nodes",
      group: "group",
      overview: "전체",
      nodeFormulas: "Node formulas",
      embedDesc: "Embed desc",
      dark: "Dark",
      openSubgraph: "Open subgraph",
    },
    en: {
      langKo: "KO",
      langEn: "EN",
      language: "Lang",
      description: "Description",
      why: "Description",
      runtime: "Runtime Behavior",
      ui: "Visualization Notes",
      open: "Open Questions",
      formula: "Formulas",
      input: "Input",
      output: "Output",
      nodes: "nodes",
      group: "group",
      overview: "All",
      nodeFormulas: "Node formulas",
      embedDesc: "Embed desc",
      dark: "Dark",
      openSubgraph: "Open subgraph",
    },
  },
  formulaTitle: {
    en: {
      "입력 텐서": "Input tensor",
      "반복 layer state": "Repeated layer state",
    },
  },
  nodeText: {
    ko: {
        "input-ids": {
            "summary": "Tokenizer output enters the model once before the decoder layer stack.",
            "notes": [
                "B는 batch size, S는 현재 forward chunk length."
            ],
            "details": {
                "why": "input_ids는 [B,S] forward chunk로 들어와 토큰 임베딩의 원천이 되고, 초반 hash routing에서는 tid2eid lookup에 직접 쓰이는 lexical prior 역할도 합니다. 다만 자주 등장하는 토큰이 특정 expert로 쏠리는 문제를 어떻게 완화했는지는 공개 코드만으로는 보이지 않습니다.",
                "runtime": "hash routing 레이어는 score top-k로 expert id를 고르지 않고 input_ids.flatten()으로 tid2eid를 조회합니다.",
                "ui": "임베딩 입력과 초반 라우팅 메타데이터라는 두 용도를 같이 보여주는 게 좋습니다.",
                "open": "tid2eid가 학습/배정될 때 자주 등장하는 토큰의 expert 쏠림을 어떻게 막았는지는 공개 inference 코드만으로는 확인되지 않습니다."
            },
            "formula": [
                {
                    "title": "입력 텐서",
                    "note": "토크나이저 결과가 batch와 sequence 축을 가진 정수 matrix로 들어옵니다."
                }
            ]
        },
        "embedding": {
            "summary": "Looks up token vectors once before the decoder layer stack begins.",
            "notes": [
                "TP에서는 vocab shard 후 all-reduce."
            ],
            "details": {
                "why": "이산 token id를 모든 sublayer가 공유할 연속 표현으로 바꾸는 표준 진입점입니다. tensor parallel에서는 vocab shard별 partial embedding을 만든 뒤 all-reduce로 같은 hidden 표현을 맞추기 때문에, decoder stack 안에서 반복 실행되는 노드가 아니라 stack 진입 전 1회 수행되는 입력 변환으로 보는 편이 정확합니다.",
                "runtime": "tensor parallel에서는 vocab shard 밖 id를 mask하고 partial embedding을 all-reduce합니다.",
                "ui": "decoder stack 안에서 매번 실행되는 노드가 아니라 stack 진입 전 1회 노드로 표시합니다."
            },
            "formula": [
                {
                    "title": "Embedding lookup",
                    "note": "TP에서는 vocab shard별 lookup 결과를 합쳐 동일한 hidden vector를 만듭니다."
                }
            ]
        },
        "hc-expand": {
            "summary": "mHC residual lanes를 4개로 확장한다.",
            "notes": [
                "Block 사이 hidden state는 [B,S,4,D]."
            ],
            "details": {
                "why": "단일 residual stream을 [B,S,4,D]의 4개 lane으로 복제해 mHC가 layer마다 read/write/mixing을 선택할 수 있는 상태 공간을 만듭니다. mHC는 residual connection을 강화하고 layer 간 signal propagation을 안정화하려는 구조이며, lane별 역할 분화는 구조상 가능한 해석입니다.",
                "runtime": "공식 forward는 [B,S,D]를 [B,S,hc_mult,D]로 repeat합니다.",
                "ui": "lane 축을 실제 tensor 차원으로 보여주고, 모델 복사본 4개처럼 보이지 않게 합니다."
            },
            "formula": [
                {
                    "title": "Lane repeat",
                    "note": "초기 hidden stream을 4개의 residual lane으로 복제합니다."
                }
            ]
        },
        "mhc-attn": {
            "summary": "Attention sublayer 앞뒤의 controller/data path.",
            "notes": [
                "pre/post/comb를 생성하고 data path를 섞는다."
            ],
            "details": {
                "why": "attention sublayer가 residual lane 전체를 직접 받는 대신 controller가 pre/post/comb를 만들고, data path가 read projection으로 필요한 stream을 읽은 뒤 write projection으로 다시 lane에 분배합니다. attention 주변에서 read/write path가 분리되기 때문에 residual 전달을 안정화하면서 attention update가 들어갈 위치도 조절할 수 있습니다.",
                "runtime": "controller coefficient는 inference에서도 현재 residual lane에서 매번 다시 계산됩니다.",
                "ui": "attention 자체가 아니라 attention 주변의 controller + data path로 표시합니다."
            },
            "formula": [
                {
                    "title": "Attention mHC wrapper",
                    "note": "attention 연산 자체보다 read/write lane projection을 감싸는 구조입니다."
                }
            ]
        },
        "attention": {
            "summary": "Q path, KV path, cache, compressor/indexer, sparse attention을 결합한다.",
            "notes": [
                "R=4는 Lightning Indexer, R=128은 compressed dense path."
            ],
            "details": {
                "why": "긴 context에서 모든 과거 token을 dense attention으로 보지 않고 SWA, compressed attention, sparse retrieval을 layer별로 조합합니다. R=4 layer는 Lightning Indexer를 써 compressed block을 고르고, R=128 layer는 더 강하게 압축된 dense compressed path를 쓰면서 long-context memory와 compute를 줄입니다.",
                "runtime": "내부 cache path는 layer ratio에 따라 CSA/c4a R=4, HCA/c128a R=128, SWA-only R=0으로 달라집니다.",
                "ui": "선택된 한 layer에서 CSA와 HCA가 동시에 켜진 것처럼 보이지 않게 합니다."
            },
            "formula": [
                {
                    "title": "Attention summary",
                    "note": "인덱스 집합 I는 SWA, CSA, HCA mode에 따라 달라집니다."
                }
            ]
        },
        "q-path": {
            "summary": "wq_a, q_norm, wq_b, q renorm, RoPE.",
            "notes": [
                "q 마지막 64 dims에 RoPE."
            ],
            "details": {
                "why": "query 생성을 low-rank latent 단계와 head expansion 단계로 나눠 큰 H*512 projection을 직접 다루는 부담을 줄입니다. q_lora_rank와 wq_a/wq_b 분리 덕분에 query projection 비용을 낮추고, 마지막 64 dimension에만 RoPE를 적용해 positional slice를 분리합니다.",
                "ui": "compact scene에서만 쓰고, overview에서는 하위 노드들을 직접 보여주는 편이 좋습니다."
            },
            "formula": [
                {
                    "title": "Low-rank query path",
                    "note": "overview에서는 하위 q-wqa/q-norm/q-wqb/q-rope 노드로 풀어 보여줍니다."
                }
            ]
        },
        "kv-path": {
            "summary": "wkv, kv_norm, RoPE, non-RoPE FP8 simulation.",
            "notes": [
                "KV는 shared [B,S,512]."
            ],
            "details": {
                "why": "head마다 별도 KV를 저장하지 않고 하나의 [B,S,512] shared KV를 key/value 양쪽에 재사용해 cache memory를 크게 줄입니다. shared KV를 쓰면 cache memory는 줄지만 key score에 필요한 RoPE phase가 value path에 남을 수 있어, value로 사용할 때는 inverse RoPE 보정이 필요합니다.",
                "runtime": "동일한 shared vector가 logit 계산에서는 key처럼, output 계산에서는 value처럼 쓰입니다.",
                "ui": "KV sharing과 inverse RoPE 이야기가 시작되는 핵심 노드입니다."
            },
            "formula": [
                {
                    "title": "Shared KV path",
                    "note": "동일한 512-dim vector가 key/value cache의 공유 표현으로 쓰입니다."
                }
            ]
        },
        "kv-cache": {
            "summary": "Window cache와 compressed cache를 하나의 buffer로 관리한다.",
            "notes": [
                "prefill과 decode write path가 다르다."
            ],
            "details": {
                "why": "최근 128개 uncompressed SWA entry와 오래된 compressed entry를 같은 attention id 공간에 놓아 kernel indexing을 단순화합니다. prefill과 decode의 write path는 다르지만, 이 layout 덕분에 long-context memory를 줄이면서 최근 local detail은 uncompressed cache로 보존합니다.",
                "runtime": "앞 128 slot은 SWA, 그 뒤 suffix는 compressed entry입니다. decode에서는 SWA를 start_pos % 128 위치에 씁니다.",
                "ui": "window prefix + compressed suffix 구조로 그립니다."
            },
            "formula": [
                {
                    "title": "Cache layout",
                    "note": "앞쪽은 최근 128개 uncompressed entry, 뒤쪽은 compressed entry 영역입니다."
                }
            ]
        },
        "compressor": {
            "summary": "wkv/wgate/ape/tail state로 compressed KV를 만든다.",
            "notes": [
                "R=4는 overlap, R=128은 non-overlap."
            ],
            "details": {
                "why": "1M context의 오래된 KV를 모두 원본으로 유지하지 않고 compressed entry로 바꿔 long-context memory를 줄입니다. R=4에서는 overlap을 둔 c4a 방식으로 더 촘촘한 compressed entry를 만들고, R=128에서는 non-overlap c128a 방식으로 훨씬 강하게 압축합니다.",
                "runtime": "R=4는 c4a overlap, R=128은 c128a non-overlap 동작입니다.",
                "ui": "선택된 layer mode에 따라 c4a/c128a label을 보여줍니다."
            },
            "formula": [
                {
                    "title": "Compression summary",
                    "note": "R=4/128 layer mode에 따라 block span과 overlap 처리가 달라집니다."
                }
            ]
        },
        "indexer": {
            "summary": "R=4 layer에서 compressed blocks top-k를 선택한다.",
            "notes": [
                "Hadamard rotation, FP4 quant, weighted ReLU score."
            ],
            "details": {
                "why": "c4a의 T/4 compressed blocks가 여전히 많기 때문에 attention 전에 중요한 compressed block만 고르는 cheap retrieval path를 둡니다. Hadamard rotation, FP4 activation quantization, weighted ReLU score로 ranking 비용을 낮춘 뒤 후보 block을 좁혀 long-context attention compute를 줄입니다.",
                "runtime": "Pro topK는 1024, Flash topK는 512입니다.",
                "ui": "CSA/c4a 모드에서만 보여줍니다."
            },
            "formula": [
                {
                    "title": "Indexer summary",
                    "note": "R=4 CSA에서 compressed block 후보를 sparse하게 고릅니다."
                }
            ]
        },
        "sparse-attn": {
            "summary": "Runs attention over Q and the selected KV set produced by SWA, CSA, or HCA mode.",
            "notes": [
                "N은 선택된 KV entry 수이며 mode에 따라 <=128, 128+topK, 128+T/R로 달라진다."
            ],
            "details": {
                "why": "SWA, CSA, HCA의 후보 선택 차이를 하나의 selected-KV attention kernel로 실행합니다. N은 mode에 따라 최근 window만 보거나, window에 topK compressed entry를 더하거나, valid compressed block 전체를 더하는 식으로 달라지며, 같은 kernel 추상화 안에서 long-context compute와 memory를 줄이고 local 정보는 유지합니다.",
                "runtime": "per-head attn_sink와 online-softmax 스타일 kernel 동작을 포함합니다.",
                "ui": "선택 모드별 후보 수를 <=128, topK+128, T/128+128처럼 보여줍니다."
            },
            "formula": [
                {
                    "title": "Attention logits"
                },
                {
                    "title": "Weighted value sum",
                    "note": "선택된 KV entry만 gather해 online-softmax kernel에서 계산합니다."
                }
            ]
        },
        "o-proj": {
            "summary": "wo_a group low-rank projection 후 wo_b로 D차원 복원.",
            "notes": [
                "각 group은 8 heads * 512 dims."
            ],
            "details": {
                "why": "sparse attention head 결과를 residual stream 크기로 되돌리면서 grouped low-rank output projection으로 비용을 낮춥니다. 각 group은 8 heads * 512 dims 단위로 접히며, wo_a/wo_b 구조가 projection 비용을 낮추고 inverse RoPE로 보정된 value semantics를 residual stream에 맞춥니다.",
                "runtime": "KV가 shared라서 grouped low-rank output projection 전에 inverse RoPE를 적용합니다.",
                "ui": "shared-KV attention 이후 value semantics를 보정하는 지점입니다."
            },
            "formula": [
                {
                    "title": "Grouped output projection",
                    "note": "head output을 group low-rank projection으로 D차원 hidden stream에 복원합니다."
                }
            ]
        },
        "mhc-ffn": {
            "summary": "MoE FFN 앞뒤의 mHC controller/data path.",
            "notes": [
                "Attention mHC와 별도 파라미터 세트."
            ],
            "details": {
                "why": "MoE sublayer도 attention과 같은 mHC read/write 구조로 감싸 residual lane 안정성을 유지합니다. FFN 쪽 mHC는 attention mHC와 목적은 같지만 별도 parameter set을 쓰므로, MoE 입력 읽기와 MoE output writeback 정책을 독립적으로 학습할 수 있습니다.",
                "runtime": "attention mHC와 별도의 FFN mHC parameter set을 씁니다."
            },
            "formula": [
                {
                    "title": "MoE mHC wrapper"
                }
            ]
        },
        "moe": {
            "summary": "Gate가 top-6 experts를 고르고 routed/shared experts를 결합한다.",
            "notes": [
                "First 3 layers는 hash routing."
            ],
            "details": {
                "why": "전체 parameter capacity는 크게 유지하되 token당 활성화 expert 수를 top-6로 제한해 inference compute를 줄입니다. routed experts는 조건부로 선택되고 shared expert는 항상 더해지며, 첫 3개 layer에서는 score top-k 대신 hash routing을 사용합니다.",
                "runtime": "공개 구현에서는 모든 decoder block이 MoE FFN을 씁니다.",
                "ui": "shared expert는 항상 active, routed expert는 conditional로 그립니다."
            },
            "formula": [
                {
                    "title": "MoE summary",
                    "note": "routed expert top-6과 always-on shared expert를 합칩니다."
                }
            ]
        },
        "gate": {
            "summary": "sqrtsoftplus score, hash/top-k selection, weight normalize.",
            "notes": [
                "bias는 selection score에만 적용."
            ],
            "details": {
                "why": "token별로 필요한 expert subset을 선택해 sparse FFN을 구성합니다. 초반에는 token-id 기반 routing으로 싸게 처리하고, 이후에는 hidden representation 기반 top-k routing을 쓰며, selection bias는 expert 선택에만 쓰고 최종 weight 계산은 bias 없는 score에서 다시 가져옵니다.",
                "runtime": "초반 hash layer는 token id로 expert id를 고르고, 이후 layer는 score top-k로 고릅니다."
            },
            "formula": [
                {
                    "title": "Routing abstraction"
                }
            ]
        },
        "routed-experts": {
            "summary": "선택된 expert별 FP4 SwiGLU FFN.",
            "notes": [
                "silu(w1(x)) * w3(x) 후 w2."
            ],
            "details": {
                "why": "선택된 expert별로 token rows를 모아 FP4 SwiGLU FFN을 실행합니다. 각 expert 내부에서는 silu(w1(x)) * w3(x) 후 w2로 hidden dimension을 복원하므로, token당 compute를 제한하면서 큰 expert pool의 parameter capacity를 조건부로 사용할 수 있습니다.",
                "runtime": "Pro는 384 routed expert, token당 6 activated expert를 사용합니다."
            },
            "formula": [
                {
                    "title": "Routed expert FFN"
                }
            ]
        },
        "shared-expert": {
            "summary": "모든 token에 항상 더해지는 shared SwiGLU expert.",
            "notes": [
                "routed expert output에 더해진다."
            ],
            "details": {
                "why": "sparse routing과 무관하게 모든 token에 공통 SwiGLU FFN 경로를 제공하고, 그 결과를 routed expert output에 항상 더합니다. 공통 변환을 흡수하는 역할로 볼 수 있지만, routed expert load를 실제로 얼마나 완화하는지는 공개 코드만으로 단정하기 어렵습니다.",
                "runtime": "모든 token에 대해 계산되고 routed expert 누적값 뒤에 더해집니다.",
                "ui": "routing gate 뒤가 아니라 routed expert와 병렬인 always-on path로 그립니다.",
                "open": "공통 변환을 흡수해 routed expert 부담을 줄일 수 있다는 해석은 가능하지만, 공개 inference 코드만으로 load-balance 메커니즘이라고 단정할 수는 없습니다."
            },
            "formula": [
                {
                    "title": "Shared expert",
                    "note": "routing과 무관하게 모든 token에서 계산됩니다."
                }
            ]
        },
        "hc-post-moe": {
            "summary": "MoE residual lane mixing 결과와 MoE output injection을 합쳐 다음 block residual lanes를 만든다.",
            "notes": [
                "다음 block input이 된다."
            ],
            "details": {
                "why": "MoE residual mixing과 output injection을 합쳐 다음 decoder block이 받을 [B,S,4,D] residual state를 만듭니다. 이 writeback 결과가 다음 block input이 되므로, sparse expert update와 기존 residual lane transport가 같은 state로 정리됩니다."
            },
            "formula": [
                {
                    "title": "MoE writeback"
                }
            ]
        },
        "head": {
            "summary": "Runs after the full decoder stack, collapsing HC lanes and projecting only the last token to vocabulary logits.",
            "notes": [
                "공식 path는 x[:, -1]만 logits 계산."
            ],
            "details": {
                "why": "모든 decoder layer가 끝난 뒤 mHC lane collapse, final norm, last-token vocab projection을 output-only stage로 묶습니다. 공식 path는 x[:, -1]만 logits 계산에 사용하므로, LM head가 매 layer나 모든 token에서 반복 실행되는 구조가 아닙니다.",
                "runtime": "공식 get_logits는 x[:, -1]만 사용하므로 마지막 token만 projection합니다."
            },
            "formula": [
                {
                    "title": "HC collapse + LM head",
                    "note": "공식 path는 마지막 token만 vocab projection합니다."
                }
            ]
        },
        "mtp": {
            "summary": "Auxiliary next-token prediction branch after the final stack state.",
            "notes": [
                "embedding path와 hidden path를 결합."
            ],
            "details": {
                "why": "최종 stack state 뒤에서 auxiliary next-token prediction을 수행해 학습/추론 보조 경로를 제공합니다. 이 branch는 token embedding path와 hidden path를 결합하지만, MTP branch의 정확한 serving 사용 범위는 공개 코드 해석이 더 필요합니다.",
                "runtime": "embedding/head module을 재사용하고 SWA-only attention mode를 가집니다."
            },
            "formula": [
                {
                    "title": "MTP branch",
                    "note": "보조 next-token prediction block이며 attention mode는 SWA-only로 표시합니다."
                }
            ]
        },
        "logits": {
            "summary": "Vocabulary scores for the final-token path; sampling is outside this graph.",
            "notes": [
                "sampling은 이 그래프 범위 밖."
            ],
            "details": {
                "why": "최종 token distribution을 만들기 위한 vocabulary score vector를 제공합니다. 이 노드는 architecture graph의 마지막 score 생성 단계이고, sampling이나 top-p 같은 decoding policy는 모델 구조 밖 runtime 단계입니다.",
                "ui": "sampling, top-p, tool decoding은 이 architecture graph 밖의 단계입니다."
            },
            "formula": [
                {
                    "title": "Vocabulary scores",
                    "note": "sampling 정책은 모델 구조 그래프 밖의 runtime 단계입니다."
                }
            ]
        },
        "hc-flatten": {
            "summary": "controller path용으로 lane 축과 hidden 축을 flatten.",
            "details": {
                "why": "controller path가 4개 lane 전체를 동시에 보고 현재 token의 read/write/mix coefficient를 만들게 합니다. 이 flatten은 coefficient 생성 경로이며, data path를 직접 키우지 않고 lane 선택만 동적으로 바꾸게 해줍니다.",
                "ui": "실제 attention data path가 아니라 coefficient 생성으로 들어가는 control flow로 그립니다."
            },
            "formula": [
                {
                    "title": "Controller flatten",
                    "note": "controller path가 4개 lane을 한 번에 보도록 lane 축을 hidden 축으로 합칩니다."
                }
            ]
        },
        "hc-controller": {
            "summary": "hc_fn linear로 mHC controller logits를 생성한다.",
            "notes": [
                "rsqrt normalization factor를 곱한다."
            ],
            "details": {
                "why": "token별 residual lane 상태에서 pre, post, comb를 한 번에 예측하는 작은 controller를 둡니다. flatten된 lane에는 controller linear 전후의 RMS scale 정규화가 들어가며, 24-channel 출력은 attention 계산 자체와 residual mixing 정책을 분리해 안정성과 표현력을 함께 확보합니다.",
                "runtime": "flatten된 lane은 controller linear 출력 전후로 RMS scale 정규화가 들어갑니다.",
                "ui": "pre, post, comb 세 갈래로 split되는 지점으로 보여줍니다."
            },
            "formula": [
                {
                    "title": "Controller linear",
                    "note": "24개 출력은 pre 4개, post 4개, comb 16개로 split됩니다."
                }
            ]
        },
        "hc-split": {
            "summary": "mixes의 24 channel을 pre 4, post 4, comb 16으로 나눈다.",
            "notes": [
                "공식 kernel index: 0:4 pre, 4:8 post, 8:24 comb."
            ],
            "details": {
                "why": "controller output을 read coefficient, write coefficient, residual lane mixing matrix라는 서로 다른 의미의 텐서로 명확히 분해합니다. 공식 kernel index 기준으로 0:4는 pre, 4:8은 post, 8:24는 comb이므로 이후 edge shape과 data dependency가 pre/post/comb로 깔끔하게 나뉩니다."
            },
            "formula": [
                {
                    "title": "Split indices"
                }
            ]
        },
        "hc-pre-sigmoid": {
            "summary": "read weight를 scaled sigmoid와 eps로 만든다.",
            "notes": [
                "pre는 hc_read에서 lane weighted sum에 쓰인다."
            ],
            "details": {
                "why": "read weight를 양수 범위로 제한해 lane weighted sum이 무제한 부호/scale로 흔들리지 않게 합니다. scaled sigmoid와 epsilon으로 만든 pre coefficient는 hc_read에서 lane 축 가중합에 직접 쓰이므로, attention 입력 stream의 scale을 안정적으로 잡아 줍니다."
            },
            "formula": [
                {
                    "title": "Pre weights",
                    "note": "read path에서 residual lanes를 하나의 hidden stream으로 읽는 coefficient입니다."
                }
            ]
        },
        "hc-post-sigmoid": {
            "summary": "sublayer output을 4개 lane에 주입할 post weight를 만든다.",
            "notes": [
                "post에는 pre와 달리 +eps가 없고 2배 sigmoid를 쓴다."
            ],
            "details": {
                "why": "sublayer output을 4개 lane에 주입할 때 lane별 강도를 bounded coefficient로 제어합니다. pre와 달리 +eps 없이 2 * sigmoid 형태를 쓰며, 이 값이 attention output writeback의 scale과 lane placement를 제한합니다."
            },
            "formula": [
                {
                    "title": "Post weights",
                    "note": "sublayer output을 4개 residual lane에 주입하는 coefficient입니다."
                }
            ]
        },
        "hc-comb-softmax": {
            "summary": "comb logits를 4x4 matrix로 보고 row softmax + eps를 적용한다.",
            "notes": [
                "Sinkhorn 반복 전 초기 row-normalized matrix."
            ],
            "details": {
                "why": "4x4 comb raw logits를 먼저 row-wise probability-like matrix로 만들어 Sinkhorn 반복의 출발점을 안정화합니다. 이 단계의 출력은 Sinkhorn 반복 전 초기 row-normalized matrix이며, epsilon까지 더해 다음 row/column normalization이 다룰 matrix를 안정적으로 만듭니다."
            },
            "formula": [
                {
                    "title": "Comb row softmax"
                }
            ]
        },
        "hc-comb-sinkhorn": {
            "summary": "row/column normalization을 반복해 comb를 doubly stochastic에 가깝게 만든다.",
            "notes": [
                "residual lane mixing에 들어가는 최종 4x4 matrix."
            ],
            "details": {
                "why": "comb matrix를 row/column 양쪽에서 반복 정규화해 residual lane transport가 특정 lane으로 붕괴하지 않게 합니다. Sinkhorn을 거친 최종 4x4 matrix가 실제 residual lane mixing에 들어가며, 이 doubly stochastic에 가까운 lane transport가 gradient와 signal propagation을 안정화하는 역할을 합니다."
            },
            "formula": [
                {
                    "title": "Sinkhorn iterations",
                    "note": "row/column normalization을 반복해 comb를 doubly stochastic에 가깝게 만듭니다."
                }
            ]
        },
        "hc-sinkhorn": {
            "summary": "TileLang kernel이 pre/post/comb를 나누고 comb를 Sinkhorn normalize.",
            "notes": [
                "comb는 [B,S,4,4]."
            ],
            "details": {
                "why": "controller logit을 실제 data path coefficient로 바꾸면서 comb에는 doubly stochastic 제약을 걸어 residual lane 이동을 안정화합니다. split과 Sinkhorn을 거치면 pre/post는 [B,S,4] vector로, comb는 [B,S,4,4] matrix로 정리되어 read/write/mix 계수가 안정적인 data path coefficient가 됩니다.",
                "runtime": "comb는 hc_split_sinkhorn kernel에서 row/column normalization을 반복해 doubly stochastic에 가깝게 만듭니다.",
                "ui": "pre/post는 vector, comb는 4x4 heatmap처럼 보여주면 좋습니다.",
                "open": "mHC constraint가 학습 안정성에 기여하는 정확한 분석은 technical report와 같이 대조해야 합니다."
            },
            "formula": [
                {
                    "title": "Split"
                },
                {
                    "title": "Doubly stochastic mixing",
                    "note": "row/column 합을 안정화해 layer 간 residual gradient transport를 덜 흔들리게 만드는 목적입니다."
                }
            ]
        },
        "hc-read": {
            "summary": "pre 가중합으로 sublayer input을 만든다.",
            "notes": [
                "sum(pre * X) over lane axis."
            ],
            "details": {
                "why": "4-lane residual state를 attention이 처리할 단일 hidden stream으로 읽습니다. 연산은 lane axis에 대한 sum(pre * X)이고, pre coefficient로 lane weighted sum을 만들기 때문에 attention compute를 4배로 늘리지 않고 mHC 표현력을 유지합니다.",
                "ui": "controller coefficient가 data path에 처음 적용되는 지점입니다."
            },
            "formula": [
                {
                    "title": "Read projection",
                    "note": "4-lane residual을 attention이 받을 단일 hidden stream으로 읽습니다."
                }
            ]
        },
        "attn-residual-mix": {
            "summary": "comb matrix가 기존 4개 residual lane을 token별로 서로 섞는다.",
            "notes": [
                "이 노드가 attention writeback의 핵심 residual lane mixing이다."
            ],
            "details": {
                "why": "attention output과 별개로 기존 residual lane 자체를 comb matrix로 운반하는 attention writeback의 핵심 residual lane mixing입니다. identity residual보다 lane 간 정보 이동을 허용하므로, attention output injection과 분리된 안정적인 residual transport 항으로 볼 수 있습니다.",
                "ui": "attention output injection과 분리해서 보여줘야 합니다."
            },
            "formula": [
                {
                    "title": "Residual lane mixing",
                    "note": "기존 residual lane을 4x4 transport matrix로 섞습니다."
                }
            ]
        },
        "attn-post-inject": {
            "summary": "attention output을 post weights로 4개 lane에 주입한다.",
            "notes": [
                "residual lane mixing과 별도 항으로 더해진다."
            ],
            "details": {
                "why": "attention output 하나를 post coefficient로 4개 lane에 배분해 sublayer 결과가 어느 lane에 남을지 조절합니다. 이 항은 residual lane mixing과 별도로 더해지는 output injection이며, post write path가 residual update의 scale과 lane placement를 제어합니다.",
                "ui": "하나의 attention stream이 4개 lane으로 fan-out되는 구조입니다."
            },
            "formula": [
                {
                    "title": "Attention injection",
                    "note": "단일 attention output을 4개 lane으로 다시 분배합니다."
                }
            ]
        },
        "hc-write": {
            "summary": "comb * residual과 post * attention output을 합쳐 다음 residual lanes를 만든다.",
            "notes": [
                "writeback = residual lane mixing + sublayer output injection."
            ],
            "details": {
                "why": "comb로 운반한 기존 residual lane mixing 결과와 post로 주입한 attention output injection을 합쳐 다음 residual lane state를 만듭니다. 즉 writeback은 residual lane mixing + sublayer output injection이며, 이 조합으로 residual 안정성과 attention update 표현력을 동시에 확보합니다."
            },
            "formula": [
                {
                    "title": "Attention writeback",
                    "note": "residual lane mixing과 attention output injection을 더해 다음 state를 만듭니다."
                }
            ]
        },
        "ffn-hc-flatten": {
            "summary": "MoE/FFN용 mHC controller 입력을 만들기 위해 residual lanes를 flatten한다.",
            "notes": [
                "attention mHC와 별도의 controller path."
            ],
            "details": {
                "why": "MoE controller가 현재 4-lane residual 상태 전체를 보고 read/write/mix coefficient를 만들게 합니다. attention mHC와 별도의 controller path를 두어 MoE 입력과 residual update를 독립적으로 조절합니다."
            },
            "formula": [
                {
                    "title": "MoE controller flatten"
                }
            ]
        },
        "ffn-hc-controller": {
            "summary": "MoE/FFN 앞뒤에서 쓸 pre/post/comb logits를 생성한다.",
            "notes": [
                "attention mHC parameter set과 분리된다."
            ],
            "details": {
                "why": "MoE sublayer 전용 pre/post/comb logits를 attention과 독립적으로 생성합니다. 구조는 attention controller와 같지만 parameter set이 분리되어 있어, sublayer별 lane mixing 정책을 따로 학습할 수 있습니다.",
                "ui": "attention controller와 같은 구조지만 독립 parameter라고 표시합니다."
            },
            "formula": [
                {
                    "title": "MoE controller linear"
                }
            ]
        },
        "ffn-hc-split": {
            "summary": "MoE-side mixes를 pre/post/comb logits로 나눈다.",
            "notes": [
                "attention-side와 같은 kernel 구조지만 별도 parameter set을 쓴다."
            ],
            "details": {
                "why": "MoE-side controller output을 read, write, comb 텐서로 분해해 router/expert data path에 각각 다른 coefficient를 공급합니다. attention-side와 같은 kernel 구조를 쓰지만 별도 parameter set을 사용하므로, MoE path의 dependency가 read/write/comb로 독립적으로 나뉩니다."
            },
            "formula": [
                {
                    "title": "MoE split indices"
                }
            ]
        },
        "ffn-hc-pre-sigmoid": {
            "summary": "MoE read weight를 scaled sigmoid와 eps로 만든다.",
            "details": {
                "why": "MoE가 읽을 hidden stream의 lane weight를 bounded positive coefficient로 만듭니다. bounded positive coefficient가 router와 expert 입력 scale을 안정화합니다."
            },
            "formula": [
                {
                    "title": "MoE pre weights"
                }
            ]
        },
        "ffn-hc-post-sigmoid": {
            "summary": "MoE output injection weight를 만든다.",
            "details": {
                "why": "MoE output을 4개 residual lane에 주입할 강도를 bounded coefficient로 제어합니다. bounded coefficient로 sparse FFN output scale을 안정화합니다."
            },
            "formula": [
                {
                    "title": "MoE post weights"
                }
            ]
        },
        "ffn-hc-comb-softmax": {
            "summary": "MoE-side comb logits에 row softmax + eps를 적용한다.",
            "details": {
                "why": "MoE residual lane mixing의 raw 4x4 logits를 Sinkhorn 가능한 row-normalized matrix로 초기화합니다. row-normalized 초기값으로 lane mixing의 수치 안정성을 높입니다."
            },
            "formula": [
                {
                    "title": "MoE comb row softmax"
                }
            ]
        },
        "ffn-hc-comb-sinkhorn": {
            "summary": "MoE-side residual lane mixing matrix를 Sinkhorn normalize한다.",
            "details": {
                "why": "MoE writeback의 comb matrix도 doubly stochastic에 가깝게 만들어 lane 붕괴를 줄입니다. doubly stochastic에 가까운 comb가 residual gradient와 signal을 안정화합니다."
            },
            "formula": [
                {
                    "title": "MoE Sinkhorn"
                }
            ]
        },
        "ffn-hc-sinkhorn": {
            "summary": "FFN용 pre/post/comb를 나누고 comb를 Sinkhorn normalize한다.",
            "notes": [
                "여기서 MoE writeback residual lane mixing weights가 나온다."
            ],
            "details": {
                "why": "MoE sublayer의 controller output을 data path coefficient로 변환하면서 attention 쪽과 같은 mHC 제약을 적용합니다. 여기서 MoE writeback에 들어갈 residual lane mixing weights가 나오며, attention 쪽과 동일한 4x4 comb matrix 제약으로 MoE에서도 residual lane stability가 유지됩니다.",
                "ui": "attention 쪽과 동일한 4x4 comb matrix 시각화를 재사용합니다."
            },
            "formula": [
                {
                    "title": "MoE split + Sinkhorn"
                }
            ]
        },
        "hc-pre-moe": {
            "summary": "MoE FFN에 들어갈 hidden state를 mHC pre weights로 lane 축에서 읽는다.",
            "notes": [
                "sum(pre * residual) over lane axis."
            ],
            "details": {
                "why": "router와 experts가 처리할 단일 hidden stream을 4-lane residual에서 읽어냅니다. 연산은 lane axis에 대한 sum(pre * residual)이고, lane을 먼저 단일 hidden stream으로 접기 때문에 MoE compute를 4개 lane마다 반복하지 않습니다."
            },
            "formula": [
                {
                    "title": "MoE read projection"
                }
            ]
        },
        "ffn-residual-mix": {
            "summary": "MoE writeback에서 기존 residual lanes를 comb matrix로 다시 섞는다.",
            "notes": [
                "attention writeback과 같은 residual lane mixing 구조지만 별도 mHC weights를 쓴다."
            ],
            "details": {
                "why": "MoE sublayer를 지나도 기존 residual lane 정보를 comb matrix로 계속 운반합니다. attention writeback과 같은 residual lane mixing 구조지만 별도 mHC weights를 쓰므로, sparse expert update와 별개로 MoE 쪽 residual signal을 안정적으로 보존합니다."
            },
            "formula": [
                {
                    "title": "MoE residual mixing"
                }
            ]
        },
        "ffn-post-inject": {
            "summary": "MoE output을 post weights로 4개 residual lane에 주입한다.",
            "details": {
                "why": "MoE output을 post coefficient로 lane별 분배해 sparse FFN 결과를 residual lane state에 주입합니다. post coefficient가 expert output scale과 lane placement를 제어합니다."
            },
            "formula": [
                {
                    "title": "MoE output injection"
                }
            ]
        },
        "q-wqa": {
            "summary": "Query low-rank A projection.",
            "details": {
                "why": "hidden state를 먼저 작은 query latent로 투영해 query 계산의 중간 rank를 제한합니다. q_lora_rank로 query 중간 rank를 제한해 projection compute와 parameter 배치를 줄입니다.",
                "runtime": "Pro의 q_lora_rank는 1536입니다.",
                "ui": "LoRA식 low-rank A projection으로 표시합니다."
            },
            "formula": [
                {
                    "title": "Query A projection"
                }
            ]
        },
        "q-norm": {
            "summary": "Low-rank query RMSNorm.",
            "details": {
                "why": "main Q expansion과 Lightning indexer가 공유하는 low-rank query latent의 scale을 분기 전에 맞춥니다. 두 path가 같은 normalized latent를 쓰므로 attention score와 retrieval score의 scale 민감도를 줄이는 배치로 볼 수 있습니다.",
                "runtime": "main query expansion과 CSA Lightning indexer query path가 둘 다 이 출력을 씁니다.",
                "ui": "fork 지점으로 보여줍니다."
            },
            "formula": [
                {
                    "title": "RMSNorm",
                    "note": "q_lora_rank 축을 기준으로 scale을 맞춘 뒤 main query와 indexer query가 이 출력을 공유합니다."
                }
            ]
        },
        "q-wqb": {
            "summary": "Query head expansion.",
            "details": {
                "why": "안정화된 query latent를 실제 attention head별 512-dim query로 확장합니다. wq_b가 low-rank compute 절감 뒤 필요한 head 표현력을 복원합니다.",
                "runtime": "Pro 기준 논리 출력은 H*512 = 128*512 = 65536 channel입니다."
            },
            "formula": [
                {
                    "title": "Query B projection"
                }
            ]
        },
        "q-reshape": {
            "summary": "Head reshape 후 per-head RMS re-normalization.",
            "details": {
                "why": "flat channel을 head 축으로 재해석하고 per-head scale을 다시 맞춰 attention kernel 입력 형태로 만듭니다. reshape 뒤 per-head renorm을 거쳐 head별 score scale을 맞춥니다.",
                "runtime": "코드에서는 reshape 후 per-head RMS re-normalization도 수행합니다.",
                "ui": "학습 파라미터가 있는 layer라기보다 axis split으로 보여줍니다."
            },
            "formula": [
                {
                    "title": "Head split + renorm",
                    "note": "projection 출력의 channel 축을 head 축과 head dim 축으로 나눕니다."
                }
            ]
        },
        "q-rope": {
            "summary": "마지막 64 dims에 RoPE 적용.",
            "details": {
                "why": "query의 일부 차원에만 위치 위상을 넣어 content dimension과 positional dimension을 분리합니다. 마지막 64 RoPE dim만 position phase를 맡아 shared-KV 설계와 맞물립니다.",
                "runtime": "마지막 64 dim만 RoPE가 적용되고 나머지 448 dim은 content dim으로 남습니다.",
                "ui": "각 head를 448 no-RoPE dim + 64 RoPE dim으로 나눠 보여줍니다."
            },
            "formula": [
                {
                    "title": "RoPE slice",
                    "note": "512차원 중 마지막 64차원에 position phase를 넣습니다."
                }
            ]
        },
        "kv-wkv": {
            "summary": "Shared KV projection.",
            "details": {
                "why": "hidden stream에서 shared key/value entry 하나를 생성합니다. single KV projection이 per-head KV cache를 피하게 해 memory를 줄입니다.",
                "runtime": "K에는 RoPE가 필요하지만 V에는 절대 위치가 섞이면 이상하므로 뒤에서 inverse RoPE로 보정합니다."
            },
            "formula": [
                {
                    "title": "Shared KV projection"
                }
            ]
        },
        "kv-norm": {
            "summary": "KV RMSNorm.",
            "details": {
                "why": "RoPE, quantization, compression, cache write로 갈라지기 전에 shared KV scale을 맞춥니다. 여러 downstream path가 같은 normalized KV를 쓰기 때문에 RoPE, compression, cache write의 scale 민감도가 줄어듭니다."
            },
            "formula": [
                {
                    "title": "KV RMSNorm",
                    "note": "RoPE, compressor, cache write 전에 shared KV scale을 맞춥니다."
                }
            ]
        },
        "kv-rope-quant": {
            "summary": "RoPE dims는 BF16, non-RoPE dims는 FP8 simulation.",
            "details": {
                "why": "key score에 필요한 RoPE 정보는 보존하면서 non-RoPE content는 저정밀 표현으로 cache 부담을 낮춥니다. BF16 RoPE dim은 score 안정성을 맡고, FP8-sim non-RoPE dim은 memory 부담을 낮춥니다.",
                "runtime": "RoPE dim은 BF16으로 유지하고 non-RoPE dim은 FP8 simulation을 적용합니다.",
                "ui": "448개 quantized content dim과 64개 BF16 RoPE dim으로 나눠 보여줍니다."
            },
            "formula": [
                {
                    "title": "RoPE + quantized content",
                    "note": "content 448 dim은 low precision simulation, RoPE 64 dim은 position-aware key로 남깁니다."
                }
            ]
        },
        "kv-slice": {
            "summary": "shared KV를 content slice와 RoPE slice로 나눈다.",
            "notes": [
                "K score에는 RoPE slice가 필요하고 V sum에는 position phase를 제거한 shared value가 필요하다."
            ],
            "details": {
                "why": "shared KV 안에서 position-aware key slice와 content/value slice를 분리해 K와 V가 서로 다른 의미를 갖게 합니다. K score에는 RoPE slice가 필요하지만 V sum에는 position phase가 없는 shared value가 필요하므로, 448/64 split으로 RoPE를 필요한 key slice에 집중시킵니다.",
                "runtime": "RoPE는 마지막 64 dim에만 들어가고, 448 dim은 content/value semantics를 유지합니다."
            },
            "formula": [
                {
                    "title": "Content / RoPE split",
                    "note": "512-dim shared KV 내부에서 position-aware key slice와 content/value slice를 분리해서 생각합니다."
                }
            ]
        },
        "window-topk": {
            "summary": "Sliding-window branch over the most recent 128 tokens; active in every attention mode.",
            "details": {
                "why": "compressed block retrieval과 무관하게 최근 128개 token을 항상 후보로 넣어 local causality와 근접 문맥을 보존합니다. 이 SWA path가 long-context 압축에서 생기는 local detail 손실을 보완합니다.",
                "runtime": "CSA, HCA, SWA-only 모든 모드에서 존재합니다.",
                "ui": "score 기반 top-k가 아니라 sliding-window index set입니다."
            },
            "formula": [
                {
                    "title": "Sliding window ids",
                    "note": "score top-k가 아니라 최근 local token index set입니다."
                }
            ]
        },
        "swa-prefill-write": {
            "summary": "prefill chunk에서 최근 128개 uncompressed KV를 window cache에 채운다.",
            "notes": [
                "긴 prompt에서는 window 밖 token은 compressed path로만 남는다."
            ],
            "details": {
                "why": "prefill 중에도 최근 local token을 압축하지 않고 보존해 short-range attention 품질을 지킵니다. 긴 prompt에서는 window 밖 token을 compressed path가 담당하고, 128-window cache는 compressed memory의 정보 손실을 local window로 보완합니다.",
                "runtime": "SWA window 밖의 오래된 token은 compressed path가 담당합니다."
            },
            "formula": [
                {
                    "title": "Prefill window write",
                    "note": "prefill에서는 chunk 끝의 local window만 uncompressed cache에 남깁니다."
                }
            ]
        },
        "swa-decode-write": {
            "summary": "decode에서는 최근 KV를 128-slot ring buffer에 rolling write한다.",
            "notes": [
                "runtime cache manager가 오래된 uncompressed entry를 덮어쓴다."
            ],
            "details": {
                "why": "decode에서 새 token KV를 고정 크기 128-slot ring buffer에 갱신해 uncompressed local cache memory를 상수로 유지합니다. runtime cache manager가 start_pos % 128 위치에 쓰면서 오래된 uncompressed entry를 자연스럽게 덮어쓰므로, bounded memory로 local attention을 유지합니다.",
                "runtime": "slot은 start_pos % 128이며 오래된 uncompressed KV는 자연스럽게 덮어씁니다."
            },
            "formula": [
                {
                    "title": "Decode ring write",
                    "note": "decode step마다 128-slot circular buffer의 한 칸을 갱신합니다."
                }
            ]
        },
        "cache-layout": {
            "summary": "SWA prefix와 compressed suffix를 attention kernel이 읽는 하나의 논리 cache로 붙인다.",
            "details": {
                "why": "SWA prefix와 compressed suffix를 하나의 logical cache로 붙여 selected id gather를 단순하게 만듭니다. 128 offset 구조가 sparse attention kernel의 indexing을 단순화합니다.",
                "runtime": "compressed id는 128 offset 뒤쪽 suffix로 매핑됩니다."
            },
            "formula": [
                {
                    "title": "Logical concat",
                    "note": "attention index는 SWA prefix와 compressed suffix를 같은 cache id 공간에서 봅니다."
                }
            ]
        },
        "attn-selected": {
            "summary": "The final KV set merges the local SWA window with the active compressed path: CSA top-k blocks, HCA all compressed blocks, or no compressed blocks for MTP.",
            "details": {
                "why": "attention이 실제로 읽을 KV 후보를 mode별로 합성합니다. CSA/HCA/SWA-only 선택 규칙을 한 노드에서 합쳐 layer mode별 compute budget을 명확히 제한합니다.",
                "runtime": "CSA = SWA ids + indexer topK, HCA = SWA ids + 모든 valid c128a ids, SWA-only = SWA ids만 사용합니다.",
                "ui": "index-set union 노드로 보여줍니다."
            },
            "formula": [
                {
                    "title": "KV index union",
                    "note": "CSA는 indexer top-k, HCA는 valid compressed block 전체, MTP는 SWA만 사용합니다."
                }
            ]
        },
        "comp-wkv": {
            "summary": "Compression candidate KV projection.",
            "details": {
                "why": "pooling될 token별 KV 후보를 만들어 압축이 raw hidden이 아니라 learned projection 위에서 일어나게 합니다. learned compressor projection을 거쳐 compressed entry 표현력을 보존합니다.",
                "runtime": "c4a overlap에서는 Coff=2, c128a에서는 Coff=1입니다."
            },
            "formula": [
                {
                    "title": "Compressor KV candidate"
                }
            ]
        },
        "comp-wgate": {
            "summary": "Softmax pooling score projection.",
            "details": {
                "why": "어떤 token/channel 정보를 compressed entry에 더 반영할지 learned score로 결정합니다. wgate와 softmax pooling으로 단순 평균보다 더 선택적으로 정보를 모읍니다.",
                "runtime": "softmax 전에 learned ape가 score에 더해집니다.",
                "ui": "weighted sum의 weight가 어디서 나오는지 보여주는 노드입니다."
            },
            "formula": [
                {
                    "title": "Compressor gate score",
                    "note": "softmax pooling weight를 만들기 위한 learned score입니다."
                }
            ]
        },
        "comp-ape": {
            "summary": "pooling score에 compressor-local absolute position embedding을 더한다.",
            "notes": [
                "block 내부 상대 위치를 softmax gate에 알려주는 path다."
            ],
            "details": {
                "why": "compressor-local block 내부 위치 정보를 gate score에 더해 pooling이 순서/위치를 완전히 잃지 않게 합니다. attention RoPE와는 별개로 softmax gate에 block 내부 상대 위치를 알려 주는 path이며, compressed entry의 block-local 위치 민감도를 보완합니다.",
                "runtime": "attention RoPE와 별개인 compressor-local score path입니다."
            },
            "formula": [
                {
                    "title": "APE score add"
                }
            ]
        },
        "comp-cutoff": {
            "summary": "현재 chunk와 이전 tail을 합쳐 완성 block과 남은 remainder를 나눈다.",
            "notes": [
                "decode에서는 대부분 remainder가 누적되다가 block boundary에서만 compressed entry가 생긴다."
            ],
            "details": {
                "why": "현재 chunk와 이전 tail을 합친 뒤 완성된 block만 compressed cache로 보내고 나머지는 tail로 넘깁니다. decode에서는 대부분 remainder가 누적되다가 block boundary에서만 compressed entry가 생기며, 이 state 흐름이 chunking과 decode step 크기에 따른 결과 흔들림을 줄입니다.",
                "runtime": "block boundary에 닿지 않으면 compressed cache write가 발생하지 않을 수 있습니다."
            },
            "formula": [
                {
                    "title": "Full block split",
                    "note": "R개 단위로 완성된 projection만 pooling으로 보내고 remainder는 tail로 남깁니다."
                }
            ]
        },
        "tail-append": {
            "summary": "다음 decode step에서 이어 쓸 미완성 compressor state를 갱신한다.",
            "details": {
                "why": "미완성 remainder를 다음 forward 호출까지 유지해 compression window가 요청 경계를 넘어서도 이어지게 합니다. stateful compressor 흐름으로 runtime chunking artifact를 줄입니다.",
                "runtime": "decode/prefill chunking에 따라 tail 길이는 0 이상 R 미만입니다."
            },
            "formula": [
                {
                    "title": "Persistent tail",
                    "note": "request별 cache state로 다음 token/chunk까지 유지됩니다."
                }
            ]
        },
        "tail-state": {
            "summary": "Buffers tail tokens until a full compressed block can be formed.",
            "details": {
                "why": "decode처럼 token이 조금씩 들어오는 상황에서 아직 R개가 차지 않은 projection을 request state로 보관합니다. block boundary에 도달했을 때만 compressed cache write가 발생하므로 streaming decode에서도 compression window가 끊기지 않습니다. tail state는 streaming compression을 위한 request별 runtime state입니다.",
                "runtime": "c4a는 overlap된 8-token 스타일 compressor state, c128a는 128-token state를 유지합니다.",
                "ui": "projection이 아니라 request별 persistent state로 표시합니다."
            },
            "formula": [
                {
                    "title": "Tail state update",
                    "note": "decode에서 아직 block이 완성되지 않은 token projection을 임시 저장합니다."
                }
            ]
        },
        "comp-block-view": {
            "summary": "projection channel을 pooling이 볼 block/token/value 축으로 재배치한다.",
            "notes": [
                "CSA는 overlap 후 span=8, HCA는 span=128로 해석한다."
            ],
            "details": {
                "why": "projection 결과를 pooling kernel이 읽기 쉬운 block/token/value 축으로 바꿔 c4a overlap과 c128a block 처리를 같은 추상화에 올립니다. CSA에서는 overlap 후 span=8처럼 해석되고 HCA에서는 128-token block으로 해석되므로, reshape와 overlap 처리가 compressor compute를 block/token/value 축에 맞춥니다.",
                "runtime": "CSA는 overlap 때문에 span이 8처럼 보이고 HCA는 128-token block이 됩니다."
            },
            "formula": [
                {
                    "title": "Block view"
                }
            ]
        },
        "overlap-transform": {
            "summary": "R=4 CSA path overlaps previous and current chunks before gated pooling.",
            "details": {
                "why": "c4a에서 stride 4로 cache entry를 만들면서도 pooling span은 8 token을 보게 해 block 경계 정보 손실을 줄입니다. overlap transform은 block boundary 정보 손실을 완화하는 쪽으로 작동합니다.",
                "runtime": "이전 block half와 현재 block half를 gated pooling 전에 재배치하며 boundary는 0 또는 -inf padding으로 처리됩니다.",
                "ui": "native span과 anchor position을 분리해서 보여줍니다."
            },
            "formula": [
                {
                    "title": "CSA overlap span",
                    "note": "c4a는 stride 4이지만 pooling span은 8-token overlap으로 볼 수 있습니다."
                }
            ]
        },
        "gated-pool": {
            "summary": "Pools R tokens into a compressed KV block using softmax weights.",
            "details": {
                "why": "여러 native token을 learned softmax weight로 하나의 compressed KV entry에 모읍니다. compressor pooling이 memory를 줄이면서 중요한 token/channel 정보를 더 남깁니다.",
                "ui": "c4a는 8-to-1, c128a는 128-to-1 pooling처럼 보여줍니다."
            },
            "formula": [
                {
                    "title": "Softmax-gated pooling",
                    "note": "여러 native token을 하나의 compressed KV entry로 줄입니다."
                }
            ]
        },
        "comp-anchor": {
            "summary": "compressed entry에 적용할 RoPE anchor position을 만든다.",
            "notes": [
                "block 내부 token마다 position을 따로 주지 않고 대표 anchor를 쓴다."
            ],
            "details": {
                "why": "compressed block 전체에 하나의 대표 position을 부여해 RoPE와 causal indexing이 가능한 cache entry로 만듭니다. block 내부 token마다 position을 따로 주지 않고 stride별 anchor id를 쓰기 때문에, compressed entry는 cache에서 다루기 쉬워지는 대신 block 내부 세부 위치는 일부 손실될 수 있습니다.",
                "runtime": "c4a는 stride 4 anchor, c128a는 stride 128 anchor를 씁니다."
            },
            "formula": [
                {
                    "title": "Anchor id",
                    "note": "c4a는 0,4,8,..., c128a는 0,128,256,... anchor를 씁니다."
                }
            ]
        },
        "comp-norm-rope": {
            "summary": "compressed KV norm 후 compressed position RoPE.",
            "details": {
                "why": "compressed KV entry도 attention key로 쓰일 수 있게 scale과 position phase를 맞춥니다. norm과 anchor RoPE가 compressed attention score를 안정화합니다.",
                "runtime": "prefill anchor는 R 간격 위치를 쓰고, decode에서는 block 완성 시 start_pos + 1 - R을 anchor로 씁니다.",
                "ui": "anchor position이라는 용어를 명시적으로 보여줍니다."
            },
            "formula": [
                {
                    "title": "Compressed norm + anchor RoPE",
                    "note": "compressed block 내부 token 위치가 아니라 anchor position을 사용합니다."
                }
            ]
        },
        "comp-cache-slot": {
            "summary": "compressed block id를 SWA prefix 뒤 cache slot으로 매핑한다.",
            "details": {
                "why": "compressed block id를 SWA prefix 뒤 logical cache slot으로 변환해 attention gather가 같은 id 체계를 쓰게 합니다. 128 offset 구조가 runtime indexing을 단순화합니다.",
                "runtime": "SWA prefix 128개 뒤에 compressed suffix가 이어집니다."
            },
            "formula": [
                {
                    "title": "Slot map",
                    "note": "attention cache id 공간에서 compressed entry는 SWA 128-slot 뒤에 배치됩니다."
                }
            ]
        },
        "comp-cache-write": {
            "summary": "Writes compressed KV after the live sliding-window cache region.",
            "details": {
                "why": "완성된 compressed KV를 long-context memory 영역에 기록해 오래된 문맥을 싼 entry로 유지합니다. compressed cache write가 long-context memory를 줄입니다.",
                "runtime": "논리 compressed index는 대략 start_pos // R이며 serving runtime은 이를 page로 다시 매핑할 수 있습니다."
            },
            "formula": [
                {
                    "title": "Compressed cache write",
                    "note": "SWA 영역 뒤쪽 compressed cache slot에 기록합니다."
                }
            ]
        },
        "idx-q": {
            "summary": "CSA-only indexer query projection from q_norm.",
            "details": {
                "why": "main attention Q를 그대로 쓰지 않고 retrieval 전용 64-head 128-dim query를 만들어 index scoring 비용을 낮춥니다. q_norm latent에서 retrieval query를 파생해 compute를 줄입니다.",
                "runtime": "최종 main Q head가 아니라 q_norm latent에서 파생됩니다."
            },
            "formula": [
                {
                    "title": "Indexer query projection"
                }
            ]
        },
        "idx-rope": {
            "summary": "Lightning index query에 position phase를 넣는다.",
            "details": {
                "why": "retrieval query도 현재 token position을 반영해 compressed block과의 시간적 관련성을 평가하게 합니다. indexer RoPE로 retrieval score가 query position을 반영합니다."
            },
            "formula": [
                {
                    "title": "Indexer RoPE"
                }
            ]
        },
        "idx-hadamard": {
            "summary": "retrieval score용 query를 cheap orthogonal rotation으로 섞는다.",
            "details": {
                "why": "저렴한 orthogonal mixing으로 index query 표현을 섞어 FP4 ranking path의 표현력을 보완합니다. Hadamard rotation은 cheap retrieval representation의 channel mixing을 보완합니다.",
                "runtime": "main attention projection이 아니라 approximate retrieval representation입니다."
            },
            "formula": [
                {
                    "title": "Hadamard rotation"
                }
            ]
        },
        "idx-fp4": {
            "summary": "indexer score path의 activation을 FP4 형태로 양자화한다.",
            "details": {
                "why": "indexer activation을 FP4로 낮춰 top-k scoring 경로의 memory와 compute를 줄입니다. FP4 activation quantization으로 scoring path의 compute와 memory를 줄입니다.",
                "runtime": "정밀한 attention 값이 아니라 topK 후보 ranking용 표현입니다."
            },
            "formula": [
                {
                    "title": "FP4 query"
                }
            ]
        },
        "idx-rotate": {
            "summary": "index query rotation and FP4 activation quant.",
            "details": {
                "why": "RoPE, Hadamard, FP4를 묶어 cheap retrieval representation을 만듭니다. 이 경로는 accurate attention 대신 후보 ranking을 싸게 만드는 데 집중합니다.",
                "runtime": "RoPE, Hadamard rotation, FP4 activation quantization을 적용합니다.",
                "ui": "main attention이 아니라 approximate retrieval scoring으로 표시합니다."
            },
            "formula": [
                {
                    "title": "Cheap rotated query",
                    "note": "정확한 attention Q가 아니라 retrieval score용 cheap representation입니다."
                }
            ]
        },
        "idx-cache-compress": {
            "summary": "main KV와 별도의 128-dim compressor로 indexer cache entry를 만든다.",
            "details": {
                "why": "main 512-dim compressed KV와 별도로 top-k ranking 전용 128-dim cache를 만들어 retrieval 비용을 낮춥니다. 전용 index compressor가 memory와 score compute를 줄입니다.",
                "runtime": "score 계산용 cache라 value sum에는 직접 들어가지 않습니다."
            },
            "formula": [
                {
                    "title": "Index compressor",
                    "note": "main 512-dim compressed KV와 별개의 128-dim retrieval cache를 만듭니다."
                }
            ]
        },
        "idx-cache-write": {
            "summary": "Lightning score가 조회할 compressed index cache에 기록한다.",
            "details": {
                "why": "새 compressed block에 대응하는 index entry를 별도 retrieval cache에 기록합니다. index cache write가 다음 token들의 top-k 검색을 가능하게 합니다."
            },
            "formula": [
                {
                    "title": "Index cache write"
                }
            ]
        },
        "idx-cache": {
            "summary": "CSA-only compressed index cache used by the Lightning indexer.",
            "details": {
                "why": "Lightning Indexer가 main KV cache를 직접 스캔하지 않고 작은 128-dim entry만 보게 합니다. 작은 index cache만 읽어 long-context retrieval memory bandwidth를 줄입니다.",
                "runtime": "head_dim=128, rotate=true인 indexer 전용 compressor를 씁니다."
            },
            "formula": [
                {
                    "title": "Index cache",
                    "note": "main 512-dim KV cache와 별도의 128-dim indexer cache입니다."
                }
            ]
        },
        "idx-einsum": {
            "summary": "ReLU dot product scores.",
            "details": {
                "why": "query token과 compressed index cache 사이의 후보 block score를 빠르게 계산합니다. ReLU dot score로 attention 전에 후보 공간을 줄입니다.",
                "runtime": "dot product score는 weighted head sum 전에 ReLU를 통과합니다."
            },
            "formula": [
                {
                    "title": "Block score",
                    "note": "candidate compressed block별 retrieval score를 계산합니다."
                }
            ]
        },
        "idx-weight": {
            "summary": "head별 score를 weighted sum.",
            "details": {
                "why": "64개 index head score를 query-dependent weight로 합쳐 block rank score 하나로 만듭니다. weights_proj가 retrieval head 중요도를 token별로 조절합니다.",
                "runtime": "weights_proj(x)가 query-dependent index-head weight를 만듭니다."
            },
            "formula": [
                {
                    "title": "Head weighted sum",
                    "note": "query-dependent head weight로 64개 index head score를 하나로 합칩니다."
                }
            ]
        },
        "idx-mask": {
            "summary": "미래 compressed block을 topK 후보에서 제외한다.",
            "details": {
                "why": "아직 생성되지 않았거나 causal boundary를 넘는 compressed block이 top-k에 들어오지 못하게 합니다. mask가 causal correctness를 지킵니다."
            },
            "formula": [
                {
                    "title": "Causal mask"
                }
            ]
        },
        "idx-topk": {
            "summary": "Applies causal masking and selects top-k compressed block ids for CSA.",
            "details": {
                "why": "c4a compressed cache 중 일부만 attention 후보로 남겨 sparse attention compute를 제한합니다. topK로 남길 compressed block 수를 제한해 long-context compute를 줄입니다.",
                "runtime": "Pro topK=1024, Flash topK=512입니다."
            },
            "formula": [
                {
                    "title": "Causal TopK",
                    "note": "Pro는 K=1024, Flash는 K=512로 표시됩니다."
                }
            ]
        },
        "idx-offset": {
            "summary": "compressed block id를 128-slot SWA prefix 뒤의 cache id로 바꾼다.",
            "details": {
                "why": "top-k block id를 SWA prefix가 붙은 logical cache id로 변환해 gather가 같은 cache space를 쓰게 합니다. offset mapping으로 SWA와 compressed cache의 indexing을 단순화합니다.",
                "runtime": "compressed entry는 SWA 128-slot 뒤에 있으므로 offset이 필요합니다."
            },
            "formula": [
                {
                    "title": "Cache id offset",
                    "note": "selected_ids가 SWA prefix와 compressed suffix를 같은 cache id로 참조하게 맞춥니다."
                }
            ]
        },
        "gate-score": {
            "summary": "linear + sqrtsoftplus expert scores.",
            "details": {
                "why": "expert affinity를 연속 score로 만들어 top-k selection과 route weight 계산의 공통 기반을 제공합니다. hash layer에서도 original score를 gather해 routing weight 계산을 일관되게 유지합니다.",
                "runtime": "hash layer에서도 route weight를 원래 score에서 gather하기 때문에 score 계산 자체는 남아 있습니다.",
                "ui": "hash layer가 모든 scoring을 생략한다고 오해하지 않게 합니다. 생략되는 것은 score top-k selection입니다."
            },
            "formula": [
                {
                    "title": "Expert score",
                    "note": "selection bias는 expert 선택에만 영향을 주고 weight는 원 score에서 gather합니다."
                }
            ]
        },
        "hash-route": {
            "summary": "first 3 layers use tid2eid lookup.",
            "details": {
                "why": "초반 layer에서 아직 hidden representation이 얕을 때 token id 기반 expert prior를 써 routing 결정을 싸게 만듭니다. tid2eid lookup을 쓰지만, frequent token 쏠림 완화 방식은 공개 문서에 남아 있지 않습니다.",
                "runtime": "tid2eid[input_ids]가 token당 6개 expert id를 반환합니다.",
                "ui": "lexical/token-prior routing으로 보여줍니다.",
                "open": "자주 등장하는 token의 expert 쏠림을 어떻게 완화했는지는 checkpoint나 report 분석이 필요합니다."
            },
            "formula": [
                {
                    "title": "Token-id routing",
                    "note": "초반 layer에서는 score top-k가 아니라 input id 기반 expert id table을 사용합니다."
                }
            ]
        },
        "route-bias": {
            "summary": "topK 선택용 score에만 route bias를 더한다.",
            "notes": [
                "weight 계산에는 bias 없는 original score를 다시 gather한다."
            ],
            "details": {
                "why": "expert 선택 확률만 조정하고 실제 mixture weight는 원래 affinity score에서 계산해 output magnitude 왜곡을 줄입니다. bias는 top-k selection score에만 더해지고, weight 계산에는 bias 없는 original score를 다시 gather하므로 selection correction과 output weighting이 분리됩니다.",
                "runtime": "bias는 topK에만 들어가고 normalize weight는 original score gather로 계산합니다."
            },
            "formula": [
                {
                    "title": "Selection score",
                    "note": "bias는 expert id 선택용이며 route weight는 original score에서 다시 gather합니다."
                }
            ]
        },
        "topk-route": {
            "summary": "later layers choose top-6 experts.",
            "details": {
                "why": "후반 layer에서 token representation 기반으로 가장 관련 있는 6개 expert만 활성화합니다. top-k routing이 MoE compute를 줄이면서 conditional capacity를 제공합니다.",
                "runtime": "bias는 selection에만 영향을 주고, route weight는 bias 없는 original score에서 gather합니다."
            },
            "formula": [
                {
                    "title": "Activation routing"
                }
            ]
        },
        "route-score-gather": {
            "summary": "선택된 expert id에 대해 bias 없는 original score를 모은다.",
            "details": {
                "why": "선택된 expert id에 대해 bias 없는 original score를 다시 가져와 mixture weight를 만들게 합니다. original score gather로 selection correction과 output weighting을 분리합니다.",
                "runtime": "초반 hash route에서도 score 계산이 필요한 이유입니다."
            },
            "formula": [
                {
                    "title": "Gather original score"
                }
            ]
        },
        "route-weights": {
            "summary": "gather original scores, normalize, apply route scale.",
            "details": {
                "why": "선택 expert들의 기여도를 normalize하고 route_scale로 전체 MoE output scale을 맞춥니다. route_scale과 normalization이 routing stability와 activation scale을 제어합니다.",
                "runtime": "sqrtsoftplus score를 선택 expert들 사이에서 normalize하고 route_scale을 곱합니다."
            },
            "formula": [
                {
                    "title": "Normalize route weights",
                    "note": "선택된 expert의 original score를 normalize한 뒤 scale을 곱합니다."
                }
            ]
        },
        "expert-counts": {
            "summary": "각 expert가 처리할 token row 수를 센다.",
            "notes": [
                "dispatch kernel sizing과 load 관찰에 필요한 runtime metadata다."
            ],
            "details": {
                "why": "expert별 token row 수를 세어 dispatch/scatter kernel sizing과 병렬 실행 크기를 정하는 runtime metadata를 만듭니다. 이 count는 routing 실행을 관리하는 데 쓰이고, 동시에 expert별 load imbalance를 관찰할 수 있는 지점이 됩니다.",
                "runtime": "load imbalance를 관찰할 수 있는 지점이기도 합니다."
            },
            "formula": [
                {
                    "title": "Expert counts"
                }
            ]
        },
        "expert-dispatch": {
            "summary": "torch.where(indices == expert_id)로 token dispatch.",
            "details": {
                "why": "선택된 expert별로 token rows를 모아 해당 expert weight로만 처리하게 합니다. expert별 token 묶음으로 sparse expert compute를 실제 batch 연산으로 실행할 수 있습니다.",
                "runtime": "코드는 torch.where(indices == expert_id)를 쓰고 parallel 환경에서는 routed output을 all-reduce합니다."
            },
            "formula": [
                {
                    "title": "Token dispatch",
                    "note": "expert id별 token row를 모아 해당 expert weight로 처리합니다."
                }
            ]
        },
        "expert-w1w3": {
            "summary": "SwiGLU의 gate와 up projection.",
            "details": {
                "why": "SwiGLU에 필요한 gate projection과 up projection을 expert별로 계산합니다. gate/up projection이 FFN 표현력을 만듭니다.",
                "runtime": "Pro에서는 expert weight가 FP4일 수 있습니다."
            },
            "formula": [
                {
                    "title": "Gate/up projection"
                }
            ]
        },
        "swiglu": {
            "summary": "clamp 후 silu(gate) * up.",
            "details": {
                "why": "gate와 up activation의 곱으로 expert FFN의 비선형 선택성을 높입니다. SiLU gate와 clamp가 비선형성을 주고 activation 폭을 제한합니다.",
                "runtime": "Pro는 swiglu_limit=10.0으로 gate/up을 clamp한 뒤 silu(gate) * up을 계산합니다."
            },
            "formula": [
                {
                    "title": "SwiGLU",
                    "note": "공개 config의 swiglu_limit을 반영해 gate/up activation을 clamp합니다."
                }
            ]
        },
        "expert-w2": {
            "summary": "expert output projection.",
            "details": {
                "why": "expert intermediate activation을 residual hidden dimension으로 되돌려 routed accumulation에 합칠 수 있게 합니다. down projection이 expert output shape을 hidden dimension으로 복원합니다."
            },
            "formula": [
                {
                    "title": "Down projection"
                }
            ]
        },
        "routed-accum": {
            "summary": "expert output을 원 token row 위치로 scatter-add하고 routing weight를 곱한다.",
            "details": {
                "why": "expert별로 흩어진 결과를 원 token 위치로 되돌리고 route weight를 곱해 하나의 routed output으로 만듭니다. dispatch/scatter 흐름이 sparse compute 결과를 dense token stream으로 복원합니다."
            },
            "formula": [
                {
                    "title": "Scatter weighted sum"
                }
            ]
        },
        "shared-w1w3": {
            "summary": "always-on shared expert의 gate/up projection.",
            "details": {
                "why": "routing과 무관하게 모든 token이 거치는 shared expert의 gate/up projection을 만듭니다. always-on shared expert가 공통 FFN 변환을 제공합니다.",
                "runtime": "routing 결과와 무관하게 모든 token에 대해 실행됩니다."
            },
            "formula": [
                {
                    "title": "Shared gate/up"
                }
            ]
        },
        "shared-swiglu": {
            "summary": "shared expert 내부의 clamp + SiLU gate.",
            "details": {
                "why": "shared expert 내부에서도 routed expert와 같은 SwiGLU 비선형성을 유지합니다. shared path도 routed expert와 같은 비선형 표현력을 유지합니다.",
                "runtime": "routed SwiGLU와 동일하게 clamp와 SiLU gate를 거칩니다."
            },
            "formula": [
                {
                    "title": "Shared SwiGLU"
                }
            ]
        },
        "shared-w2": {
            "summary": "shared expert output projection.",
            "details": {
                "why": "shared expert activation을 hidden dimension으로 복원해 routed output과 더할 수 있게 합니다. shared output을 routed output과 더할 수 있게 shape을 맞춥니다."
            },
            "formula": [
                {
                    "title": "Shared down"
                }
            ]
        },
        "expert-combine": {
            "summary": "routed outputs accumulate, shared expert added.",
            "details": {
                "why": "token별 conditional routed output과 universal shared output을 하나의 MoE output으로 합칩니다. routed specialization과 common transform을 동시에 씁니다."
            },
            "formula": [
                {
                    "title": "Routed + shared combine"
                }
            ]
        },
        "moe-allreduce": {
            "summary": "tensor/expert parallel shard의 MoE output을 합친다.",
            "notes": [
                "단일 GPU 개념 그래프에서는 identity처럼 보이지만 distributed inference에서는 명시적 동기화 지점이다."
            ],
            "details": {
                "why": "tensor/expert parallel 환경에서 shard별 MoE 결과를 동일한 hidden stream으로 합칩니다. 단일 GPU 개념 그래프에서는 identity처럼 보일 수 있지만, distributed inference에서는 all-reduce가 분산 shard의 MoE output을 맞추는 명시적 동기화 지점입니다.",
                "runtime": "single-device conceptual graph에서는 거의 identity처럼 보일 수 있습니다."
            },
            "formula": [
                {
                    "title": "Parallel reduce",
                    "note": "distributed runtime에서 shard별 output을 합치는 단계입니다."
                }
            ]
        },
        "attn-gather": {
            "summary": "선택된 SWA/compressed KV entry만 attention kernel 입력으로 gather한다.",
            "details": {
                "why": "전체 cache를 dense하게 읽지 않고 선택된 id만 모아 attention kernel 입력으로 만듭니다. selected KV만 gather해 memory bandwidth와 attention compute를 줄입니다.",
                "runtime": "SWA entry와 compressed entry는 같은 512-dim KV shape라 gather 후 concat됩니다."
            },
            "formula": [
                {
                    "title": "Selected gather",
                    "note": "attention kernel은 전체 cache가 아니라 선택된 entry만 읽습니다."
                }
            ]
        },
        "attn-score": {
            "summary": "query와 selected key의 scaled dot product를 계산한다.",
            "details": {
                "why": "selected key에 대해서만 QK logit을 계산해 sparse attention의 핵심 compute를 제한하고 dense T 길이 score matrix를 피합니다.",
                "runtime": "KV sharing 때문에 key 역할에서는 RoPE가 들어간 representation을 씁니다."
            },
            "formula": [
                {
                    "title": "QK score"
                }
            ]
        },
        "attn-mask-sink": {
            "summary": "causal/window mask와 head별 attention sink bias를 더한다.",
            "details": {
                "why": "sparse 후보 안에서도 causal/window 제약을 지키고 head별 attention sink로 확률 질량을 조절합니다. mask는 causal/window 제약을 지키고, attn_sink는 head별 score bias를 제공합니다.",
                "runtime": "local SWA mask와 compressed causal mask가 최종 score 공간에서 합쳐집니다."
            },
            "formula": [
                {
                    "title": "Mask + sink",
                    "note": "head별 attention sink와 causal/window mask를 같은 logit 공간에 더합니다."
                }
            ]
        },
        "attn-softmax": {
            "summary": "선택 KV set 위에서 attention probability를 정규화한다.",
            "details": {
                "why": "selected KV set 안에서만 확률을 정규화해 dense cache softmax를 대체합니다. selected KV set 안에서만 softmax를 계산해 long-context compute와 memory traffic을 줄입니다.",
                "runtime": "긴 context용 kernel은 전체 cache dense softmax가 아니라 selected entry softmax입니다."
            },
            "formula": [
                {
                    "title": "Softmax"
                }
            ]
        },
        "attn-value-sum": {
            "summary": "shared KV value를 attention probability로 가중합한다.",
            "details": {
                "why": "sparse softmax 결과로 selected shared value entry만 가중합합니다. 선택된 문맥만 value sum에 쓰므로 attention output 계산이 작아집니다.",
                "runtime": "이 단계 이후에는 key-position semantics보다 value semantics가 중요합니다."
            },
            "formula": [
                {
                    "title": "Value sum"
                }
            ]
        },
        "attn-inv-rope": {
            "summary": "KV sharing에서 value 역할에는 position phase가 남지 않도록 보정한다.",
            "notes": [
                "K score에는 RoPE가 필요하지만 V sum에서는 위치 phase를 제거한다."
            ],
            "details": {
                "why": "shared KV가 key로 쓰일 때 들어간 RoPE phase가 value output에 남는 문제를 보정합니다. K score에는 RoPE가 필요하지만 V sum에서는 위치 phase를 제거해야 하므로, inverse RoPE path가 KV sharing의 memory 이득을 유지하면서 value phase 부작용을 줄입니다.",
                "runtime": "K에는 RoPE가 필요하지만 V에는 위치 회전이 직접 섞이면 부자연스럽기 때문에 output path에서 보정합니다."
            },
            "formula": [
                {
                    "title": "Inverse RoPE",
                    "note": "shared KV가 value로 쓰일 때 position phase가 남는 문제를 보정하는 path입니다."
                }
            ]
        },
        "o-woa": {
            "summary": "head group별 low-rank output latent를 만든다.",
            "details": {
                "why": "head output을 group별 low-rank latent로 먼저 접어 output projection의 중간 표현을 줄입니다. wo_a group projection이 compute와 parameter 사용을 줄입니다.",
                "runtime": "Pro는 G=16, Flash는 G=8 group 구성을 씁니다."
            },
            "formula": [
                {
                    "title": "Group low-rank A"
                }
            ]
        },
        "o-wob": {
            "summary": "group latent를 hidden size D로 복원한다.",
            "details": {
                "why": "group low-rank latent를 다시 residual hidden dimension으로 복원합니다. wo_b output projection이 attention output을 mHC writeback 가능한 [B,S,D] stream으로 맞춥니다."
            },
            "formula": [
                {
                    "title": "Output B"
                }
            ]
        },
        "hc-head-collapse": {
            "summary": "최종 4-lane residual state를 단일 hidden stream으로 접는다.",
            "details": {
                "why": "최종 4-lane residual state를 LM head가 받을 단일 hidden stream으로 접습니다. head collapse가 mHC 내부 표현을 일반 output projection 형태로 바꿉니다."
            },
            "formula": [
                {
                    "title": "HC head",
                    "note": "4-lane 최종 residual을 단일 hidden stream으로 접습니다."
                }
            ]
        },
        "final-rmsnorm": {
            "summary": "LM head 전에 최종 hidden scale을 맞춘다.",
            "details": {
                "why": "vocab projection 직전 hidden scale을 맞춰 logits magnitude를 안정화합니다. final RMSNorm은 vocab projection 직전 logits magnitude를 안정화합니다."
            },
            "formula": [
                {
                    "title": "Final RMSNorm"
                }
            ]
        },
        "last-token": {
            "summary": "decode logits는 마지막 token hidden만 vocab projection한다.",
            "notes": [
                "모든 layer나 모든 token에서 LM head를 매번 계산하는 구조가 아니다."
            ],
            "details": {
                "why": "autoregressive decode에서 필요한 마지막 token hidden만 vocab projection해 output compute를 줄입니다. 모든 layer나 모든 token에서 LM head를 매번 계산하는 구조가 아니며, x[:, -1]만 projection하는 이 경계가 그래프의 output path 의미를 명확히 합니다.",
                "runtime": "이 노드가 'LM head가 매 layer마다 돈다'는 오해를 막는 핵심 경계입니다."
            },
            "formula": [
                {
                    "title": "Last token only"
                }
            ]
        },
        "lm-project": {
            "summary": "최종 hidden을 vocabulary shard/output head로 투영한다.",
            "details": {
                "why": "최종 hidden을 vocabulary logit 공간으로 사영해 sampling 가능한 score를 만듭니다. LM head가 architecture 내부 표현을 token distribution으로 변환합니다.",
                "runtime": "TP 환경에서는 vocab shard projection 후 gather/reduce가 붙을 수 있습니다."
            },
            "formula": [
                {
                    "title": "LM projection"
                }
            ]
        },
        "mtp-embed": {
            "summary": "MTP branch가 next-token 보조 학습/추론에 쓸 token embedding path.",
            "details": {
                "why": "MTP branch가 token id 기반 정보를 별도 embedding path로 다시 사용하게 합니다. MTP embedding path가 auxiliary prediction 입력을 구성합니다."
            },
            "formula": [
                {
                    "title": "MTP embedding path"
                }
            ]
        },
        "mtp-hidden-proj": {
            "summary": "최종 hidden/lane state를 MTP block 입력 공간으로 투영한다.",
            "details": {
                "why": "최종 decoder hidden/lane state를 MTP block이 받을 공간으로 맞춥니다. MTP hidden path가 main stack representation을 auxiliary branch에 연결합니다."
            },
            "formula": [
                {
                    "title": "MTP hidden path"
                }
            ]
        },
        "mtp-combine": {
            "summary": "embedding path와 hidden projection path를 결합한다.",
            "details": {
                "why": "token embedding 정보와 final hidden 정보를 합쳐 MTP block 입력을 만듭니다. 두 입력을 합쳐 auxiliary next-token objective에 더 풍부한 입력을 줍니다."
            },
            "formula": [
                {
                    "title": "MTP combine"
                }
            ]
        },
        "mtp-block": {
            "summary": "보조 next-token prediction용 block을 통과한다. attention mode는 SWA-only로 표시한다.",
            "details": {
                "why": "main output 뒤에 보조 prediction block을 붙여 multi-token prediction 신호를 제공합니다. 그래프의 R=0 표시는 MTP block의 SWA-only 흐름을 반영합니다.",
                "runtime": "그래프에서는 R=0 SWA-only attention mode로 표시합니다."
            },
            "formula": [
                {
                    "title": "MTP block"
                }
            ]
        },
        "mtp-head": {
            "summary": "MTP branch의 auxiliary vocabulary projection.",
            "details": {
                "why": "MTP branch output을 auxiliary vocabulary logits로 변환합니다. MTP head가 main logits와 별도의 보조 prediction score를 만듭니다."
            },
            "formula": [
                {
                    "title": "MTP logits"
                }
            ]
        },
        "stack-entry": {
            "summary": "The expanded graph below is one representative decoder layer selected by the layer-mode control; this is not a second input stream inside every layer.",
            "notes": [
                "Input and embedding are outside the repeated decoder block."
            ],
            "details": {
                "why": "입력 처리와 반복 decoder layer 내부를 분리해, input과 embedding이 매 layer 안에 다시 생기는 것처럼 보이는 오해를 막습니다. 이 노드는 반복 block 내부의 두 번째 input stream이 아니라 대표 decoder layer로 들어가는 최종 stack state의 시작점을 표시합니다.",
                "ui": "레이어 모드 버튼은 대표 layer 내부 경로만 바꾸며 input node 자체를 바꾸는 것이 아닙니다."
            },
            "formula": [
                {
                    "title": "반복 layer state",
                    "note": "아래 그래프는 선택된 대표 decoder layer의 내부 전개입니다."
                }
            ]
        },
        "stack-exit": {
            "summary": "Only after the decoder stack finishes does the graph branch to HC head / LM head and MTP.",
            "notes": [
                "The official path computes logits from x[:, -1], not from every layer."
            ],
            "details": {
                "why": "모든 decoder layer를 지난 최종 residual lane state와 output head를 분리합니다. 공식 path는 every layer가 아니라 전체 stack이 끝난 뒤 x[:, -1]에서 logits를 계산하므로, 이 노드가 per-layer LM head처럼 보이는 오해를 줄입니다.",
                "ui": "LM head가 매 layer마다 실행되는 것처럼 보이는 오해를 막습니다."
            },
            "formula": [
                {
                    "title": "Final decoder state",
                    "note": "LM head는 각 layer가 아니라 최종 stack state 뒤에 붙습니다."
                }
            ]
        },
        "hca-all-compressed": {
            "summary": "R=128 HCA layers read all valid heavily-compressed blocks instead of running the Lightning indexer.",
            "notes": [
                "No Lightning indexer is used in HCA mode."
            ],
            "details": {
                "why": "R=128처럼 매우 강하게 압축된 layer에서는 Lightning indexer를 쓰지 않고 valid compressed block 전체를 읽어 retrieval overhead를 없앱니다. HCA mode에서는 Attention.indexer가 없으므로, compressed memory 전체와 SWA window에 attend하는 구조로 보는 편이 맞습니다.",
                "runtime": "HCA에는 Lightning indexer가 없고 R=128에서는 Attention.indexer가 None입니다.",
                "ui": "compressed memory 전체 + SWA에 attend하는 구조로 표시합니다."
            },
            "formula": [
                {
                    "title": "HCA compressed set",
                    "note": "HCA layer에서는 Lightning indexer 없이 valid c128a block 전체를 사용합니다."
                }
            ]
        }
    },
    en: {},
  },
  groupPurpose: {
    ko: {
    "Model entry (once)": "Decoder layer 반복에 들어가기 전, token id를 dense hidden state로 바꾸고 DeepSeek V4의 기본 운반 형식인 4-lane residual stream을 여는 진입 구간입니다. 여기서 mHC(manifold / hyper-connection 계열의 4-lane residual 구조)가 처음 등장하며, 이후 모든 attention과 MoE sublayer는 단일 `[B,S,D]` stream이 아니라 `[B,S,4,D]` lane state를 읽고 다시 쓰는 방식으로 동작합니다. 이 구간의 의도는 입력 token 정보를 일반 embedding으로 끝내지 않고, layer 사이를 더 안정적으로 이동할 수 있는 다중 residual lane 상태로 올려놓는 것입니다.",
    "mHC controller + read path": "Attention sublayer 앞에서 mHC(manifold / hyper-connection 계열 residual lane controller)가 4개 lane 전체를 보고 read/write/mix coefficient를 만드는 제어 구간입니다. attention 자체를 lane마다 4번 실행하면 compute가 커지므로, controller가 먼저 어떤 lane 조합을 읽을지 정하고 data path는 그 결과를 단일 `[B,S,D]` hidden stream으로 접습니다. 동시에 post coefficient와 doubly-stochastic comb matrix도 준비해 두기 때문에, 이 구간은 attention 입력 생성뿐 아니라 attention 이후 residual lane 안정화까지 미리 설계하는 역할을 합니다.",
    "Attention Q/KV paths": "Long-context attention의 핵심 경로로, query low-rank projection, shared KV, SWA(Sliding Window Attention) local window, compressed cache, sparse attention, grouped output projection이 한 흐름 안에서 연결됩니다. 의도는 1M token context를 dense KV attention처럼 모두 읽지 않고, 최근 token은 정확하게 보존하고 오래된 token은 compressed entry와 선택된 block만 읽게 만드는 것입니다. 그래서 이 그룹은 단순한 Q/K/V 생성이 아니라 memory 절감, cache layout, sparse candidate selection, output 복원을 한 번에 보여주는 attention 본체에 가깝습니다.",
    "SWA cache write path": "SWA(Sliding Window Attention) cache는 최근 token을 압축하지 않은 KV 형태로 보존하는 local memory path입니다. compressed attention이 오래된 문맥을 저렴하게 다루는 동안, 최근 128 token 근처의 세부 정보는 손실 없이 남겨야 다음 decode step에서 local coherence가 유지됩니다. 이 구간은 모델 수식만으로 끝나는 부분이 아니라 runtime이 ring buffer처럼 최신 KV를 갱신하고 오래된 local entry를 밀어내야 하는 cache 관리 책임까지 드러냅니다.",
    "KV compressor + tail state": "긴 문맥을 compressed KV entry로 바꿔 memory footprint를 줄이는 경로입니다. C4A류 compression은 일정 token block을 모아 하나의 cache entry로 만들지만, streaming decode에서는 항상 block boundary가 딱 맞지 않으므로 tail state가 아직 완성되지 않은 remainder token의 projection과 score state를 보관합니다. 이 그룹의 의도는 오래된 문맥을 버리는 것이 아니라, attention이 나중에 다시 읽을 수 있는 더 작은 KV 단위로 재표현하고 runtime chunking 문제까지 함께 처리하는 것입니다.",
    "Lightning indexer": "CSA(Compressed Sparse Attention 계열) layer에서 많은 compressed KV block 중 실제 attention이 볼 후보만 빠르게 고르는 retrieval gate입니다. Lightning Indexer는 value sum을 직접 수행하는 attention이 아니라, compressed block ranking을 위해 별도 query와 별도 index cache를 사용하는 경량 검색 경로입니다. long-context에서 compute 병목은 cache를 저장하는 것뿐 아니라 어떤 block을 읽을지 고르는 데서도 생기므로, 이 그룹은 attention 전에 candidate set을 줄여 sparse attention의 비용을 제한하는 역할을 합니다.",
    "mHC attention residual mixing": "Attention output을 4-lane residual state에 다시 쓰는 mHC writeback 구간입니다. mHC에서는 기존 residual lane을 그대로 다음 layer로 넘기는 항과 attention output을 새로 주입하는 항이 분리되어 있고, comb matrix가 기존 lane 사이의 residual transport를 담당합니다. 이 설계의 의도는 sublayer output이 residual stream을 덮어쓰는 느낌이 아니라, 기존 lane 신호를 안정적으로 운반하면서 필요한 만큼만 attention update를 각 lane에 분배하는 것입니다.",
    "mHC MoE controller + read path": "MoE(Mixture-of-Experts) sublayer 앞에서 4-lane residual state를 단일 FFN input stream으로 읽는 FFN-side mHC controller 구간입니다. attention 쪽과 같은 read/write/mix 구조를 쓰지만 parameter set은 분리되어 있어, attention에 적합한 lane 읽기 정책과 sparse FFN에 적합한 lane 읽기 정책을 따로 학습할 수 있습니다. 이 그룹의 의도는 MoE compute를 lane마다 반복하지 않으면서도, MoE가 현재 residual lane 전체를 보고 필요한 정보를 골라 들어가게 만드는 것입니다.",
    "MoE routing + SwiGLU experts": "DeepSeek V4의 sparse capacity를 담당하는 MoE(Mixture-of-Experts) 본체입니다. token마다 일부 routed expert만 실행해 token당 compute를 제한하고, shared expert는 항상 더해 공통 변환 경로를 유지합니다. 이 그룹은 router score와 expert id 선택, selected token dispatch, routed SwiGLU expert 계산, shared expert 결합까지 이어지며, 큰 parameter capacity를 실제 실행 비용과 분리하려는 MoE의 핵심 의도를 보여줍니다.",
    "mHC MoE residual mixing": "MoE 결과를 다시 4-lane residual stream으로 되돌리는 FFN-side writeback 구간입니다. routed/shared expert가 만든 update는 post coefficient로 각 lane에 주입되고, 기존 residual lane은 comb matrix를 통해 별도로 이동합니다. attention writeback과 같은 구조를 MoE 뒤에도 반복하는 이유는 sparse expert update가 강하게 들어와도 residual lane gradient와 signal propagation이 갑자기 불안정해지지 않도록 하기 위해서입니다.",
    "Final output + MTP": "반복 decoder stack이 모두 끝난 뒤 main LM head와 MTP(Multi-Token Prediction) branch로 나뉘는 최종 output 구간입니다. LM head는 매 layer마다 실행되는 것이 아니라 최종 state 뒤에서 last token에 대해서만 vocabulary logits를 계산하고, MTP branch는 별도의 auxiliary prediction path로 final hidden과 token embedding 정보를 다시 결합합니다. 이 그룹은 모델 내부 반복 계산과 최종 prediction head의 경계를 분명히 하며, main autoregressive decode와 보조 next-token prediction이 어디서 갈라지는지 보여줍니다.",
    "mHC attention controller + read path": "Attention 상세 scene에서 mHC controller와 read data path만 확대해 보여주는 구간입니다. mHC는 여기서 manifold / hyper-connection residual lane 구조를 뜻하며, controller path는 coefficient를 만들고 data path는 실제 tensor를 읽는 식으로 역할이 분리됩니다. attention 계산 자체보다 중요한 점은 attention이 어떤 residual lane 조합을 입력으로 받는지, 그리고 이후 writeback을 위해 post/comb coefficient가 어떻게 함께 준비되는지입니다.",
    "mHC attention entry/exit": "Attention sublayer를 mHC wrapper 관점에서 감싸 보여주는 입구와 출구입니다. wrapper의 입구에서는 `[B,S,4,D]` lane state를 attention용 `[B,S,D]` stream으로 읽고, 출구에서는 attention output과 기존 residual lane transport를 합쳐 다시 `[B,S,4,D]`로 복원합니다. 이 그룹은 attention kernel의 세부보다 sublayer boundary에서 hidden state 형식이 어떻게 바뀌고 다시 돌아오는지 이해하는 데 초점을 둡니다.",
    "Query LoRA + RoPE": "Attention query를 저비용으로 만들면서 position 정보를 필요한 slice에만 주입하는 query 생성 구간입니다. LoRA(low-rank adaptation식 저랭크 projection) 형태의 query latent는 projection 비용과 parameter 부담을 줄이고, RMSNorm은 main attention과 indexer가 공유할 query latent의 scale을 안정화합니다. RoPE(Rotary Position Embedding)는 query의 position-sensitive slice에만 적용되어, content dimension과 position phase를 구분한 채 long-context attention score를 만들 수 있게 합니다.",
    "Shared KV + SWA": "Head별 KV cache를 모두 들고 가지 않고 shared KV representation과 SWA(Sliding Window Attention) local window를 함께 쓰는 memory 절감 구간입니다. shared KV는 multi-head query가 하나의 compact KV entry를 공유하게 만들어 cache 크기를 줄이고, SWA window는 최근 128 token을 uncompressed로 남겨 가까운 문맥의 정확도를 보존합니다. 이 그룹의 의도는 오래된 context는 압축/선택으로 싸게 다루되, 방금 생성된 local context는 손실 없이 attention 후보에 유지하는 균형입니다.",
    "Compressed selection": "Attention이 읽을 compressed memory 후보를 정하는 선택 구간입니다. CSA 계열에서는 Lightning Indexer가 compressed block을 sparse retrieval하고, HCA 계열에서는 더 강하게 압축된 block 전체를 후보로 넣는 식으로 layer schedule에 따라 선택 방식이 달라집니다. 이 그룹은 compressed cache가 저장되어 있다는 사실보다, 각 layer가 그 compressed memory를 어떤 정책으로 attention candidate set에 포함시키는지를 보여주는 데 의미가 있습니다.",
    "Core attention kernel": "이미 선택된 KV 후보 위에서 실제 sparse attention을 수행하는 kernel 구간입니다. dense context 전체가 아니라 selected cache entry만 gather한 뒤, QK score, causal/window mask, attention sink, softmax, value sum을 순서대로 적용합니다. 이 그룹의 의도는 long-context attention이 결국 같은 attention 수식을 쓰더라도, 수식이 적용되는 domain이 전체 context가 아니라 선택된 cache subset이라는 점을 명확히 보여주는 것입니다.",
    "KV sharing output fix": "Shared KV 설계 때문에 output 쪽에서 필요한 value phase 보정을 처리하는 구간입니다. key score를 만들기 위해 KV representation 일부에는 RoPE phase가 들어가지만, value sum 결과는 position phase가 그대로 섞이면 의미가 어색해질 수 있으므로 inverse RoPE 계열 보정이 필요합니다. 이후 grouped low-rank output projection이 attention head 결과를 residual hidden size로 되돌려 mHC writeback이 받을 수 있는 `[B,S,D]` stream으로 정리합니다.",
    "SWA window cache": "Cache 상세 scene에서 SWA(Sliding Window Attention) local uncompressed memory를 담당하는 구간입니다. compressed cache가 오래된 문맥을 줄여 저장하는 동안, SWA window는 최근 token의 정확한 KV를 유지해 short-range dependency를 안정적으로 처리합니다. 이 그룹은 모델 graph와 runtime cache layout이 만나는 부분으로, prefill에서는 마지막 window를 남기고 decode에서는 ring buffer처럼 새 token KV를 쓰는 책임을 보여줍니다.",
    "Compressor projections": "Compression이 단순 평균이 아니라 learned projection과 learned gate score 위에서 일어난다는 점을 보여주는 구간입니다. compressor KV projection은 compressed entry의 내용이 될 vector를 만들고, gate projection은 여러 token 중 어떤 정보를 더 강하게 남길지 정하는 pooling score를 만듭니다. APE 계열 block-local position signal까지 더해지므로, 이 구간의 의도는 압축 전부터 content path와 selection weight path를 분리해 더 정보량 있는 compressed memory를 만드는 것입니다.",
    "Tail / cutoff runtime state": "Streaming 입력을 compression block 단위로 자를 때 필요한 runtime state 구간입니다. compressor는 window 8, stride 4 같은 block 규칙을 쓰기 때문에 현재 chunk 끝에 남은 token이 아직 완성 block을 이루지 못할 수 있고, tail state는 그 remainder token의 projection과 score를 다음 호출까지 보관합니다. 이 그룹은 모델 수식만이 아니라 dynamic decode runtime에서 if/cutoff/state carry가 왜 필요한지 보여줍니다.",
    "Block pooling": "여러 token representation을 하나의 compressed KV entry로 합치는 실제 pooling 구간입니다. c4a식 overlap transform은 stride보다 넓은 token span을 보게 해 boundary 손실을 줄이고, softmax-gated pooling은 learned score로 중요한 token/channel 정보를 더 크게 반영합니다. 이 그룹의 의도는 오래된 문맥을 균등하게 뭉개는 것이 아니라, block 내부에서 중요도가 높은 정보를 weighted sum으로 남겨 attention cache entry로 재표현하는 것입니다.",
    "Compressed entry write": "Pooling된 compressed representation을 attention cache에서 읽을 수 있는 정식 cache entry로 마무리하는 구간입니다. compressed block은 내부 token 각각의 position을 모두 유지하지 않고 anchor position을 대표 위치로 쓰며, RoPE와 normalization을 거쳐 key로 읽힐 수 있는 형태가 됩니다. slot mapping과 cache write는 SWA prefix 뒤에 compressed suffix를 배치하므로, 이 그룹은 compressed memory가 실제 attention id space에 편입되는 마지막 단계입니다.",
    "Attention consumer": "Cache/compressor가 만든 결과가 attention kernel로 들어가는 소비 지점입니다. SWA cache, compressed KV cache, indexer가 고른 selected block이 서로 다른 방식으로 만들어졌더라도, attention kernel 입장에서는 selected KV gather라는 하나의 입력 인터페이스로 합쳐집니다. 이 그룹은 생산자별 세부 경로를 지나 최종적으로 sparse attention이 실제로 읽는 cache entry가 무엇인지 연결해 줍니다.",
    "Indexer query path": "Lightning Indexer가 main attention과 별도의 cheap retrieval query를 만드는 구간입니다. main attention query는 value sum까지 이어지는 고품질 attention score를 위한 것이고, indexer query는 compressed block ranking을 빠르게 만들기 위한 별도 표현입니다. RoPE, Hadamard rotation, FP4 activation을 거치며 retrieval용 근사 표현으로 바뀌기 때문에, 이 그룹은 정확한 attention output보다 top-k 후보를 싸게 고르는 데 초점이 있습니다.",
    "Indexer compressed KV cache": "Main KV cache와 분리된 indexer 전용 compressed cache를 보여주는 구간입니다. 이 cache는 value sum에 직접 쓰이지 않고, Lightning Indexer가 query와 dot score를 계산해 어떤 compressed block을 볼지 정하는 검색 메모리로 쓰입니다. 분리된 작은 retrieval memory를 두는 의도는 main compressed KV cache를 매번 전부 attention score 대상으로 삼지 않고, 먼저 더 싼 공간에서 후보를 줄이는 것입니다.",
    "Score + head weighting": "Indexer query와 index cache를 비교해 compressed block ranking score를 만드는 구간입니다. 여러 index head는 서로 다른 retrieval 관점을 제공하고, token별 head weighting은 이 점수들을 하나의 block score로 합쳐 top-k selection에 넘깁니다. 이 그룹은 sparse attention의 품질이 단순 dot product 하나가 아니라, head별 후보 평가와 query-dependent weighting을 거쳐 결정된다는 점을 보여줍니다.",
    "Masked TopK selected blocks": "Indexer score에서 causal correctness와 top-k budget을 적용하는 선택 마무리 구간입니다. future block을 먼저 제거해 decode causality를 지키고, 남은 compressed block 중 제한된 수만 골라 attention compute를 예산 안에 묶습니다. 마지막 offset 변환은 compressed block id를 실제 attention cache id로 바꾸므로, 이 그룹은 retrieval score가 실행 가능한 KV gather 목록으로 바뀌는 경계입니다.",
    "mHC MoE entry/exit": "MoE scene에서 sparse FFN을 감싸는 mHC wrapper의 경계를 보여줍니다. mHC는 manifold / hyper-connection residual lane 구조로, MoE가 독립적인 FFN처럼 보이더라도 실제로는 `[B,S,4,D]` lane state를 읽고 다시 쓰는 wrapper 안에서 실행됩니다. 이 그룹의 의도는 routing과 expert 계산만 보면 놓치기 쉬운 residual lane read/write boundary를 MoE 앞뒤에 명확히 드러내는 것입니다.",
    "Router scores + ids": "MoE(Mixture-of-Experts)에서 어떤 expert를 실행할지 결정하는 routing control 구간입니다. 초기 layer의 hash routing은 token id 기반 expert prior를 쓰고, 일반 layer의 top-k routing은 score, bias, original score gather를 통해 expert 선택과 output weighting을 분리합니다. 이 그룹은 token이 어떤 expert로 갈지 정하는 control path이며, compute 절감뿐 아니라 expert load와 token specialization을 좌우하는 핵심 의사결정 지점입니다.",
    "Routed expert dispatch": "선택된 expert별로 token rows를 실제 실행 가능한 batch 형태로 모으는 runtime 구간입니다. routing 결과는 논리적으로는 token마다 expert id 목록이지만, 실제 matmul은 expert별 token 묶음으로 재배치되어야 효율적으로 실행됩니다. 이 그룹은 sparse MoE에서 자주 숨겨지는 dispatch/gather 비용을 드러내며, routing decision이 실제 expert compute layout으로 바뀌는 지점을 보여줍니다.",
    "Routed SwiGLU internals": "Routed expert 하나가 token hidden을 어떻게 FFN 변환하는지 보여주는 expert 내부 구간입니다. SwiGLU(Swish-Gated Linear Unit)는 gate projection과 up projection을 곱해 비선형성을 만들고, down projection은 다시 hidden dimension으로 복원합니다. 이 그룹의 의도는 MoE expert가 단순 linear layer가 아니라 token별로 선택된 작은 FFN이며, routed capacity의 실제 표현력은 이 gated projection 내부에서 나온다는 점을 보여주는 것입니다.",
    "Shared expert + combine": "Sparse routing과 무관하게 모든 token이 거치는 shared expert path와 routed output 결합을 보여줍니다. shared expert는 token별 top-k expert가 놓칠 수 있는 공통 변환을 항상 제공하고, routed experts는 token-specific specialization을 담당합니다. 이 그룹은 MoE가 완전히 sparse expert만 믿는 구조가 아니라, 공통 경로와 선택 경로를 더해 안정적인 기본 변환과 큰 sparse capacity를 함께 쓰는 설계라는 점을 설명합니다.",
    "Final stack state": "Decoder 반복이 끝난 상태를 output branch로 넘기는 경계입니다. 마지막 mHC lane state, 원래 input token identity, output branch가 만나는 위치이므로 LM head와 MTP(Multi-Token Prediction) branch가 어디서 시작되는지 구분해 줍니다. 이 그룹은 layer 내부 반복과 최종 head 계산을 분리해, logits가 매 layer에서 만들어지는 것이 아니라 stack exit 이후에만 계산된다는 흐름을 명확히 합니다.",
    "LM head path": "Main autoregressive prediction을 만드는 최종 head 경로입니다. 4-lane mHC residual state를 단일 hidden stream으로 collapse하고 final RMSNorm으로 scale을 맞춘 뒤, 마지막 token만 vocabulary projection해 logits를 만듭니다. 이 그룹의 의도는 output head가 전체 sequence 전체나 매 layer에서 반복되는 무거운 계산이 아니라, decode path에서 최종 last-token state를 vocab score로 바꾸는 좁은 단계임을 보여주는 것입니다.",
    "MTP branch": "MTP(Multi-Token Prediction) branch는 main logits와 별도로 final hidden state와 token embedding 정보를 결합해 auxiliary next-token prediction을 수행하는 보조 경로입니다. 공개 graph에서는 MTP block이 SWA-only attention mode(R=0)로 표시되며, main decoder stack 뒤에 붙어 추가 prediction signal을 제공합니다. 이 그룹은 main autoregressive head와 별도의 보조 prediction branch가 어떤 입력을 받아 어떻게 logits로 이어지는지 보여주는 데 목적이 있습니다.",
    },
    en: {
      default: "{group} collects related graph nodes that work together as one model subsystem.",
      "Model entry (once)": "This is the one-time entry path before the repeated decoder stack. Token ids become dense hidden vectors, then the model enters the 4-lane residual state used by the mHC hyper-connection machinery. The point is to start the stack with a residual representation that can be read, mixed, and written by later attention and MoE sublayers.",
      "mHC controller + read path": "This group shows the mHC controller before attention. The controller looks at all four residual lanes, predicts read/write/mix coefficients, and lets the data path read only the lane mixture needed by the attention block. It keeps attention compute close to a single hidden stream while preserving the extra residual-lane structure.",
      "Attention Q/KV paths": "This is the main long-context attention pipeline. Query low-rank projection, shared KV, the local SWA window, compressed cache selection, sparse attention, and grouped output projection work together so the model can read very long context without dense attention over every cached token.",
      "SWA cache write path": "This group covers the local Sliding Window Attention cache writes. Recent tokens stay as uncompressed KV entries, while older context is handled by compressed memory. The runtime has to keep this window fresh during prefill and decode.",
      "KV compressor + tail state": "This group turns older context into compressed KV entries. Projection, tail-state carry, block pooling, anchor positions, RoPE, and cache writes are all part of the path that makes long context cheaper while still readable by later attention.",
      "Lightning indexer": "This is the lightweight retrieval path used by CSA layers. It does not produce the final attention output; it ranks compressed blocks so the sparse attention kernel only reads a smaller candidate set.",
      "mHC attention residual mixing": "This group is the attention writeback side of mHC. Existing residual lanes are transported through the learned mixing matrix, while the attention output is injected separately into lanes with write coefficients.",
      "mHC MoE controller + read path": "This is the MoE-side mHC read controller. It mirrors the attention-side controller but uses separate parameters, allowing the sparse FFN to read a different lane mixture from the one used by attention.",
      "MoE routing + SwiGLU experts": "This is the sparse MoE body. Routing chooses a small set of experts per token, dispatch groups token rows by expert, routed SwiGLU experts compute specialized updates, and the always-on shared expert adds a common transformation.",
      "mHC MoE residual mixing": "This group writes the MoE result back into the 4-lane residual state. It combines residual lane transport with expert-output injection so sparse FFN updates do not simply overwrite the carried residual signal.",
      "Final output + MTP": "This is the output boundary after the decoder stack. The main LM head is applied after the final state, not after every layer, and the MTP branch adds an auxiliary Multi-Token Prediction path.",
      "mHC attention controller + read path": "This detailed mHC view focuses on the attention controller and read data path. The controller produces coefficient tensors, while the data path applies the read coefficients to form the actual attention input.",
      "mHC attention entry/exit": "This group shows the wrapper boundary around attention. The 4-lane state is read into a single attention stream at entry and restored to a 4-lane state at exit.",
      "Query LoRA + RoPE": "This group builds attention queries cheaply and adds position information only where it is needed. The low-rank query latent reduces projection cost, and RoPE applies a rotary phase to the position-sensitive slice.",
      "Shared KV + SWA": "This group shows how KV memory is reduced while local context is preserved. Shared KV avoids per-head KV cache expansion, and the SWA window keeps recent tokens uncompressed.",
      "Compressed selection": "This group decides which compressed memory entries attention will read. CSA uses the Lightning Indexer for sparse retrieval, while HCA consumes the heavily compressed blocks more directly.",
      "Core attention kernel": "This group is the actual sparse attention computation over already selected KV entries. Gather, score, mask, softmax, and value sum operate on a candidate subset rather than the full context.",
      "KV sharing output fix": "This group fixes the value-output side effects of shared KV and RoPE. Key scoring needs positional phase, but value output must be brought back into a residual hidden stream.",
      "SWA window cache": "This group is the runtime-maintained local KV memory. It preserves recent tokens exactly while compressed paths handle older context.",
      "Compressor projections": "This group prepares compression content and compression weights separately. One projection creates candidate KV content, while another produces scores for learned pooling.",
      "Tail / cutoff runtime state": "This group explains why compression needs persistent tail state. Decode chunks do not always align with compression block boundaries, so incomplete tokens must be carried forward.",
      "Block pooling": "This group performs the weighted pooling that turns token blocks into compressed KV entries. Overlap and softmax-gated pooling reduce boundary loss and keep important information.",
      "Compressed entry write": "This group finalizes compressed memory. It assigns anchor positions, applies normalization and RoPE, maps entries to cache slots, and writes them into the compressed cache.",
      "Attention consumer": "This is the point where cache producers meet the attention kernel. Window cache, compressed cache, and selected block ids all become a gathered KV input.",
      "Indexer query path": "This group builds the retrieval query for the Lightning Indexer. It is optimized for ranking compressed blocks, not for producing the final attention value sum.",
      "Indexer compressed KV cache": "This group is the indexer-only retrieval cache. It is separate from the main value cache and stores compact vectors used for block scoring.",
      "Score + head weighting": "This group turns indexer query/cache comparisons into block scores. Multiple index heads are weighted and reduced into a single ranking score.",
      "Masked TopK selected blocks": "This group applies causal constraints and top-k selection to indexer scores, then converts selected block ids into attention cache ids.",
      "mHC MoE entry/exit": "This group shows the mHC wrapper boundary around sparse FFN execution. MoE is run inside the residual lane read/write structure.",
      "Router scores + ids": "This group is the MoE routing control path. It creates expert scores, applies hash or top-k routing depending on layer, and separates expert selection from output weighting.",
      "Routed expert dispatch": "This group converts routing decisions into executable expert batches. Token rows are counted and grouped by selected expert.",
      "Routed SwiGLU internals": "This group shows the actual routed expert FFN. Gate/up projections feed SwiGLU, down projection restores hidden size, and routed accumulation scatters weighted outputs back.",
      "Shared expert + combine": "This group combines the always-on shared expert with routed expert outputs. It gives every token a common FFN path in addition to sparse specialization.",
      "Final stack state": "This group is the shared boundary between the repeated decoder stack and the output heads. It separates internal layer flow from final prediction paths.",
      "LM head path": "This group is the main autoregressive output head. It collapses the 4-lane final state, normalizes it, slices the last token, and projects to vocabulary logits.",
      "MTP branch": "This group is the auxiliary Multi-Token Prediction branch. It combines token embedding information with final hidden state and runs an extra prediction block.",
    },
  },
  groupNarrative: {
    ko: {
    "Model entry (once)": (group, docs, n) => [
      `${n("input-ids", "토큰 id")}는 모델에 한 번 들어오는 이산 입력이고, ${n("embedding", "embedding lookup")}이 이를 hidden vector로 바꾼 뒤 ${n("hc-expand", "4-lane residual state")}로 확장합니다. 이 상태가 ${n("stack-entry", "대표 decoder layer")}의 시작점이 되므로, 반복 layer 내부가 아니라 전체 stack 진입부를 보여줍니다.`,
    ],
    "mHC controller + read path": (group, docs, n) => [
      `${n("hc-flatten", "lane flatten")}은 4개 residual lane을 controller가 볼 수 있는 하나의 vector로 펼치고, ${n("hc-controller", "controller linear")}가 read/write/mix 정책을 한 번에 예측합니다. 그 출력은 ${n("hc-split", "pre, post, comb split")}으로 갈라지고, ${n("hc-pre-sigmoid", "read coefficient")}와 ${n("hc-post-sigmoid", "write coefficient")}는 bounded scalar로 정리됩니다.`,
      `Residual lane mixing 쪽은 ${n("hc-comb-softmax", "row-normalized comb seed")}를 만든 뒤 ${n("hc-comb-sinkhorn", "Sinkhorn-normalized comb")}로 안정화됩니다. 최종적으로 ${n("hc-read", "read data path")}가 pre coefficient를 이용해 4-lane state를 attention이 받을 단일 hidden stream으로 읽습니다.`,
    ],
    "Attention Q/KV paths": (group, docs, n) => [
      `${n("q-wqa", "query low-rank projection")}과 ${n("q-norm", "query normalization")}은 attention query와 indexer query가 공유할 안정적인 latent를 만들고, ${n("q-wqb", "head expansion")}, ${n("q-reshape", "head reshape")}, ${n("q-rope", "query RoPE")}가 이를 실제 multi-head query로 바꿉니다.`,
      `KV 쪽에서는 ${n("kv-wkv", "shared KV projection")}과 ${n("kv-norm", "KV normalization")}이 하나의 512-dim shared cache entry를 만들고, ${n("kv-slice", "content/RoPE split")}와 ${n("kv-rope-quant", "RoPE/quantized KV")}가 key score와 value semantics를 분리합니다. 최근 token 후보는 ${n("window-topk", "SWA window")}와 ${n("cache-layout", "logical cache layout")}에서 오고, HCA에서는 ${n("hca-all-compressed", "all compressed blocks")}가 indexer 없이 들어갑니다.`,
      `선택된 후보는 ${n("attn-selected", "selected ids")}와 ${n("attn-gather", "KV gather")}를 거쳐 ${n("attn-score", "QK score")}, ${n("attn-mask-sink", "mask/sink")}, ${n("attn-softmax", "selected softmax")}, ${n("attn-value-sum", "value sum")}으로 처리됩니다. 마지막으로 ${n("attn-inv-rope", "inverse RoPE value fix")}가 shared-KV value phase를 보정하고 ${n("o-woa", "grouped low-rank output A")}, ${n("o-wob", "output projection B")}가 residual stream 크기로 복원합니다.`,
    ],
    "SWA cache write path": (group, docs, n) => [
      `${n("swa-prefill-write", "prefill write")}는 긴 prompt 중 최근 local window만 uncompressed cache에 남기고, ${n("swa-decode-write", "decode ring write")}는 새 token KV를 128-slot ring buffer에 갱신합니다. ${n("cache-layout", "logical cache layout")}은 이 SWA prefix를 compressed suffix와 같은 id 공간에 놓고, ${n("window-topk", "window ids")}가 score top-k와 무관하게 최근 local token을 항상 후보로 보존합니다.`,
    ],
    "KV compressor + tail state": (group, docs, n) => [
      `${n("comp-wkv", "compressor KV projection")}과 ${n("comp-wgate", "pooling score projection")}은 compressed entry를 만들 재료와 weight를 따로 만들고, ${n("comp-ape", "compressor APE")}가 block 내부 위치 정보를 gate score에 더합니다. Streaming decode에서는 ${n("tail-state", "tail state")}, ${n("comp-cutoff", "cutoff/remainder split")}, ${n("tail-append", "tail append")}가 아직 block을 이루지 못한 token projection을 다음 호출까지 이어 줍니다.`,
      `완성된 block은 ${n("comp-block-view", "block view")}와 ${n("overlap-transform", "c4a overlap transform")}을 거쳐 pooling 축으로 재배치되고, ${n("gated-pool", "softmax-gated pooling")}이 여러 token을 하나의 compressed KV entry로 모읍니다. 그 entry는 ${n("comp-anchor", "anchor position")}, ${n("comp-norm-rope", "compressed norm/RoPE")}, ${n("comp-cache-slot", "compressed slot map")}, ${n("comp-cache-write", "compressed cache write")}를 통해 attention cache에서 읽을 수 있는 형태로 저장됩니다.`,
    ],
    "Lightning indexer": (group, docs, n) => [
      `${n("idx-q", "indexer query")}는 main attention Q가 아니라 q latent에서 cheap retrieval query를 만들고, ${n("idx-rope", "indexer RoPE")}, ${n("idx-hadamard", "Hadamard rotation")}, ${n("idx-fp4", "FP4 activation")}가 ranking용 표현을 가볍게 만듭니다. 별도의 ${n("idx-cache-compress", "index cache compressor")}, ${n("idx-cache-write", "index cache write")}, ${n("idx-cache", "index cache")}는 main KV와 분리된 작은 retrieval memory를 유지합니다.`,
      `${n("idx-einsum", "ReLU dot score")}가 query와 compressed index cache를 비교하고, ${n("idx-weight", "head weighting")}가 여러 index head를 하나의 block score로 합칩니다. 이후 ${n("idx-mask", "causal mask")}, ${n("idx-topk", "compressed topK")}, ${n("idx-offset", "cache offset")}이 future block을 제거하고 SWA prefix 뒤 cache id로 변환합니다.`,
    ],
    "mHC attention residual mixing": (group, docs, n) => [
      `${n("attn-residual-mix", "residual lane mixing")}은 attention output과 별개로 기존 4-lane residual state를 comb matrix로 운반하고, ${n("attn-post-inject", "attention output injection")}은 단일 attention result를 post coefficient로 각 lane에 분배합니다. ${n("hc-write", "attention writeback")}은 이 두 항을 더해 다음 sublayer가 받을 residual lane state를 만듭니다.`,
    ],
    "mHC MoE controller + read path": (group, docs, n) => [
      `${n("ffn-hc-flatten", "MoE lane flatten")}과 ${n("ffn-hc-controller", "MoE controller")}는 attention과 별도 parameter set으로 MoE sublayer 전용 coefficient를 만듭니다. ${n("ffn-hc-split", "MoE split")} 후 ${n("ffn-hc-pre-sigmoid", "MoE read coefficient")}와 ${n("ffn-hc-post-sigmoid", "MoE write coefficient")}가 bounded scale로 정리되고, ${n("ffn-hc-comb-softmax", "MoE comb seed")}와 ${n("ffn-hc-comb-sinkhorn", "MoE Sinkhorn comb")}이 residual lane transport matrix를 안정화합니다.`,
      `마지막으로 ${n("hc-pre-moe", "MoE read path")}가 4-lane residual을 router와 experts가 처리할 단일 hidden stream으로 읽어 MoE compute가 lane마다 반복되지 않게 합니다.`,
    ],
    "MoE routing + SwiGLU experts": (group, docs, n) => [
      `${n("gate-score", "router score")}는 expert affinity를 만들고, 초기 layer에서는 ${n("hash-route", "hash routing")}가 token id 기반 expert prior를 사용합니다. 후반 layer에서는 ${n("route-bias", "selection bias")}와 ${n("topk-route", "top-k routing")}가 expert id를 고르며, ${n("route-score-gather", "original score gather")}와 ${n("route-weights", "route weights")}가 output magnitude를 위한 mixture weight를 다시 계산합니다.`,
      `${n("expert-counts", "expert counts")}와 ${n("expert-dispatch", "dispatch")}는 token rows를 expert별로 모으고, routed path는 ${n("expert-w1w3", "expert gate/up projection")}, ${n("swiglu", "SwiGLU")}, ${n("expert-w2", "expert down projection")}, ${n("routed-accum", "routed accumulation")}으로 sparse FFN 결과를 원래 token stream에 되돌립니다. 병렬 always-on path에서는 ${n("shared-w1w3", "shared gate/up")}, ${n("shared-swiglu", "shared SwiGLU")}, ${n("shared-w2", "shared down")}가 공통 transform을 만들고, ${n("expert-combine", "expert combine")}과 ${n("moe-allreduce", "MoE all-reduce")}가 routed/shared 결과를 하나로 맞춥니다.`,
    ],
    "mHC MoE residual mixing": (group, docs, n) => [
      `${n("ffn-residual-mix", "MoE residual lane mixing")}은 sparse expert update와 별개로 기존 residual lane을 comb matrix로 운반하고, ${n("ffn-post-inject", "MoE output injection")}은 MoE result를 각 lane에 주입합니다. ${n("hc-post-moe", "MoE writeback")}은 두 항을 합쳐 다음 decoder block이 받을 4-lane residual state를 만듭니다.`,
    ],
    "Final output + MTP": (group, docs, n) => [
      `${n("stack-exit", "final stack state")} 이후에만 output path가 시작되고, ${n("hc-head-collapse", "HC head collapse")}와 ${n("final-rmsnorm", "final RMSNorm")}이 4-lane state를 logits용 hidden stream으로 정리합니다. ${n("last-token", "last-token slice")}와 ${n("lm-project", "LM projection")}는 모든 token이 아니라 마지막 token의 vocabulary score를 만들고, ${n("logits", "logits")}가 main decode score가 됩니다.`,
      `보조 branch에서는 ${n("mtp-embed", "MTP embedding path")}, ${n("mtp-hidden-proj", "MTP hidden projection")}, ${n("mtp-combine", "MTP combine")}, ${n("mtp-block", "MTP block")}, ${n("mtp-head", "MTP head")}가 final state 뒤에서 auxiliary next-token prediction을 구성합니다.`,
    ],
    "mHC attention controller + read path": (group, docs, n) => [
      `${n("hc-flatten", "flattened residual lanes")}를 기준으로 ${n("hc-controller", "attention mHC controller")}가 attention 앞뒤의 coefficient를 예측합니다. ${n("hc-split", "split")}은 그 결과를 read/write/mix 계열로 나누고, ${n("hc-pre-sigmoid", "pre coefficient")}와 ${n("hc-post-sigmoid", "post coefficient")}는 lane read와 output injection의 scale을 안정화합니다.`,
      `${n("hc-comb-softmax", "comb seed")}와 ${n("hc-comb-sinkhorn", "Sinkhorn comb")}는 residual lane transport matrix를 만들며, ${n("hc-read", "read path")}가 이 controller 결과 중 read coefficient를 실제 attention input stream에 적용합니다.`,
    ],
    "mHC attention entry/exit": (group, docs, n) => [
      `${n("mhc-attn", "attention mHC wrapper")}는 attention sublayer를 mHC read/write 구조로 감싸는 개념적 경계이고, ${n("hc-write", "HC writeback")}는 attention 결과와 residual lane transport를 합쳐 wrapper 밖으로 내보내는 출구입니다.`,
    ],
    "Query LoRA + RoPE": (group, docs, n) => [
      `${n("q-wqa", "low-rank query A projection")}이 hidden stream을 compact latent로 낮추고 ${n("q-norm", "query RMSNorm")}이 main attention과 indexer가 공유할 scale을 맞춥니다. 이후 ${n("q-wqb", "query B projection")}와 ${n("q-reshape", "head reshape")}가 multi-head query를 만들고, ${n("q-rope", "RoPE slice")}가 position phase를 query의 일부 dimension에만 넣습니다.`,
    ],
    "Shared KV + SWA": (group, docs, n) => [
      `${n("kv-wkv", "shared KV projection")}과 ${n("kv-norm", "KV RMSNorm")}은 head별 KV가 아니라 하나의 shared 512-dim cache entry를 만듭니다. ${n("kv-slice", "content/RoPE split")}와 ${n("kv-rope-quant", "KV RoPE/quantization")}는 key score와 value semantics를 분리하고, ${n("cache-layout", "cache layout")}과 ${n("window-topk", "SWA window")}가 최근 128개 local token을 항상 attention 후보로 유지합니다.`,
    ],
    "Compressed selection": (group, docs, n) => [
      `${n("compressor", "KV compressor")}가 오래된 context를 compressed memory로 바꾸고, CSA에서는 ${n("indexer", "Lightning indexer")}와 ${n("idx-offset", "compressed cache offset")}이 중요한 compressed block만 고릅니다. HCA에서는 ${n("hca-all-compressed", "all compressed blocks")}가 indexer 없이 compressed memory 전체를 후보로 넣고, 최종 후보 집합은 ${n("attn-selected", "selected KV ids")}에 모입니다.`,
    ],
    "Core attention kernel": (group, docs, n) => [
      `${n("attn-gather", "selected KV gather")}가 cache에서 필요한 entry만 모으고, ${n("attn-score", "QK score")}가 sparse candidate set 위에서 logit을 계산합니다. ${n("attn-mask-sink", "mask and sink")}가 causal/window 제약과 attention sink를 더한 뒤 ${n("attn-softmax", "selected softmax")}가 확률을 만들고, ${n("attn-value-sum", "value sum")}이 selected value만 가중합합니다.`,
    ],
    "KV sharing output fix": (group, docs, n) => [
      `${n("attn-inv-rope", "inverse RoPE")}는 shared KV가 key로 쓰일 때 들어간 position phase를 value output에서 보정합니다. 그 뒤 ${n("o-woa", "grouped output A projection")}와 ${n("o-wob", "output B projection")}가 attention head 결과를 residual hidden dimension으로 복원합니다.`,
    ],
    "SWA window cache": (group, docs, n) => [
      `${n("kv-path", "shared KV path")}와 ${n("kv-cache", "KV cache")}가 window/compressed cache의 기본 표현을 만들고, ${n("swa-prefill-write", "prefill window write")}와 ${n("swa-decode-write", "decode ring write")}가 최근 local KV를 uncompressed 상태로 유지합니다. ${n("cache-layout", "logical cache layout")}과 ${n("window-topk", "window ids")}는 이 SWA region을 compressed suffix와 같은 attention id 체계에 올립니다.`,
    ],
    "Compressor projections": (group, docs, n) => [
      `${n("comp-wkv", "compressor KV projection")}은 pooling될 value 후보를 만들고, ${n("comp-wgate", "gate projection")}은 어떤 token/channel을 더 남길지 정하는 score를 만듭니다. ${n("comp-ape", "compressor APE")}는 이 score에 block-local 위치 정보를 더해 compressed entry가 내부 순서를 완전히 잃지 않게 합니다.`,
    ],
    "Tail / cutoff runtime state": (group, docs, n) => [
      `${n("tail-state", "tail state")}는 아직 compression block을 채우지 못한 remainder를 request state로 보관합니다. ${n("comp-cutoff", "cutoff split")}은 완성 block과 남은 tail을 나누고, ${n("tail-append", "tail append")}는 다음 decode/prefill chunk에서 이어 쓸 state를 갱신합니다.`,
    ],
    "Block pooling": (group, docs, n) => [
      `${n("comp-block-view", "block view")}가 projection channel을 pooling kernel이 읽을 block/token/value 축으로 바꾸고, c4a에서는 ${n("overlap-transform", "overlap transform")}이 stride 4 entry가 8-token span을 보게 만듭니다. ${n("gated-pool", "softmax-gated pool")}은 learned weight로 여러 token을 하나의 compressed KV entry로 합칩니다.`,
    ],
    "Compressed entry write": (group, docs, n) => [
      `${n("comp-anchor", "anchor position")}는 compressed block 전체에 대표 position을 붙이고, ${n("comp-norm-rope", "compressed norm/RoPE")}가 attention key로 쓸 수 있게 scale과 phase를 맞춥니다. ${n("comp-cache-slot", "slot map")}은 SWA prefix 뒤 logical slot으로 옮기고, ${n("comp-cache-write", "cache write")}가 compressed memory에 기록합니다.`,
    ],
    "Attention consumer": (group, docs, n) => [
      `${n("attn-gather", "attention gather")}는 compressor나 SWA가 만든 cache entry를 실제 attention kernel 입력으로 소비하는 접점입니다.`,
    ],
    "Indexer query path": (group, docs, n) => [
      `${n("q-norm", "normalized query latent")}에서 ${n("idx-q", "indexer query projection")}가 retrieval 전용 query를 만들고, ${n("idx-rope", "indexer RoPE")}가 query position을 반영합니다. ${n("idx-hadamard", "Hadamard rotation")}와 ${n("idx-fp4", "FP4 activation")}는 top-k ranking을 싸게 만들기 위한 approximate representation을 구성합니다.`,
    ],
    "Indexer compressed KV cache": (group, docs, n) => [
      `${n("kv-path", "shared KV path")}와 별개로 ${n("idx-cache-compress", "index cache compressor")}가 retrieval score 전용 128-dim entry를 만들고, ${n("idx-cache-write", "index cache write")}가 이를 저장합니다. 이후 ${n("idx-cache", "index cache")}는 main value sum이 아니라 compressed block ranking에만 쓰입니다.`,
    ],
    "Score + head weighting": (group, docs, n) => [
      `${n("idx-einsum", "index score")}이 query와 compressed index cache 사이의 ReLU dot score를 만들고, ${n("idx-weight", "head weighting")}가 64개 index head의 점수를 query-dependent weight로 합쳐 block score를 만듭니다.`,
    ],
    "Masked TopK selected blocks": (group, docs, n) => [
      `${n("idx-mask", "causal block mask")}가 아직 볼 수 없는 compressed block을 제거하고, ${n("idx-topk", "compressed topK")}가 제한된 수의 block만 남깁니다. ${n("idx-offset", "cache offset")}은 그 block id를 SWA prefix 뒤 cache id로 바꾸고, ${n("attn-selected", "selected ids")}가 window 후보와 합칩니다.`,
    ],
    "mHC MoE entry/exit": (group, docs, n) => [
      `${n("mhc-ffn", "MoE mHC wrapper")}는 sparse FFN 주변의 read/write 구조를 나타내고, ${n("hc-post-moe", "post-MoE writeback")}는 MoE result와 residual lane transport를 합친 다음 block input state를 나타냅니다.`,
    ],
    "Router scores + ids": (group, docs, n) => [
      `${n("gate-score", "router score")}가 expert affinity를 만들고, 초기 layer에서는 ${n("hash-route", "hash route")}가 token id 기반 expert ids를 사용합니다. 일반 layer에서는 ${n("route-bias", "selection bias")}와 ${n("topk-route", "top-k route")}가 expert ids를 고르고, ${n("route-score-gather", "original score gather")}와 ${n("route-weights", "normalized route weights")}가 실제 mixture weight를 만듭니다.`,
    ],
    "Routed expert dispatch": (group, docs, n) => [
      `${n("expert-counts", "expert counts")}는 expert별 token row 수를 세고, ${n("expert-dispatch", "expert dispatch")}는 선택된 rows를 해당 expert 연산으로 모아 sparse FFN을 batch 연산처럼 실행할 수 있게 합니다.`,
    ],
    "Routed SwiGLU internals": (group, docs, n) => [
      `${n("expert-w1w3", "expert gate/up projection")}이 routed expert의 intermediate features를 만들고, ${n("swiglu", "SwiGLU")}가 gate와 up activation을 곱해 비선형성을 줍니다. ${n("expert-w2", "expert down projection")}는 hidden dimension으로 복원하고, ${n("routed-accum", "routed accumulation")}은 route weight를 곱해 원 token 위치로 되돌립니다.`,
    ],
    "Shared expert + combine": (group, docs, n) => [
      `${n("shared-w1w3", "shared gate/up projection")}, ${n("shared-swiglu", "shared SwiGLU")}, ${n("shared-w2", "shared down projection")}는 모든 token이 항상 거치는 common FFN path를 구성합니다. ${n("expert-combine", "expert combine")}은 shared output과 routed output을 합치고, ${n("moe-allreduce", "MoE all-reduce")}는 parallel shard 결과를 동일한 hidden stream으로 맞춥니다.`,
    ],
    "Final stack state": (group, docs, n) => [
      `${n("input-ids", "input ids")}는 output/MTP branch에서도 token identity를 참조할 수 있는 원천이고, ${n("hc-post-moe", "post-MoE state")}는 decoder stack 마지막 residual lane state를 제공합니다. ${n("stack-exit", "stack exit")}는 반복 layer가 끝나고 output head로 넘어가는 경계를 표시합니다.`,
    ],
    "LM head path": (group, docs, n) => [
      `${n("hc-head-collapse", "HC head collapse")}가 4-lane residual을 단일 hidden stream으로 접고, ${n("final-rmsnorm", "final RMSNorm")}이 vocab projection 전 scale을 맞춥니다. ${n("last-token", "last-token slice")}는 마지막 token hidden만 고르고, ${n("lm-project", "LM projection")}와 ${n("logits", "logits")}가 main vocabulary score를 만듭니다.`,
    ],
    "MTP branch": (group, docs, n) => [
      `${n("mtp-embed", "MTP embedding")}은 token id 기반 branch 입력을 만들고, ${n("mtp-hidden-proj", "hidden projection")}는 final hidden state를 auxiliary block에 맞춥니다. ${n("mtp-combine", "MTP combine")}이 두 입력을 합친 뒤 ${n("mtp-block", "MTP block")}과 ${n("mtp-head", "MTP head")}가 auxiliary next-token logits를 만듭니다.`,
    ],
    },
    en: {
      "Model entry (once)": (group, docs, n) => [
        `${n("input-ids", "input ids")} are the discrete token inputs that enter the model once. ${n("embedding", "embedding lookup")} turns them into dense hidden vectors, ${n("hc-expand", "4-lane residual expansion")} opens the mHC residual-lane representation, and ${n("stack-entry", "decoder stack entry")} marks the boundary before repeated layers begin.`,
      ],
      "mHC controller + read path": (group, docs, n) => [
        `${n("hc-flatten", "lane flatten")} exposes all four residual lanes to the controller. ${n("hc-controller", "controller linear")} predicts the lane policy, then ${n("hc-split", "pre/post/comb split")} separates read weights, write weights, and the residual mixing matrix.`,
        `${n("hc-pre-sigmoid", "read coefficient")} and ${n("hc-post-sigmoid", "write coefficient")} bound the scalar gates used on the data path. ${n("hc-comb-softmax", "comb row seed")} initializes the 4x4 transport matrix, ${n("hc-comb-sinkhorn", "Sinkhorn comb")} makes it close to doubly stochastic, and ${n("hc-read", "read data path")} folds the lane state into the single stream consumed by attention.`,
      ],
      "Attention Q/KV paths": (group, docs, n) => [
        `${n("q-wqa", "query A projection")} creates the low-rank query latent, ${n("q-norm", "query RMSNorm")} stabilizes the shared latent, and ${n("q-wqb", "query B projection")} expands it to head width. ${n("q-reshape", "head reshape")} and ${n("q-rope", "query RoPE")} then form position-aware multi-head queries.`,
        `The KV side starts with ${n("kv-wkv", "shared KV projection")} and ${n("kv-norm", "KV RMSNorm")}, then ${n("kv-slice", "content/RoPE split")} separates content/value dimensions from positional key dimensions. ${n("kv-rope-quant", "RoPE and quantized KV")} prepares cache entries, ${n("window-topk", "SWA window ids")} keeps the local window explicit, ${n("cache-layout", "logical cache layout")} defines the shared id space, and ${n("hca-all-compressed", "all HCA compressed blocks")} covers the HCA path without indexer selection.`,
        `${n("attn-selected", "selected KV ids")} becomes the sparse candidate set and ${n("attn-gather", "KV gather")} materializes it for the kernel. ${n("attn-score", "QK score")}, ${n("attn-mask-sink", "mask and sink")}, ${n("attn-softmax", "selected softmax")}, and ${n("attn-value-sum", "value sum")} perform the actual attention computation. ${n("attn-inv-rope", "inverse RoPE value fix")} corrects the shared-KV value path before ${n("o-woa", "output low-rank A")} and ${n("o-wob", "output projection B")} restore the residual hidden stream.`,
      ],
      "SWA cache write path": (group, docs, n) => [
        `${n("swa-prefill-write", "prefill window write")} stores the most recent local prompt entries, while ${n("swa-decode-write", "decode ring write")} updates the 128-slot rolling window during decode. ${n("cache-layout", "logical cache layout")} places that SWA prefix next to compressed memory, and ${n("window-topk", "window ids")} keeps local tokens as guaranteed candidates rather than retrieval results.`,
      ],
      "KV compressor + tail state": (group, docs, n) => [
        `${n("comp-wkv", "compressor KV projection")} creates candidate compressed content and ${n("comp-wgate", "pooling score projection")} creates the learned pooling scores. ${n("comp-ape", "compressor APE")} injects block-local position information before runtime state enters through ${n("tail-state", "tail state")}, ${n("comp-cutoff", "cutoff split")}, and ${n("tail-append", "tail append")} for incomplete blocks.`,
        `Completed tokens are reshaped by ${n("comp-block-view", "block view")}, widened into the c4a span by ${n("overlap-transform", "overlap transform")}, and collapsed by ${n("gated-pool", "softmax-gated pooling")}. ${n("comp-anchor", "anchor position")}, ${n("comp-norm-rope", "compressed norm and RoPE")}, ${n("comp-cache-slot", "compressed slot map")}, and ${n("comp-cache-write", "compressed cache write")} finish the cache entry used by later attention.`,
      ],
      "Lightning indexer": (group, docs, n) => [
        `${n("idx-q", "indexer query projection")} builds a retrieval query separate from the main attention query. ${n("idx-rope", "indexer RoPE")}, ${n("idx-hadamard", "Hadamard rotation")}, and ${n("idx-fp4", "FP4 activation")} make that query cheap enough for block ranking.`,
        `The retrieval memory is separate as well: ${n("idx-cache-compress", "index cache compressor")}, ${n("idx-cache-write", "index cache write")}, and ${n("idx-cache", "index cache")} maintain compact block vectors. ${n("idx-einsum", "ReLU dot score")} scores blocks, ${n("idx-weight", "head weighting")} reduces index-head scores, ${n("idx-mask", "causal mask")} removes future blocks, ${n("idx-topk", "compressed topK")} selects candidates, and ${n("idx-offset", "cache offset")} converts block ids into attention cache ids.`,
      ],
      "mHC attention residual mixing": (group, docs, n) => [
        `${n("attn-residual-mix", "residual lane mixing")} transports the previous 4-lane state through the Sinkhorn comb matrix. ${n("attn-post-inject", "attention output injection")} distributes the attention update with write coefficients, and ${n("hc-write", "attention writeback")} combines transport plus injection into the next residual-lane state.`,
      ],
      "mHC MoE controller + read path": (group, docs, n) => [
        `${n("ffn-hc-flatten", "MoE lane flatten")} and ${n("ffn-hc-controller", "MoE controller")} repeat the mHC control pattern with FFN-specific parameters. ${n("ffn-hc-split", "MoE split")} separates the outputs into ${n("ffn-hc-pre-sigmoid", "MoE read coefficient")}, ${n("ffn-hc-post-sigmoid", "MoE write coefficient")}, ${n("ffn-hc-comb-softmax", "MoE comb seed")}, and ${n("ffn-hc-comb-sinkhorn", "MoE Sinkhorn comb")}. ${n("hc-pre-moe", "MoE read path")} then forms the single hidden stream used by router and experts.`,
      ],
      "MoE routing + SwiGLU experts": (group, docs, n) => [
        `${n("gate-score", "router score")} produces expert affinities. Early layers can use ${n("hash-route", "hash route")}, while later layers use ${n("route-bias", "route bias")} and ${n("topk-route", "top-k route")} to choose experts; ${n("route-score-gather", "score gather")} and ${n("route-weights", "route weights")} recover the mixture weights used on outputs.`,
        `${n("expert-counts", "expert counts")} and ${n("expert-dispatch", "expert dispatch")} turn routing decisions into expert batches. Routed experts run ${n("expert-w1w3", "expert gate/up projection")}, ${n("swiglu", "SwiGLU")}, ${n("expert-w2", "expert down projection")}, and ${n("routed-accum", "routed accumulation")} before the always-on path ${n("shared-w1w3", "shared gate/up projection")}, ${n("shared-swiglu", "shared SwiGLU")}, ${n("shared-w2", "shared down projection")}, ${n("expert-combine", "expert combine")}, and ${n("moe-allreduce", "MoE all-reduce")} merge shared and sparse results.`,
      ],
      "mHC MoE residual mixing": (group, docs, n) => [
        `${n("ffn-residual-mix", "MoE residual lane mixing")} carries the existing residual lanes through the FFN-side comb matrix. ${n("ffn-post-inject", "MoE output injection")} adds the expert result to lanes, and ${n("hc-post-moe", "post-MoE writeback")} produces the residual state for the next decoder block.`,
      ],
      "Final output + MTP": (group, docs, n) => [
        `${n("stack-exit", "stack exit")} is the boundary after repeated decoder layers. ${n("hc-head-collapse", "HC head collapse")}, ${n("final-rmsnorm", "final RMSNorm")}, ${n("last-token", "last-token slice")}, and ${n("lm-project", "LM projection")} create the main ${n("logits", "logits")} path only at the output boundary.`,
        `The auxiliary branch starts with ${n("mtp-embed", "MTP embedding")} and ${n("mtp-hidden-proj", "MTP hidden projection")}, joins them in ${n("mtp-combine", "MTP combine")}, then runs ${n("mtp-block", "MTP block")} and ${n("mtp-head", "MTP head")} for Multi-Token Prediction.`,
      ],
      "mHC attention controller + read path": (group, docs, n) => [
        `${n("hc-flatten", "flattened residual lanes")} feeds ${n("hc-controller", "attention mHC controller")}, whose output is split by ${n("hc-split", "split")} into read, write, and mixing tensors. ${n("hc-pre-sigmoid", "pre coefficient")} and ${n("hc-post-sigmoid", "post coefficient")} gate the data path, while ${n("hc-comb-softmax", "comb seed")} and ${n("hc-comb-sinkhorn", "Sinkhorn comb")} prepare residual-lane transport before ${n("hc-read", "read path")} forms the attention input.`,
      ],
      "mHC attention entry/exit": (group, docs, n) => [
        `${n("mhc-attn", "attention mHC wrapper")} is the conceptual wrapper around the attention sublayer, and ${n("hc-write", "attention writeback")} is the exit point where attention output and residual-lane transport return to the 4-lane state.`,
      ],
      "Query LoRA + RoPE": (group, docs, n) => [
        `${n("q-wqa", "low-rank query A")} compresses the hidden stream, ${n("q-norm", "query RMSNorm")} normalizes the shared query latent, ${n("q-wqb", "query B")} expands it to head channels, ${n("q-reshape", "head reshape")} restores head axes, and ${n("q-rope", "RoPE slice")} adds position phase only to the query slice that needs it.`,
      ],
      "Shared KV + SWA": (group, docs, n) => [
        `${n("kv-wkv", "shared KV projection")} and ${n("kv-norm", "KV RMSNorm")} create a compact shared cache entry. ${n("kv-slice", "content/RoPE split")} and ${n("kv-rope-quant", "KV RoPE and quantization")} separate positional key scoring from value content, while ${n("cache-layout", "cache layout")} and ${n("window-topk", "SWA window")} keep recent tokens available as exact local context.`,
      ],
      "Compressed selection": (group, docs, n) => [
        `${n("compressor", "KV compressor")} creates compressed memory, ${n("indexer", "Lightning indexer")} ranks it in CSA layers, and ${n("idx-offset", "compressed cache offset")} maps selected blocks into cache ids. HCA layers instead expose ${n("hca-all-compressed", "all compressed blocks")}; both paths meet at ${n("attn-selected", "selected KV ids")}.`,
      ],
      "Core attention kernel": (group, docs, n) => [
        `${n("attn-gather", "selected KV gather")} materializes sparse cache entries, ${n("attn-score", "QK score")} computes logits, ${n("attn-mask-sink", "mask and sink")} applies causal/window constraints and sink terms, ${n("attn-softmax", "selected softmax")} normalizes the selected set, and ${n("attn-value-sum", "value sum")} produces the attention result.`,
      ],
      "KV sharing output fix": (group, docs, n) => [
        `${n("attn-inv-rope", "inverse RoPE")} corrects the value path after shared KV has carried positional phase for key scoring. ${n("o-woa", "grouped output A")} and ${n("o-wob", "output B")} then project the attention result back to residual hidden width.`,
      ],
      "SWA window cache": (group, docs, n) => [
        `${n("kv-path", "shared KV path")} produces the entries stored in ${n("kv-cache", "KV cache")}. ${n("swa-prefill-write", "prefill window write")} and ${n("swa-decode-write", "decode ring write")} maintain the exact local window, while ${n("cache-layout", "logical cache layout")} and ${n("window-topk", "window ids")} expose that window to attention.`,
      ],
      "Compressor projections": (group, docs, n) => [
        `${n("comp-wkv", "compressor KV projection")} creates the content candidate, ${n("comp-wgate", "gate projection")} creates pooling scores, and ${n("comp-ape", "compressor APE")} adds local position bias before block pooling.`,
      ],
      "Tail / cutoff runtime state": (group, docs, n) => [
        `${n("tail-state", "tail state")} keeps incomplete compression fragments across calls. ${n("comp-cutoff", "cutoff split")} separates full blocks from remainder tokens, and ${n("tail-append", "tail append")} carries those remainder projections into the next chunk.`,
      ],
      "Block pooling": (group, docs, n) => [
        `${n("comp-block-view", "block view")} reshapes full projections into block form, ${n("overlap-transform", "overlap transform")} applies the c4a overlapping span, and ${n("gated-pool", "softmax-gated pool")} collapses each block into one compressed KV entry.`,
      ],
      "Compressed entry write": (group, docs, n) => [
        `${n("comp-anchor", "anchor position")} assigns the block position, ${n("comp-norm-rope", "compressed norm/RoPE")} prepares the key representation, ${n("comp-cache-slot", "slot map")} places it after the SWA prefix, and ${n("comp-cache-write", "cache write")} stores the compressed entry.`,
      ],
      "Attention consumer": (group, docs, n) => [
        `${n("attn-gather", "attention gather")} is the consumer boundary where SWA entries, compressed entries, and selected ids become the actual KV tensor read by sparse attention.`,
      ],
      "Indexer query path": (group, docs, n) => [
        `${n("q-norm", "normalized query latent")} is reused by the retrieval path. ${n("idx-q", "indexer query projection")}, ${n("idx-rope", "indexer RoPE")}, ${n("idx-hadamard", "Hadamard rotation")}, and ${n("idx-fp4", "FP4 activation")} transform it into a cheap block-ranking query.`,
      ],
      "Indexer compressed KV cache": (group, docs, n) => [
        `${n("kv-path", "shared KV path")} is separate from the indexer retrieval memory. ${n("idx-cache-compress", "index cache compressor")} creates compact retrieval vectors, ${n("idx-cache-write", "index cache write")} stores them, and ${n("idx-cache", "index cache")} is the memory scored by the Lightning Indexer.`,
      ],
      "Score + head weighting": (group, docs, n) => [
        `${n("idx-einsum", "index score")} compares the query with compressed index cache entries, and ${n("idx-weight", "head weighting")} combines multiple index-head scores into the block ranking used by selection.`,
      ],
      "Masked TopK selected blocks": (group, docs, n) => [
        `${n("idx-mask", "causal block mask")} removes future compressed blocks, ${n("idx-topk", "compressed topK")} keeps the budgeted candidates, ${n("idx-offset", "cache offset")} maps block ids into cache ids, and ${n("attn-selected", "selected ids")} merges them with local-window candidates.`,
      ],
      "mHC MoE entry/exit": (group, docs, n) => [
        `${n("mhc-ffn", "MoE mHC wrapper")} marks the sparse FFN as being executed inside an mHC read/write wrapper, and ${n("hc-post-moe", "post-MoE writeback")} is the exit state returned to the decoder stack.`,
      ],
      "Router scores + ids": (group, docs, n) => [
        `${n("gate-score", "router score")} creates expert affinities. ${n("hash-route", "hash route")} covers early input-id based matching, while ${n("route-bias", "selection bias")} and ${n("topk-route", "top-k route")} select later-layer experts; ${n("route-score-gather", "score gather")} and ${n("route-weights", "route weights")} recover the output mixture weights.`,
      ],
      "Routed expert dispatch": (group, docs, n) => [
        `${n("expert-counts", "expert counts")} determines how many token rows go to each expert, and ${n("expert-dispatch", "expert dispatch")} packs those rows so sparse expert matmuls can run as expert-local batches.`,
      ],
      "Routed SwiGLU internals": (group, docs, n) => [
        `${n("expert-w1w3", "expert gate/up projection")} builds the routed expert intermediate, ${n("swiglu", "SwiGLU")} applies the gated nonlinearity, ${n("expert-w2", "expert down projection")} restores hidden width, and ${n("routed-accum", "routed accumulation")} scatters weighted expert outputs back to token order.`,
      ],
      "Shared expert + combine": (group, docs, n) => [
        `${n("shared-w1w3", "shared gate/up projection")}, ${n("shared-swiglu", "shared SwiGLU")}, and ${n("shared-w2", "shared down projection")} form the always-on FFN path. ${n("expert-combine", "expert combine")} adds shared and routed outputs, and ${n("moe-allreduce", "MoE all-reduce")} reconciles parallel shards.`,
      ],
      "Final stack state": (group, docs, n) => [
        `${n("input-ids", "input ids")} remain available to output-side auxiliary paths, ${n("hc-post-moe", "post-MoE state")} is the final residual-lane state from the stack, and ${n("stack-exit", "stack exit")} marks the transition from repeated layers to output heads.`,
      ],
      "LM head path": (group, docs, n) => [
        `${n("hc-head-collapse", "HC head collapse")} converts four lanes back to one hidden stream, ${n("final-rmsnorm", "final RMSNorm")} stabilizes the final representation, ${n("last-token", "last-token slice")} selects the decode position, and ${n("lm-project", "LM projection")} produces ${n("logits", "logits")}.`,
      ],
      "MTP branch": (group, docs, n) => [
        `${n("mtp-embed", "MTP embedding")} brings token-id information into the auxiliary path, ${n("mtp-hidden-proj", "hidden projection")} adapts final hidden state, ${n("mtp-combine", "MTP combine")} joins them, and ${n("mtp-block", "MTP block")} plus ${n("mtp-head", "MTP head")} produce auxiliary prediction logits.`,
      ],
    },
  },
};
