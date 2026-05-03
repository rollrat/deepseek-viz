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
      default: (group, docs, n, helpers) => {
        const { escapeHtml, resolve } = helpers;
        if (!docs.length) return [`${escapeHtml(group.label)} has no visible nodes in the current graph view.`];
        const chunks = [];
        for (let index = 0; index < docs.length; index += 5) {
          const part = docs
            .slice(index, index + 5)
            .map((doc) => {
              const input = escapeHtml(resolve(doc.input || ""));
              const output = escapeHtml(resolve(doc.output || ""));
              return `${n(doc.id)} (${input} -> ${output})`;
            })
            .join(", ");
          const prefix = index === 0 ? "The linked flow starts with " : "It then continues through ";
          chunks.push(`${prefix}${part}.`);
        }
        return chunks;
      },
    },
  },
};
