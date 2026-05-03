# DeepSeek V4 visualization docs

이 폴더는 DeepSeek V4만 대상으로 한 그래프형 웹페이지의 정보 설계 문서다. 목표는 화면 중앙에 큰 모델 그래프를 두고, 사용자가 각 데이터 흐름 요소를 클릭하면 해당 요소의 tensor shape, 역할, 내부 연산, 시각화 포인트를 확대해서 볼 수 있게 하는 것이다.

## 문서 구성

- [01-overview.md](./01-overview.md): V4 모델 패밀리와 전체 데이터 흐름.
- [02-shape-glossary.md](./02-shape-glossary.md): 문서 전체에서 쓰는 shape 기호와 Pro/Flash 수치.
- [03-token-embedding.md](./03-token-embedding.md): input ids, embedding, hyper-connection 초기 확장.
- [04-hyper-connections.md](./04-hyper-connections.md): mHC/HC pre-post 흐름.
- [05-hybrid-attention.md](./05-hybrid-attention.md): attention layer의 q, kv, window, sparse attention, output projection.
- [06-compression-csa-hca.md](./06-compression-csa-hca.md): compression ratio 4/128의 CSA/HCA식 KV 압축 흐름.
- [07-indexer.md](./07-indexer.md): CSA 계층에서 compressed block top-k를 고르는 indexer.
- [08-moe-router-experts.md](./08-moe-router-experts.md): MoE gate, routed experts, shared expert.
- [09-output-mtp.md](./09-output-mtp.md): head logits와 MTP block.
- [10-graph-spec.md](./10-graph-spec.md): 웹 그래프 노드/엣지 설계.

## 핵심 출처

- DeepSeek V4 model card: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro
- DeepSeek V4 Flash config: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json
- DeepSeek V4 Pro config: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json
- Official inference implementation: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/inference/model.py
- Hugging Face technical explainer: https://huggingface.co/blog/deepseekv4

## 신뢰도 규칙

그래프 UI에서는 모든 정보에 source badge를 붙인다.

- `official-config`: `config.json`에서 직접 온 수치.
- `official-code`: 공식 inference 코드에서 직접 온 데이터 흐름.
- `official-card`: 모델 카드의 모델 패밀리, 파라미터, precision, context 정보.
- `explainer`: HF 블로그/기술 해설의 설명. 코드나 config와 맞을 때만 주요 설명으로 사용한다.
- `derived`: 위 값들로부터 계산한 shape. 예: `nope_dim = head_dim - rope_dim`.

