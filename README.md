<div align="center">

# ⚖️ SafeAgent

**노동법 전문 RAG 기반 AI 보조원**

근로기준법 · 산업안전보건법 · 산업재해보상보험법 · 직업교육훈련 촉진법

질문을 입력하면 관련 법률 조문을 검색하고, 조문 근거만을 바탕으로 답변합니다.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.62-red?logo=streamlit)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-purple)
![Ollama](https://img.shields.io/badge/Ollama-gemma3:4b-black)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## 📌 프로젝트 소개

SafeAgent는 대한민국 노동법 4개 법률을 대상으로 **RAG(Retrieval-Augmented Generation)** 기반으로 동작하는 AI 법률 보조원입니다.

- 사용자 질문을 `BAAI/bge-m3` 임베딩으로 변환하여 법률 조문 DB를 검색
- 검색된 조문만을 근거로 LLM이 답변을 생성 (Hallucination 방지)
- 답변에 반드시 법률명과 조문 번호를 명시
- 모든 추론은 **로컬 환경에서 완전히 실행** (인터넷 연결 불필요)

> 💡 외부 API 없이 Ollama + Qdrant + SentenceTransformer로 완전 로컬 동작

---

## 🏗️ 시스템 구조

```
사용자 질문
    │
    ▼
bge-m3 임베딩 (BAAI/bge-m3, 1024-dim)
    │
    ▼
Qdrant Dense Search + Law Filter
    │
    ├─ Dense-only → Top-5 조문
    │
    └─ Reranker 모드 → Top-20 후보
                           │
                       Cross-Encoder (BAAI/bge-reranker-base)
                           │
                       Top-5 재정렬
    │
    ▼
Context 구성 ([순번] 법률명 조문번호 \n 본문)
    │
    ▼
Prompt 구성 (조문 근거 답변 원칙 적용)
    │
    ▼
LLM 호출 (gemma3:4b via Ollama)
    │
    ▼
답변 + 출처 반환
```

---

## 📊 평가 결과 (Dense-only 기준선)

> 평가셋: 4개 법률 × 각 5개 = 20개 질문 | 모델: gemma3:4b | 임베딩: BAAI/bge-m3

| 지표 | 결과 |
|---|---|
| 🎯 검색 히트 (Top-5) | **18 / 20 (90.0%)** |
| 🥇 검색 순위 Top-1 | **16 / 20 (80.0%)** |
| 📎 출처 포함 답변 | **20 / 20 (100.0%)** |
| ⚠️ Hallucination 의심 | 2 / 20 (10.0%) — False Positive 추정 |
| ⏱️ 평균 검색 시간 | 0.14초 |
| 🤖 평균 LLM 응답 시간 | 14.84초 (CPU 환경) |

> Reranker 통합 후 비교 평가 진행 중 (`python evaluate_rag.py --mode both`)

---

## 🗂️ 프로젝트 구조

```
SafeAgent/
├── app/
│   └── rag/
│       ├── rag_pipeline.py      # 핵심 — 전체 RAG 파이프라인 (단일 파일)
│       ├── evaluate_rag.py      # 20개 질문 자동 평가 (Dense / Reranker / 비교)
│       ├── embed_bge_m3.py      # bge-m3 임베딩 → Qdrant 저장
│       ├── hybrid_search.py     # Hybrid Search (BM25 + Dense + RRF)
│       ├── reranker.py          # Cross-Encoder Reranker (독립 실험용)
│       └── ingestion.py         # PDF 파싱 및 조문 단위 청킹
├── frontend/
│   └── app.py                   # Streamlit UI
├── backend/
│   └── main.py                  # FastAPI (확장 예정)
├── data/
│   ├── documents/               # 원본 법률 PDF (gitignore)
│   ├── processed/               # 청킹 결과 JSON
│   └── evaluation/
│       ├── legal_queriesV3.json     # 평가 질문 20개
│       └── rag_eval_results.json    # 평가 결과
├── .gitignore
├── ISSUES.md                    # 개선 사항 추적
├── error_log.md                 # 에러 로그 및 해결 과정
└── README.md
```

---

## ⚙️ 기술 스택

| 구성 요소 | 선택 | 비고 |
|---|---|---|
| 임베딩 모델 | `BAAI/bge-m3` | 다국어, 1024-dim |
| Reranker | `BAAI/bge-reranker-base` | Cross-Encoder |
| 벡터 DB | Qdrant | 로컬 Docker |
| LLM | `gemma3:4b` | Ollama 로컬 추론 |
| UI | Streamlit 1.62 | |
| 백엔드 | FastAPI | 확장 예정 |
| 언어 | Python 3.11 | Conda `safeagent` 환경 |

---

## 📚 법률 데이터

| 법률명 | 법률 번호 | 시행일 | 청크 수 |
|---|---|---|---|
| 근로기준법 | 제21533호 | 2027.01.01 | 130 |
| 산업안전보건법 | 제21374호 | 2026.08.01 | 187 |
| 산업재해보상보험법 | 제21375호 | 2026.07.01 | 161 |
| 직업교육훈련 촉진법 | 제21065호 | 2025.10.01 | 36 |
| **합계** | | | **514** |

---

## 🚀 실행 방법

### 1. 환경 준비

```bash
# Conda 환경 활성화 (Windows)
conda activate safeagent

# 또는 python.exe 직접 지정
# C:\Users\{username}\miniconda3\envs\safeagent\python.exe
```

### 2. Qdrant 실행

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 3. Ollama 모델 준비

```bash
ollama pull gemma3:4b
```

### 4. Streamlit UI 실행

```bash
cd SafeAgent
streamlit run frontend/app.py
```

브라우저에서 `http://localhost:8501` 접속

### 5. RAG 파이프라인 단독 테스트

```bash
cd SafeAgent/app/rag
python rag_pipeline.py
```

### 6. 평가 실행

```bash
cd SafeAgent/app/rag

# Dense-only 평가
python evaluate_rag.py --mode dense

# Reranker 평가
python evaluate_rag.py --mode reranker

# Dense vs Reranker 비교
python evaluate_rag.py --mode both
```

---

## 🔍 사용 예시

```python
from rag_pipeline import init_pipeline, run_rag

# 초기화 (최초 1회)
init_pipeline()

# Dense 검색
result = run_rag(
    question="근로자와 사용자는 어떻게 정의되나요?",
    law="근로기준법",
)

# Reranker 사용
result = run_rag(
    question="산업안전보건법은 어떤 사업장에 적용되나요?",
    law="산업안전보건법",
    use_reranker=True,
)

print(result["answer"])
print(result["sources"])
# ['산업안전보건법 제3조', ...]
```

---

## 💬 프롬프트 설계 원칙

```
1. 제공된 [법률 조문]에 명시된 내용만 근거로 사용
2. 조문에 없는 내용은 추측·생성 금지
3. 답변 불가 시 "제공된 조문에서 해당 내용을 확인할 수 없습니다" 명시
4. 답변 마지막에 법률명 + 조문번호 반드시 명시
5. Context와 사용자 질문을 명확히 구분
```

---

## 🗺️ 로드맵

- [x] bge-m3 임베딩 + Qdrant 구축 (514 chunks)
- [x] Dense Search + Law Filter
- [x] RAG 파이프라인 (검색 → Context → LLM)
- [x] 20개 질문 자동 평가
- [x] Reranker (Cross-Encoder) 통합
- [ ] Dense vs Reranker 비교 평가
- [ ] Hybrid Search (BM25 + Dense + RRF) 적용
- [ ] Hallucination 평가 로직 개선
- [ ] FastAPI 백엔드 연동
- [ ] Vision 모듈 (현장 이미지 분석)

---

## 📄 라이선스

MIT License

---

<div align="center">
  <sub>SafeAgent — 노동법 전문 로컬 AI 보조원</sub>
</div>
