# SafeAgent — Issues & 개선 사항 추적

> 상태 표기
> - 🔴 미해결 (Open)
> - 🟡 진행 중 (In Progress)
> - 🟢 해결됨 (Resolved)
> - 🔵 다음 단계 (Planned)

---

## 검색 품질

### #001 🟡 적용범위 질문 Dense 검색 미스

**현상**
Dense-only 평가에서 "적용 범위" 질문 2건이 Top-5 검색에서 기대 조문을 놓침.

```
[08] 산업안전보건법은 어떤 사업장에 적용되나요?
     기대: 제3조(적용범위) → 실제 Top-1: 제24조

[13] 산업재해보상보험법은 어떤 사업에 적용되나요?
     기대: 제6조(적용범위) → 실제 Top-1: 제1조
```

**원인 분석**
bge-m3 Dense Search는 의미적 유사도 기준으로 검색.
"어떤 사업장에 적용되나요?" 질문이 "적용범위" 조문 제목보다
조문 본문의 구체적 내용(제24조 보건관리자 등)에 더 높게 매칭됨.

**현재 상태**
Reranker(BAAI/bge-reranker-base) 통합 완료. 평가 재실행 대기 중.

**예상 효과**
Dense Top-20 후보 안에는 정답 조문이 포함될 가능성 높음.
Cross-Encoder가 (질문, 조문제목+본문) 쌍을 직접 비교하여 재정렬 → Top-5 진입 기대.

**다음 액션**
```powershell
python evaluate_rag.py --mode both
```
Dense vs Reranker 비교 결과로 개선 여부 확인.

---

### #002 🔴 Hallucination 평가 False Positive

**현상**
평가 로직이 답변 내 조문번호를 추출하여 검색 조문 범위 밖이면 Hallucination으로 판정.
실제 Hallucination이 아닌 조문 본문 내 상호참조 구문을 오탐함.

```
[09] 답변에 "제76조에 따라..." 포함 → 미지원 조문으로 판정
     실제: 산업안전보건법 제4조 본문에 "제76조" 상호참조 포함

[15] 답변에 "제1조" 포함 → 미지원 조문으로 판정
     실제: 조문 본문에 다른 조항 인용 구문 포함
```

**현재 상태**
미해결. 평가 결과에서 Hallucination 10% (2/20)으로 기록되어 있으나 실제 Hallucination 0건으로 추정.

**개선 방향**
조문 본문에서 상호참조 번호를 사전 추출하여 허용 목록 확장:
```python
# 현재
source_articles = {조문의 article 필드에서 추출한 번호}

# 개선
source_articles = {article 필드} ∪ {본문 내 상호참조 번호}
```

또는 평가 기준을 완화:
- 답변에 등장하는 조문번호가 검색된 조문의 **법률명과 일치하면** 허용

---

### #003 🔵 Hybrid Search 미적용

**현상**
현재 파이프라인은 Dense Search만 사용. BM25 키워드 매칭이 없어
"적용", "범위", "목적" 같은 조문 제목 키워드 일치 검색에 약점.

**이미 구현된 것**
`hybrid_search.py` — BM25 + Dense + RRF 결합 코드 존재.
`reranker.py` — 독립 파일로 존재.

**적용 조건**
Reranker 평가 결과에서도 검색 미스가 남는 경우 도입 검토.

**구조**
```
BM25 Top-K + Dense Top-K
        ↓
    RRF 결합
        ↓
   Hybrid Top-20
        ↓
    Reranker
        ↓
     Top-5
```

---

## 평가 품질

### #004 🔴 평가 질문 수 부족

**현상**
현재 평가 데이터가 20개로, 법률당 5개(제1조~제5조 또는 유사 조문).
실제 사용 패턴(중간 조문, 복합 질문, 법률 간 비교 등)을 반영하지 못함.

**개선 방향**
- 법률당 10~20개로 확장
- 적용범위, 벌칙, 절차, 복합 조문 등 다양한 유형 포함
- 정답이 여러 조문에 걸쳐 있는 복합 질문 추가

---

### #005 🔴 LLM 답변 품질 정량 평가 미흡

**현상**
현재 평가 지표:
- 검색 히트 (chunk_id 일치)
- 출처 포함 여부 (조문번호 패턴)
- Hallucination (조문번호 범위 초과)

LLM이 조문 내용을 얼마나 정확히 반영했는지, 답변이 질문에 충분히 응답했는지는 평가하지 않음.

**개선 방향**
- ROUGE / BERTScore 기반 답변-조문 유사도 측정
- LLM-as-Judge: 다른 LLM이 답변 품질을 0~5 점수로 평가
- 사람이 직접 검토하는 Human Eval 샘플 추가

---

## 성능

### #006 🔴 LLM 응답 속도 느림

**현상**
평균 LLM 응답 시간 14.84초 (gemma3:4b, CPU 추론).
일부 질문은 40초 이상 소요 (복잡한 조문 Context 포함 시).

**환경**
- CPU 추론 (NVIDIA GPU 없음, AMD Radeon 860M — CUDA 미지원)
- gemma3:4b 모델, temperature=0

**개선 방향**
- 더 작은 모델 테스트 (gemma3:1b, qwen2.5:3b 등)
- Context 길이 최적화 (현재 Top-5 전체 → 핵심 조문만 선택)
- GPU 환경에서 재테스트

---

### #007 🔴 Streamlit 실시간 스트리밍 미구현

**현상**
`frontend/app.py`에서 ChatOllama는 동기 invoke로 전체 답변 완성 후 반환.
단어 단위 점진 출력으로 스트리밍 효과를 모사하고 있으나 실제 스트리밍 아님.
답변 생성 완료 전까지 사용자가 빈 화면을 봄.

**개선 방향**
`ChatOllama.astream()` 또는 `stream()` 사용:
```python
for chunk in _llm.stream(prompt):
    streamed += chunk.content
    placeholder.markdown(streamed + "▌")
```

---

## 코드 구조

### #008 🟢 retriever.py 분리 → rag_pipeline.py 통합 (해결됨)

**현상**
초기 구현에서 `retriever.py`와 `rag_pipeline.py`로 분리.
실행 위치에 따라 임포트 경로 불일치로 오류 발생.

**해결**
모든 RAG 로직을 `rag_pipeline.py` 단일 파일로 통합. `retriever.py` 삭제.

---

### #009 🔵 FastAPI 백엔드 미연동

**현상**
`backend/main.py`에 FastAPI 기본 구조만 있고 RAG 파이프라인과 연결되지 않음.

**개선 방향**
```python
# backend/main.py
@app.post("/query")
async def query(request: QueryRequest):
    result = run_rag(request.question, law=request.law)
    return result
```

---

### #010 🔵 Vision 모듈 미구현

**현상**
`app/vision/` 폴더가 비어 있음. 현장 이미지 분석 기능 미구현.
`frontend/app.py`에 이미지 업로드 UI가 있으나 연결 없음.

**개선 방향**
- 이미지 → LLaVA 또는 Qwen2-VL로 위험 요소 분석
- 분석 결과 + 관련 법령 검색 연계

---

## 개선 우선순위

| 우선순위 | 이슈 | 난이도 | 예상 효과 |
|:---:|---|:---:|---|
| 1 | #001 Reranker 평가 실행 | 낮음 | 검색 Top-5 히트율 개선 |
| 2 | #002 Hallucination 평가 로직 수정 | 낮음 | 평가 신뢰도 향상 |
| 3 | #007 실시간 스트리밍 | 낮음 | UX 개선 |
| 4 | #005 LLM 답변 품질 평가 | 중간 | 평가 완성도 향상 |
| 5 | #003 Hybrid Search 적용 | 중간 | 검색 품질 추가 개선 |
| 6 | #004 평가 데이터 확장 | 중간 | 평가 신뢰도 향상 |
| 7 | #009 FastAPI 연동 | 중간 | 서비스 구조 완성 |
| 8 | #006 LLM 속도 개선 | 높음 | 사용성 개선 |
| 9 | #010 Vision 모듈 | 높음 | 핵심 기능 완성 |
