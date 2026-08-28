# SafeAgent — 에러 로그 및 해결 과정

> 구현 단계 순서로 기록. 각 항목은 발생 맥락 → 에러 메시지 → 원인 분석 → 해결 방법 순으로 작성.

---

## 구현 단계 1. 환경 구축

### ERR-01 🟢 PowerShell에서 `conda` 명령 미인식

**발생 맥락**
Kiro IDE 터미널(PowerShell)에서 `conda activate safeagent` 실행 시.

**에러 메시지**
```
conda : 'conda' 용어가 cmdlet, 함수, 스크립트 파일 또는 실행할 수 있는 프로그램
이름으로 인식되지 않습니다.
```

**원인 분석**
Miniconda가 `C:\Users\guswj\miniconda3`에 설치되어 있으나,
PowerShell `PATH`에 conda Scripts 경로가 등록되지 않은 상태.
Conda 초기화(`conda init powershell`)가 실행되지 않았음.

**해결 방법**
`conda.exe` 절대 경로로 환경 목록 확인 후, 해당 환경의 `python.exe`를 직접 호출.

```powershell
# 환경 목록 확인
& "C:\Users\guswj\miniconda3\Scripts\conda.exe" env list
# → safeagent  C:\Users\guswj\miniconda3\envs\safeagent

# 이후 모든 실행에 safeagent python.exe 직접 지정
& "C:\Users\guswj\miniconda3\envs\safeagent\python.exe" app\rag\rag_pipeline.py
```

---

### ERR-02 🟢 `ModuleNotFoundError: No module named 'sentence_transformers'`

**발생 맥락**
`python app\rag\rag_pipeline.py` 실행 시. ERR-01 해결 전 시스템 Python으로 실행된 경우.

**에러 메시지**
```
Traceback (most recent call last):
  File "C:\SafeAgent\app\rag\rag_pipeline.py", line 1, in <module>
    from sentence_transformers import SentenceTransformer
ModuleNotFoundError: No module named 'sentence_transformers'
```

**원인 분석**
시스템 Python(`C:\Users\guswj\AppData\Local\Programs\Python\Python311\python.exe`)으로
실행. `sentence_transformers`, `qdrant_client`, `langchain_ollama` 등 프로젝트 의존성은
모두 Conda `safeagent` 환경에만 설치됨.

**해결 방법**
ERR-01 해결(safeagent python.exe 직접 호출)로 함께 해결됨.

---

### ERR-03 🟢 CUDA 미인식 — CPU 추론 전환

**발생 맥락**
LLM 연결 및 임베딩 모델 로드 초기 테스트 단계.

**확인 내용**
```python
import torch
print(torch.cuda.is_available())  # False
```

**원인 분석**
탑재 GPU가 AMD Radeon 860M. CUDA는 NVIDIA GPU 전용이므로 미지원.
PyTorch CUDA 빌드가 설치되어 있어도 AMD GPU에서는 `cuda.is_available() = False`.

**영향**
- bge-m3 임베딩: CPU 추론 → 처리 속도 저하
- LLM (gemma3:4b via Ollama): Ollama가 내부적으로 CPU 추론 처리
- Reranker (CrossEncoder): CPU 추론

**해결 방법**
GPU 가속 없이 CPU 추론으로 진행. 모델 크기를 4B로 제한하여 허용 범위 내 속도 확보.

---

## 구현 단계 2. LLM 연결

### ERR-04 🟢 Ollama 모델이 OpenWebUI에 표시되지 않음

**발생 맥락**
Ollama에 `gemma3:4b` pull 후 OpenWebUI에서 모델 선택 불가.

**원인 분석**
OpenWebUI와 Ollama의 연결 설정 불일치.
OpenWebUI가 Ollama API 엔드포인트(`http://localhost:11434`)에 접근하지 못하거나
모델 목록 갱신이 필요한 상태.

**해결 방법**
1. Ollama 서비스 재시작
2. OpenWebUI 설정에서 Ollama API URL 재확인 (`http://localhost:11434`)
3. 브라우저 새로고침 후 모델 목록 재조회

---

### ERR-05 🟢 PowerShell 스크립트 실행 정책 오류

**발생 맥락**
OpenWebUI `.venv` 환경 활성화 시 (`.\venv\Scripts\Activate.ps1`).

**에러 메시지**
```
.\venv\Scripts\Activate.ps1 : 이 시스템에서 스크립트를 실행할 수 없으므로...
파일을 로드할 수 없습니다.
```

**원인 분석**
Windows PowerShell 기본 실행 정책이 `Restricted`로 설정되어 있어 `.ps1` 스크립트 실행 차단.

**해결 방법**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

또는 활성화 없이 직접 `python.exe` 절대 경로 사용으로 우회.

---

## 구현 단계 3. PDF 처리 및 청킹

### ERR-06 🟢 `chunks.json` 경로 오류

**발생 맥락**
`ingestion.py` 실행 시 처리된 청크 JSON 파일을 찾지 못함.

**에러 메시지**
```
FileNotFoundError: [Errno 2] No such file or directory:
'data/processed/근로기준법.../chunks.json'
```

**원인 분석**
`BASE_DIR` 설정이 스크립트 위치(`app/rag/`) 기준이 아닌 실행 위치 기준으로 구성됨.
`app/rag/`에서 실행 시 `data/processed/` 경로가 `app/rag/data/...`로 해석됨.

**해결 방법**
`Path(__file__).resolve().parents[N]`으로 프로젝트 루트를 정확히 지정.

```python
# 수정 전
BASE_DIR = Path("data/processed")

# 수정 후
BASE_DIR = Path(__file__).resolve().parents[2]  # C:\SafeAgent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
```

---

### ERR-07 🟢 법률 PDF 한글 파일명 경로 처리 문제

**발생 맥락**
`data/documents/` 내 한글 파일명 PDF 처리 시.

**파일명 예시**
```
근로기준법(법률)(제21533호)(20270101).pdf
산업안전보건법(법률)(제21374호)(20260801).pdf
```

**원인 분석**
Windows 파일 시스템에서는 문제없으나, 일부 Python 라이브러리에서 한글 경로 처리 시
인코딩 문제 또는 경로 파싱 오류 발생 가능.

**해결 방법**
`pathlib.Path` 사용으로 플랫폼 독립적 경로 처리. 파일명을 직접 문자열 연산하지 않고
`Path` 객체로 일관되게 처리.

---

## 구현 단계 4. Embedding 및 Qdrant 구축

### ERR-08 🟢 bge-m3 임베딩 중 `KeyboardInterrupt`

**발생 맥락**
`embed_bge_m3.py`로 법률 청크 임베딩 중 수동 중단.

**상황**
```
근로기준법  130개 → 약 56초 완료
산업안전보건법 187개 → 약 1분 35초 경과 후 Ctrl+C
```

**원인 분석**
CPU 환경에서 bge-m3(1024차원) 임베딩 처리 속도가 느림.
초기 구현이 배치 처리 없이 순차 처리 → 중간 저장 없이 진행되어 중단 시 전체 손실.

**해결 방법**
1. **배치 단위 처리**: `batch_size=32`로 청크를 나눠서 처리
2. **즉시 저장**: 배치 완료 시 즉시 Qdrant에 저장 (중간 손실 방지)
3. **중복 제거**: 재실행 시 이미 저장된 `chunk_id` skip

```python
# 기존 chunk_id 조회
existing_ids = get_existing_chunk_ids(client, collection_name)

# 이미 처리된 청크 건너뜀
for batch in batches:
    batch = [c for c in batch if c["chunk_id"] not in existing_ids]
    if not batch:
        continue
    # 임베딩 후 저장
    process_batch(batch, model, client, collection_name)
```

**결과**
재실행 시 중단 지점부터 이어서 처리 가능. 전체 514개 정상 구축 완료.

```
근로기준법             130
산업안전보건법         187
산업재해보상보험법     161
직업교육훈련 촉진법     36
────────────────────────
전체                  514
```

---

### ERR-09 🟢 Qdrant Collection 재생성 충돌

**발생 맥락**
bge-m3 임베딩 실험 코드 초기 버전 실행 시.

**원인 분석**
초기 코드가 실행 시마다 Collection을 `recreate_collection()`으로 재생성.
기존 `legal_documents_bge_m3`에 저장된 데이터가 삭제됨.

**에러 메시지** (간접 확인)
```
# Collection 재생성 후 Point 수 = 0
Collection: legal_documents_bge_m3
Point Count: 0
```

**해결 방법**
Collection 존재 여부 확인 후 없을 때만 생성.

```python
def prepare_collection(client, collection_name):
    if client.collection_exists(collection_name):
        print(f"Collection 이미 존재: {collection_name}")
        return  # 재생성하지 않음

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
```

---

### ERR-10 🟢 HuggingFace Hub 인증 경고

**발생 맥락**
`SentenceTransformer("BAAI/bge-m3")` 및 `CrossEncoder("BAAI/bge-reranker-base")` 로드 시.

**에러 메시지** (경고)
```
Warning: You are sending unauthenticated requests to the HF Hub.
Please set a HF_TOKEN to enable higher rate limits and faster downloads.
```

**원인 분석**
HuggingFace 토큰 미설정 시 출력되는 경고. 기능에는 영향 없음.
모델이 이미 로컬 캐시(`~/.cache/huggingface/`)에 저장되어 있어 실제 다운로드 발생하지 않음.

**해결 방법**
개발 환경에서는 무시. 운영 환경에서는 환경변수 설정.

```powershell
$env:HF_TOKEN = "hf_..."
```

---

## 구현 단계 5. RAG 파이프라인 구현

### ERR-11 🟢 `retriever.py` 임포트 실패 (test_rag_pipeline.py)

**발생 맥락**
기존 `test_rag_pipeline.py` 실행 시.

**관련 코드**
```python
try:
    from app.rag.retriever import search_query
except ImportError:
    # fallback: 하드코딩 더미 데이터 반환
    def search_query(...):
        return [{"law": "근로기준법", "article": "제2조(정의)", "text": "..."}]
```

**원인 분석**
`retriever.py`가 아직 구현되지 않아 `ImportError` 발생.
`except` 블록의 더미 데이터로 실행되어 표면적으로 정상 동작처럼 보였음.
실제로는 Qdrant 검색 없이 하드코딩 데이터를 LLM에 전달하고 있었음.

**해결 방법**
1. `retriever.py` 신규 구현
2. 이후 `rag_pipeline.py` 단일 파일 통합으로 구조 자체를 개선
3. `test_rag_pipeline.py`는 레거시 파일로 분류, `rag_pipeline.py`로 대체

---

### ERR-12 🟢 실행 위치별 임포트 경로 불일치

**발생 맥락**
`retriever.py`와 `rag_pipeline.py`가 분리된 상태에서 다양한 경로로 실행 시.

**문제 상황**
```
# app/rag/ 에서 실행 시
from app.rag.retriever import search_query  # ❌ 실패 (app.rag 없음)

# 프로젝트 루트에서 실행 시
from app.rag.retriever import search_query  # ✓ 성공
```

**원인 분석**
임포트 경로가 실행 위치(cwd)에 따라 달라짐.
IDE에서 파일을 직접 실행하면 파일 위치 기준으로 cwd가 설정됨.

**해결 방법**
모든 RAG 로직을 `rag_pipeline.py` 단일 파일로 통합.
`retriever.py` 삭제. 외부에서 임포트 시 `sys.path` 명시적 설정.

```python
# evaluate_rag.py
_HERE = Path(__file__).resolve().parent  # app/rag/
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from rag_pipeline import init_pipeline, run_rag  # 항상 성공
```

---

## 구현 단계 6. 평가

### ERR-13 🟡 Dense 검색 미스 — 적용범위 질문 (20개 평가 결과)

**발생 맥락**
`evaluate_rag.py` 전체 20개 질문 평가 실행 결과.

**실제 출력**
```
[08/20] 산업안전보건법은 어떤 사업장에 적용되나요?
  검색 히트  [✗] 미포함  |  기대: ...본문_제3조
  → 실제 Top-3: 제24조, 제22조, 제49조

[13/20] 산업재해보상보험법은 어떤 사업에 적용되나요?
  검색 히트  [✗] 미포함  |  기대: ...본문_제6조
  → 실제 Top-3: 제1조, 제124조, 제2조
```

**원인 분석**
Dense Search 특성상 "어떤 사업장에 적용되나요?"라는 질문이
"적용범위"를 명시하는 제3조/제6조보다 다른 조문 본문과 더 높은 코사인 유사도를 가짐.

**현재 상태**
Reranker 통합 완료. 평가 재실행으로 개선 여부 확인 예정. → ISSUES.md #001

---

### ERR-14 🟡 Hallucination 평가 False Positive

**발생 맥락**
`evaluate_rag.py` 결과에서 Hallucination 의심 2건 발생.

**실제 출력**
```
[09/20] Hallucination [⚠] 의심: ['제76조']
  근거 조문: ['제1조', '제24조', '제4조', '제62조', '제6조']
  답변 조문: ['제4조', '제76조']

[15/20] Hallucination [⚠] 의심: ['제1조']
  근거 조문: ['제107조', '제121조', '제2조', '제8조', '제95조']
  답변 조문: ['제1조', '제95조']
```

**원인 분석**
평가 로직이 LLM 답변에서 `제\d+조` 패턴을 추출하여
검색 조문의 `article` 필드와 비교. 검색 조문의 **본문 내 상호참조 구문**
(예: "제76조에 따른 기술지원...") 은 `article` 필드에 없으므로 미지원으로 판정됨.

실제로 LLM은 조문 본문을 충실히 인용했고, 조문 본문에 상호참조 번호가 포함되어 있음.
→ 실제 Hallucination 0건으로 판단.

**현재 상태**
평가 로직 개선 필요. → ISSUES.md #002

---

## 전체 평가 기준선 (Dense-only)

```
모델    : gemma3:4b (Ollama, CPU)
임베딩  : BAAI/bge-m3 (CPU)
DB      : Qdrant local (514 chunks)
평가셋  : 20개 질문 (legal_queriesV3.json)

검색 Top-5 히트율  : 90.0%  (18/20)
검색 Top-1 히트율  : 80.0%  (16/20)
출처 포함 답변     : 100.0% (20/20)
Hallucination 의심 : 10.0%  (2/20) ← False Positive로 추정
평균 검색 시간     : 0.14초
평균 LLM 시간      : 14.84초
총 소요 시간       : 299.6초
```

**다음 비교 기준선 (Reranker)**
```powershell
cd C:\SafeAgent\app\rag
& "C:\Users\guswj\miniconda3\envs\safeagent\python.exe" evaluate_rag.py --mode both
```
