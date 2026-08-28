"""
rag_pipeline.py — SafeAgent 전체 RAG 파이프라인 (단일 파일)

Dense-only 모드:
    질문 → bge-m3 → Qdrant Dense(Top-5) → Context → LLM

Reranker 모드:
    질문 → bge-m3 → Qdrant Dense(Top-20) → Cross-Encoder → Top-5 → Context → LLM

사용 예:
    from rag_pipeline import init_pipeline, run_rag

    # Dense-only (기본)
    init_pipeline()
    result = run_rag("근로자와 사용자는 어떻게 정의되나요?", law="근로기준법")

    # Reranker 사용
    init_pipeline(use_reranker=True)
    result = run_rag("어떤 사업장에 적용되나요?", law="산업안전보건법", use_reranker=True)
"""

from __future__ import annotations

import time
from typing import Optional

from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate


# ============================================================
# 설정
# ============================================================

QDRANT_HOST     = "localhost"
QDRANT_PORT     = 6333
COLLECTION_NAME = "legal_documents_bge_m3"

EMBEDDING_MODEL = "BAAI/bge-m3"
VECTOR_SIZE     = 1024

RERANKER_MODEL  = "BAAI/bge-reranker-base"
RERANKER_MAX_LEN = 512

LLM_MODEL       = "gemma3:4b"
LLM_TEMPERATURE = 0

# Dense-only 모드: Qdrant에서 바로 가져올 조문 수
DENSE_TOP_K     = 5

# Reranker 모드: Dense 후보 수 → Cross-Encoder → 최종 조문 수
DENSE_CANDIDATE_K = 20   # Qdrant에서 가져올 후보
RERANK_TOP_K      = 5    # Cross-Encoder 통과 후 최종


# ============================================================
# Prompt
# ============================================================

PROMPT_TEMPLATE = PromptTemplate.from_template(
    """당신은 대한민국 노동법 전문 AI 보조원입니다.
아래 [법률 조문]에 제공된 내용만을 근거로 사용자 질문에 답변하십시오.

[답변 원칙]
1. 반드시 아래 [법률 조문]에 명시된 내용만 근거로 사용하십시오.
2. 조문에 없는 내용은 절대 추측하거나 만들어내지 마십시오.
3. 제공된 조문만으로 답변할 수 없는 경우 "제공된 조문에서 해당 내용을 확인할 수 없습니다."라고 답하십시오.
4. 답변 마지막에 참고한 법률명과 조문 번호를 반드시 명시하십시오.
5. 답변은 명확하고 간결하게 작성하되, 조문의 핵심 내용을 빠짐없이 포함하십시오.

[법률 조문]
{context}

[사용자 질문]
{question}

[답변]
"""
)


# ============================================================
# 싱글톤
# ============================================================

_embedding_model: Optional[SentenceTransformer] = None
_reranker_model:  Optional[CrossEncoder]         = None
_qdrant_client:   Optional[QdrantClient]         = None
_llm:             Optional[ChatOllama]            = None


# ============================================================
# 초기화
# ============================================================

def init_pipeline(
    use_reranker:    bool  = False,
    qdrant_host:     str   = QDRANT_HOST,
    qdrant_port:     int   = QDRANT_PORT,
    embedding_model: str   = EMBEDDING_MODEL,
    reranker_model:  str   = RERANKER_MODEL,
    llm_model:       str   = LLM_MODEL,
    temperature:     float = LLM_TEMPERATURE,
) -> None:
    """
    bge-m3 임베딩, Qdrant, LLM을 초기화한다.
    use_reranker=True 이면 Cross-Encoder도 함께 로드한다.
    이미 초기화된 리소스는 재사용한다.
    """
    global _embedding_model, _reranker_model, _qdrant_client, _llm

    if _embedding_model is None:
        print(f"[RAG] bge-m3 로드 중: {embedding_model}")
        _embedding_model = SentenceTransformer(embedding_model)
        print("[RAG] bge-m3 로드 완료")

    if use_reranker and _reranker_model is None:
        print(f"[RAG] Reranker 로드 중: {reranker_model}")
        _reranker_model = CrossEncoder(
            reranker_model,
            max_length=RERANKER_MAX_LEN,
        )
        print("[RAG] Reranker 로드 완료")

    if _qdrant_client is None:
        print(f"[RAG] Qdrant 연결: {qdrant_host}:{qdrant_port}")
        _qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        info = _qdrant_client.get_collection(COLLECTION_NAME)
        print(
            f"[RAG] Qdrant 연결 완료 "
            f"(Collection={COLLECTION_NAME}, Points={info.points_count})"
        )

    if _llm is None:
        print(f"[RAG] LLM 로드: {llm_model}")
        _llm = ChatOllama(model=llm_model, temperature=temperature)
        print("[RAG] LLM 준비 완료")


# ============================================================
# Dense Search
# ============================================================

def _search(
    query: str,
    law:   Optional[str] = None,
    top_k: int           = DENSE_TOP_K,
) -> list[dict]:
    """
    bge-m3로 질문을 임베딩하고 Qdrant에서 유사 조문을 검색한다.

    Returns
    -------
    각 항목: chunk_id, law, article, article_title, text, score
    """
    query_vector = _embedding_model.encode(  # type: ignore[union-attr]
        query,
        normalize_embeddings=True,
    ).tolist()

    query_filter = None
    if law:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="law",
                    match=MatchValue(value=law),
                )
            ]
        )

    points = _qdrant_client.query_points(  # type: ignore[union-attr]
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    ).points

    docs = []
    for rank, point in enumerate(points, start=1):
        payload = point.payload or {}
        docs.append(
            {
                "dense_rank":    rank,
                "chunk_id":      payload.get("chunk_id", ""),
                "law":           payload.get("law", ""),
                "article":       payload.get("article", ""),
                "article_title": payload.get("article_title", ""),
                "text":          payload.get("text", ""),
                "score":         round(float(point.score), 4),
            }
        )
    return docs


# ============================================================
# Reranker
# ============================================================

def _rerank(
    query:      str,
    candidates: list[dict],
    top_k:      int = RERANK_TOP_K,
) -> list[dict]:
    """
    Dense 후보 리스트를 Cross-Encoder 점수로 재정렬한다.

    passage 형식: "[조문번호 제목] 본문" — 조문 제목을 앞에 붙여
    Cross-Encoder가 질문 키워드(정의, 적용범위 등)와 직접 비교하도록 한다.

    Returns
    -------
    rerank_score 기준 내림차순 정렬된 상위 top_k 결과.
    각 항목에 rerank_score, rerank_rank, dense_rank 포함.
    """
    if not candidates:
        return []

    # passage 구성: "[조문번호 제목] 본문"
    pairs = []
    for item in candidates:
        article = item.get("article", "")
        title   = item.get("article_title", "")
        text    = item.get("text", "")

        if article and title:
            passage = f"[{article} {title}] {text}"
        elif article:
            passage = f"[{article}] {text}"
        else:
            passage = text

        pairs.append((query, passage))

    scores = _reranker_model.predict(pairs)  # type: ignore[union-attr]

    scored = sorted(
        zip(scores, candidates),
        key=lambda x: float(x[0]),
        reverse=True,
    )

    results = []
    for rerank_rank, (score, item) in enumerate(scored[:top_k], start=1):
        result = dict(item)
        result["rerank_score"] = round(float(score), 4)
        result["rerank_rank"]  = rerank_rank
        # dense_rank 는 _search 에서 이미 포함됨
        results.append(result)

    return results


# ============================================================
# Context 구성
# ============================================================

def _build_context(docs: list[dict]) -> str:
    """
    조문 리스트를 LLM 입력용 Context 문자열로 변환한다.

    형식:
        [1] 근로기준법 제2조(정의)
        제2조(정의) 이 법에서 사용하는 용어의 뜻은...
    """
    if not docs:
        return "검색된 관련 조문이 없습니다."

    blocks = []
    for idx, doc in enumerate(docs, start=1):
        law     = doc.get("law", "")
        article = doc.get("article", "")
        title   = doc.get("article_title", "")
        text    = doc.get("text", "")

        if title:
            header = f"[{idx}] {law} {article}({title})"
        elif article:
            header = f"[{idx}] {law} {article}"
        else:
            header = f"[{idx}] {law}"

        blocks.append(f"{header}\n{text}")

    return "\n\n".join(blocks)


# ============================================================
# 핵심 함수 — 전체 파이프라인
# ============================================================

def run_rag(
    question:     str,
    law:          Optional[str] = None,
    use_reranker: bool          = False,
    verbose:      bool          = False,
) -> dict:
    """
    질문 → 검색 → (Reranker) → Context → LLM 전체 파이프라인.

    Parameters
    ----------
    question     : 사용자 질문
    law          : 법률명 필터, None 이면 전체 법률 검색
    use_reranker : True 이면 Dense Top-20 → Reranker → Top-5
                   False 이면 Dense Top-5 직접 사용
    verbose      : True 이면 단계별 로그 출력

    Returns
    -------
    {
        "question"      : 입력 질문,
        "law_filter"    : 적용된 법률 필터,
        "mode"          : "reranker" | "dense",
        "docs"          : 최종 조문 리스트 (Context 구성에 사용된 것),
        "context"       : LLM 에 전달된 Context,
        "sources"       : 출처 요약 리스트,
        "answer"        : LLM 답변,
        "retrieval_sec" : 검색(+재정렬) 소요 시간(초),
        "llm_sec"       : LLM 소요 시간(초),
    }
    """
    # 초기화 보장
    if _embedding_model is None or _qdrant_client is None or _llm is None:
        init_pipeline(use_reranker=use_reranker)

    if use_reranker and _reranker_model is None:
        init_pipeline(use_reranker=True)

    # --------------------------------------------------------
    # Step 1. Dense Search
    # --------------------------------------------------------
    t0 = time.perf_counter()

    if use_reranker:
        # Reranker 모드: 후보 20개 가져오기
        candidates = _search(query=question, law=law, top_k=DENSE_CANDIDATE_K)

        if verbose:
            print(
                f"[RAG] Dense 후보 {len(candidates)}개 검색 완료"
            )
            for i, d in enumerate(candidates[:5], 1):
                print(
                    f"  [{i}] score={d['score']:.4f}  {d['chunk_id']}"
                )
            if len(candidates) > 5:
                print(f"  ... +{len(candidates) - 5}개 더")

        # --------------------------------------------------------
        # Step 2. Reranker
        # --------------------------------------------------------
        docs = _rerank(query=question, candidates=candidates, top_k=RERANK_TOP_K)
        mode = "reranker"

        if verbose:
            print(f"[RAG] Reranker Top-{RERANK_TOP_K}:")
            for d in docs:
                print(
                    f"  [{d['rerank_rank']}] "
                    f"rerank={d['rerank_score']:.4f} "
                    f"dense={d['dense_rank']}위  "
                    f"{d['chunk_id']}"
                )

    else:
        # Dense-only 모드: Top-5 직접 사용
        docs = _search(query=question, law=law, top_k=DENSE_TOP_K)
        mode = "dense"

        if verbose:
            print(f"[RAG] Dense Top-{DENSE_TOP_K}:")
            for d in docs:
                print(
                    f"  [{d['dense_rank']}] "
                    f"score={d['score']:.4f}  "
                    f"{d['chunk_id']}"
                )

    retrieval_sec = round(time.perf_counter() - t0, 2)

    if verbose:
        print(f"[RAG] 검색 완료 ({retrieval_sec}초)")

    # --------------------------------------------------------
    # Step 3. Context 구성
    # --------------------------------------------------------
    context = _build_context(docs)
    sources = [
        f"{d['law']} {d['article']}"
        for d in docs
        if d.get("law") and d.get("article")
    ]

    # --------------------------------------------------------
    # Step 4. LLM 호출
    # --------------------------------------------------------
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    t1 = time.perf_counter()
    response = _llm.invoke(prompt)  # type: ignore[union-attr]
    llm_sec  = round(time.perf_counter() - t1, 2)
    answer   = response.content.strip()

    if verbose:
        print(f"[RAG] LLM 완료 ({llm_sec}초)")

    return {
        "question":      question,
        "law_filter":    law,
        "mode":          mode,
        "docs":          docs,
        "context":       context,
        "sources":       sources,
        "answer":        answer,
        "retrieval_sec": retrieval_sec,
        "llm_sec":       llm_sec,
    }


# ============================================================
# 직접 실행 — Dense vs Reranker 비교 테스트
# ============================================================

if __name__ == "__main__":

    # Dense-only + Reranker 모두 초기화
    init_pipeline(use_reranker=True)

    # 검색 미스 케이스에 집중하여 Reranker 효과 확인
    test_cases = [
        # Dense 검색 미스 케이스
        ("산업안전보건법은 어떤 사업장에 적용되나요?",    "산업안전보건법"),
        ("산업재해보상보험법은 어떤 사업에 적용되나요?",  "산업재해보상보험법"),
        # 정상 케이스
        ("근로기준법에서 근로자와 사용자는 어떻게 정의되나요?", "근로기준법"),
    ]

    for question, law in test_cases:
        print()
        print("=" * 60)
        print(f"질문 : {question}")
        print(f"필터 : {law}")
        print("=" * 60)

        dense_result = run_rag(
            question=question, law=law, use_reranker=False, verbose=True
        )
        print(f"\n[Dense]    출처: {', '.join(dense_result['sources'][:3])}")
        print(f"[Dense]    소요: 검색 {dense_result['retrieval_sec']}초")

        print()
        rerank_result = run_rag(
            question=question, law=law, use_reranker=True, verbose=True
        )
        print(f"\n[Reranker] 출처: {', '.join(rerank_result['sources'][:3])}")
        print(f"[Reranker] 소요: 검색 {rerank_result['retrieval_sec']}초")

        print()
        print("── 최종 답변 (Reranker) ──")
        print(rerank_result["answer"])
