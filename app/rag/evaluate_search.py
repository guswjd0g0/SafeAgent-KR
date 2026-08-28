import json
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from hybrid_search import (
    hybrid_search,
    CANDIDATE_K,
    RRF_K,
)

from reranker import (
    Reranker,
    attach_texts,
)


# ============================================================
# Path
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

EVALUATION_PATH = (
    BASE_DIR
    / "data"
    / "evaluation"
    / "legal_queriesV2.json"
)

# Dense 전체 검색 결과
RESULT_PATH_DENSE = (
    BASE_DIR
    / "data"
    / "evaluation"
    / "search_results_dense.json"
)

# Dense + Law Filter 검색 결과
RESULT_PATH_FILTER = (
    BASE_DIR
    / "data"
    / "evaluation"
    / "search_results_filter.json"
)

# Hybrid + Law Filter 검색 결과
RESULT_PATH_HYBRID = (
    BASE_DIR
    / "data"
    / "evaluation"
    / "search_results_hybrid.json"
)

# Hybrid + Law Filter + Reranker 검색 결과
RESULT_PATH_RERANKER = (
    BASE_DIR
    / "data"
    / "evaluation"
    / "search_results_reranker.json"
)


# ============================================================
# Hybrid Top-K (Reranker 후보 확보용)
# ============================================================

HYBRID_RECALL_K = 20


# ============================================================
# Qdrant 설정
# ============================================================

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

COLLECTION_NAME = "legal_documents"


# ============================================================
# Embedding 설정
# ============================================================

EMBEDDING_MODEL = "nomic-embed-text"


# ============================================================
# Search 설정
# ============================================================

TOP_K = 5


# ============================================================
# 평가 데이터 로드
# ============================================================

def load_evaluation_queries():

    print()
    print("=" * 70)
    print("평가 데이터 로드")
    print("=" * 70)

    print()
    print(f"파일: {EVALUATION_PATH}")

    # --------------------------------------------------------
    # 파일 존재 확인
    # --------------------------------------------------------

    if not EVALUATION_PATH.exists():

        raise FileNotFoundError(
            "\n평가 파일을 찾을 수 없습니다.\n"
            f"경로: {EVALUATION_PATH}"
        )

    # --------------------------------------------------------
    # JSON 로드
    # --------------------------------------------------------

    with open(
        EVALUATION_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        queries = json.load(f)

    # --------------------------------------------------------
    # 구조 확인
    # --------------------------------------------------------

    if not isinstance(queries, list):

        raise ValueError(
            "평가 파일의 최상위 구조는 "
            "list여야 합니다."
        )

    if not queries:

        raise ValueError(
            "평가 질문이 비어 있습니다."
        )

    print(
        f"평가 질문 수: "
        f"{len(queries)}"
    )

    # --------------------------------------------------------
    # 각 질문 필수 필드 검사
    # --------------------------------------------------------

    required_fields = [
        "id",
        "query",
        "expected_law",
        "expected_article",
        "expected_chunk_id",
    ]

    for index, item in enumerate(
        queries,
        start=1,
    ):

        if not isinstance(item, dict):

            raise ValueError(
                f"{index}번째 평가 데이터가 "
                "dictionary가 아닙니다."
            )

        for field in required_fields:

            if not item.get(field):

                raise ValueError(
                    f"{index}번째 평가 데이터에 "
                    f"'{field}'가 없습니다."
                )

    return queries


# ============================================================
# Qdrant 연결
# ============================================================

def connect_qdrant():

    print()
    print("=" * 70)
    print("Qdrant 연결")
    print("=" * 70)

    print()
    print(
        f"Host: {QDRANT_HOST}"
    )

    print(
        f"Port: {QDRANT_PORT}"
    )

    # --------------------------------------------------------
    # Qdrant Client
    # --------------------------------------------------------

    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
    )

    # --------------------------------------------------------
    # Collection 존재 확인
    # --------------------------------------------------------

    if not client.collection_exists(
        COLLECTION_NAME
    ):

        raise ValueError(
            "\nCollection이 없습니다.\n"
            f"Collection: {COLLECTION_NAME}"
        )

    # --------------------------------------------------------
    # Collection 정보
    # --------------------------------------------------------

    collection_info = client.get_collection(
        collection_name=COLLECTION_NAME
    )

    print()
    print("Qdrant 연결 성공")

    print(
        f"Collection: "
        f"{COLLECTION_NAME}"
    )

    print(
        f"Point Count: "
        f"{collection_info.points_count}"
    )

    return client


# ============================================================
# 단일 질문 검색
# ============================================================

def search_query(
    client,
    embedding,
    query,
    law_filter=None,
):

    # --------------------------------------------------------
    # Query Embedding
    # --------------------------------------------------------

    query_vector = embedding.embed_query(
        query
    )

    # --------------------------------------------------------
    # Law Filter 구성
    #
    # law_filter 가 주어지면 해당 법률명만 검색
    # --------------------------------------------------------

    qdrant_filter = None

    if law_filter:

        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="law",
                    match=MatchValue(
                        value=law_filter
                    ),
                )
            ]
        )

    # --------------------------------------------------------
    # Qdrant Vector Search
    # --------------------------------------------------------

    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=qdrant_filter,
        limit=TOP_K,
        with_payload=True,
    )

    return result.points


# ============================================================
# 정답 여부 확인
# ============================================================

def is_correct_result(
    payload,
    expected_chunk_id,
):

    if not payload:

        return False

    retrieved_chunk_id = payload.get(
        "chunk_id",
        "",
    )

    # --------------------------------------------------------
    # 가장 정확한 평가 기준
    #
    # expected_chunk_id와
    # 검색된 chunk_id가 완전히 동일해야 정답
    # --------------------------------------------------------

    return (
        retrieved_chunk_id
        == expected_chunk_id
    )


# ============================================================
# MRR 계산용 Reciprocal Rank
# ============================================================

def calculate_reciprocal_rank(
    correct_rank
):

    if correct_rank is None:

        return 0.0

    return 1.0 / correct_rank


# ============================================================
# 전체 평가
# ============================================================

def evaluate(
    client,
    embedding,
    queries,
    use_law_filter=False,
    use_hybrid=False,
    use_reranker=False,
    reranker=None,
):

    total = len(queries)

    # ========================================================
    # 정확도 카운터
    # ========================================================

    top1_correct = 0
    top3_correct = 0
    top5_correct = 0

    # ========================================================
    # MRR
    # ========================================================

    reciprocal_rank_sum = 0.0

    # ========================================================
    # Hybrid Top-20 Recall 카운터
    # (Reranker 모드에서만 의미 있음)
    # ========================================================

    top20_recall_correct = 0

    # ========================================================
    # 상세 결과
    # ========================================================

    results = []

    # ========================================================
    # 질문별 평가
    # ========================================================

    for index, item in enumerate(
        queries,
        start=1,
    ):

        query_id = item["id"]

        query = item["query"]

        expected_law = item[
            "expected_law"
        ]

        expected_article = item[
            "expected_article"
        ]

        expected_chunk_id = item[
            "expected_chunk_id"
        ]

        # ====================================================
        # 검색
        # ====================================================

        # Law Filter: expected_law 를 필터 조건으로 사용
        law_filter = (
            expected_law
            if (use_law_filter or use_hybrid or use_reranker)
            else None
        )

        # --------------------------------------------------------
        # Reranker: Hybrid Top-20 → Cross-Encoder 재정렬
        # --------------------------------------------------------

        if use_reranker:

            # Hybrid Top-20 후보 확보
            hybrid_candidates = hybrid_search(
                client=client,
                embedding=embedding,
                query=query,
                law_filter=law_filter,
                top_k=HYBRID_RECALL_K,
                candidate_k=CANDIDATE_K,
                rrf_k=RRF_K,
            )

            # Top-20 Recall 측정
            in_top20 = any(
                c["chunk_id"] == expected_chunk_id
                for c in hybrid_candidates
            )

            if in_top20:
                top20_recall_correct += 1

            # text([제N조 제목] 형식) 보강
            candidates_with_text = attach_texts(
                candidates=hybrid_candidates,
                client=client,
                collection_name=COLLECTION_NAME,
            )

            # Reranker 재정렬 → Top-5
            reranked = reranker.rerank(
                query=query,
                candidates=candidates_with_text,
                top_k=TOP_K,
            )

            points = reranked
            is_hybrid = True     # dict 형식이므로 동일 처리

        # --------------------------------------------------------
        # Hybrid Search
        # --------------------------------------------------------

        elif use_hybrid:

            hybrid_points = hybrid_search(
                client=client,
                embedding=embedding,
                query=query,
                law_filter=law_filter,
                top_k=TOP_K,
                candidate_k=CANDIDATE_K,
                rrf_k=RRF_K,
            )

            points = hybrid_points
            is_hybrid = True

        # --------------------------------------------------------
        # Dense Search (전체 or Law Filter)
        # --------------------------------------------------------

        else:

            points = search_query(
                client=client,
                embedding=embedding,
                query=query,
                law_filter=law_filter,
            )
            is_hybrid = False

        # ====================================================
        # 정답 순위 탐색
        # ====================================================

        correct_rank = None

        retrieved = []

        for rank, point in enumerate(
            points,
            start=1,
        ):

            # ------------------------------------------------
            # Hybrid: dict / Dense: Qdrant point 객체
            # ------------------------------------------------

            if is_hybrid:

                retrieved_chunk_id = point.get(
                    "chunk_id", ""
                )
                retrieved_law = point.get(
                    "law", ""
                )
                retrieved_article = point.get(
                    "article", ""
                )
                retrieved_file = point.get(
                    "file", ""
                )
                # Reranker 결과면 rerank_score, 아니면 rrf_score
                score = point.get(
                    "rerank_score",
                    point.get("rrf_score", 0.0),
                )

            else:

                payload = point.payload or {}

                retrieved_chunk_id = payload.get(
                    "chunk_id", ""
                )
                retrieved_law = payload.get(
                    "law", ""
                )
                retrieved_article = payload.get(
                    "article", ""
                )
                retrieved_file = payload.get(
                    "file", ""
                )
                score = point.score

            correct = is_correct_result(
                payload=(
                    {"chunk_id": retrieved_chunk_id}
                ),
                expected_chunk_id=expected_chunk_id,
            )

            # ------------------------------------------------
            # 정답 순위
            # ------------------------------------------------

            if (
                correct
                and correct_rank is None
            ):

                correct_rank = rank

            # ------------------------------------------------
            # 검색 결과 저장
            # ------------------------------------------------

            retrieved.append(
                {
                    "rank": rank,
                    "score": round(score, 4),
                    "chunk_id": retrieved_chunk_id,
                    "law": retrieved_law,
                    "article": retrieved_article,
                    "file": retrieved_file,
                    "correct": correct,
                }
            )

        # ====================================================
        # Accuracy
        # ====================================================

        top1 = (
            correct_rank == 1
        )

        top3 = (
            correct_rank is not None
            and correct_rank <= 3
        )

        top5 = (
            correct_rank is not None
            and correct_rank <= 5
        )

        if top1:

            top1_correct += 1

        if top3:

            top3_correct += 1

        if top5:

            top5_correct += 1

        # ====================================================
        # Reciprocal Rank
        # ====================================================

        reciprocal_rank = (
            calculate_reciprocal_rank(
                correct_rank
            )
        )

        reciprocal_rank_sum += (
            reciprocal_rank
        )

        # ====================================================
        # 결과 출력
        # ====================================================

        print()
        print("=" * 70)

        print(
            f"[{index}/{total}] "
            f"질문 ID: {query_id}"
        )

        print()

        print(
            f"질문: {query}"
        )

        print()

        print(
            f"정답 법률: "
            f"{expected_law}"
        )

        print(
            f"정답 조문: "
            f"{expected_article}"
        )

        print(
            f"정답 Chunk ID: "
            f"{expected_chunk_id}"
        )

        # ----------------------------------------------------
        # 정답 순위
        # ----------------------------------------------------

        if correct_rank is None:

            print()
            print(
                "정답 조문: "
                "Top-5에 없음"
            )

        else:

            print()
            print(
                f"정답 조문 검색 순위: "
                f"{correct_rank}위"
            )

        print(
            f"Reciprocal Rank: "
            f"{reciprocal_rank:.4f}"
        )

        # ----------------------------------------------------
        # 검색 결과
        # ----------------------------------------------------

        print()
        print("검색 결과:")

        for result in retrieved:

            marker = (
                "✓"
                if result["correct"]
                else " "
            )

            print(
                f"{marker} "
                f"{result['rank']}위 | "
                f"Score={result['score']:.4f} | "
                f"{result['chunk_id']}"
            )

        # ====================================================
        # 결과 저장
        # ====================================================

        results.append(
            {
                "id": query_id,

                "query": query,

                "expected_law": expected_law,

                "expected_article": expected_article,

                "expected_chunk_id": expected_chunk_id,

                "correct_rank": correct_rank,

                "reciprocal_rank": round(
                    reciprocal_rank,
                    4,
                ),

                "top1": top1,

                "top3": top3,

                "top5": top5,

                "retrieved": retrieved,
            }
        )

    # ========================================================
    # 최종 지표
    # ========================================================

    top1_accuracy = (
        top1_correct / total
    )

    top3_accuracy = (
        top3_correct / total
    )

    top5_accuracy = (
        top5_correct / total
    )

    mean_reciprocal_rank = (
        reciprocal_rank_sum / total
    )

    # ========================================================
    # 최종 출력
    # ========================================================

    mode_label = (
        "Hybrid + Reranker"
        if use_reranker
        else (
            "Hybrid + Law Filter"
            if use_hybrid
            else (
                "Dense + Law Filter"
                if use_law_filter
                else "Dense 전체"
            )
        )
    )

    print()
    print()
    print("#" * 70)
    print(f"검색 평가 결과  [{mode_label}]")
    print("#" * 70)

    print()

    print(
        f"전체 질문       : "
        f"{total}"
    )

    if use_reranker:

        print(
            f"Top-20 Recall   : "
            f"{top20_recall_correct}/{total} "
            f"({top20_recall_correct / total:.1%})"
        )

    print(
        f"Top-1 정확도    : "
        f"{top1_correct}/{total} "
        f"({top1_accuracy:.1%})"
    )

    print(
        f"Top-3 정확도    : "
        f"{top3_correct}/{total} "
        f"({top3_accuracy:.1%})"
    )

    print(
        f"Top-5 정확도    : "
        f"{top5_correct}/{total} "
        f"({top5_accuracy:.1%})"
    )

    print(
        f"MRR             : "
        f"{mean_reciprocal_rank:.4f}"
    )

    print("#" * 70)

    # ========================================================
    # 평가 요약도 함께 반환
    # ========================================================

    summary = {
        "total_questions": total,

        "top20_recall": top20_recall_correct,

        "top20_recall_rate": round(
            top20_recall_correct / total,
            4,
        ),

        "top1_correct": top1_correct,

        "top3_correct": top3_correct,

        "top5_correct": top5_correct,

        "top1_accuracy": round(
            top1_accuracy,
            4,
        ),

        "top3_accuracy": round(
            top3_accuracy,
            4,
        ),

        "top5_accuracy": round(
            top5_accuracy,
            4,
        ),

        "mrr": round(
            mean_reciprocal_rank,
            4,
        ),
    }

    return results, summary


# ============================================================
# 평가 결과 저장
# ============================================================

def save_results(
    results,
    summary,
    result_path,
):

    result_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 결과 구조
    # --------------------------------------------------------

    output = {
        "summary": summary,
        "results": results,
    }

    # --------------------------------------------------------
    # JSON 저장
    # --------------------------------------------------------

    with open(
        result_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        f"평가 결과 저장: "
        f"{result_path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("Legal RAG Search Evaluation")
    print("=" * 70)

    # ========================================================
    # 1. 평가 데이터
    # ========================================================

    queries = load_evaluation_queries()

    # ========================================================
    # 2. Embedding
    # ========================================================

    embedding = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    print()
    print(
        f"Embedding Model: "
        f"{EMBEDDING_MODEL}"
    )

    # ========================================================
    # 3. Qdrant
    # ========================================================

    client = connect_qdrant()

    # ========================================================
    # 4-A. Dense 전체 검색 평가 (Baseline)
    # ========================================================

    print()
    print()
    print("*" * 70)
    print("평가 1/4 — Dense 전체 검색 (Baseline)")
    print("*" * 70)

    results_dense, summary_dense = evaluate(
        client=client,
        embedding=embedding,
        queries=queries,
        use_law_filter=False,
        use_hybrid=False,
        use_reranker=False,
    )

    save_results(
        results=results_dense,
        summary=summary_dense,
        result_path=RESULT_PATH_DENSE,
    )

    # ========================================================
    # 4-B. Dense + Law Filter 검색 평가
    # ========================================================

    print()
    print()
    print("*" * 70)
    print("평가 2/4 — Dense + Law Filter 검색")
    print("*" * 70)

    results_filter, summary_filter = evaluate(
        client=client,
        embedding=embedding,
        queries=queries,
        use_law_filter=True,
        use_hybrid=False,
        use_reranker=False,
    )

    save_results(
        results=results_filter,
        summary=summary_filter,
        result_path=RESULT_PATH_FILTER,
    )

    # ========================================================
    # 4-C. Hybrid + Law Filter 검색 평가
    # ========================================================

    print()
    print()
    print("*" * 70)
    print("평가 3/4 — Hybrid + Law Filter 검색")
    print(f"         (candidate_k={CANDIDATE_K}, rrf_k={RRF_K})")
    print("*" * 70)

    results_hybrid, summary_hybrid = evaluate(
        client=client,
        embedding=embedding,
        queries=queries,
        use_law_filter=False,
        use_hybrid=True,
        use_reranker=False,
    )

    save_results(
        results=results_hybrid,
        summary=summary_hybrid,
        result_path=RESULT_PATH_HYBRID,
    )

    # ========================================================
    # 4-D. Hybrid + Law Filter + Reranker 평가
    # ========================================================

    print()
    print()
    print("*" * 70)
    print("평가 4/4 — Hybrid + Law Filter + Reranker")
    print(f"         (Top-{HYBRID_RECALL_K} → Reranker → Top-{TOP_K})")
    print("*" * 70)

    reranker_model = Reranker()

    results_reranker, summary_reranker = evaluate(
        client=client,
        embedding=embedding,
        queries=queries,
        use_law_filter=False,
        use_hybrid=False,
        use_reranker=True,
        reranker=reranker_model,
    )

    save_results(
        results=results_reranker,
        summary=summary_reranker,
        result_path=RESULT_PATH_RERANKER,
    )

    # ========================================================
    # 5. 4-way 비교 출력
    # ========================================================

    total = summary_dense["total_questions"]

    print()
    print()
    print("=" * 88)
    print("검색 성능 비교")
    print("=" * 88)

    print()
    print(
        f"{'':18s} "
        f"{'Dense':>12s} "
        f"{'+ Filter':>12s} "
        f"{'Hybrid':>12s} "
        f"{'+ Reranker':>12s}"
    )

    print("-" * 72)

    metrics = [
        ("Top-1 정확도", "top1_accuracy", "top1_correct"),
        ("Top-3 정확도", "top3_accuracy", "top3_correct"),
        ("Top-5 정확도", "top5_accuracy", "top5_correct"),
    ]

    summaries = [
        summary_dense,
        summary_filter,
        summary_hybrid,
        summary_reranker,
    ]

    for label, acc_key, cnt_key in metrics:

        row = f"{label:18s}"

        for s in summaries:

            acc = s[acc_key]
            cnt = s[cnt_key]
            row += f"  {cnt}/{total}({acc:.0%})  "

        print(row)

    # MRR 행
    mrr_row = f"{'MRR':18s}"

    for s in summaries:

        mrr_row += f"  {s['mrr']:.4f}      "

    print(mrr_row)

    # Top-20 Recall 행 (Reranker만)
    r20 = summary_reranker["top20_recall"]
    r20_rate = summary_reranker["top20_recall_rate"]

    print(
        f"\n{'Top-20 Recall':18s}"
        f"  {'—':>10s}  "
        f"  {'—':>10s}  "
        f"  {'—':>10s}  "
        f"  {r20}/{total}({r20_rate:.0%})  "
    )

    print("=" * 88)

    # ========================================================
    # 6. 완료
    # ========================================================

    print()
    print(
        "검색 평가가 완료되었습니다."
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()