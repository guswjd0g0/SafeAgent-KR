"""
bge-m3 Embedding 기반 4-way 검색 평가 스크립트

evaluate_search.py의 평가 로직을 그대로 재사용하되
Embedding 모델과 Collection만 bge-m3 전용으로 교체한다.

비교 구조:
  Dense (bge-m3)
  Dense + Law Filter (bge-m3)
  Hybrid + Law Filter (bge-m3)
  Hybrid + Reranker  (bge-m3 + bge-reranker-base)
"""

import json
import sys
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# ============================================================
# 같은 디렉토리의 모듈 임포트
# ============================================================

from hybrid_search import (
    hybrid_search,
    SentenceTransformerEmbedding,
    COLLECTION_NAME_BGE_M3,
    CANDIDATE_K,
    RRF_K,
)

from reranker import Reranker, attach_texts

# evaluate_search의 평가 로직 재사용
import evaluate_search as ev


# ============================================================
# bge-m3 전용 설정
# ============================================================

EMBEDDING_MODEL   = "BAAI/bge-m3"
COLLECTION_NAME   = COLLECTION_NAME_BGE_M3     # legal_documents_bge_m3

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

TOP_K           = 5
HYBRID_RECALL_K = 20


# ============================================================
# Path
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

EVALUATION_PATH = (
    BASE_DIR / "data" / "evaluation" / "legal_queriesV3.json"
)

RESULT_PATH_DENSE    = BASE_DIR / "data" / "evaluation" / "v3_bge_search_results_dense.json"
RESULT_PATH_FILTER   = BASE_DIR / "data" / "evaluation" / "v3_bge_search_results_filter.json"
RESULT_PATH_HYBRID   = BASE_DIR / "data" / "evaluation" / "v3_bge_search_results_hybrid.json"
RESULT_PATH_RERANKER = BASE_DIR / "data" / "evaluation" / "v3_bge_search_results_reranker.json"


# ============================================================
# Dense 단일 질문 검색 (bge-m3 전용)
# ============================================================

def search_query_bge(
    client,
    embedding,
    query,
    law_filter=None,
):
    """
    bge-m3 Collection으로 Dense 검색을 수행한다.
    evaluate_search.search_query()와 동일한 인터페이스.
    """

    query_vector = embedding.embed_query(query)

    qdrant_filter = None

    if law_filter:

        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="law",
                    match=MatchValue(value=law_filter),
                )
            ]
        )

    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=qdrant_filter,
        limit=TOP_K,
        with_payload=True,
    )

    return result.points


# ============================================================
# 4-way 평가 (bge-m3)
# ============================================================

def evaluate_all(client, embedding, queries, reranker_model):
    """
    Dense / Dense+Filter / Hybrid / Hybrid+Reranker
    4가지 방식으로 평가하고 각 summary를 반환한다.
    """

    total = len(queries)

    summaries = {}
    all_results = {}

    # --------------------------------------------------------
    # 공통 내부 평가 함수
    # --------------------------------------------------------

    def _run(mode: str):

        top1 = top3 = top5 = 0
        rr_sum = 0.0
        top20_recall = 0
        results = []

        for idx, item in enumerate(queries, start=1):

            query            = item["query"]
            expected_law     = item["expected_law"]
            expected_id      = item["expected_chunk_id"]
            expected_article = item["expected_article"]

            # ------------------------------------------------
            # 모드별 검색
            # ------------------------------------------------

            is_hybrid_result = False

            if mode == "dense":

                points = search_query_bge(
                    client, embedding, query, law_filter=None
                )

            elif mode == "filter":

                points = search_query_bge(
                    client, embedding, query,
                    law_filter=expected_law,
                )

            elif mode == "hybrid":

                points = hybrid_search(
                    client=client,
                    embedding=embedding,
                    query=query,
                    law_filter=expected_law,
                    top_k=TOP_K,
                    candidate_k=CANDIDATE_K,
                    rrf_k=RRF_K,
                    collection_name=COLLECTION_NAME,
                )
                is_hybrid_result = True

            elif mode == "reranker":

                candidates = hybrid_search(
                    client=client,
                    embedding=embedding,
                    query=query,
                    law_filter=expected_law,
                    top_k=HYBRID_RECALL_K,
                    candidate_k=CANDIDATE_K,
                    rrf_k=RRF_K,
                    collection_name=COLLECTION_NAME,
                )

                in_top20 = any(
                    c["chunk_id"] == expected_id
                    for c in candidates
                )

                if in_top20:
                    top20_recall += 1

                candidates_with_text = attach_texts(
                    candidates=candidates,
                    client=client,
                    collection_name=COLLECTION_NAME,
                )

                points = reranker_model.rerank(
                    query=query,
                    candidates=candidates_with_text,
                    top_k=TOP_K,
                )
                is_hybrid_result = True

            # ------------------------------------------------
            # 정답 순위 탐색
            # ------------------------------------------------

            correct_rank = None
            retrieved    = []

            for rank, point in enumerate(points, start=1):

                if is_hybrid_result:
                    cid     = point.get("chunk_id", "")
                    law     = point.get("law", "")
                    article = point.get("article", "")
                    file_   = point.get("file", "")
                    score   = point.get(
                        "rerank_score",
                        point.get("rrf_score", 0.0),
                    )
                else:
                    payload = point.payload or {}
                    cid     = payload.get("chunk_id", "")
                    law     = payload.get("law", "")
                    article = payload.get("article", "")
                    file_   = payload.get("file", "")
                    score   = point.score

                correct = (cid == expected_id)

                if correct and correct_rank is None:
                    correct_rank = rank

                retrieved.append({
                    "rank":    rank,
                    "score":   round(score, 4),
                    "chunk_id": cid,
                    "law":     law,
                    "article": article,
                    "file":    file_,
                    "correct": correct,
                })

            # ------------------------------------------------
            # 지표 집계
            # ------------------------------------------------

            t1 = (correct_rank == 1)
            t3 = (correct_rank is not None and correct_rank <= 3)
            t5 = (correct_rank is not None and correct_rank <= 5)
            rr = (1.0 / correct_rank) if correct_rank else 0.0

            if t1: top1 += 1
            if t3: top3 += 1
            if t5: top5 += 1
            rr_sum += rr

            # ------------------------------------------------
            # 개별 결과 출력
            # ------------------------------------------------

            print()
            print("=" * 70)
            print(
                f"[{idx}/{total}] "
                f"Q{item['id']} | "
                f"{query}"
            )
            print(
                f"  정답: {expected_law} {expected_article}"
            )

            if correct_rank:
                print(f"  → {correct_rank}위  (RR={rr:.4f})")
            else:
                print(f"  → Top-{TOP_K} 밖")

            for r in retrieved:
                marker = "✓" if r["correct"] else " "
                print(
                    f"  {marker} "
                    f"{r['rank']}위 "
                    f"score={r['score']:.4f} "
                    f"{r['chunk_id']}"
                )

            results.append({
                "id":                  item["id"],
                "query":               query,
                "expected_law":        expected_law,
                "expected_article":    expected_article,
                "expected_chunk_id":   expected_id,
                "correct_rank":        correct_rank,
                "reciprocal_rank":     round(rr, 4),
                "top1": t1, "top3": t3, "top5": t5,
                "retrieved": retrieved,
            })

        # ------------------------------------------------
        # 요약
        # ------------------------------------------------

        summary = {
            "mode":              mode,
            "total_questions":   total,
            "top20_recall":      top20_recall,
            "top20_recall_rate": round(top20_recall / total, 4),
            "top1_correct":      top1,
            "top3_correct":      top3,
            "top5_correct":      top5,
            "top1_accuracy":     round(top1 / total, 4),
            "top3_accuracy":     round(top3 / total, 4),
            "top5_accuracy":     round(top5 / total, 4),
            "mrr":               round(rr_sum / total, 4),
        }

        return results, summary

    # --------------------------------------------------------
    # 4가지 모드 순서대로 실행
    # --------------------------------------------------------

    modes = [
        ("dense",    "1/4 — Dense (bge-m3)"),
        ("filter",   "2/4 — Dense + Law Filter (bge-m3)"),
        ("hybrid",   "3/4 — Hybrid + Law Filter (bge-m3)"),
        ("reranker", "4/4 — Hybrid + Reranker  (bge-m3)"),
    ]

    for mode, label in modes:

        print()
        print()
        print("*" * 70)
        print(f"평가 {label}")
        print("*" * 70)

        results, summary = _run(mode)

        all_results[mode]  = results
        summaries[mode]    = summary

        # ---- 소계 출력 ----
        print()
        print("#" * 70)
        print(f"결과  [{label}]")
        print("#" * 70)
        print(f"  전체 질문   : {total}")

        if mode == "reranker":
            r20 = summary["top20_recall"]
            print(
                f"  Top-20 Recall: {r20}/{total} "
                f"({summary['top20_recall_rate']:.1%})"
            )

        for k in (1, 3, 5):
            cnt = summary[f"top{k}_correct"]
            acc = summary[f"top{k}_accuracy"]
            print(f"  Top-{k} 정확도 : {cnt}/{total} ({acc:.1%})")

        print(f"  MRR          : {summary['mrr']:.4f}")
        print("#" * 70)

        # ---- JSON 저장 ----
        path_map = {
            "dense":    RESULT_PATH_DENSE,
            "filter":   RESULT_PATH_FILTER,
            "hybrid":   RESULT_PATH_HYBRID,
            "reranker": RESULT_PATH_RERANKER,
        }

        out_path = path_map[mode]
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {"summary": summary, "results": results},
                f, ensure_ascii=False, indent=2,
            )

        print(f"  저장: {out_path.name}")

    return summaries


# ============================================================
# 비교 표 출력
# ============================================================

def print_comparison(summaries: dict):

    total = summaries["dense"]["total_questions"]

    print()
    print()
    print("=" * 88)
    print("bge-m3  검색 성능 4-way 비교")
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

    for k in (1, 3, 5):
        row = f"Top-{k} 정확도{'':<9s}"
        for mode in ("dense", "filter", "hybrid", "reranker"):
            cnt = summaries[mode][f"top{k}_correct"]
            acc = summaries[mode][f"top{k}_accuracy"]
            row += f"  {cnt}/{total}({acc:.0%})  "
        print(row)

    # MRR
    mrr_row = f"{'MRR':<18s}"
    for mode in ("dense", "filter", "hybrid", "reranker"):
        mrr_row += f"  {summaries[mode]['mrr']:.4f}      "
    print(mrr_row)

    # Top-20 Recall
    r20    = summaries["reranker"]["top20_recall"]
    r20_rt = summaries["reranker"]["top20_recall_rate"]
    print(
        f"\n{'Top-20 Recall':<18s}"
        f"  {'—':>10s}  "
        f"  {'—':>10s}  "
        f"  {'—':>10s}  "
        f"  {r20}/{total}({r20_rt:.0%})  "
    )

    print("=" * 88)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("Legal RAG — bge-m3 4-way 검색 평가")
    print(f"  Embedding  : {EMBEDDING_MODEL}")
    print(f"  Collection : {COLLECTION_NAME}")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. 평가 데이터 로드
    # --------------------------------------------------------

    with open(EVALUATION_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    print(f"\n평가 질문 수: {len(queries)}")

    # --------------------------------------------------------
    # 2. bge-m3 Embedding 초기화
    # --------------------------------------------------------

    print()
    print("bge-m3 모델 로드 중...")

    embedding = SentenceTransformerEmbedding(EMBEDDING_MODEL)

    print("bge-m3 로드 완료")

    # --------------------------------------------------------
    # 3. Qdrant 연결
    # --------------------------------------------------------

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    info = client.get_collection(COLLECTION_NAME)
    print(
        f"\nQdrant 연결 성공 "
        f"— {COLLECTION_NAME} ({info.points_count} points)"
    )

    # --------------------------------------------------------
    # 4. Reranker 로드
    # --------------------------------------------------------

    print()
    reranker_model = Reranker()

    # --------------------------------------------------------
    # 5. 4-way 평가 실행
    # --------------------------------------------------------

    summaries = evaluate_all(
        client=client,
        embedding=embedding,
        queries=queries,
        reranker_model=reranker_model,
    )

    # --------------------------------------------------------
    # 6. 비교 표 출력
    # --------------------------------------------------------

    print_comparison(summaries)

    print()
    print("평가 완료")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()
