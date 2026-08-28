"""
evaluate_rag.py — SafeAgent RAG 응답 품질 평가

모드:
    --mode dense     : Dense-only 평가  (기본값)
    --mode reranker  : Dense→Reranker 평가
    --mode both      : 두 모드 동시 실행 후 비교

출력:
    data/evaluation/rag_eval_results.json        (dense)
    data/evaluation/rag_eval_results_reranker.json (reranker)
    data/evaluation/rag_eval_comparison.json     (both)
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

# ============================================================
# 경로 설정
# ============================================================
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from rag_pipeline import init_pipeline, run_rag  # noqa: E402


# ============================================================
# 경로 상수
# ============================================================
QUERIES_PATH       = _ROOT / "data" / "evaluation" / "legal_queriesV3.json"
OUTPUT_DENSE       = _ROOT / "data" / "evaluation" / "rag_eval_results.json"
OUTPUT_RERANKER    = _ROOT / "data" / "evaluation" / "rag_eval_results_reranker.json"
OUTPUT_COMPARISON  = _ROOT / "data" / "evaluation" / "rag_eval_comparison.json"


# ============================================================
# 평가 보조 함수
# ============================================================

def _check_retrieval_hit(
    docs: list[dict],
    expected_chunk_id: str,
) -> tuple[bool, int | None]:
    """검색 결과에 expected_chunk_id 가 있는지 확인. (hit, rank)"""
    for rank, doc in enumerate(docs, start=1):
        if doc.get("chunk_id") == expected_chunk_id:
            return True, rank
    return False, None


def _check_answer_has_source(answer: str, sources: list[str]) -> bool:
    """답변에 출처(법률명 + 조문번호)가 하나라도 포함되는지 확인."""
    for src in sources:
        parts = src.split()
        if all(p in answer for p in parts):
            return True
    return bool(re.search(r"제\d+조", answer))


def _check_hallucination(answer: str, docs: list[dict]) -> dict:
    """
    답변의 조문번호가 검색 범위 밖인지 확인.

    알려진 한계: 조문 본문 내 상호참조 구문(예: '제76조에 따라...')도
    미지원 조문으로 판정될 수 있어 False Positive 발생 가능.
    → ISSUES.md #002 참고
    """
    source_articles: set[str] = set()
    for doc in docs:
        m = re.search(r"제\d+조", doc.get("article", ""))
        if m:
            source_articles.add(m.group())

    answer_articles: set[str] = set(re.findall(r"제\d+조", answer))
    unsupported = answer_articles - source_articles

    return {
        "flag":            bool(unsupported),
        "answer_articles": sorted(answer_articles),
        "source_articles": sorted(source_articles),
        "unsupported":     sorted(unsupported),
    }


# ============================================================
# 단일 질문 평가
# ============================================================

def _evaluate_one(
    item:         dict,
    use_reranker: bool = False,
    verbose:      bool = True,
) -> dict:
    q_id              = item["id"]
    query             = item["query"]
    expected_law      = item["expected_law"]
    expected_article  = item["expected_article"]
    expected_chunk_id = item["expected_chunk_id"]

    if verbose:
        mode_label = "Reranker" if use_reranker else "Dense"
        print(f"\n[{q_id:02d}/20] [{mode_label}] {query}")

    result = run_rag(
        question=query,
        law=expected_law,
        use_reranker=use_reranker,
        verbose=False,
    )

    docs    = result["docs"]
    sources = result["sources"]
    answer  = result["answer"]

    hit, rank     = _check_retrieval_hit(docs, expected_chunk_id)
    has_source    = _check_answer_has_source(answer, sources)
    hallucination = _check_hallucination(answer, docs)

    if verbose:
        print(
            f"  검색 히트     [{'✓' if hit else '✗'}] "
            f"{'순위 ' + str(rank) + '위' if rank else '미포함'}"
            f"  |  기대: {expected_chunk_id}"
        )
        print(f"  출처 포함     [{'✓' if has_source else '✗'}]")
        print(
            f"  Hallucination [{'⚠' if hallucination['flag'] else '✓'}]"
            + (f"  의심: {hallucination['unsupported']}" if hallucination["flag"] else "")
        )
        print(
            f"  소요: 검색 {result['retrieval_sec']}초 | LLM {result['llm_sec']}초"
        )

    return {
        "id":                q_id,
        "query":             query,
        "expected_law":      expected_law,
        "expected_article":  expected_article,
        "expected_chunk_id": expected_chunk_id,
        "mode":              result["mode"],
        "retrieved_chunks": [
            {
                "rank":         i + 1,
                "chunk_id":     d["chunk_id"],
                "law":          d["law"],
                "article":      d["article"],
                "score":        d.get("score"),
                "rerank_score": d.get("rerank_score"),
                "dense_rank":   d.get("dense_rank"),
                "text":         d["text"][:200],
            }
            for i, d in enumerate(docs)
        ],
        "retrieval_hit":     hit,
        "retrieval_rank":    rank,
        "answer_has_source": has_source,
        "hallucination":     hallucination,
        "answer":            answer,
        "retrieval_sec":     result["retrieval_sec"],
        "llm_sec":           result["llm_sec"],
    }


# ============================================================
# 집계 요약 생성
# ============================================================

def _summarize(results: list[dict], total_sec: float) -> dict:
    total  = len(results)
    n_hit  = sum(1 for r in results if r["retrieval_hit"])
    n_src  = sum(1 for r in results if r["answer_has_source"])
    n_hall = sum(1 for r in results if r["hallucination"]["flag"])

    ranks = [r["retrieval_rank"] for r in results if r["retrieval_rank"]]
    top1  = sum(1 for rk in ranks if rk == 1)
    top3  = sum(1 for rk in ranks if rk <= 3)
    top5  = sum(1 for rk in ranks if rk <= 5)

    avg_ret = round(sum(r["retrieval_sec"] for r in results) / total, 2)
    avg_llm = round(sum(r["llm_sec"]       for r in results) / total, 2)

    return {
        "total_questions":     total,
        "retrieval_hit":       n_hit,
        "retrieval_hit_rate":  round(n_hit / total * 100, 1),
        "top1_rate":           round(top1  / total * 100, 1),
        "top3_rate":           round(top3  / total * 100, 1),
        "top5_rate":           round(top5  / total * 100, 1),
        "answer_has_source":   n_src,
        "source_rate":         round(n_src  / total * 100, 1),
        "hallucination_count": n_hall,
        "hallucination_rate":  round(n_hall / total * 100, 1),
        "avg_retrieval_sec":   avg_ret,
        "avg_llm_sec":         avg_llm,
        "total_sec":           total_sec,
    }


# ============================================================
# 콘솔 출력
# ============================================================

def _print_summary(summary: dict, title: str = "평가 요약") -> None:
    print()
    print("=" * 60)
    print(f"▶ {title}")
    print("=" * 60)
    print(f"  총 질문 수          : {summary['total_questions']}개")
    print(f"  검색 히트 (Top-5)   : {summary['retrieval_hit']}/{summary['total_questions']}  ({summary['retrieval_hit_rate']}%)")
    print(f"  검색 순위 Top-1     : {round(summary['top1_rate'] * summary['total_questions'] / 100)}/{summary['total_questions']}  ({summary['top1_rate']}%)")
    print(f"  검색 순위 Top-3     : {round(summary['top3_rate'] * summary['total_questions'] / 100)}/{summary['total_questions']}  ({summary['top3_rate']}%)")
    print(f"  출처 포함 답변      : {summary['answer_has_source']}/{summary['total_questions']}  ({summary['source_rate']}%)")
    print(f"  Hallucination 의심  : {summary['hallucination_count']}/{summary['total_questions']}  ({summary['hallucination_rate']}%)")
    print(f"  평균 검색 시간      : {summary['avg_retrieval_sec']}초")
    print(f"  평균 LLM 시간       : {summary['avg_llm_sec']}초")
    print(f"  총 소요 시간        : {summary['total_sec']}초")


def _print_failures(results: list[dict]) -> None:
    failed = [r for r in results if not r["retrieval_hit"]]
    if failed:
        print()
        print("=" * 60)
        print("▶ 검색 미스 사례")
        print("=" * 60)
        for r in failed:
            top3 = ", ".join(
                c["chunk_id"].split("_본문_")[-1]
                for c in r["retrieved_chunks"][:3]
            )
            print(f"  [{r['id']:02d}] {r['query']}")
            print(f"       기대  : {r['expected_chunk_id']}")
            print(f"       검색  : {top3}")

    hall_cases = [r for r in results if r["hallucination"]["flag"]]
    if hall_cases:
        print()
        print("=" * 60)
        print("▶ Hallucination 의심 사례")
        print("=" * 60)
        for r in hall_cases:
            h = r["hallucination"]
            print(f"  [{r['id']:02d}] {r['query']}")
            print(f"       근거 조문 : {h['source_articles']}")
            print(f"       답변 조문 : {h['answer_articles']}")
            print(f"       미지원    : {h['unsupported']}")


# ============================================================
# 단일 모드 평가
# ============================================================

def run_evaluation(
    use_reranker: bool  = False,
    queries_path: Path  = QUERIES_PATH,
    output_path:  Path  = OUTPUT_DENSE,
    verbose:      bool  = True,
) -> dict:
    """Dense 또는 Reranker 단일 모드로 20개 질문을 평가한다."""

    with open(queries_path, encoding="utf-8") as f:
        queries = json.load(f)

    total      = len(queries)
    mode_label = "Reranker" if use_reranker else "Dense-only"

    print("=" * 60)
    print(f"SafeAgent RAG 응답 품질 평가  [{mode_label}]")
    print(f"총 {total}개 질문 | LLM: gemma3:4b | 임베딩: bge-m3")
    print("=" * 60)

    init_pipeline(use_reranker=use_reranker)

    t_start = time.perf_counter()
    results: list[dict] = []

    for item in queries:
        results.append(
            _evaluate_one(item, use_reranker=use_reranker, verbose=verbose)
        )

    total_sec = round(time.perf_counter() - t_start, 1)
    summary   = _summarize(results, total_sec)

    _print_summary(summary, title=f"평가 요약  [{mode_label}]")
    _print_failures(results)

    # 저장
    out_path = output_path if not use_reranker else OUTPUT_RERANKER
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out_path}")

    return {"summary": summary, "results": results}


# ============================================================
# Both 모드 — Dense vs Reranker 비교
# ============================================================

def run_comparison(
    queries_path: Path = QUERIES_PATH,
    verbose:      bool = True,
) -> dict:
    """Dense와 Reranker를 모두 실행하고 비교 결과를 저장한다."""

    print("=" * 60)
    print("SafeAgent RAG 비교 평가  [Dense vs Reranker]")
    print("=" * 60)

    # Dense
    print("\n--- [1/2] Dense-only ---")
    dense_out = run_evaluation(
        use_reranker=False,
        queries_path=queries_path,
        verbose=verbose,
    )

    # Reranker
    print("\n--- [2/2] Reranker ---")
    reranker_out = run_evaluation(
        use_reranker=True,
        queries_path=queries_path,
        verbose=verbose,
    )

    # 비교 요약
    d_sum = dense_out["summary"]
    r_sum = reranker_out["summary"]

    comparison = {
        "dense":    d_sum,
        "reranker": r_sum,
        "diff": {
            "retrieval_hit_rate":  round(r_sum["retrieval_hit_rate"]  - d_sum["retrieval_hit_rate"],  1),
            "top1_rate":           round(r_sum["top1_rate"]           - d_sum["top1_rate"],           1),
            "top3_rate":           round(r_sum["top3_rate"]           - d_sum["top3_rate"],           1),
            "hallucination_rate":  round(r_sum["hallucination_rate"]  - d_sum["hallucination_rate"],  1),
            "avg_retrieval_sec":   round(r_sum["avg_retrieval_sec"]   - d_sum["avg_retrieval_sec"],   2),
        },
    }

    print()
    print("=" * 60)
    print("▶ Dense vs Reranker 비교")
    print("=" * 60)
    print(f"  {'지표':<25} {'Dense':>10} {'Reranker':>10} {'차이':>8}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*8}")
    for key, label in [
        ("retrieval_hit_rate", "검색 히트율 (Top-5)"),
        ("top1_rate",          "Top-1 히트율"),
        ("top3_rate",          "Top-3 히트율"),
        ("hallucination_rate", "Hallucination"),
        ("avg_retrieval_sec",  "평균 검색 시간(초)"),
    ]:
        d_val  = d_sum.get(key, 0)
        r_val  = r_sum.get(key, 0)
        diff   = comparison["diff"].get(key, 0)
        sign   = "+" if diff > 0 else ""
        unit   = "초" if "sec" in key else "%"
        print(
            f"  {label:<25} {str(d_val) + unit:>10} "
            f"{str(r_val) + unit:>10} {sign + str(diff) + unit:>8}"
        )

    # 저장
    OUTPUT_COMPARISON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_COMPARISON, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"\n비교 결과 저장: {OUTPUT_COMPARISON}")

    return comparison


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SafeAgent RAG 평가")
    parser.add_argument(
        "--mode",
        choices=["dense", "reranker", "both"],
        default="dense",
        help="평가 모드 (기본값: dense)",
    )
    args = parser.parse_args()

    if args.mode == "dense":
        run_evaluation(use_reranker=False)
    elif args.mode == "reranker":
        run_evaluation(use_reranker=True)
    elif args.mode == "both":
        run_comparison()
