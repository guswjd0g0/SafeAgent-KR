"""
bge-m3 Dense 실패 사례 분석 스크립트

Top-5 실패 케이스를 추출하고
각 케이스의 정답 조문 원문을 함께 출력한다.

출력 항목:
  - 질문
  - 정답 법률 / 조문 / Chunk ID
  - 실제 조문 텍스트 (chunks.json에서 조회)
  - 검색 결과 Top-5 (score + chunk_id + 해당 조문 텍스트)
"""

import json
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


# ============================================================
# 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DENSE_RESULT_PATH = (
    BASE_DIR / "data" / "evaluation"
    / "bge_search_results_dense.json"
)

COLLECTION_NAME = "legal_documents_bge_m3"

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333


# ============================================================
# Qdrant에서 chunk_id로 텍스트 조회
# ============================================================

def get_text(client, chunk_id: str) -> str:

    result = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="chunk_id",
                    match=MatchValue(value=chunk_id),
                )
            ]
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )

    points = result[0]

    if not points:
        return "(조회 실패)"

    payload = points[0].payload
    article  = payload.get("article", "")
    title    = payload.get("article_title", "") or ""
    text     = payload.get("text", "")

    header = f"[{article} {title}]" if title else f"[{article}]"
    return f"{header}\n{text}"


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # 결과 로드
    # --------------------------------------------------------

    with open(DENSE_RESULT_PATH, encoding="utf-8") as f:
        data = json.load(f)

    all_results = data["results"]
    total       = len(all_results)
    summary     = data["summary"]

    print("=" * 70)
    print("bge-m3 Dense 실패 분석")
    print("=" * 70)
    print(
        f"\n전체: {total}개  |  "
        f"Top-5 성공: {summary['top5_correct']}개  |  "
        f"Top-5 실패: {total - summary['top5_correct']}개"
    )
    print(f"MRR: {summary['mrr']:.4f}")

    # --------------------------------------------------------
    # Qdrant 연결
    # --------------------------------------------------------

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # --------------------------------------------------------
    # 실패 케이스 추출
    # --------------------------------------------------------

    failures = [r for r in all_results if not r["top5"]]

    print(f"\n실패 케이스 {len(failures)}개:")

    for item in failures:

        qid      = item["id"]
        query    = item["query"]
        exp_law  = item["expected_law"]
        exp_art  = item["expected_article"]
        exp_cid  = item["expected_chunk_id"]

        print()
        print("=" * 70)
        print(
            f"[ID {qid}]  {exp_law}  {exp_art}"
        )
        print(f"질문: {query}")
        print()

        # 정답 조문 원문
        print("[ 정답 조문 원문 ]")
        answer_text = get_text(client, exp_cid)
        for line in answer_text.split("\n"):
            print(f"  {line}")

        print()
        print("[ 검색 결과 Top-5 ]")

        for r in item["retrieved"]:
            marker = "✓" if r["correct"] else " "
            retrieved_text = get_text(client, r["chunk_id"])
            # 첫 줄(헤더)만 표시
            first_line = retrieved_text.split("\n")[0]
            # 본문 앞 60자
            body_preview = retrieved_text.split("\n")[-1][:60] if "\n" in retrieved_text else retrieved_text[:60]
            print(
                f"  {marker} {r['rank']}위  "
                f"score={r['score']:.4f}  "
                f"{first_line}  "
                f"| {body_preview}..."
            )

    # --------------------------------------------------------
    # 성공 케이스도 간략히 출력 (참고용)
    # --------------------------------------------------------

    successes = [r for r in all_results if r["top5"]]

    print()
    print()
    print("=" * 70)
    print(f"성공 케이스 {len(successes)}개 (참고)")
    print("=" * 70)

    for item in successes:
        rank_str = (
            f"{item['correct_rank']}위"
            if item["correct_rank"]
            else "?"
        )
        print(
            f"  ID {item['id']:2d}  "
            f"{item['expected_law']:12s}  "
            f"{item['expected_article']:8s}  "
            f"→ {rank_str}  "
            f"(RR={item['reciprocal_rank']:.4f})"
        )

    print()
    print("분석 완료")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()
