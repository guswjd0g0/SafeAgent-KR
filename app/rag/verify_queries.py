"""
평가셋 전수 검증 스크립트

20개 질문 각각에 대해:
  - query
  - expected_law / expected_article
  - 실제 조문 제목 + 본문 전체
를 나란히 출력하여 불일치를 육안으로 확인한다.
"""

import json
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


# ============================================================
# 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

EVAL_PATH = (
    BASE_DIR / "data" / "evaluation" / "legal_queriesV2.json"
)

COLLECTION_NAME = "legal_documents_bge_m3"
QDRANT_HOST     = "localhost"
QDRANT_PORT     = 6333


# ============================================================
# Qdrant에서 chunk_id로 payload 전체 조회
# ============================================================

def get_payload(client, chunk_id: str) -> dict:

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
        return {}

    return points[0].payload


# ============================================================
# Main
# ============================================================

def main():

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    with open(EVAL_PATH, encoding="utf-8") as f:
        queries = json.load(f)

    print("=" * 72)
    print("평가셋 전수 검증 — query vs 실제 조문 원문")
    print("=" * 72)
    print(f"총 {len(queries)}개\n")

    for item in queries:

        qid      = item["id"]
        query    = item["query"]
        exp_law  = item["expected_law"]
        exp_art  = item["expected_article"]
        exp_cid  = item["expected_chunk_id"]

        payload  = get_payload(client, exp_cid)

        article       = payload.get("article", "")
        article_title = payload.get("article_title", "") or ""
        text          = payload.get("text", "")

        # ------------------------------------------------
        # 출력
        # ------------------------------------------------

        print("─" * 72)
        print(
            f"[ID {qid:02d}]  "
            f"{exp_law}  "
            f"{exp_art}  "
            f"({article_title})"
        )
        print()
        print(f"  질문: {query}")
        print()
        print(f"  조문 제목 : {article} {article_title}")
        print(f"  조문 본문 : {text[:200]}{'...' if len(text) > 200 else ''}")
        print()

    print("=" * 72)
    print("출력 완료")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()
