from sentence_transformers import CrossEncoder


# ============================================================
# Reranker 설정
# ============================================================

# 한국어 포함 다국어 Cross-Encoder
# BAAI/bge-reranker-base
#
# 벤치마크:
# 제2조(정의) 쿼리에서 제1조 대비 약 3배 높은 점수
# 추론 속도: 쌍당 약 0.07초 (CPU)
RERANKER_MODEL = "BAAI/bge-reranker-base"

# Cross-Encoder 최대 입력 길이
MAX_LENGTH = 512

# 최종 반환 Top-K
TOP_K = 5


# ============================================================
# Reranker
# ============================================================

class Reranker:
    """
    Cross-Encoder 기반 Reranker.

    Hybrid / Dense 검색의 후보 리스트를
    (query, passage) 쌍 점수로 재정렬한다.

    사용 예:
        reranker = Reranker()
        results  = reranker.rerank(query, candidates, top_k=5)
    """

    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
        max_length: int = MAX_LENGTH,
    ):

        print(
            f"Reranker 로드 중: "
            f"{model_name}"
        )

        self.model = CrossEncoder(
            model_name,
            max_length=max_length,
        )

        self.model_name = model_name

        print("Reranker 로드 완료")

    # ============================================================
    # Rerank
    # ============================================================

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = TOP_K,
    ) -> list[dict]:
        """
        후보 리스트를 Cross-Encoder 점수로 재정렬한다.

        Parameters
        ----------
        query      : 검색 질문
        candidates : hybrid_search() 또는 search_query()
                     결과 리스트.
                     각 항목은 "chunk_id", "text" 키를 포함해야 한다.
        top_k      : 최종 반환 개수

        Returns
        -------
        rerank_score 기준으로 정렬된 결과 리스트.
        각 항목에 "rerank_score"와 "rerank_rank" 필드가 추가된다.
        """

        if not candidates:
            return []

        # --------------------------------------------------------
        # (query, passage) 쌍 구성
        # --------------------------------------------------------

        pairs = [
            (query, item.get("text", ""))
            for item in candidates
        ]

        # --------------------------------------------------------
        # Cross-Encoder 점수 계산
        # --------------------------------------------------------

        scores = self.model.predict(pairs)

        # --------------------------------------------------------
        # 점수 기준 내림차순 정렬
        # --------------------------------------------------------

        scored = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )

        # --------------------------------------------------------
        # 결과 구성
        # --------------------------------------------------------

        results = []

        for rerank_rank, (score, item) in enumerate(
            scored[:top_k],
            start=1,
        ):

            result = dict(item)

            result["rerank_score"] = round(
                float(score), 4
            )

            result["rerank_rank"] = rerank_rank

            # 이전 Hybrid 순위 보존
            result["hybrid_rank"] = item.get(
                "rank", None
            )

            results.append(result)

        return results


# ============================================================
# 편의 함수 — candidates에 text 필드 보강
# ============================================================

def attach_texts(
    candidates: list[dict],
    client,
    collection_name: str = "legal_documents",
) -> list[dict]:
    """
    hybrid_search() 결과에는 text 필드가 없으므로
    Qdrant에서 조회하여 보강한다.

    passage 형식:
        "[제2조 정의] 제2조(정의) 이 법에서..."

    조문 번호와 제목을 앞에 명시함으로써
    Cross-Encoder가 조문 제목 키워드와
    질문의 의미를 더 잘 매칭하도록 한다.
    """

    from qdrant_client.models import (
        Filter,
        FieldCondition,
        MatchValue,
    )

    enriched = []

    for item in candidates:

        chunk_id = item.get("chunk_id", "")

        if not chunk_id:
            enriched.append(item)
            continue

        # --------------------------------------------------------
        # chunk_id payload 필터로 조회
        # --------------------------------------------------------

        result = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="chunk_id",
                        match=MatchValue(
                            value=chunk_id
                        ),
                    )
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )

        points = result[0]

        if points:

            payload = points[0].payload

            raw_text = payload.get("text", "") or ""
            article  = payload.get("article", "") or ""
            title    = payload.get("article_title", "") or ""

            # ------------------------------------------------
            # "[제N조 제목] 본문텍스트" 형식으로 구성
            #
            # 조문 번호와 제목을 앞에 붙이면 Cross-Encoder가
            # 질문의 핵심 키워드(정의, 적용범위 등)와
            # 조문 제목을 직접 비교할 수 있다.
            # ------------------------------------------------

            if article and title:
                text = f"[{article} {title}] {raw_text}"
            elif article:
                text = f"[{article}] {raw_text}"
            else:
                text = raw_text

        else:

            text = ""

        enriched_item = dict(item)
        enriched_item["text"] = text
        enriched.append(enriched_item)

    return enriched


# ============================================================
# Main — 동작 확인용
# ============================================================

def main():

    from qdrant_client import QdrantClient
    from langchain_ollama import OllamaEmbeddings
    from hybrid_search import (
        hybrid_search,
        EMBEDDING_MODEL,
        CANDIDATE_K,
        RRF_K,
    )

    print("=" * 60)
    print("Reranker 테스트")
    print("=" * 60)

    # --------------------------------------------------------
    # 모델 로드
    # --------------------------------------------------------

    client = QdrantClient(host="localhost", port=6333)
    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)
    reranker = Reranker()

    # --------------------------------------------------------
    # 테스트 케이스
    # --------------------------------------------------------

    test_cases = [
        (
            "근로기준법에서 근로자와 사용자는 어떻게 정의되나요?",
            "근로기준법",
            "근로기준법_법률_제21533호_20270101_본문_제2조",
        ),
        (
            "산업안전보건법에서 사용하는 주요 용어는 어떻게 정의되어 있나요?",
            "산업안전보건법",
            "산업안전보건법_법률_제21374호_20260801_본문_제2조",
        ),
        (
            "산업재해보상보험과 관련하여 국가가 부담해야 하는 비용은 무엇인가요?",
            "산업재해보상보험법",
            "산업재해보상보험법_법률_제21375호_20260701_본문_제4조",
        ),
    ]

    for query, law_filter, expected in test_cases:

        print()
        print("-" * 60)
        print(f"질문: {query}")
        print(f"정답: {expected}")

        # ------------------------------------------------
        # 1. Hybrid Top-20
        # ------------------------------------------------

        candidates = hybrid_search(
            client=client,
            embedding=embedding,
            query=query,
            law_filter=law_filter,
            top_k=20,
            candidate_k=CANDIDATE_K,
            rrf_k=RRF_K,
        )

        hybrid_rank = next(
            (
                c["rank"]
                for c in candidates
                if c["chunk_id"] == expected
            ),
            None,
        )

        print(
            f"Hybrid Top-20: "
            f"{'정답 포함 (' + str(hybrid_rank) + '위)' if hybrid_rank else '정답 없음'}"
        )

        # ------------------------------------------------
        # 2. text 보강
        # ------------------------------------------------

        candidates_with_text = attach_texts(
            candidates=candidates,
            client=client,
        )

        # ------------------------------------------------
        # 3. Reranker
        # ------------------------------------------------

        reranked = reranker.rerank(
            query=query,
            candidates=candidates_with_text,
            top_k=5,
        )

        rerank_rank = next(
            (
                r["rerank_rank"]
                for r in reranked
                if r["chunk_id"] == expected
            ),
            None,
        )

        print()
        print("Reranker Top-5:")

        for r in reranked:

            marker = (
                "✓"
                if r["chunk_id"] == expected
                else " "
            )

            print(
                f"{marker} "
                f"{r['rerank_rank']}위 | "
                f"rerank={r['rerank_score']:.4f} | "
                f"rrf={r.get('rrf_score', 0):.4f} | "
                f"{r['chunk_id']}"
            )

        if rerank_rank:
            print(f"\n→ 정답 순위: {rerank_rank}위")
        else:
            print("\n→ 정답: Top-5에 없음")

    print()
    print("=" * 60)
    print("테스트 완료")
    print("=" * 60)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()
