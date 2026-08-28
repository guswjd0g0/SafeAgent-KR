import re
from typing import Optional

from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from rank_bm25 import BM25Okapi


from sentence_transformers import SentenceTransformer


# ============================================================
# Qdrant 설정
# ============================================================

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

# 기본 Collection (nomic-embed-text)
COLLECTION_NAME = "legal_documents"

# bge-m3 실험용 Collection
COLLECTION_NAME_BGE_M3 = "legal_documents_bge_m3"


# ============================================================
# Embedding 설정
# ============================================================

EMBEDDING_MODEL = "nomic-embed-text"


# ============================================================
# Search 설정
# ============================================================

TOP_K = 5

# RRF 상수 (작을수록 BM25 고순위 항목 영향력 증가)
RRF_K = 5

# BM25/Dense 각각 가져올 후보 수
# 최종 Top-K보다 충분히 크게
CANDIDATE_K = 50


# ============================================================
# 한국어 토크나이저
#
# 형태소 분석기 없이 사용 가능한 수준의 토크나이저.
#
# 처리 순서:
# 1. 소문자 정규화
# 2. 특수문자 공백 치환
# 3. 공백 기준 분리
# 4. 조사 제거          (에서, 은/는, 이/가 등)
# 5. 동사 어근 추출     (정의되나요 → 정의, 적용되 → 적용)
# 6. 1글자 이하 제거
# ============================================================

# 한국어 조사 패턴
_JOSA_PATTERN = re.compile(
    r"(은|는|이|가|을|를|의|에|에서|에게|으로|로|와|과|도|만|이나|나|란"
    r"|이란|까지|부터|보다|처럼|에서는|에는|에도|에서도|으로는|으로도"
    r"|에서의|에의|와의|과의|이고|이며|이라|으로서|로서|으로써|로써"
    r"|이어서|여서|에서부터|로부터|은요|는요|나요|인가요|인지|이야"
    r"|이에요|인데|이라면|라면|라도|이라도)$"
)

# 한국어 동사/형용사 어미 패턴 (어근 추출용)
_VERB_SUFFIX = re.compile(
    r"(되나요|됩니까|됩니다|되어야|되어서|되어있|되었|되어|되는|됩|되고"
    r"|되며|되면|되지|되다|해야|합니까|합니다|하나요|하여야|하여|하는"
    r"|하고|하며|하면|하지|하다|합|이다|이며|있나요|있어야|있습니까"
    r"|있습니다|있는|있어|됩니다|됩니까|되)$"
)


def _extract_stem(token: str) -> str | None:
    """동사 어미를 제거한 어근을 반환한다. 어근이 2글자 미만이면 None."""
    m = _VERB_SUFFIX.search(token)
    if m and (len(token) - len(m.group())) >= 2:
        return token[: m.start()]
    return None


def tokenize(text: str) -> list[str]:
    """
    한국어 법률 텍스트를 토큰 리스트로 변환한다.

    조사 제거 + 동사 어근 추출을 적용하여
    질문과 문서 간 표면형 불일치를 최소화한다.

    예:
        "근로자와 사용자는 어떻게 정의되나요"
        → ['근로자', '사용자', '어떻게', '정의되', '정의']
    """

    if not text:
        return []

    # --------------------------------------------------------
    # 1. 소문자 정규화
    # --------------------------------------------------------

    text = text.lower()

    # --------------------------------------------------------
    # 2. 특수문자를 공백으로 치환
    # --------------------------------------------------------

    text = re.sub(r"[^\w\s가-힣]", " ", text)

    # --------------------------------------------------------
    # 3. 공백 기준 분리
    # --------------------------------------------------------

    raw_tokens = text.split()

    # --------------------------------------------------------
    # 4~6. 조사 제거 + 어근 추출 + 길이 필터
    # --------------------------------------------------------

    tokens: list[str] = []
    seen: set[str] = set()

    for t in raw_tokens:

        if len(t) <= 1:
            continue

        # 조사 제거
        stripped = _JOSA_PATTERN.sub("", t)
        base = stripped if len(stripped) > 1 else t

        if base not in seen:
            tokens.append(base)
            seen.add(base)

        # 동사 어근도 추가
        stem = _extract_stem(base)
        if stem and len(stem) > 1 and stem not in seen:
            tokens.append(stem)
            seen.add(stem)

    return tokens


# ============================================================
# BM25 인덱스
# ============================================================

class BM25Index:
    """
    Qdrant의 chunk를 로드하여 BM25 인덱스를 구축한다.

    law 필터를 적용하면 해당 법률 chunk만 인덱싱한다.
    """

    def __init__(
        self,
        client: QdrantClient,
        law_filter: Optional[str] = None,
        collection_name: str = COLLECTION_NAME,
    ):

        # --------------------------------------------------------
        # Qdrant에서 chunk 로드
        # --------------------------------------------------------

        self.chunks = self._load_chunks(
            client=client,
            law_filter=law_filter,
            collection_name=collection_name,
        )

        # --------------------------------------------------------
        # BM25 corpus 구성
        #
        # text + article_title을 함께 인덱싱
        # article_title을 5회 반복하여
        # 조문 제목 키워드에 강한 가중치 부여
        #
        # 예: 제2조(정의) → "정의"가 5회 등장
        #     → "정의되나요" 질문에서 높은 BM25 점수
        # --------------------------------------------------------

        corpus = []

        for chunk in self.chunks:

            text = chunk["text"]

            article_title = chunk.get(
                "article_title",
                "",
            ) or ""

            # 제목을 텍스트 앞에 5회 반복 추가
            combined = (
                f"{article_title} " * 5
                + text
            )

            corpus.append(
                tokenize(combined)
            )

        self.corpus = corpus

        # --------------------------------------------------------
        # BM25 모델 생성
        # --------------------------------------------------------

        self.bm25 = BM25Okapi(corpus)

    # ============================================================
    # Qdrant chunk 로드
    # ============================================================

    def _load_chunks(
        self,
        client: QdrantClient,
        law_filter: Optional[str],
        collection_name: str = COLLECTION_NAME,
    ) -> list[dict]:
        """
        Qdrant에서 전체 또는 특정 법률의 chunk를 로드한다.
        """

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

        # 최대 1000개 (현재 514개)
        points, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=qdrant_filter,
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )

        chunks = []

        for point in points:

            payload = point.payload or {}

            chunks.append(
                {
                    "point_id": str(point.id),
                    "chunk_id": payload.get(
                        "chunk_id", ""
                    ),
                    "law": payload.get(
                        "law", ""
                    ),
                    "article": payload.get(
                        "article", ""
                    ),
                    "article_title": payload.get(
                        "article_title", ""
                    ),
                    "file": payload.get(
                        "file", ""
                    ),
                    "text": payload.get(
                        "text", ""
                    ),
                }
            )

        return chunks

    # ============================================================
    # BM25 검색
    # ============================================================

    def search(
        self,
        query: str,
        top_k: int = CANDIDATE_K,
    ) -> list[dict]:
        """
        BM25 점수 기준으로 상위 chunk를 반환한다.

        반환 형태:
        [
            {
                "rank": 1,
                "bm25_score": 3.14,
                "chunk_id": "...",
                ...
            },
            ...
        ]
        """

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(
            query_tokens
        )

        # 점수 기준 내림차순 정렬
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results = []

        for rank, idx in enumerate(
            ranked_indices[:top_k],
            start=1,
        ):

            chunk = self.chunks[idx]

            results.append(
                {
                    "rank": rank,
                    "bm25_score": float(
                        scores[idx]
                    ),
                    "chunk_id": chunk["chunk_id"],
                    "law": chunk["law"],
                    "article": chunk["article"],
                    "article_title": chunk[
                        "article_title"
                    ],
                    "file": chunk["file"],
                    "text": chunk["text"],
                }
            )

        return results


# ============================================================
# RRF (Reciprocal Rank Fusion)
#
# 두 랭킹 리스트를 결합한다.
#
# RRF score = Σ 1 / (k + rank_i)
# ============================================================

def reciprocal_rank_fusion(
    dense_results: list,
    bm25_results: list,
    rrf_k: int = RRF_K,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Dense 검색 결과와 BM25 검색 결과를
    RRF로 결합하여 최종 순위를 반환한다.

    Parameters
    ----------
    dense_results : Qdrant query_points 결과 (points 리스트)
    bm25_results  : BM25Index.search() 결과 리스트
    rrf_k         : RRF 상수 (default 60)
    top_k         : 최종 반환 개수

    Returns
    -------
    chunk_id 기준으로 정렬된 결과 리스트
    """

    # --------------------------------------------------------
    # chunk_id → RRF 점수 누적
    # --------------------------------------------------------

    rrf_scores: dict[str, float] = {}

    # chunk_id → payload 매핑 (출력용)
    chunk_meta: dict[str, dict] = {}

    # --------------------------------------------------------
    # Dense 결과 반영
    # --------------------------------------------------------

    for rank, point in enumerate(
        dense_results,
        start=1,
    ):

        payload = point.payload or {}

        chunk_id = payload.get(
            "chunk_id", str(point.id)
        )

        rrf_scores[chunk_id] = (
            rrf_scores.get(chunk_id, 0.0)
            + 1.0 / (rrf_k + rank)
        )

        if chunk_id not in chunk_meta:

            chunk_meta[chunk_id] = {
                "chunk_id": chunk_id,
                "law": payload.get("law", ""),
                "article": payload.get(
                    "article", ""
                ),
                "file": payload.get("file", ""),
                "dense_score": round(
                    point.score, 4
                ),
                "bm25_score": 0.0,
                "dense_rank": rank,
                "bm25_rank": None,
            }

    # --------------------------------------------------------
    # BM25 결과 반영
    # --------------------------------------------------------

    for item in bm25_results:

        chunk_id = item["chunk_id"]

        rank = item["rank"]

        rrf_scores[chunk_id] = (
            rrf_scores.get(chunk_id, 0.0)
            + 1.0 / (rrf_k + rank)
        )

        if chunk_id not in chunk_meta:

            chunk_meta[chunk_id] = {
                "chunk_id": chunk_id,
                "law": item.get("law", ""),
                "article": item.get(
                    "article", ""
                ),
                "file": item.get("file", ""),
                "dense_score": 0.0,
                "bm25_score": round(
                    item["bm25_score"], 4
                ),
                "dense_rank": None,
                "bm25_rank": rank,
            }

        else:

            chunk_meta[chunk_id][
                "bm25_score"
            ] = round(item["bm25_score"], 4)

            chunk_meta[chunk_id][
                "bm25_rank"
            ] = rank

    # --------------------------------------------------------
    # RRF 점수 기준 정렬
    # --------------------------------------------------------

    ranked = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    # --------------------------------------------------------
    # 최종 결과 구성
    # --------------------------------------------------------

    results = []

    for final_rank, (chunk_id, rrf_score) in enumerate(
        ranked[:top_k],
        start=1,
    ):

        meta = chunk_meta[chunk_id]

        results.append(
            {
                "rank": final_rank,
                "rrf_score": round(rrf_score, 6),
                "chunk_id": chunk_id,
                "law": meta["law"],
                "article": meta["article"],
                "file": meta["file"],
                "dense_score": meta["dense_score"],
                "bm25_score": meta["bm25_score"],
                "dense_rank": meta["dense_rank"],
                "bm25_rank": meta["bm25_rank"],
            }
        )

    return results


# ============================================================
# Hybrid Search (메인 함수)
# ============================================================

def hybrid_search(
    client: QdrantClient,
    embedding,
    query: str,
    law_filter: Optional[str] = None,
    top_k: int = TOP_K,
    candidate_k: int = CANDIDATE_K,
    rrf_k: int = RRF_K,
    collection_name: str = COLLECTION_NAME,
) -> list[dict]:
    """
    Dense Search + BM25 Search 를 RRF로 결합한다.

    Parameters
    ----------
    client          : Qdrant 클라이언트
    embedding       : OllamaEmbeddings 또는 SentenceTransformer 래퍼
    query           : 검색 질문
    law_filter      : 법률명 필터 (예: "근로기준법"), None이면 전체 검색
    top_k           : 최종 반환 개수
    candidate_k     : Dense/BM25 각각 가져올 후보 수
    rrf_k           : RRF 상수
    collection_name : 검색할 Qdrant Collection 이름

    Returns
    -------
    RRF 결합된 결과 리스트
    """

    # --------------------------------------------------------
    # 1. Query Embedding
    #
    # OllamaEmbeddings: embed_query(str) → list[float]
    # SentenceTransformer 래퍼: embed_query(str) → list[float]
    # --------------------------------------------------------

    query_vector = embedding.embed_query(query)

    # --------------------------------------------------------
    # 2. Dense Search (Qdrant)
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

    dense_result = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=qdrant_filter,
        limit=candidate_k,
        with_payload=True,
    )

    dense_points = dense_result.points

    # --------------------------------------------------------
    # 3. BM25 Search
    # --------------------------------------------------------

    bm25_index = BM25Index(
        client=client,
        law_filter=law_filter,
        collection_name=collection_name,
    )

    bm25_results = bm25_index.search(
        query=query,
        top_k=candidate_k,
    )

    # --------------------------------------------------------
    # 4. RRF 결합
    # --------------------------------------------------------

    final_results = reciprocal_rank_fusion(
        dense_results=dense_points,
        bm25_results=bm25_results,
        rrf_k=rrf_k,
        top_k=top_k,
    )

    return final_results


# ============================================================
# Qdrant 연결
# ============================================================

def connect_qdrant() -> QdrantClient:

    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
    )

    return client


# ============================================================
# SentenceTransformer embed_query 래퍼
#
# OllamaEmbeddings와 동일한 인터페이스(embed_query)를 제공하여
# hybrid_search()에서 두 모델을 동일하게 사용할 수 있다.
# ============================================================

class SentenceTransformerEmbedding:
    """
    SentenceTransformer를 OllamaEmbeddings와
    동일한 인터페이스로 감싸는 래퍼.

    사용 예:
        embedding = SentenceTransformerEmbedding("BAAI/bge-m3")
        vector = embedding.embed_query("근로기준법 제2조")
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
    ):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed_query(self, text: str) -> list[float]:
        """단일 쿼리를 임베딩하고 float 리스트를 반환한다."""
        vector = self.model.encode(
            [text],
            normalize_embeddings=True,
        )[0]
        return vector.tolist()

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """문서 리스트를 임베딩한다."""
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
        )
        return [v.tolist() for v in vectors]


# ============================================================
# Main — 동작 확인용
# ============================================================

def main():

    print("=" * 60)
    print("Hybrid Search 테스트")
    print("=" * 60)

    embedding = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    client = connect_qdrant()

    # --------------------------------------------------------
    # 테스트 질문
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

    for query, law_filter, expected_chunk_id in test_cases:

        print()
        print("-" * 60)
        print(f"질문: {query}")
        print(f"법률 필터: {law_filter}")
        print(f"정답 Chunk ID: {expected_chunk_id}")
        print()

        results = hybrid_search(
            client=client,
            embedding=embedding,
            query=query,
            law_filter=law_filter,
            top_k=TOP_K,
        )

        correct_rank = None

        for result in results:

            marker = (
                "✓"
                if result["chunk_id"] == expected_chunk_id
                else " "
            )

            if (
                result["chunk_id"] == expected_chunk_id
                and correct_rank is None
            ):
                correct_rank = result["rank"]

            print(
                f"{marker} "
                f"{result['rank']}위 | "
                f"RRF={result['rrf_score']:.4f} | "
                f"Dense={result['dense_score']:.4f}(rank={result['dense_rank']}) | "
                f"BM25={result['bm25_score']:.4f}(rank={result['bm25_rank']}) | "
                f"{result['chunk_id']}"
            )

        if correct_rank:
            print(f"\n→ 정답 순위: {correct_rank}위")
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
