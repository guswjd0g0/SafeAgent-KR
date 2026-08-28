from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient


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
# Qdrant 연결
# ============================================================

def connect_qdrant():

    print("Qdrant 연결 중...")

    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
    )

    try:

        collections = client.get_collections()

        print("Qdrant 연결 성공")
        print(
            f"Collection 개수: "
            f"{len(collections.collections)}"
        )

    except Exception as e:

        raise ConnectionError(
            "\nQdrant 연결 실패\n"
            "Docker Qdrant가 실행 중인지 확인하세요.\n"
            f"오류: {e}"
        )

    return client


# ============================================================
# Collection 확인
# ============================================================

def check_collection(client):

    print()
    print("Collection 확인...")

    if not client.collection_exists(
        COLLECTION_NAME
    ):

        raise ValueError(
            f"\nCollection이 존재하지 않습니다.\n"
            f"Collection: {COLLECTION_NAME}"
        )

    collection_info = client.get_collection(
        collection_name=COLLECTION_NAME
    )

    print(
        f"Collection: "
        f"{COLLECTION_NAME}"
    )

    print(
        f"Point Count: "
        f"{collection_info.points_count}"
    )

    return collection_info


# ============================================================
# Similarity Search
# ============================================================

def similarity_search(
    client,
    embedding,
    query,
    top_k=TOP_K,
):

    print()
    print("=" * 60)
    print("Similarity Search")
    print("=" * 60)

    print()
    print(
        f"질문: {query}"
    )

    # ========================================================
    # 1. Query Embedding
    # ========================================================

    print()
    print("Query Embedding 생성...")

    query_vector = embedding.embed_query(
        query
    )

    print(
        f"Query Vector Dimension: "
        f"{len(query_vector)}"
    )

    # ========================================================
    # 2. Qdrant 검색
    # ========================================================

    print()
    print(
        f"Qdrant 검색 시작 "
        f"(Top-{top_k})..."
    )

    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    points = search_result.points

    # ========================================================
    # 3. 검색 결과 확인
    # ========================================================

    print()
    print(
        f"검색 결과: "
        f"{len(points)}개"
    )

    # ========================================================
    # 4. 결과 출력
    # ========================================================

    for rank, point in enumerate(
        points,
        start=1,
    ):

        payload = point.payload or {}

        text = payload.get(
            "text",
            "",
        )

        chunk_id = payload.get(
            "chunk_id",
            "N/A",
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        file_name = payload.get(
            "file",
            "N/A",
        )

        page_start = payload.get(
            "page_start",
            "N/A",
        )

        page_end = payload.get(
            "page_end",
            "N/A",
        )

        section = payload.get(
            "section",
            "N/A",
        )

        article = payload.get(
            "article",
            "N/A",
        )

        # ====================================================
        # 출력
        # ====================================================

        print()
        print("-" * 60)

        print(
            f"[{rank}] "
            f"Score: {point.score:.4f}"
        )

        print(
            f"Chunk ID : {chunk_id}"
        )

        print(
            f"File     : {file_name}"
        )

        print(
            f"Page     : "
            f"{page_start} ~ {page_end}"
        )

        print(
            f"Section  : {section}"
        )

        print(
            f"Article  : {article}"
        )

        print()
        print("Text:")

        print(text)

    print()
    print("=" * 60)

    return points


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("Legal RAG Similarity Search")
    print("=" * 60)

    # ========================================================
    # 1. Embedding Model
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
    # 2. Qdrant 연결
    # ========================================================

    client = connect_qdrant()

    # ========================================================
    # 3. Collection 확인
    # ========================================================

    check_collection(
        client
    )

    # ========================================================
    # 4. 테스트 질문
    # ========================================================

    query = (
        "근로자가 받을 수 있는 "
        "연차 유급휴가는 며칠인가?"
    )

    # ========================================================
    # 5. Similarity Search
    # ========================================================

    similarity_search(
        client=client,
        embedding=embedding,
        query=query,
        top_k=TOP_K,
    )

    # ========================================================
    # 완료
    # ========================================================

    print()
    print(
        "Similarity Search 테스트 완료"
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()