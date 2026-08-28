import json
import re
from pathlib import Path
from uuid import uuid5, NAMESPACE_URL

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)


# ============================================================
# Path
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]


# ============================================================
# 임베딩할 법률 청킹 파일
# ============================================================

JSON_PATHS = [

    # --------------------------------------------------------
    # 근로기준법
    # --------------------------------------------------------

    (
        BASE_DIR
        / "data"
        / "processed"
        / "근로기준법(법률)(제21533호)(20270101)"
        / "chunks.json"
    ),

    # --------------------------------------------------------
    # 산업안전보건법
    # --------------------------------------------------------

    (
        BASE_DIR
        / "data"
        / "processed"
        / "산업안전보건법(법률)(제21374호)(20260801)"
        / "chunks.json"
    ),

    # --------------------------------------------------------
    # 산업재해보상보험법
    # --------------------------------------------------------

    (
        BASE_DIR
        / "data"
        / "processed"
        / "산업재해보상보험법(법률)(제21375호)(20260701)"
        / "chunks.json"
    ),

    # --------------------------------------------------------
    # 직업교육훈련 촉진법
    # --------------------------------------------------------

    (
        BASE_DIR
        / "data"
        / "processed"
        / "직업교육훈련 촉진법(법률)(제21065호)(20251001)"
        / "chunks.json"
    ),
]


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

# test_embedding.py에서 확인한 실제 dimension
VECTOR_SIZE = 768


# ============================================================
# Metadata 허용 타입
# ============================================================

def is_valid_metadata_value(value):
    """
    Qdrant Payload에 저장 가능한 metadata 타입인지 확인한다.
    """

    return (
        isinstance(value, (str, int, float, bool))
        or value is None
        or (
            isinstance(value, list)
            and all(
                isinstance(item, (str, int, float, bool))
                or item is None
                for item in value
            )
        )
    )


# ============================================================
# Metadata 평탄화
# ============================================================

def flatten_metadata(raw_metadata):
    """
    중첩된 metadata를 평탄한 dictionary 형태로 변환한다.

    예:

    {
        "source": {
            "file": "법률.pdf",
            "page_start": 1,
            "page_end": 1,
            "section": "본문",
            "article": "제1조"
        }
    }

    ↓

    {
        "file": "법률.pdf",
        "page_start": 1,
        "page_end": 1,
        "section": "본문",
        "article": "제1조"
    }
    """

    metadata = {}

    if not isinstance(raw_metadata, dict):
        return metadata

    for key, value in raw_metadata.items():

        # ----------------------------------------------------
        # 중첩 dictionary
        # ----------------------------------------------------

        if isinstance(value, dict):

            for sub_key, sub_value in value.items():

                if is_valid_metadata_value(sub_value):

                    metadata[sub_key] = sub_value

        # ----------------------------------------------------
        # 일반 값
        # ----------------------------------------------------

        else:

            if is_valid_metadata_value(value):

                metadata[key] = value

    return metadata


# ============================================================
# law_name 에서 법률 단순명 추출
# ============================================================

def extract_law_short_name(law_name: str) -> str:
    """
    law_name 에서 괄호 앞 법률 단순명을 추출한다.

    예:
        "근로기준법(법률)(제21533호)(20270101)"
        → "근로기준법"

        "직업교육훈련 촉진법(법률)(제21065호)(20251001)"
        → "직업교육훈련 촉진법"
    """

    if not law_name:
        return ""

    # 첫 번째 '(' 이전 부분만 추출
    match = re.match(r"^([^(]+)", law_name)

    if match:
        return match.group(1).strip()

    return law_name.strip()


# ============================================================
# JSON → LangChain Documents
# ============================================================

def load_documents(json_path):

    print()
    print("=" * 60)
    print("JSON 파일 처리")
    print("=" * 60)

    print()
    print(f"JSON 파일: {json_path}")

    # --------------------------------------------------------
    # 파일 존재 확인
    # --------------------------------------------------------

    if not json_path.exists():

        raise FileNotFoundError(
            "\n청킹 JSON 파일을 찾을 수 없습니다.\n"
            f"경로: {json_path}"
        )

    # --------------------------------------------------------
    # JSON 읽기
    # --------------------------------------------------------

    with open(
        json_path,
        "r",
        encoding="utf-8",
    ) as f:

        chunks = json.load(f)

    # --------------------------------------------------------
    # JSON 구조 확인
    # --------------------------------------------------------

    if not isinstance(chunks, list):

        raise ValueError(
            "chunks.json의 최상위 구조가 list가 아닙니다."
        )

    print(
        f"JSON 청크 개수: "
        f"{len(chunks)}"
    )

    documents = []
    ids = []

    # ========================================================
    # Chunk 처리
    # ========================================================

    for index, chunk in enumerate(chunks):

        # ----------------------------------------------------
        # ID 검사
        # ----------------------------------------------------

        chunk_id = chunk.get("id")

        if not chunk_id:

            raise ValueError(
                f"{index}번째 청크에 id가 없습니다."
            )

        # ----------------------------------------------------
        # 중복 ID 검사
        # ----------------------------------------------------

        if chunk_id in ids:

            raise ValueError(
                f"중복 chunk ID 발견:\n"
                f"{chunk_id}"
            )

        ids.append(chunk_id)

        # ----------------------------------------------------
        # Text 검사
        # ----------------------------------------------------

        text = chunk.get("text")

        if not isinstance(text, str):

            raise ValueError(
                f"청크 {chunk_id}의 "
                f"text가 문자열이 아닙니다."
            )

        text = text.strip()

        if not text:

            raise ValueError(
                f"청크 {chunk_id}의 "
                f"text가 비어 있습니다."
            )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        raw_metadata = chunk.get(
            "metadata",
            {}
        )

        metadata = flatten_metadata(
            raw_metadata
        )

        # ----------------------------------------------------
        # 기존 Chunk ID 보존
        #
        # Qdrant Point ID와 별도로 저장
        # ----------------------------------------------------

        metadata["chunk_id"] = chunk_id

        # ----------------------------------------------------
        # law 단순명 추가
        #
        # "근로기준법(법률)(제21533호)(20270101)"
        # → "근로기준법"
        #
        # Qdrant Filter에서 사용
        # ----------------------------------------------------

        raw_law_name = metadata.get(
            "law_name",
            "",
        )

        metadata["law"] = extract_law_short_name(
            raw_law_name
        )

        # ----------------------------------------------------
        # LangChain Document
        # ----------------------------------------------------

        document = Document(
            page_content=text,
            metadata=metadata,
        )

        documents.append(document)

    # ========================================================
    # 최종 검사
    # ========================================================

    if len(documents) != len(ids):

        raise RuntimeError(
            "Document 개수와 ID 개수가 "
            "일치하지 않습니다."
        )

    print()
    print(
        f"Document 생성 완료: "
        f"{len(documents)}개"
    )

    return documents, ids


# ============================================================
# Qdrant 연결
# ============================================================

def connect_qdrant():

    print()
    print("Qdrant 연결 중...")

    # --------------------------------------------------------
    # Docker Qdrant
    # localhost:6333
    # --------------------------------------------------------

    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
    )

    # --------------------------------------------------------
    # 연결 테스트
    # --------------------------------------------------------

    try:

        collections = client.get_collections()

        print("Qdrant 연결 성공")

        print(
            f"현재 Collection 개수: "
            f"{len(collections.collections)}"
        )

    except Exception as e:

        raise ConnectionError(
            "\nQdrant 서버에 연결할 수 없습니다.\n"
            "Docker Qdrant가 실행 중인지 확인하세요.\n\n"
            f"Host: {QDRANT_HOST}\n"
            f"Port: {QDRANT_PORT}\n"
            f"원본 오류: {e}"
        )

    return client


# ============================================================
# Collection 생성 / 확인
# ============================================================

def create_collection(client):

    print()
    print(
        f"Collection 확인: "
        f"{COLLECTION_NAME}"
    )

    # ========================================================
    # 기존 Collection 확인
    # ========================================================

    if client.collection_exists(
        COLLECTION_NAME
    ):

        collection_info = client.get_collection(
            collection_name=COLLECTION_NAME
        )

        print()
        print(
            f"Collection 이미 존재: "
            f"{COLLECTION_NAME}"
        )

        print(
            f"현재 Point 수: "
            f"{collection_info.points_count}"
        )

        # ----------------------------------------------------
        # 기존 Vector Dimension 확인
        # ----------------------------------------------------

        existing_size = (
            collection_info
            .config
            .params
            .vectors
            .size
        )

        print(
            f"기존 Vector dimension: "
            f"{existing_size}"
        )

        if existing_size != VECTOR_SIZE:

            raise ValueError(
                "\n기존 Collection의 "
                "Vector dimension이 "
                "현재 Embedding과 다릅니다.\n\n"
                f"기존 Collection: "
                f"{existing_size}\n"
                f"현재 Embedding: "
                f"{VECTOR_SIZE}"
            )

        return

    # ========================================================
    # Collection이 없는 경우 생성
    # ========================================================

    client.create_collection(
        collection_name=COLLECTION_NAME,

        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )

    print()
    print("Collection 생성 완료")

    print(
        f"Collection       : "
        f"{COLLECTION_NAME}"
    )

    print(
        f"Vector dimension : "
        f"{VECTOR_SIZE}"
    )

    print(
        f"Distance         : "
        f"COSINE"
    )


# ============================================================
# Vector + Payload 저장
# ============================================================

def save_to_qdrant(
    client,
    embedding,
    documents,
    ids,
):

    print()
    print("Embedding 생성 시작...")

    # ========================================================
    # Text 추출
    # ========================================================

    texts = [
        document.page_content
        for document in documents
    ]

    # ========================================================
    # Embedding 생성
    # ========================================================

    vectors = embedding.embed_documents(
        texts
    )

    # --------------------------------------------------------
    # 개수 확인
    # --------------------------------------------------------

    if len(vectors) != len(documents):

        raise RuntimeError(
            "Embedding 개수와 "
            "Document 개수가 다릅니다."
        )

    print()
    print(
        f"Embedding 생성 완료: "
        f"{len(vectors)}개"
    )

    # ========================================================
    # Vector Dimension 검증
    # ========================================================

    for index, vector in enumerate(vectors):

        actual_dimension = len(vector)

        if actual_dimension != VECTOR_SIZE:

            raise ValueError(
                f"\n{index}번째 Vector dimension 오류\n"
                f"예상: {VECTOR_SIZE}\n"
                f"실제: {actual_dimension}"
            )

    # ========================================================
    # Qdrant Point 생성
    # ========================================================

    points = []

    for document, chunk_id, vector in zip(
        documents,
        ids,
        vectors,
    ):

        # ----------------------------------------------------
        # Qdrant Point ID
        #
        # chunk_id → UUID5
        #
        # 같은 chunk_id라면
        # 항상 동일한 UUID가 생성된다.
        # ----------------------------------------------------

        point_id = str(
            uuid5(
                NAMESPACE_URL,
                chunk_id,
            )
        )

        # ----------------------------------------------------
        # Payload
        #
        # text + metadata
        # ----------------------------------------------------

        payload = {
            "text": document.page_content,
            **document.metadata,
        }

        # ----------------------------------------------------
        # Point 생성
        # ----------------------------------------------------

        point = PointStruct(
            id=point_id,
            vector=vector,
            payload=payload,
        )

        points.append(point)

    # ========================================================
    # Point 개수 확인
    # ========================================================

    if len(points) != len(documents):

        raise RuntimeError(
            "Point 개수와 "
            "Document 개수가 다릅니다."
        )

    # ========================================================
    # Qdrant 저장
    # ========================================================

    print()
    print("Qdrant Vector 저장 시작...")

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    print()
    print(
        "Qdrant Vector 저장 완료"
    )


# ============================================================
# Collection 상태 출력
# ============================================================

def print_collection_status(client):

    collection_info = client.get_collection(
        collection_name=COLLECTION_NAME
    )

    print()
    print("=" * 60)
    print("Qdrant 저장 완료")
    print("=" * 60)

    print(
        f"Qdrant Host      : "
        f"{QDRANT_HOST}"
    )

    print(
        f"Qdrant Port      : "
        f"{QDRANT_PORT}"
    )

    print(
        f"Collection       : "
        f"{COLLECTION_NAME}"
    )

    print(
        f"Embedding Model  : "
        f"{EMBEDDING_MODEL}"
    )

    print(
        f"Vector Dimension : "
        f"{VECTOR_SIZE}"
    )

    print(
        f"Point Count      : "
        f"{collection_info.points_count}"
    )

    print("=" * 60)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("Legal RAG Embedding 시작")
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
    # 2. Embedding Dimension 확인
    # ========================================================

    print()
    print("Embedding dimension 확인...")

    test_vector = embedding.embed_query(
        "근로기준법 제1조의 목적은 무엇인가?"
    )

    actual_dimension = len(
        test_vector
    )

    print(
        f"Embedding dimension: "
        f"{actual_dimension}"
    )

    # --------------------------------------------------------
    # Dimension 검증
    # --------------------------------------------------------

    if actual_dimension != VECTOR_SIZE:

        raise ValueError(
            "\nEmbedding dimension이 "
            "설정값과 다릅니다.\n"
            f"설정값: {VECTOR_SIZE}\n"
            f"실제값: {actual_dimension}"
        )

    # ========================================================
    # 3. Qdrant 연결
    # ========================================================

    client = connect_qdrant()

    # ========================================================
    # 4. Collection 생성 / 확인
    # ========================================================

    create_collection(client)

    # ========================================================
    # 5. 여러 법률 JSON 처리
    # ========================================================

    total_documents = 0

    total_files = len(
        JSON_PATHS
    )

    # --------------------------------------------------------
    # 각 법률 처리
    # --------------------------------------------------------

    for file_index, json_path in enumerate(
        JSON_PATHS,
        start=1,
    ):

        print()
        print("=" * 60)

        print(
            f"법률 처리 "
            f"[{file_index}/{total_files}]"
        )

        print("=" * 60)

        # ----------------------------------------------------
        # JSON → Documents
        # ----------------------------------------------------

        documents, ids = load_documents(
            json_path
        )

        print()
        print(
            f"임베딩 대상 청크: "
            f"{len(documents)}개"
        )

        # ----------------------------------------------------
        # Embedding + Qdrant 저장
        # ----------------------------------------------------

        save_to_qdrant(
            client=client,
            embedding=embedding,
            documents=documents,
            ids=ids,
        )

        # ----------------------------------------------------
        # 누적 처리 개수
        # ----------------------------------------------------

        total_documents += len(
            documents
        )

        # ----------------------------------------------------
        # 현재 Qdrant 상태
        # ----------------------------------------------------

        collection_info = client.get_collection(
            collection_name=COLLECTION_NAME
        )

        print()
        print(
            f"현재 Qdrant Point 수: "
            f"{collection_info.points_count}"
        )

    # ========================================================
    # 6. 최종 결과
    # ========================================================

    print_collection_status(
        client
    )

    print()
    print(
        f"이번 실행에서 처리한 "
        f"전체 Chunk: "
        f"{total_documents}개"
    )

    print()
    print(
        "Legal RAG Embedding 작업이 "
        "정상적으로 완료되었습니다."
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()