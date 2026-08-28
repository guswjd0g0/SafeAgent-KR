"""
bge-m3 Embedding 구축 스크립트

기존 legal_documents (nomic-embed-text) Collection은 보존하고
legal_documents_bge_m3 Collection을 별도로 구축한다.

특징:
  - Embedding: SentenceTransformer(BAAI/bge-m3)
  - Vector Dimension: 1024
  - Distance: COSINE
  - Collection: legal_documents_bge_m3

중요:
  - 기존 bge-m3 Collection을 삭제하지 않는다.
  - 이미 저장된 chunk는 다시 임베딩하지 않는다.
  - batch 단위로 embedding + Qdrant 저장한다.
  - 중간에 Ctrl+C로 중단해도 이미 저장된 batch는 유지된다.
  - 재실행하면 저장되지 않은 chunk부터 이어서 처리한다.
"""

import json
import re
import time
from pathlib import Path
from uuid import uuid5, NAMESPACE_URL
from collections import Counter

from sentence_transformers import SentenceTransformer

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

    BASE_DIR
    / "data"
    / "processed"
    / "근로기준법(법률)(제21533호)(20270101)"
    / "chunks.json",

    BASE_DIR
    / "data"
    / "processed"
    / "산업안전보건법(법률)(제21374호)(20260801)"
    / "chunks.json",

    BASE_DIR
    / "data"
    / "processed"
    / "산업재해보상보험법(법률)(제21375호)(20260701)"
    / "chunks.json",

    BASE_DIR
    / "data"
    / "processed"
    / "직업교육훈련 촉진법(법률)(제21065호)(20251001)"
    / "chunks.json",
]


# ============================================================
# Qdrant 설정
# ============================================================

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

COLLECTION_NAME = "legal_documents_bge_m3"


# ============================================================
# Embedding 설정
# ============================================================

EMBEDDING_MODEL = "BAAI/bge-m3"

VECTOR_SIZE = 1024

# CPU 환경을 고려하여 작게 설정
BATCH_SIZE = 8


# ============================================================
# 법률명 단순화
# ============================================================

def extract_law_short_name(law_name: str) -> str:
    """
    예:
    근로기준법(법률)(제21533호)(20270101)
    →
    근로기준법
    """

    if not law_name:
        return ""

    match = re.match(r"^([^(]+)", law_name)

    if match:
        return match.group(1).strip()

    return law_name.strip()


# ============================================================
# JSON → chunks
# ============================================================

def load_chunks(json_path: Path) -> list[dict]:

    print()
    print(f"JSON 로드: {json_path.name}")

    if not json_path.exists():
        raise FileNotFoundError(
            f"파일 없음: {json_path}"
        )

    with open(
        json_path,
        "r",
        encoding="utf-8",
    ) as f:

        chunks = json.load(f)

    if not isinstance(chunks, list):

        raise ValueError(
            "chunks.json 최상위 구조가 "
            "list가 아닙니다."
        )

    for chunk in chunks:

        metadata = chunk.get(
            "metadata",
            {},
        )

        law_name = metadata.get(
            "law_name",
            "",
        )

        metadata["law"] = (
            extract_law_short_name(
                law_name
            )
        )

        chunk["metadata"] = metadata

    print(
        f"  청크 수: {len(chunks)}"
    )

    return chunks


# ============================================================
# Qdrant 연결
# ============================================================

def connect_qdrant() -> QdrantClient:

    print()
    print("Qdrant 연결 중...")

    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
    )

    try:

        collections = (
            client.get_collections()
        )

        print(
            "Qdrant 연결 성공 "
            f"(Collection 수: "
            f"{len(collections.collections)})"
        )

    except Exception as e:

        raise ConnectionError(
            "\nQdrant 연결 실패.\n"
            "Docker 실행 여부를 확인하세요.\n"
            f"오류: {e}"
        )

    return client


# ============================================================
# Collection 준비
# ============================================================

def prepare_collection(
    client: QdrantClient,
) -> None:

    print()
    print(
        f"Collection 준비: "
        f"{COLLECTION_NAME}"
    )

    # --------------------------------------------------------
    # 이미 존재하면 절대 삭제하지 않는다.
    # --------------------------------------------------------

    if client.collection_exists(
        COLLECTION_NAME
    ):

        info = client.get_collection(
            COLLECTION_NAME
        )

        print(
            "  기존 Collection 유지"
        )

        print(
            f"  현재 Point: "
            f"{info.points_count}"
        )

        return

    # --------------------------------------------------------
    # 없으면 새로 생성
    # --------------------------------------------------------

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )

    print(
        "  Collection 생성 완료"
    )

    print(
        f"  Dimension: {VECTOR_SIZE}"
    )

    print(
        "  Distance: COSINE"
    )


# ============================================================
# 기존 chunk_id 조회
# ============================================================

def get_existing_chunk_ids(
    client: QdrantClient,
) -> set[str]:
    """
    현재 Qdrant에 저장되어 있는
    chunk_id 전체를 가져온다.

    이미 저장된 chunk는
    다시 embedding하지 않는다.
    """

    print()
    print(
        "기존 chunk 확인 중..."
    )

    existing_ids = set()

    offset = None

    while True:

        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            break

        for point in points:

            payload = (
                point.payload or {}
            )

            chunk_id = payload.get(
                "chunk_id"
            )

            if chunk_id:
                existing_ids.add(
                    chunk_id
                )

        if offset is None:
            break

    print(
        f"  기존 chunk: "
        f"{len(existing_ids)}개"
    )

    return existing_ids


# ============================================================
# metadata → payload
# ============================================================

def build_payload(
    chunk: dict,
) -> dict:

    text = chunk.get(
        "text",
        "",
    ).strip()

    metadata = chunk.get(
        "metadata",
        {},
    )

    payload = {
        "text": text
    }

    for key, value in metadata.items():

        if isinstance(
            value,
            dict,
        ):

            for sub_key, sub_value in value.items():

                if (
                    isinstance(
                        sub_value,
                        (
                            str,
                            int,
                            float,
                            bool,
                        ),
                    )
                    or sub_value is None
                ):

                    payload[sub_key] = (
                        sub_value
                    )

        elif (
            isinstance(
                value,
                (
                    str,
                    int,
                    float,
                    bool,
                ),
            )
            or value is None
        ):

            payload[key] = value

    chunk_id = chunk.get(
        "id",
        "",
    )

    payload["chunk_id"] = chunk_id

    return payload


# ============================================================
# batch embedding + Qdrant 저장
# ============================================================

def process_batch(
    client: QdrantClient,
    model: SentenceTransformer,
    chunks: list[dict],
    batch_number: int,
    total_batches: int,
) -> int:

    texts = [
        chunk.get(
            "text",
            "",
        ).strip()
        for chunk in chunks
    ]

    if not all(texts):

        raise ValueError(
            "빈 text가 포함되어 있습니다."
        )

    print()
    print(
        f"  Batch "
        f"[{batch_number}/{total_batches}] "
        f"{len(chunks)}개 embedding..."
    )

    start_time = time.time()

    # --------------------------------------------------------
    # Embedding
    # --------------------------------------------------------

    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    elapsed = time.time() - start_time

    print(
        f"  Embedding 완료 "
        f"({elapsed:.1f}초)"
    )

    # --------------------------------------------------------
    # Dimension 확인
    # --------------------------------------------------------

    if vectors.shape[1] != VECTOR_SIZE:

        raise ValueError(
            "Vector dimension 불일치: "
            f"예상={VECTOR_SIZE}, "
            f"실제={vectors.shape[1]}"
        )

    # --------------------------------------------------------
    # Point 생성
    # --------------------------------------------------------

    points = []

    for chunk, vector in zip(
        chunks,
        vectors,
    ):

        chunk_id = chunk.get(
            "id",
            "",
        )

        if not chunk_id:

            raise ValueError(
                "chunk에 id가 없습니다."
            )

        point_id = str(
            uuid5(
                NAMESPACE_URL,
                chunk_id,
            )
        )

        payload = build_payload(
            chunk
        )

        points.append(
            PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload=payload,
            )
        )

    # --------------------------------------------------------
    # Qdrant 저장
    # --------------------------------------------------------

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )

    print(
        f"  Qdrant 저장 완료 "
        f"({len(points)}개)"
    )

    return len(points)


# ============================================================
# 법률 하나 처리
# ============================================================

def embed_and_save(
    client: QdrantClient,
    model: SentenceTransformer,
    chunks: list[dict],
    existing_ids: set[str],
) -> int:

    # --------------------------------------------------------
    # 이미 저장된 chunk 제거
    # --------------------------------------------------------

    pending_chunks = []

    skipped = 0

    for chunk in chunks:

        chunk_id = chunk.get(
            "id",
            "",
        )

        if not chunk_id:

            raise ValueError(
                "chunk에 id가 없습니다."
            )

        if chunk_id in existing_ids:

            skipped += 1

        else:

            pending_chunks.append(
                chunk
            )

    print()
    print(
        f"  기존 저장 chunk: "
        f"{skipped}개"
    )

    print(
        f"  신규 embedding 대상: "
        f"{len(pending_chunks)}개"
    )

    # --------------------------------------------------------
    # 전부 존재하는 경우
    # --------------------------------------------------------

    if not pending_chunks:

        print(
            "  → 이미 모두 저장되어 있습니다. SKIP"
        )

        return 0

    # --------------------------------------------------------
    # batch 처리
    # --------------------------------------------------------

    total_batches = (
        len(pending_chunks)
        + BATCH_SIZE
        - 1
    ) // BATCH_SIZE

    saved = 0

    for start in range(
        0,
        len(pending_chunks),
        BATCH_SIZE,
    ):

        batch = pending_chunks[
            start:start + BATCH_SIZE
        ]

        batch_number = (
            start // BATCH_SIZE
        ) + 1

        count = process_batch(
            client=client,
            model=model,
            chunks=batch,
            batch_number=batch_number,
            total_batches=total_batches,
        )

        saved += count

        # ----------------------------------------------------
        # 메모리상 existing_ids도 즉시 갱신
        # ----------------------------------------------------

        for chunk in batch:

            existing_ids.add(
                chunk["id"]
            )

        # ----------------------------------------------------
        # 현재 상태
        # ----------------------------------------------------

        info = client.get_collection(
            COLLECTION_NAME
        )

        print(
            f"  현재 전체 Point: "
            f"{info.points_count}"
        )

    return saved


# ============================================================
# 법률별 Point 수 확인
# ============================================================

def print_law_counts(
    client: QdrantClient,
) -> None:

    print()
    print("=" * 60)
    print("법률별 Point 확인")
    print("=" * 60)

    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=1000,
        with_payload=True,
        with_vectors=False,
    )

    law_counts = Counter()

    for point in points:

        payload = (
            point.payload or {}
        )

        law = payload.get(
            "law",
            "MISSING",
        )

        law_counts[law] += 1

    for law, count in sorted(
        law_counts.items()
    ):

        print(
            f"  {law:30s}: "
            f"{count}개"
        )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("bge-m3 Embedding 구축")
    print("=" * 60)

    print(
        f"  모델      : "
        f"{EMBEDDING_MODEL}"
    )

    print(
        f"  차원      : "
        f"{VECTOR_SIZE}"
    )

    print(
        f"  Batch Size: "
        f"{BATCH_SIZE}"
    )

    print(
        f"  Collection: "
        f"{COLLECTION_NAME}"
    )

    print("=" * 60)

    # ========================================================
    # 1. 모델 로드
    # ========================================================

    print()
    print(
        "bge-m3 모델 로드 중..."
    )

    start_time = time.time()

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print(
        "모델 로드 완료 "
        f"({time.time() - start_time:.1f}초)"
    )

    # ========================================================
    # 2. Qdrant
    # ========================================================

    client = connect_qdrant()

    # ========================================================
    # 3. Collection 준비
    # ========================================================

    prepare_collection(
        client
    )

    # ========================================================
    # 4. 기존 chunk 확인
    # ========================================================

    existing_ids = (
        get_existing_chunk_ids(
            client
        )
    )

    # ========================================================
    # 5. 법률별 처리
    # ========================================================

    total_saved = 0

    for index, json_path in enumerate(
        JSON_PATHS,
        start=1,
    ):

        print()
        print("=" * 60)

        print(
            f"법률 처리 "
            f"[{index}/{len(JSON_PATHS)}]"
        )

        print("=" * 60)

        chunks = load_chunks(
            json_path
        )

        saved = embed_and_save(
            client=client,
            model=model,
            chunks=chunks,
            existing_ids=existing_ids,
        )

        total_saved += saved

        info = client.get_collection(
            COLLECTION_NAME
        )

        print()
        print(
            f"법률 처리 완료"
        )

        print(
            f"  이번 실행 저장: "
            f"{saved}개"
        )

        print(
            f"  전체 Point: "
            f"{info.points_count}개"
        )

    # ========================================================
    # 6. 최종 확인
    # ========================================================

    info = client.get_collection(
        COLLECTION_NAME
    )

    print()
    print("=" * 60)
    print("bge-m3 Embedding 구축 완료")
    print("=" * 60)

    print(
        f"  Collection : "
        f"{COLLECTION_NAME}"
    )

    print(
        f"  Embedding  : "
        f"{EMBEDDING_MODEL}"
    )

    print(
        f"  Dimension  : "
        f"{VECTOR_SIZE}"
    )

    print(
        f"  이번 실행 저장: "
        f"{total_saved}개"
    )

    print(
        f"  전체 Point : "
        f"{info.points_count}개"
    )

    print("=" * 60)

    # ========================================================
    # 7. 법률별 확인
    # ========================================================

    print_law_counts(
        client
    )

    print()
    print(
        "작업이 완료되었습니다."
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()