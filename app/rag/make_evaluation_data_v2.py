import json
from pathlib import Path


# ============================================================
# Path
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR = BASE_DIR / "data" / "evaluation"

OUTPUT_PATH = OUTPUT_DIR / "legal_queriesV2.json"


# ============================================================
# 법률 목록
# ============================================================

LAW_FILES = [
    (
        "근로기준법",
        PROCESSED_DIR
        / "근로기준법(법률)(제21533호)(20270101)"
        / "chunks.json",
    ),
    (
        "산업안전보건법",
        PROCESSED_DIR
        / "산업안전보건법(법률)(제21374호)(20260801)"
        / "chunks.json",
    ),
    (
        "산업재해보상보험법",
        PROCESSED_DIR
        / "산업재해보상보험법(법률)(제21375호)(20260701)"
        / "chunks.json",
    ),
    (
        "직업교육훈련 촉진법",
        PROCESSED_DIR
        / "직업교육훈련 촉진법(법률)(제21065호)(20251001)"
        / "chunks.json",
    ),
]


# ============================================================
# JSON 로드
# ============================================================

def load_chunks(law_name, json_path):

    print()
    print("=" * 60)
    print(f"법률: {law_name}")
    print(f"파일: {json_path}")

    if not json_path.exists():

        raise FileNotFoundError(
            f"\n파일을 찾을 수 없습니다.\n"
            f"{json_path}"
        )

    with open(
        json_path,
        "r",
        encoding="utf-8",
    ) as f:

        chunks = json.load(f)

    print(
        f"청크 수: {len(chunks)}"
    )

    return chunks


# ============================================================
# 조문 추출
# ============================================================

def extract_articles():

    articles = []

    for law_name, json_path in LAW_FILES:

        chunks = load_chunks(
            law_name,
            json_path,
        )

        for chunk in chunks:

            metadata = chunk.get(
                "metadata",
                {},
            )

            article = metadata.get(
                "article"
            )

            chunk_id = chunk.get(
                "id"
            )

            text = chunk.get(
                "text",
                "",
            )

            if not article:
                continue

            articles.append(
                {
                    "law": law_name,
                    "article": article,
                    "chunk_id": chunk_id,
                    "text": text,
                }
            )

    return articles


# ============================================================
# 조문 목록 출력
# ============================================================

def print_articles(articles):

    print()
    print("=" * 70)
    print("실제 조문 목록")
    print("=" * 70)

    current_law = None

    for item in articles:

        if item["law"] != current_law:

            current_law = item["law"]

            print()
            print(
                f"[{current_law}]"
            )

        print(
            f"  {item['article']} "
            f"| {item['chunk_id']}"
        )

    print()
    print(
        f"전체 조문 청크: "
        f"{len(articles)}개"
    )


# ============================================================
# 평가 데이터 템플릿 생성
# ============================================================

def create_template(articles):

    """
    실제 조문을 기반으로 평가 데이터의
    기본 구조를 생성한다.

    질문 내용은 사람이 작성한다.
    """

    evaluation_data = []

    # --------------------------------------------------------
    # 법률별 최대 5개
    # --------------------------------------------------------

    law_count = {}

    evaluation_id = 1

    for article in articles:

        law = article["law"]

        if law not in law_count:
            law_count[law] = 0

        if law_count[law] >= 5:
            continue

        evaluation_data.append(
            {
                "id": evaluation_id,

                "query": "",

                "expected_law": law,

                "expected_article": article[
                    "article"
                ],

                "expected_chunk_id": article[
                    "chunk_id"
                ],
            }
        )

        law_count[law] += 1

        evaluation_id += 1

    return evaluation_data


# ============================================================
# 저장
# ============================================================

def save_template(data):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 70)
    print("평가 데이터 템플릿 생성 완료")
    print("=" * 70)

    print(
        f"파일: {OUTPUT_PATH}"
    )

    print(
        f"질문 수: {len(data)}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("Legal RAG Evaluation Dataset Generator")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. 실제 조문 읽기
    # --------------------------------------------------------

    articles = extract_articles()

    # --------------------------------------------------------
    # 2. 실제 조문 확인
    # --------------------------------------------------------

    print_articles(
        articles
    )

    # --------------------------------------------------------
    # 3. 평가 템플릿 생성
    # --------------------------------------------------------

    evaluation_data = create_template(
        articles
    )

    # --------------------------------------------------------
    # 4. 저장
    # --------------------------------------------------------

    save_template(
        evaluation_data
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()