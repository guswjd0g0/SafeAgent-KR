import json
import re
from pathlib import Path


# ============================================================
# SafeAgent
# Legal RAG Evaluation Dataset Generator v2
# ============================================================


# ============================================================
# Path
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR = BASE_DIR / "data" / "evaluation"

OUTPUT_PATH = OUTPUT_DIR / "legal_queries_v2.json"


# ============================================================
# 평가 데이터 설정
# ============================================================

# 법률당 평가할 조문 수
ARTICLES_PER_LAW = 10

# 조문당 생성할 질문 수
QUERIES_PER_ARTICLE = 3


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
        / "직업교육훈련(법률)(제21065호)(20251001)"
        / "chunks.json",
    ),
]


# ============================================================
# 중요도 키워드
# ============================================================

IMPORTANT_KEYWORDS = [

    "목적",
    "정의",
    "적용 범위",
    "적용범위",
    "국가",
    "지방자치단체",
    "기본",
    "원칙",
    "사업주",
    "사용자",
    "근로자",
    "보호",
    "책임",
    "의무",
    "보험",
    "보험료",
    "급여",
    "재해",
    "안전",
    "보건",
    "교육",
    "훈련",
    "권리",
]


# ============================================================
# JSON 로드
# ============================================================

def load_chunks(law_name, json_path):

    print()
    print("=" * 70)
    print(f"법률: {law_name}")
    print(f"파일: {json_path}")

    if not json_path.exists():

        raise FileNotFoundError(
            "\n파일을 찾을 수 없습니다.\n"
            f"{json_path}"
        )

    with open(
        json_path,
        "r",
        encoding="utf-8",
    ) as f:

        chunks = json.load(f)

    if not isinstance(chunks, list):

        raise ValueError(
            f"{json_path}\n"
            "최상위 JSON 구조가 list가 아닙니다."
        )

    print(
        f"전체 청크 수: {len(chunks)}"
    )

    return chunks


# ============================================================
# 조문 번호 정규화
# ============================================================

def article_number(article):

    """
    제1조
    제2조
    제4조의2
    제76조의2
    등의 순서를 비교하기 위한 숫자를 추출한다.
    """

    if not article:
        return 999999

    match = re.search(
        r"제\s*(\d+)\s*조",
        article,
    )

    if match:

        return int(
            match.group(1)
        )

    return 999999


# ============================================================
# 조문 제목 추출
# ============================================================

def extract_article_title(article, text):

    """
    예:

    제1조(목적) 이 법은 ...

    → 목적

    제2조(정의) 이 법에서 사용하는 ...

    → 정의
    """

    if not text:

        return ""

    # --------------------------------------------------------
    # 제1조(목적)
    # --------------------------------------------------------

    pattern = re.compile(
        r"제\s*\d+(?:조의\d+)?\s*"
        r"\(([^)]+)\)"
    )

    match = pattern.search(text)

    if match:

        return match.group(1).strip()

    return ""


# ============================================================
# 조문 추출
# ============================================================

def extract_articles():

    articles_by_law = {}

    for law_name, json_path in LAW_FILES:

        chunks = load_chunks(
            law_name,
            json_path,
        )

        articles = []

        seen_chunk_ids = set()

        for index, chunk in enumerate(chunks):

            if not isinstance(chunk, dict):

                continue

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

            # ------------------------------------------------
            # 필수 데이터 검사
            # ------------------------------------------------

            if not article:
                continue

            if not chunk_id:
                continue

            if not isinstance(text, str):
                continue

            text = text.strip()

            if not text:
                continue

            # ------------------------------------------------
            # 중복 chunk 제거
            # ------------------------------------------------

            if chunk_id in seen_chunk_ids:

                print(
                    f"중복 chunk ID 무시: "
                    f"{chunk_id}"
                )

                continue

            seen_chunk_ids.add(
                chunk_id
            )

            # ------------------------------------------------
            # 조문 제목
            # ------------------------------------------------

            title = extract_article_title(
                article,
                text,
            )

            articles.append(
                {
                    "law": law_name,

                    "article": article,

                    "article_number":
                        article_number(article),

                    "title": title,

                    "chunk_id": chunk_id,

                    "text": text,
                }
            )

        # ----------------------------------------------------
        # 조문 번호 순서 정렬
        # ----------------------------------------------------

        articles.sort(
            key=lambda x: (
                x["article_number"],
                x["article"],
            )
        )

        articles_by_law[
            law_name
        ] = articles

        print(
            f"추출된 조문 수: "
            f"{len(articles)}"
        )

    return articles_by_law


# ============================================================
# 조문 중요도 계산
# ============================================================

def calculate_article_score(article):

    """
    평가 대상으로 적합한 조문을 자동 선정하기 위한
    간단한 중요도 점수.

    점수가 높을수록 평가 대상으로 우선 선택한다.
    """

    text = article["text"]

    title = article["title"]

    article_num = article[
        "article_number"
    ]

    score = 0

    # ========================================================
    # 1. 초반 조문 가산점
    # ========================================================

    if article_num <= 5:

        score += 30

    elif article_num <= 10:

        score += 15

    elif article_num <= 20:

        score += 5

    # ========================================================
    # 2. 조문 제목 기반 중요도
    # ========================================================

    for keyword in IMPORTANT_KEYWORDS:

        if keyword in title:

            score += 15

    # ========================================================
    # 3. 본문 기반 중요도
    # ========================================================

    keyword_count = 0

    for keyword in IMPORTANT_KEYWORDS:

        if keyword in text:

            keyword_count += 1

    score += min(
        keyword_count * 2,
        20,
    )

    # ========================================================
    # 4. 너무 짧은 조문은 낮게 평가
    # ========================================================

    if len(text) < 30:

        score -= 10

    # ========================================================
    # 5. 너무 긴 조문은 약간 낮춤
    #
    # 너무 복잡한 조문은 자동 질문 생성 품질이
    # 떨어질 가능성이 있기 때문
    # ========================================================

    if len(text) > 2000:

        score -= 5

    return score


# ============================================================
# 평가 조문 선정
# ============================================================

def select_evaluation_articles(
    articles_by_law
):

    selected_by_law = {}

    print()
    print("=" * 70)
    print("평가 대상 조문 선정")
    print("=" * 70)

    for law_name, articles in articles_by_law.items():

        scored_articles = []

        for article in articles:

            score = calculate_article_score(
                article
            )

            scored_articles.append(
                (
                    score,
                    article,
                )
            )

        # ----------------------------------------------------
        # 중요도 순 정렬
        # ----------------------------------------------------

        scored_articles.sort(
            key=lambda x: (
                -x[0],
                x[1]["article_number"],
            )
        )

        selected = []

        # ----------------------------------------------------
        # 우선 중요도 순으로 선택
        # ----------------------------------------------------

        for score, article in scored_articles:

            if len(selected) >= ARTICLES_PER_LAW:

                break

            selected.append(
                article
            )

        # ----------------------------------------------------
        # 최종적으로 조문 번호 순 정렬
        # ----------------------------------------------------

        selected.sort(
            key=lambda x: (
                x["article_number"],
                x["article"],
            )
        )

        selected_by_law[
            law_name
        ] = selected

        # ----------------------------------------------------
        # 출력
        # ----------------------------------------------------

        print()
        print(
            f"[{law_name}]"
        )

        for article in selected:

            score = calculate_article_score(
                article
            )

            title = article[
                "title"
            ]

            if title:

                print(
                    f"  {article['article']} "
                    f"({title}) "
                    f"| score={score}"
                )

            else:

                print(
                    f"  {article['article']} "
                    f"| score={score}"
                )

        print(
            f"선정 조문: "
            f"{len(selected)}개"
        )

    return selected_by_law


# ============================================================
# 질문 생성 - Direct
# ============================================================

def generate_direct_query(
    law_name,
    article,
    title,
):

    if title:

        return (
            f"{law_name}에서 "
            f"{article}({title})의 "
            f"내용은 무엇인가요?"
        )

    return (
        f"{law_name}의 "
        f"{article}에서는 "
        f"무엇을 규정하고 있나요?"
    )


# ============================================================
# 질문 생성 - Paraphrase
# ============================================================

def generate_paraphrase_query(
    law_name,
    article,
    title,
):

    if title:

        return (
            f"{law_name}에서 "
            f"{title}에 관한 "
            f"규정은 어떻게 되어 있나요?"
        )

    return (
        f"{law_name}의 "
        f"{article}에서 정하고 있는 "
        f"주요 내용이 궁금합니다."
    )


# ============================================================
# 질문 생성 - Natural
# ============================================================

def generate_natural_query(
    law_name,
    article,
    title,
    text,
):

    """
    법률명을 일부러 제거해서
    실제 사용자의 자연어 검색 상황을 테스트한다.

    단, 자동 생성의 한계상 완전히 자연스러운
    의미 기반 질문을 만드는 것이 아니라
    조문 제목을 활용한다.
    """

    if title:

        return (
            f"{title}에 대해서 "
            f"법에서는 어떻게 정하고 있나요?"
        )

    return (
        f"{article}에서 정하고 있는 "
        f"내용이 무엇인지 알려주세요."
    )


# ============================================================
# 질문 품질 검사
# ============================================================

def is_valid_query(query):

    if not isinstance(
        query,
        str,
    ):

        return False

    query = query.strip()

    if not query:

        return False

    # --------------------------------------------------------
    # 너무 짧은 질문 방지
    # --------------------------------------------------------

    if len(query) < 10:

        return False

    return True


# ============================================================
# 조문별 평가 데이터 생성
# ============================================================

def create_evaluation_data(
    selected_by_law
):

    evaluation_data = []

    evaluation_id = 1

    for law_name, articles in selected_by_law.items():

        for article in articles:

            article_number_text = article[
                "article"
            ]

            title = article[
                "title"
            ]

            chunk_id = article[
                "chunk_id"
            ]

            text = article[
                "text"
            ]

            # =================================================
            # 질문 3종류
            # =================================================

            queries = [

                (
                    "direct",
                    generate_direct_query(
                        law_name,
                        article_number_text,
                        title,
                    ),
                ),

                (
                    "paraphrase",
                    generate_paraphrase_query(
                        law_name,
                        article_number_text,
                        title,
                    ),
                ),

                (
                    "natural",
                    generate_natural_query(
                        law_name,
                        article_number_text,
                        title,
                        text,
                    ),
                ),
            ]

            # =================================================
            # 난이도
            # =================================================

            difficulty_map = {

                "direct": "easy",

                "paraphrase": "medium",

                "natural": "hard",
            }

            for query_type, query in queries:

                if not is_valid_query(
                    query
                ):

                    continue

                evaluation_data.append(
                    {
                        "id":
                            evaluation_id,

                        "query":
                            query,

                        "expected_law":
                            law_name,

                        "expected_article":
                            article_number_text,

                        "expected_chunk_id":
                            chunk_id,

                        "query_type":
                            query_type,

                        "difficulty":
                            difficulty_map[
                                query_type
                            ],
                    }
                )

                evaluation_id += 1

    return evaluation_data


# ============================================================
# 중복 질문 검사
# ============================================================

def validate_evaluation_data(
    evaluation_data
):

    print()
    print("=" * 70)
    print("평가 데이터 검증")
    print("=" * 70)

    query_set = set()
    chunk_set = set()

    duplicate_queries = []

    for item in evaluation_data:

        query = item[
            "query"
        ]

        chunk_id = item[
            "expected_chunk_id"
        ]

        # ----------------------------------------------------
        # Query 중복
        # ----------------------------------------------------

        if query in query_set:

            duplicate_queries.append(
                query
            )

        query_set.add(
            query
        )

        # ----------------------------------------------------
        # Chunk ID 확인
        # ----------------------------------------------------

        if not chunk_id:

            raise ValueError(
                f"ID {item['id']}에 "
                "expected_chunk_id가 없습니다."
            )

        chunk_set.add(
            chunk_id
        )

    if duplicate_queries:

        raise ValueError(
            "중복 질문이 발견되었습니다:\n"
            + "\n".join(
                duplicate_queries
            )
        )

    print(
        "중복 질문: 없음"
    )

    print(
        f"평가 질문 수: "
        f"{len(evaluation_data)}"
    )

    print(
        f"평가 대상 조문 수: "
        f"{len(chunk_set)}"
    )


# ============================================================
# 평가 데이터 통계
# ============================================================

def print_statistics(
    evaluation_data
):

    print()
    print("=" * 70)
    print("평가 데이터 통계")
    print("=" * 70)

    law_counts = {}

    type_counts = {}

    difficulty_counts = {}

    for item in evaluation_data:

        law = item[
            "expected_law"
        ]

        query_type = item[
            "query_type"
        ]

        difficulty = item[
            "difficulty"
        ]

        law_counts[law] = (
            law_counts.get(
                law,
                0,
            ) + 1
        )

        type_counts[query_type] = (
            type_counts.get(
                query_type,
                0,
            ) + 1
        )

        difficulty_counts[difficulty] = (
            difficulty_counts.get(
                difficulty,
                0,
            ) + 1
        )

    print()
    print("[법률별 질문 수]")

    for law, count in law_counts.items():

        print(
            f"  {law}: {count}"
        )

    print()
    print("[질문 유형별]")

    for query_type, count in type_counts.items():

        print(
            f"  {query_type}: {count}"
        )

    print()
    print("[난이도별]")

    for difficulty, count in difficulty_counts.items():

        print(
            f"  {difficulty}: {count}"
        )


# ============================================================
# JSON 저장
# ============================================================

def save_evaluation_data(
    evaluation_data
):

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
            evaluation_data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 70)
    print("평가 데이터셋 생성 완료")
    print("=" * 70)

    print(
        f"파일: {OUTPUT_PATH}"
    )

    print(
        f"질문 수: "
        f"{len(evaluation_data)}"
    )


# ============================================================
# 미리보기
# ============================================================

def print_preview(
    evaluation_data,
    count=15,
):

    print()
    print("=" * 70)
    print(
        f"평가 데이터 미리보기 "
        f"(처음 {count}개)"
    )
    print("=" * 70)

    for item in evaluation_data[
        :count
    ]:

        print()
        print(
            f"[ID {item['id']}]"
        )

        print(
            f"질문       : "
            f"{item['query']}"
        )

        print(
            f"법률       : "
            f"{item['expected_law']}"
        )

        print(
            f"조문       : "
            f"{item['expected_article']}"
        )

        print(
            f"Chunk ID   : "
            f"{item['expected_chunk_id']}"
        )

        print(
            f"질문 유형  : "
            f"{item['query_type']}"
        )

        print(
            f"난이도     : "
            f"{item['difficulty']}"
        )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("Legal RAG Evaluation Dataset Generator v2")
    print("=" * 70)

    print()
    print(
        f"법률당 평가 조문 수: "
        f"{ARTICLES_PER_LAW}"
    )

    print(
        f"조문당 질문 수: "
        f"{QUERIES_PER_ARTICLE}"
    )

    print(
        f"예상 최대 질문 수: "
        f"{len(LAW_FILES) * ARTICLES_PER_LAW * QUERIES_PER_ARTICLE}"
    )

    # ========================================================
    # 1. 모든 법률 조문 추출
    # ========================================================

    articles_by_law = extract_articles()

    # ========================================================
    # 2. 평가 대상 조문 선정
    # ========================================================

    selected_by_law = select_evaluation_articles(
        articles_by_law
    )

    # ========================================================
    # 3. 평가 질문 생성
    # ========================================================

    evaluation_data = create_evaluation_data(
        selected_by_law
    )

    # ========================================================
    # 4. 데이터 검증
    # ========================================================

    validate_evaluation_data(
        evaluation_data
    )

    # ========================================================
    # 5. 통계
    # ========================================================

    print_statistics(
        evaluation_data
    )

    # ========================================================
    # 6. 미리보기
    # ========================================================

    print_preview(
        evaluation_data
    )

    # ========================================================
    # 7. JSON 저장
    # ========================================================

    save_evaluation_data(
        evaluation_data
    )

    # ========================================================
    # 완료
    # ========================================================

    print()
    print(
        "Legal RAG Evaluation Dataset "
        "v2 생성이 완료되었습니다."
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()