import json
import re
import hashlib
import unicodedata
from pathlib import Path
from collections import Counter


# ============================================================
# SafeAgent - Legal PDF RAG Ingestion
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
DOCUMENT_DIR = BASE_DIR / "data" / "documents"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

# 너무 긴 조문만 추가 분할
MAX_CHUNK_CHARS = 7000

# 너무 짧은 텍스트는 제외
MIN_CHUNK_CHARS = 20


# ============================================================
# 1. PDF 파일 탐색
# ============================================================

def find_pdf_files():
    """
    data/documents 아래의 모든 PDF를 찾는다.
    """
    if not DOCUMENT_DIR.exists():
        print("[ERROR] PDF 폴더가 없습니다.")
        print(f"        {DOCUMENT_DIR}")
        return []

    return sorted(DOCUMENT_DIR.glob("*.pdf"))


# ============================================================
# 2. 파일명 -> 문서명
# ============================================================

def get_document_name(pdf_path: Path):
    """
    PDF 파일명을 법률 document 이름으로 사용한다.
    """
    return pdf_path.stem.strip()


# ============================================================
# 3. 문자열 기본 정리
# ============================================================

def normalize_unicode(text: str):
    """
    유니코드 정규화
    """
    text = unicodedata.normalize("NFKC", text)

    # Zero Width 문자 제거
    text = text.replace("\u200b", "")
    text = text.replace("\u200c", "")
    text = text.replace("\u200d", "")
    text = text.replace("\ufeff", "")

    # 특수 공백
    text = text.replace("\u00a0", " ")
    text = text.replace("\u3000", " ")

    return text


def clean_line(line: str):
    """
    한 줄 단위 PDF 추출 텍스트 정리
    """
    line = normalize_unicode(line)

    # 반복 공백 제거
    line = re.sub(r"[ \t]+", " ", line)

    line = line.strip()

    return line


def clean_text(text: str):
    """
    최종 Chunk 텍스트 정리
    """
    text = normalize_unicode(text)

    # 반복 공백
    text = re.sub(r"[ \t]+", " ", text)

    # 줄바꿈 주변 공백
    text = re.sub(r"\s*\n\s*", "\n", text)

    # 지나친 빈 줄
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# 4. PDF Header / Footer 제거
# ============================================================

def remove_pdf_noise(line: str):
    """
    국가법령정보센터 PDF에서 반복적으로 등장하는
    헤더/푸터를 제거한다.
    """

    line = line.strip()

    if not line:
        return ""

    # 국가법령정보센터
    line = re.sub(
        r"국가법령정보센터",
        "",
        line
    )

    # 법제처
    line = re.sub(
        r"법제처\s*\d+",
        "",
        line
    )

    # 단독 법제처
    line = re.sub(
        r"법제처",
        "",
        line
    )

    return line.strip()


# ============================================================
# 5. PDF 페이지 추출
# ============================================================

def extract_pages(pdf_path: Path):
    """
    페이지별 텍스트 추출

    반환:
    [
        {
            "page": 1,
            "lines": [...]
        }
    ]
    """

    import pymupdf

    pages = []

    with pymupdf.open(pdf_path) as pdf:

        print(f"페이지 수: {len(pdf)}")

        for page_index, page in enumerate(pdf):

            page_number = page_index + 1

            raw_text = page.get_text(
                "text",
                sort=True
            )

            raw_lines = raw_text.splitlines()

            lines = []

            for raw_line in raw_lines:

                line = remove_pdf_noise(raw_line)
                line = clean_line(line)

                if not line:
                    continue

                lines.append(line)

            pages.append({
                "page": page_number,
                "lines": lines
            })

    return pages


# ============================================================
# 6. Chapter 판별
# ============================================================

CHAPTER_PATTERN = re.compile(
    r"^제\d+장(?:\s+.+)?$"
)


def is_chapter(line: str):
    return CHAPTER_PATTERN.match(line)


def normalize_chapter(line: str):
    return clean_text(line)


# ============================================================
# 7. 부칙 판별
# ============================================================

def is_supplementary_start(line: str):
    """
    부칙 시작 여부 판별.

    대응 예:

    부칙
    부칙 <법률 제12345호, 2026. 1. 1.>
    부칙 <제정 2026. 1. 1.>
    """

    line = clean_line(line)

    if not line:
        return False

    # 가장 기본적인 형태
    if re.match(r"^부칙$", line):
        return True

    # 부칙 + 시행일/법률번호 등의 정보
    if re.match(r"^부칙\s*(?:<.*>)?$", line):
        return True

    # PDF에서 부칙 뒤에 불필요한 문자가 붙는 경우
    if re.match(r"^부칙\s+", line):
        return True

    return False


def is_supplementary_related(line: str):
    """
    부칙 제목의 다음 줄에 나오는 법령 정보 등을 감지.

    예:
    <법률 제21533호, 2026. 8. 1.>
    """

    line = clean_line(line)

    return bool(
        re.match(
            r"^<.*(?:법률|대통령령|총리령|부령|시행령|시행규칙).*?>$",
            line
        )
    )


# ============================================================
# 8. Article 판별
# ============================================================

ARTICLE_PATTERN = re.compile(
    r"^(제\d+조)"
    r"(?:\s*\(([^)]*)\))?"
    r"(?:\s+|$)"
    r"(.*)$"
)


# PDF 추출 오류 대응
#
# 제7조제7조(다른 법률의 개정)
#
DUPLICATED_ARTICLE_PATTERN = re.compile(
    r"^(제\d+조)(제\d+조)"
    r"(?:\s*\(([^)]*)\))?"
    r"(?:\s+|$)"
    r"(.*)$"
)


# 부칙에서 흔히 등장
#
# 제2조부터 제6조까지 생략
#
RANGE_ARTICLE_PATTERN = re.compile(
    r"^(제\d+조부터\s+제\d+조까지)"
    r"(?:\s+(.+))?$"
)


def parse_article(line: str):
    """
    Article 시작 여부 판별
    """

    line = clean_line(line)

    # --------------------------------------------------------
    # 1. PDF 추출 중복
    # --------------------------------------------------------

    match = DUPLICATED_ARTICLE_PATTERN.match(line)

    if match:

        first_article = match.group(1)
        second_article = match.group(2)

        if first_article == second_article:

            title = match.group(3)
            content = match.group(4).strip()

            return {
                "article": first_article,
                "title": title.strip() if title else None,
                "content": content,
                "duplicated_prefix": True
            }

    # --------------------------------------------------------
    # 2. 일반 Article
    # --------------------------------------------------------

    match = ARTICLE_PATTERN.match(line)

    if match:

        article = match.group(1)
        title = match.group(2)
        content = match.group(3).strip()

        return {
            "article": article,
            "title": title.strip() if title else None,
            "content": content,
            "duplicated_prefix": False
        }

    # --------------------------------------------------------
    # 3. 부칙 조문 범위
    # --------------------------------------------------------

    match = RANGE_ARTICLE_PATTERN.match(line)

    if match:

        return {
            "article": match.group(1),
            "title": None,
            "content": match.group(2).strip()
            if match.group(2)
            else "",
            "duplicated_prefix": False
        }

    return None


# ============================================================
# 9. Article 원문 생성
# ============================================================

def build_article_text(article_data):

    article = article_data["article"]
    title = article_data["title"]
    lines = article_data["lines"]

    if title:
        header = f"{article}({title})"
    else:
        header = article

    body = " ".join(lines)

    if body:

        return clean_text(
            f"{header} {body}"
        )

    return clean_text(header)


# ============================================================
# 10. 페이지 범위 계산
# ============================================================

def get_page_range(article_data):

    pages = article_data.get("pages", [])

    if not pages:
        return None, None

    return min(pages), max(pages)


# ============================================================
# 11. 긴 Article 분할
# ============================================================

PARAGRAPH_PATTERN = re.compile(
    r"(?=(?:①|②|③|④|⑤|⑥|⑦|⑧|⑨|⑩|⑪|⑫|⑬|⑭|⑮|⑯|⑰|⑱|⑲|⑳))"
)


def split_long_article(text: str):

    if len(text) <= MAX_CHUNK_CHARS:
        return [text]

    parts = PARAGRAPH_PATTERN.split(text)

    # 항 구분이 없는 경우
    if len(parts) <= 1:

        chunks = []

        start = 0

        while start < len(text):

            end = start + MAX_CHUNK_CHARS

            if end < len(text):

                candidates = [
                    text.rfind("다.", start, end),
                    text.rfind("한다.", start, end),
                    text.rfind("한다", start, end),
                    text.rfind(".", start, end),
                ]

                split_at = max(candidates)

                if split_at > start + 1000:
                    end = split_at + 2

            chunks.append(
                text[start:end].strip()
            )

            start = end

        return chunks

    # 항 단위 재조립
    chunks = []

    current = ""

    for part in parts:

        part = part.strip()

        if not part:
            continue

        if not current:

            current = part
            continue

        if len(current) + len(part) <= MAX_CHUNK_CHARS:

            current += " " + part

        else:

            chunks.append(
                current.strip()
            )

            current = part

    if current:
        chunks.append(
            current.strip()
        )

    return chunks


# ============================================================
# 12. ID 생성
# ============================================================

def make_safe_id(text: str):

    text = re.sub(
        r"[^\w가-힣\-]+",
        "_",
        text
    )

    text = re.sub(
        r"_+",
        "_",
        text
    )

    return text.strip("_")


def make_chunk_id(
    document,
    section_type,
    article,
    chunk_index=1
):

    base = (
        f"{document}_"
        f"{section_type}_"
        f"{article}"
    )

    if chunk_index > 1:
        base += f"_part{chunk_index}"

    return make_safe_id(base)


# ============================================================
# 13. 중복 텍스트 fingerprint
# ============================================================

def text_fingerprint(text: str):

    normalized = re.sub(
        r"\s+",
        "",
        text
    )

    return hashlib.sha1(
        normalized.encode("utf-8")
    ).hexdigest()


# ============================================================
# 14. PDF -> Article 구조화
# ============================================================

def parse_document(
    pages,
    document
):

    articles = []

    current_section = "본문"
    current_chapter = None
    current_article = None

    supplementary_found = False

    supplementary_pages = []

    for page_data in pages:

        page_number = page_data["page"]

        for raw_line in page_data["lines"]:

            line = clean_line(raw_line)

            if not line:
                continue

            # =================================================
            # 부칙 시작
            # =================================================

            if is_supplementary_start(line):

                supplementary_found = True
                supplementary_pages.append(
                    page_number
                )

                # 기존 본문 Article 저장
                if current_article is not None:

                    articles.append(
                        current_article
                    )

                    current_article = None

                current_section = "부칙"
                current_chapter = None

                continue

            # =================================================
            # 부칙 제목 다음의 법령 정보
            # =================================================

            if (
                current_section == "부칙"
                and is_supplementary_related(line)
            ):

                supplementary_found = True
                supplementary_pages.append(
                    page_number
                )

                # 현재 Article이 없다면
                # 부칙 메타정보로만 처리
                continue

            # =================================================
            # Chapter
            # =================================================

            chapter_match = is_chapter(line)

            if (
                chapter_match
                and current_section == "본문"
            ):

                current_chapter = normalize_chapter(
                    line
                )

                continue

            # =================================================
            # Article
            # =================================================

            parsed = parse_article(line)

            if parsed:

                # 기존 Article 저장
                if current_article is not None:

                    articles.append(
                        current_article
                    )

                current_article = {

                    "document": document,

                    "section_type":
                        current_section,

                    "chapter":
                        current_chapter,

                    "article":
                        parsed["article"],

                    "title":
                        parsed["title"],

                    "lines": [],

                    "pages":
                        [page_number]
                }

                if parsed["content"]:

                    current_article["lines"].append(
                        parsed["content"]
                    )

                continue

            # =================================================
            # 현재 Article 내용
            # =================================================

            if current_article is not None:

                current_article["lines"].append(
                    line
                )

                current_article["pages"].append(
                    page_number
                )

    # =========================================================
    # 마지막 Article
    # =========================================================

    if current_article is not None:

        articles.append(
            current_article
        )

    return {
        "articles": articles,
        "supplementary_found": supplementary_found,
        "supplementary_pages": sorted(
            set(supplementary_pages)
        )
    }


# ============================================================
# 15. Article -> Chunk
# ============================================================

def articles_to_chunks(articles):

    chunks = []

    seen_fingerprints = set()

    for article_data in articles:

        text = build_article_text(
            article_data
        )

        if len(text) < MIN_CHUNK_CHARS:
            continue

        document = article_data["document"]
        section_type = article_data["section_type"]
        chapter = article_data["chapter"]
        article = article_data["article"]
        title = article_data["title"]

        page_start, page_end = get_page_range(
            article_data
        )

        # 긴 Article 분할
        article_parts = split_long_article(
            text
        )

        total_parts = len(article_parts)

        for index, part in enumerate(
            article_parts,
            start=1
        ):

            part = clean_text(part)

            if len(part) < MIN_CHUNK_CHARS:
                continue

            fingerprint = text_fingerprint(
                part
            )

            # 동일 PDF 내부 완전 중복 제거
            if fingerprint in seen_fingerprints:
                continue

            seen_fingerprints.add(
                fingerprint
            )

            chunk_id = make_chunk_id(
                document=document,
                section_type=section_type,
                article=article,
                chunk_index=index
            )

            chunk = {

                "id": chunk_id,

                "text": part,

                "metadata": {

                    "document":
                        document,

                    "section_type":
                        section_type,

                    "chapter":
                        chapter,

                    "article":
                        article,

                    "title":
                        title,

                    "page_start":
                        page_start,

                    "page_end":
                        page_end,

                    "chunk_index":
                        index,

                    "chunk_count":
                        total_parts
                }
            }

            chunks.append(chunk)

    return chunks


# ============================================================
# 16. 부칙 검증
# ============================================================

def validate_supplementary(
    parse_result,
    chunks
):

    supplementary_found = (
        parse_result["supplementary_found"]
    )

    supplementary_pages = (
        parse_result["supplementary_pages"]
    )

    supplementary_chunks = [
        chunk
        for chunk in chunks
        if chunk["metadata"]["section_type"]
        == "부칙"
    ]

    print()
    print("-" * 70)
    print("부칙 검사")
    print("-" * 70)

    print(
        f"PDF에서 부칙 제목 발견 : "
        f"{'YES' if supplementary_found else 'NO'}"
    )

    print(
        f"부칙 발견 페이지       : "
        f"{supplementary_pages if supplementary_pages else '없음'}"
    )

    print(
        f"생성된 부칙 Chunk       : "
        f"{len(supplementary_chunks)}"
    )

    if supplementary_chunks:

        print()
        print("부칙 조문:")

        for chunk in supplementary_chunks:

            metadata = chunk["metadata"]

            print(
                f"  - "
                f"{metadata['article']} "
                f"{metadata['title'] or ''} "
                f"(p.{metadata['page_start']})"
            )

        print()
        print("✓ 부칙이 정상적으로 구조화되었습니다.")

        return True

    if supplementary_found:

        print()
        print(
            "⚠ PDF에는 부칙이 발견되었지만 "
            "부칙 Article을 생성하지 못했습니다."
        )

        return False

    print()
    print("ℹ PDF 텍스트상 부칙이 발견되지 않았습니다.")

    return True


# ============================================================
# 17. Validation
# ============================================================

def validate_chunks(
    chunks,
    document,
    total_pages
):

    print()
    print("=" * 70)
    print(f"VALIDATION : {document}")
    print("=" * 70)

    # ========================================================
    # ID 중복
    # ========================================================

    ids = [
        chunk["id"]
        for chunk in chunks
    ]

    id_counter = Counter(ids)

    duplicate_ids = [
        key
        for key, count in id_counter.items()
        if count > 1
    ]

    # ========================================================
    # 동일 영역 + 동일 Article
    # ========================================================

    article_keys = [
        (
            chunk["metadata"]["section_type"],
            chunk["metadata"]["article"]
        )
        for chunk in chunks
    ]

    article_counter = Counter(
        article_keys
    )

    duplicate_articles = [
        key
        for key, count in article_counter.items()
        if count > 1
    ]

    # ========================================================
    # 페이지 오류
    # ========================================================

    page_errors = []

    for chunk in chunks:

        metadata = chunk["metadata"]

        start = metadata["page_start"]
        end = metadata["page_end"]

        if (
            start is None
            or end is None
            or start < 1
            or end < start
            or end > total_pages
        ):

            page_errors.append(
                chunk["id"]
            )

    # ========================================================
    # Chapter 미지정
    # ========================================================

    chapter_missing = [

        chunk["id"]

        for chunk in chunks

        if (
            chunk["metadata"]["section_type"]
            == "본문"
            and not chunk["metadata"]["chapter"]
        )
    ]

    # ========================================================
    # 빈 Chunk
    # ========================================================

    empty_chunks = [

        chunk["id"]

        for chunk in chunks

        if len(
            chunk["text"].strip()
        ) < MIN_CHUNK_CHARS
    ]

    # ========================================================
    # 과대 Chunk
    # ========================================================

    oversized_chunks = [

        chunk["id"]

        for chunk in chunks

        if len(
            chunk["text"]
        ) > MAX_CHUNK_CHARS
    ]

    # ========================================================
    # 결과
    # ========================================================

    print(
        f"Chunk 수              : "
        f"{len(chunks)}"
    )

    print(
        f"중복 ID               : "
        f"{len(duplicate_ids)}"
    )

    print(
        f"동일 영역 Article 중복 : "
        f"{len(duplicate_articles)}"
    )

    print(
        f"Chapter 미지정        : "
        f"{len(chapter_missing)}"
    )

    print(
        f"페이지 범위 오류       : "
        f"{len(page_errors)}"
    )

    print(
        f"빈 Chunk              : "
        f"{len(empty_chunks)}"
    )

    print(
        f"과대 Chunk            : "
        f"{len(oversized_chunks)}"
    )

    # ========================================================
    # 상세 출력
    # ========================================================

    if duplicate_ids:

        print()
        print("중복 ID:")

        for item in duplicate_ids:
            print(f"  - {item}")

    if duplicate_articles:

        print()
        print("동일 영역 Article 중복:")

        for section, article in duplicate_articles:

            count = article_counter[
                (section, article)
            ]

            print(
                f"  - [{section}] "
                f"{article} ({count}개)"
            )

    if chapter_missing:

        print()
        print("Chapter 미지정:")

        for item in chapter_missing[:20]:
            print(f"  - {item}")

    if page_errors:

        print()
        print("페이지 범위 오류:")

        for item in page_errors[:20]:
            print(f"  - {item}")

    # ========================================================
    # PASS / WARNING
    # ========================================================

    validation_failed = any([

        duplicate_ids,

        chapter_missing,

        page_errors,

        empty_chunks,

        oversized_chunks
    ])

    print()

    if validation_failed:
        print("⚠ 검증 실패")
    else:
        print("✓ 검증 통과")

    return not validation_failed


# ============================================================
# 18. Preview
# ============================================================

def print_preview(
    chunks,
    limit=5
):

    print()
    print("=" * 70)
    print("CHUNK PREVIEW")
    print("=" * 70)

    for chunk in chunks[:limit]:

        metadata = chunk["metadata"]

        print()

        print(
            f"ID      : "
            f"{chunk['id']}"
        )

        print(
            f"Section : "
            f"{metadata['section_type']}"
        )

        print(
            f"Chapter : "
            f"{metadata['chapter']}"
        )

        print(
            f"Article : "
            f"{metadata['article']}"
        )

        print(
            f"Title   : "
            f"{metadata['title']}"
        )

        print(
            f"Page    : "
            f"{metadata['page_start']} "
            f"~ "
            f"{metadata['page_end']}"
        )

        print(
            f"Part    : "
            f"{metadata['chunk_index']}/"
            f"{metadata['chunk_count']}"
        )

        print("-" * 70)

        preview = chunk["text"]

        if len(preview) > 1000:
            preview = preview[:1000] + "..."

        print(preview)


# ============================================================
# 19. JSON 저장
# ============================================================

def save_chunks(
    chunks,
    document
):

    output_dir = (
        OUTPUT_DIR / document
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir / "chunks.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            chunks,
            f,
            ensure_ascii=False,
            indent=2
        )

    return output_path


# ============================================================
# 20. 법률 하나 처리
# ============================================================

def process_pdf(
    pdf_path: Path
):

    document = get_document_name(
        pdf_path
    )

    print()
    print("=" * 70)
    print(
        f"DOCUMENT : {document}"
    )
    print("=" * 70)

    # ========================================================
    # PDF 추출
    # ========================================================

    print()
    print("PDF 읽는 중...")

    pages = extract_pages(
        pdf_path
    )

    # ========================================================
    # 구조 분석
    # ========================================================

    print()
    print("법률 구조 분석 중...")

    parse_result = parse_document(
        pages=pages,
        document=document
    )

    articles = parse_result["articles"]

    print(
        f"발견된 Article 수 : "
        f"{len(articles)}"
    )

    # ========================================================
    # 부칙 원본 탐지 결과
    # ========================================================

    print()

    if parse_result["supplementary_found"]:

        print(
            "✓ PDF에서 '부칙'을 발견했습니다."
        )

        print(
            "  발견 페이지 : "
            f"{parse_result['supplementary_pages']}"
        )

    else:

        print(
            "ℹ PDF 텍스트에서 '부칙'을 발견하지 못했습니다."
        )

    # ========================================================
    # Chunk 생성
    # ========================================================

    print()
    print("Chunk 생성 중...")

    chunks = articles_to_chunks(
        articles
    )

    print(
        f"생성된 Chunk 수 : "
        f"{len(chunks)}"
    )

    # ========================================================
    # 부칙 검증
    # ========================================================

    supplementary_ok = validate_supplementary(
        parse_result,
        chunks
    )

    # ========================================================
    # 일반 Validation
    # ========================================================

    validation_ok = validate_chunks(
        chunks=chunks,
        document=document,
        total_pages=len(pages)
    )

    # ========================================================
    # 최종 Validation
    # ========================================================

    final_validation = (
        validation_ok
        and supplementary_ok
    )

    # ========================================================
    # 저장
    # ========================================================

    output_path = save_chunks(
        chunks=chunks,
        document=document
    )

    print()
    print(
        f"저장 위치 : "
        f"{output_path}"
    )

    # ========================================================
    # Preview
    # ========================================================

    print_preview(
        chunks,
        limit=5
    )

    return {

        "document":
            document,

        "chunks":
            len(chunks),

        "supplementary_found":
            parse_result["supplementary_found"],

        "supplementary_pages":
            parse_result["supplementary_pages"],

        "supplementary_chunks":
            sum(
                1
                for chunk in chunks
                if chunk["metadata"]["section_type"]
                == "부칙"
            ),

        "validation":
            final_validation,

        "output":
            str(output_path)
    }


# ============================================================
# 21. 전체 PDF 처리
# ============================================================

def main():

    print("=" * 70)
    print(
        "SafeAgent - Legal RAG Document Ingestion"
    )
    print("=" * 70)

    print()

    print(
        f"PDF 폴더 : "
        f"{DOCUMENT_DIR}"
    )

    pdf_files = find_pdf_files()

    if not pdf_files:

        print()
        print(
            "처리할 PDF가 없습니다."
        )

        return

    print()

    print(
        f"발견된 PDF : "
        f"{len(pdf_files)}개"
    )

    results = []

    # ========================================================
    # 모든 PDF 처리
    # ========================================================

    for pdf_path in pdf_files:

        try:

            result = process_pdf(
                pdf_path
            )

            results.append(
                result
            )

        except Exception as e:

            print()
            print("=" * 70)

            print(
                f"ERROR : "
                f"{pdf_path.name}"
            )

            print("=" * 70)

            print(
                f"{type(e).__name__}: {e}"
            )

            results.append({

                "document":
                    pdf_path.stem,

                "chunks":
                    0,

                "supplementary_found":
                    False,

                "supplementary_pages":
                    [],

                "supplementary_chunks":
                    0,

                "validation":
                    False,

                "output":
                    None
            })

    # ========================================================
    # 전체 결과
    # ========================================================

    print()
    print("=" * 70)
    print("전체 처리 결과")
    print("=" * 70)

    for result in results:

        status = (
            "PASS"
            if result["validation"]
            else "WARNING"
        )

        print()

        print(
            f"[{status}] "
            f"{result['document']}"
        )

        print(
            f"       Chunk : "
            f"{result['chunks']}"
        )

        print(
            f"       부칙 발견 : "
            f"{result['supplementary_found']}"
        )

        print(
            f"       부칙 페이지 : "
            f"{result['supplementary_pages']}"
        )

        print(
            f"       부칙 Chunk : "
            f"{result['supplementary_chunks']}"
        )

        print(
            f"       Output: "
            f"{result['output']}"
        )

    print()
    print("=" * 70)
    print("완료")
    print("=" * 70)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()