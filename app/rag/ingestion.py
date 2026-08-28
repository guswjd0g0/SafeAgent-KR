import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf


# ============================================================
# SafeAgent - Legal PDF RAG Ingestion
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
DOCUMENT_DIR = BASE_DIR / "data" / "documents"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

# 일반 조문은 하나의 Article Chunk
# 아주 긴 조문만 항/호 단위로 분할
MAX_CHUNK_CHARS = 7000
MIN_CHUNK_CHARS = 20


# ============================================================
# 1. PDF 탐색
# ============================================================

def find_pdf_files():
    if not DOCUMENT_DIR.exists():
        print("[ERROR] PDF 폴더가 없습니다.")
        print(DOCUMENT_DIR)
        return []

    return sorted(DOCUMENT_DIR.glob("*.pdf"))


def get_document_name(pdf_path: Path) -> str:
    return pdf_path.stem.strip()


# ============================================================
# 2. 법령 Metadata
# ============================================================

def parse_law_metadata(document: str):
    """
    파일명에서 법령 기본정보를 추출한다.

    예:
        근로기준법_법률_제21533호_20270101_본문

    결과:
        law_name       = 근로기준법
        law_type       = 법률
        law_number     = 제21533호
        effective_date = 2027-01-01
    """

    law_name = document
    law_type = None
    law_number = None
    effective_date = None

    # 법령명
    name_match = re.match(r"^([^_]+)", document)
    if name_match:
        law_name = name_match.group(1).strip()

    # 법령 종류
    type_match = re.search(
        r"(?:^|_)(법률|대통령령|총리령|부령|헌법|법규명령)(?:_|$)",
        document,
    )

    if type_match:
        law_type = type_match.group(1)

    # 법령 번호
    number_match = re.search(
        r"(제\d+호)",
        document,
    )

    if number_match:
        law_number = number_match.group(1)

    # 시행일
    date_match = re.search(
        r"(20\d{6})",
        document,
    )

    if date_match:
        raw_date = date_match.group(1)

        effective_date = (
            f"{raw_date[:4]}-"
            f"{raw_date[4:6]}-"
            f"{raw_date[6:8]}"
        )

    return {
        "law_name": law_name,
        "law_type": law_type,
        "law_number": law_number,
        "effective_date": effective_date,
    }


# ============================================================
# 3. 텍스트 정규화
# ============================================================

def normalize_unicode(text: str) -> str:
    """
    한글 깨짐 방지를 위한 Unicode 정규화.
    """

    text = unicodedata.normalize("NFKC", text)

    for char in (
        "\u200b",   # zero width space
        "\u200c",
        "\u200d",
        "\ufeff",   # BOM
    ):
        text = text.replace(char, "")

    text = text.replace("\u00a0", " ")
    text = text.replace("\u3000", " ")

    return text


def clean_line(text: str) -> str:
    text = normalize_unicode(text)

    # 탭 및 연속 공백 정리
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def clean_text(text: str) -> str:
    text = normalize_unicode(text)

    # 연속 공백
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    # 줄 앞뒤 공백 제거
    text = re.sub(
        r"[ \t]*\n[ \t]*",
        "\n",
        text,
    )

    # 3줄 이상의 빈 줄 → 2줄
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# ============================================================
# 4. Header / Footer / 연락처 제거
# ============================================================

PHONE_PATTERN = re.compile(
    r"\b\d{2,4}-\d{3,4}-\d{4}\b"
)


def is_noise_line(line: str) -> bool:
    line = clean_line(line)

    if not line:
        return True

    # 국가법령정보센터
    if "국가법령정보센터" in line:
        return True

    # 법제처 + 페이지 번호
    if re.fullmatch(
        r"법제처\s*\d*",
        line,
    ):
        return True

    # 전화번호가 포함된 담당부서 라인
    if PHONE_PATTERN.search(line):
        return True

    return False


# ============================================================
# 5. PDF 페이지 추출
# ============================================================

def extract_pages(pdf_path: Path):
    """
    PDF를 페이지 단위로 추출한다.

    반환:
        [
            {
                "page": 1,
                "lines": [...]
            }
        ]
    """

    pages = []

    with pymupdf.open(pdf_path) as pdf:
        print(f"페이지 수: {len(pdf)}")

        for page_index, page in enumerate(pdf):

            page_number = page_index + 1

            raw_text = page.get_text(
                "text",
                sort=True,
            )

            lines = []

            for raw_line in raw_text.splitlines():

                line = clean_line(raw_line)

                if not line:
                    continue

                if is_noise_line(line):
                    continue

                lines.append(line)

            pages.append({
                "page": page_number,
                "lines": lines,
            })

    return pages


# ============================================================
# 6. 법령 구조 판별
# ============================================================

CHAPTER_PATTERN = re.compile(
    r"^(제\d+장)(?:\s+(.+))?$"
)

SECTION_PATTERN = re.compile(
    r"^(제\d+절)(?:\s+(.+))?$"
)

ARTICLE_PATTERN = re.compile(
    r"^(제\d+조(?:의\d+)?)"
    r"(?:\(([^)]*)\))?"
    r"(?:\s+(.*))?$"
)

PARAGRAPH_PATTERN = re.compile(
    r"(①|②|③|④|⑤|⑥|⑦|⑧|⑨|⑩|"
    r"⑪|⑫|⑬|⑭|⑮|⑯|⑰|⑱|⑲|⑳)"
)

ITEM_PATTERN = re.compile(
    r"(?m)(?<!\S)(\d+)\.\s+"
)


def parse_chapter(line: str):

    match = CHAPTER_PATTERN.match(
        clean_line(line)
    )

    if not match:
        return None

    return {
        "number": match.group(1),
        "title": (
            match.group(2).strip()
            if match.group(2)
            else None
        ),
    }


def parse_section_title(line: str):

    match = SECTION_PATTERN.match(
        clean_line(line)
    )

    if not match:
        return None

    return {
        "number": match.group(1),
        "title": (
            match.group(2).strip()
            if match.group(2)
            else None
        ),
    }


# ============================================================
# 7. 부칙 판별
# ============================================================

def is_supplementary_start(line: str):
    """
    다음 형태를 모두 허용한다.

        부칙
        부칙 <제12345호, 2026. 1. 1.>
    """

    return bool(
        re.match(
            r"^부칙(?:\s+.*)?$",
            clean_line(line),
        )
    )


# ============================================================
# 8. Article 판별
# ============================================================

def parse_article(line: str):

    line = clean_line(line)

    match = ARTICLE_PATTERN.match(line)

    if not match:
        return None

    article = match.group(1)
    title = match.group(2)
    content = match.group(3)

    # 제목이 없는 경우
    if title is None and content:

        content = content.strip()

        # 법령에서 자주 등장하는 특수 상태
        if content in {
            "삭제",
            "신설",
            "생략",
        }:
            return {
                "article": article,
                "title": None,
                "content": content,
            }

        # 제목 없이 일반 문장이 붙은 경우
        # Article로 오인식하지 않는다.
        return None

    return {
        "article": article,
        "title": (
            title.strip()
            if title
            else None
        ),
        "content": (
            content.strip()
            if content
            else ""
        ),
    }


# ============================================================
# 9. Article 생성
# ============================================================

def create_article(
    document,
    section,
    chapter,
    subsection,
    parsed,
    page_number,
):

    return {
        "document": document,
        "section": section,
        "chapter": chapter,
        "subsection": subsection,
        "article": parsed["article"],
        "title": parsed["title"],
        "lines": [],
        "pages": [page_number],
    }


# ============================================================
# 10. Article 텍스트 생성
# ============================================================

def build_article_text(article_data):

    article = article_data["article"]
    title = article_data["title"]

    if title:
        header = f"{article}({title})"
    else:
        header = article

    body = " ".join(
        line.strip()
        for line in article_data["lines"]
        if line.strip()
    )

    if body:
        return clean_text(
            f"{header} {body}"
        )

    return clean_text(header)


# ============================================================
# 11. Article 파싱
# ============================================================

def parse_document(
    pages,
    document,
):
    """
    법령 PDF의 구조를 분석한다.

    구조:

        본문
          └─ 장
              └─ 절
                  └─ 조

        부칙
          └─ 조
    """

    articles = []

    section = "본문"
    chapter = None
    subsection = None
    current_article = None

    supplementary_found = False
    supplementary_pages = []

    def flush_current_article():

        nonlocal current_article

        if current_article is None:
            return

        if current_article["lines"]:

            current_article["pages"] = sorted(
                set(current_article["pages"])
            )

            articles.append(
                current_article
            )

        current_article = None

    for page_data in pages:

        page_number = page_data["page"]

        for line in page_data["lines"]:

            line = clean_line(line)

            if not line:
                continue

            # ------------------------------------------------
            # 부칙
            # ------------------------------------------------

            if is_supplementary_start(line):

                flush_current_article()

                section = "부칙"
                chapter = None
                subsection = None

                supplementary_found = True

                supplementary_pages.append(
                    page_number
                )

                continue

            # ------------------------------------------------
            # Chapter
            # ------------------------------------------------

            chapter_data = parse_chapter(line)

            if (
                chapter_data
                and section == "본문"
            ):

                flush_current_article()

                chapter = chapter_data
                subsection = None

                continue

            # ------------------------------------------------
            # Section / 절
            # ------------------------------------------------

            subsection_data = parse_section_title(line)

            if (
                subsection_data
                and section == "본문"
            ):

                flush_current_article()

                subsection = subsection_data

                continue

            # ------------------------------------------------
            # Article
            # ------------------------------------------------

            parsed = parse_article(line)

            if parsed:

                flush_current_article()

                current_article = create_article(
                    document=document,
                    section=section,
                    chapter=chapter,
                    subsection=subsection,
                    parsed=parsed,
                    page_number=page_number,
                )

                if parsed["content"]:
                    current_article["lines"].append(
                        parsed["content"]
                    )

                continue

            # ------------------------------------------------
            # Article 본문
            # ------------------------------------------------

            if current_article is not None:

                current_article["lines"].append(
                    line
                )

                current_article["pages"].append(
                    page_number
                )

    flush_current_article()

    return {
        "articles": articles,
        "supplementary_found": supplementary_found,
        "supplementary_pages": sorted(
            set(supplementary_pages)
        ),
    }


# ============================================================
# 12. Article 중복 병합
# ============================================================

def merge_articles(articles):
    """
    동일한 법령의 동일 Article을 병합한다.

    병합 기준:
        section
        article

    Chapter / 절 / 제목은 누락된 경우 보완한다.
    """

    merged = {}
    order = []

    for article in articles:

        key = (
            article["section"],
            article["article"],
        )

        if key not in merged:

            merged[key] = {
                "document": article["document"],
                "section": article["section"],
                "chapter": article["chapter"],
                "subsection": article["subsection"],
                "article": article["article"],
                "title": article["title"],
                "lines": [],
                "pages": [],
            }

            order.append(key)

        target = merged[key]

        # Chapter 보완
        if (
            target["chapter"] is None
            and article["chapter"] is not None
        ):
            target["chapter"] = article["chapter"]

        # 절 보완
        if (
            target["subsection"] is None
            and article["subsection"] is not None
        ):
            target["subsection"] = article["subsection"]

        # Title 보완
        if (
            target["title"] is None
            and article["title"] is not None
        ):
            target["title"] = article["title"]

        # 내용 병합
        for line in article["lines"]:
            if line not in target["lines"]:
                target["lines"].append(line)

        # 페이지 병합
        target["pages"].extend(
            article["pages"]
        )

    result = []

    for key in order:

        item = merged[key]

        item["pages"] = sorted(
            set(item["pages"])
        )

        result.append(item)

    return result


# ============================================================
# 13. Article 내부 구조 분석
# ============================================================

def parse_structural_units(text: str):
    """
    Article 내부의 항 / 호 구조를 분석한다.

    예:

        제36조(금품 청산) ...
        ① ...
        1. ...
        2. ...
        ② ...

    결과:

        paragraph = 제1항
        item      = 제1호
    """

    text = clean_text(text)

    paragraph_matches = list(
        PARAGRAPH_PATTERN.finditer(text)
    )

    if not paragraph_matches:
        return []

    units = []

    # --------------------------------------------------------
    # 항 이전의 Article 본문
    # --------------------------------------------------------

    prefix = text[
        :paragraph_matches[0].start()
    ].strip()

    if prefix:

        units.append({
            "paragraph": None,
            "item": None,
            "text": prefix,
        })

    # --------------------------------------------------------
    # 항 처리
    # --------------------------------------------------------

    for index, match in enumerate(
        paragraph_matches
    ):

        start = match.start()

        if index + 1 < len(
            paragraph_matches
        ):
            end = paragraph_matches[
                index + 1
            ].start()
        else:
            end = len(text)

        paragraph_text = text[
            start:end
        ].strip()

        symbol = match.group(1)

        paragraph_number = (
            ord(symbol)
            - ord("①")
            + 1
        )

        paragraph_id = (
            f"제{paragraph_number}항"
        )

        # ----------------------------------------------------
        # 호 분석
        # ----------------------------------------------------

        item_matches = list(
            ITEM_PATTERN.finditer(
                paragraph_text
            )
        )

        if item_matches:

            # 항 시작부터 첫 번째 호까지
            paragraph_prefix = paragraph_text[
                :item_matches[0].start()
            ].strip()

            if paragraph_prefix:

                units.append({
                    "paragraph": paragraph_id,
                    "item": None,
                    "text": paragraph_prefix,
                })

            for item_index, item_match in enumerate(
                item_matches
            ):

                item_start = item_match.start()

                if item_index + 1 < len(
                    item_matches
                ):
                    item_end = item_matches[
                        item_index + 1
                    ].start()
                else:
                    item_end = len(
                        paragraph_text
                    )

                item_text = paragraph_text[
                    item_start:item_end
                ].strip()

                item_number = item_match.group(1)

                units.append({
                    "paragraph": paragraph_id,
                    "item": f"제{item_number}호",
                    "text": item_text,
                })

        else:

            units.append({
                "paragraph": paragraph_id,
                "item": None,
                "text": paragraph_text,
            })

    return units


# ============================================================
# 14. 문장 단위 분할
# ============================================================

def split_by_sentence(text: str):
    """
    MAX_CHUNK_CHARS를 넘는 텍스트를
    가능한 한 문장 경계에서 분할한다.
    """

    text = clean_text(text)

    chunks = []

    start = 0

    while start < len(text):

        remaining = len(text) - start

        if remaining <= MAX_CHUNK_CHARS:

            chunks.append(
                text[start:].strip()
            )

            break

        end = min(
            start + MAX_CHUNK_CHARS,
            len(text),
        )

        candidates = [
            text.rfind(
                "다.",
                start,
                end,
            ),
            text.rfind(
                "한다.",
                start,
                end,
            ),
            text.rfind(
                "한다",
                start,
                end,
            ),
            text.rfind(
                ".",
                start,
                end,
            ),
        ]

        split_at = max(candidates)

        # 너무 앞에서 잘리는 경우
        if split_at <= start + 1000:

            split_at = end

        else:

            split_at += 1

        chunk = text[
            start:split_at
        ].strip()

        if chunk:
            chunks.append(chunk)

        start = split_at

    return [
        clean_text(chunk)
        for chunk in chunks
        if len(chunk.strip())
        >= MIN_CHUNK_CHARS
    ]


# ============================================================
# 15. 긴 Article 분할
# ============================================================

def split_long_article(text: str):
    """
    일반 Article:
        Article Chunk 1개

    긴 Article:
        Article
          ├─ Paragraph
          └─ Item

    구조로 분할한다.
    """

    text = clean_text(text)

    # --------------------------------------------------------
    # 일반 Article
    # --------------------------------------------------------

    if len(text) <= MAX_CHUNK_CHARS:

        return [{
            "text": text,
            "paragraph": None,
            "item": None,
            "chunk_type": "article",
        }]

    # --------------------------------------------------------
    # 항 / 호 분석
    # --------------------------------------------------------

    units = parse_structural_units(text)

    if units:

        chunks = []

        for unit in units:

            unit_text = clean_text(
                unit["text"]
            )

            if not unit_text:
                continue

            paragraph = unit["paragraph"]
            item = unit["item"]

            if item:
                chunk_type = "item"

            elif paragraph:
                chunk_type = "paragraph"

            else:
                chunk_type = "article"

            # ------------------------------------------------
            # 단위 자체가 충분히 짧은 경우
            # ------------------------------------------------

            if len(unit_text) <= MAX_CHUNK_CHARS:

                chunks.append({
                    "text": unit_text,
                    "paragraph": paragraph,
                    "item": item,
                    "chunk_type": chunk_type,
                })

                continue

            # ------------------------------------------------
            # 단위가 너무 긴 경우
            # ------------------------------------------------

            sentence_parts = split_by_sentence(
                unit_text
            )

            for part_index, sentence_part in enumerate(
                sentence_parts,
                start=1,
            ):

                chunks.append({
                    "text": sentence_part,
                    "paragraph": paragraph,
                    "item": item,
                    "chunk_type": chunk_type,
                    "part_index": part_index,
                })

        return chunks

    # --------------------------------------------------------
    # 항 / 호가 없는 긴 Article
    # --------------------------------------------------------

    sentence_parts = split_by_sentence(text)

    return [
        {
            "text": part,
            "paragraph": None,
            "item": None,
            "chunk_type": "article",
            "part_index": index,
        }
        for index, part in enumerate(
            sentence_parts,
            start=1,
        )
    ]


# ============================================================
# 16. ID
# ============================================================

def make_safe_id(text: str):

    text = re.sub(
        r"[^\w가-힣\-]+",
        "_",
        text,
    )

    text = re.sub(
        r"_+",
        "_",
        text,
    )

    return text.strip("_")


def make_article_id(
    law_name,
    section,
    article,
):

    return make_safe_id(
        f"{law_name}_{section}_{article}"
    )


def make_chunk_id(
    law_name,
    section,
    article,
    paragraph=None,
    item=None,
    part_index=None,
):

    parts = [
        law_name,
        section,
        article,
    ]

    if paragraph:
        parts.append(paragraph)

    if item:
        parts.append(item)

    if part_index is not None:
        parts.append(
            f"part{part_index}"
        )

    return make_safe_id(
        "_".join(parts)
    )


# ============================================================
# 17. Fingerprint
# ============================================================

def text_fingerprint(text: str):

    normalized = re.sub(
        r"\s+",
        "",
        text,
    )

    return hashlib.sha1(
        normalized.encode("utf-8")
    ).hexdigest()


# ============================================================
# 18. Article -> Chunk
# ============================================================

def articles_to_chunks(
    articles,
    document,
):
    """
    Article을 최종 RAG Chunk로 변환한다.

    핵심 구조:

        Article Chunk
             │
             ├── Paragraph Chunk
             │       ├── Item Chunk
             │       └── Item Chunk
             │
             └── Paragraph Chunk
    """

    chunks = []
    global_seen = set()

    law_metadata = parse_law_metadata(
        document
    )

    law_name = law_metadata["law_name"]

    for article_data in articles:

        article_text = build_article_text(
            article_data
        )

        if len(article_text) < MIN_CHUNK_CHARS:
            continue

        section = article_data["section"]
        chapter = article_data["chapter"]
        subsection = article_data["subsection"]
        article = article_data["article"]
        article_title = article_data["title"]

        pages = article_data["pages"]

        page_start = min(pages)
        page_end = max(pages)

        # ----------------------------------------------------
        # Article Parent ID
        # ----------------------------------------------------

        parent_article_id = make_article_id(
            law_name=law_name,
            section=section,
            article=article,
        )

        # ----------------------------------------------------
        # Article 분할 여부 확인
        # ----------------------------------------------------

        parts = split_long_article(
            article_text
        )

        has_structural_children = any(
            part.get("chunk_type")
            in {"paragraph", "item"}
            for part in parts
        )

        # ====================================================
        # 18-1. Article Parent Chunk
        # ====================================================

        if has_structural_children:

            article_fingerprint = text_fingerprint(
                article_text
            )

            if article_fingerprint not in global_seen:

                global_seen.add(
                    article_fingerprint
                )

                article_metadata = {

                    "law_name":
                        law_metadata["law_name"],

                    "law_type":
                        law_metadata["law_type"],

                    "law_number":
                        law_metadata["law_number"],

                    "effective_date":
                        law_metadata["effective_date"],

                    "section":
                        section,

                    "chapter":
                        (
                            chapter["number"]
                            if chapter
                            else None
                        ),

                    "chapter_title":
                        (
                            chapter["title"]
                            if chapter
                            else None
                        ),

                    "subsection":
                        (
                            subsection["number"]
                            if subsection
                            else None
                        ),

                    "subsection_title":
                        (
                            subsection["title"]
                            if subsection
                            else None
                        ),

                    "article":
                        article,

                    "article_title":
                        article_title,

                    "paragraph":
                        None,

                    "item":
                        None,

                    "chunk_type":
                        "article",

                    "parent_id":
                        None,

                    "page_start":
                        page_start,

                    "page_end":
                        page_end,

                    "chunk_index":
                        1,

                    "chunk_count":
                        1,

                    "source": {
                        "file":
                            f"{document}.pdf",

                        "page_start":
                            page_start,

                        "page_end":
                            page_end,

                        "section":
                            section,

                        "article":
                            article,
                    },
                }

                chunks.append({
                    "id":
                        parent_article_id,

                    "text":
                        article_text,

                    "metadata":
                        article_metadata,
                })

        # ====================================================
        # 18-2. Article 자체 Chunk
        # ====================================================

        for index, part_data in enumerate(
            parts,
            start=1,
        ):

            part_text = clean_text(
                part_data["text"]
            )

            if len(part_text) < MIN_CHUNK_CHARS:
                continue

            chunk_type = part_data.get(
                "chunk_type",
                "article",
            )

            # ------------------------------------------------
            # Article Chunk
            # ------------------------------------------------

            if chunk_type == "article":

                # 구조 자식이 없는 경우
                # Article 자체가 최종 Chunk

                if not has_structural_children:

                    fingerprint = text_fingerprint(
                        part_text
                    )

                    if fingerprint in global_seen:
                        continue

                    global_seen.add(
                        fingerprint
                    )

                    chunk_id = make_chunk_id(
                        law_name=law_name,
                        section=section,
                        article=article,
                    )

                    metadata = {

                        "law_name":
                            law_metadata["law_name"],

                        "law_type":
                            law_metadata["law_type"],

                        "law_number":
                            law_metadata["law_number"],

                        "effective_date":
                            law_metadata["effective_date"],

                        "section":
                            section,

                        "chapter":
                            (
                                chapter["number"]
                                if chapter
                                else None
                            ),

                        "chapter_title":
                            (
                                chapter["title"]
                                if chapter
                                else None
                            ),

                        "subsection":
                            (
                                subsection["number"]
                                if subsection
                                else None
                            ),

                        "subsection_title":
                            (
                                subsection["title"]
                                if subsection
                                else None
                            ),

                        "article":
                            article,

                        "article_title":
                            article_title,

                        "paragraph":
                            None,

                        "item":
                            None,

                        "chunk_type":
                            "article",

                        "parent_id":
                            None,

                        "page_start":
                            page_start,

                        "page_end":
                            page_end,

                        "chunk_index":
                            index,

                        "chunk_count":
                            len(parts),

                        "source": {
                            "file":
                                f"{document}.pdf",

                            "page_start":
                                page_start,

                            "page_end":
                                page_end,

                            "section":
                                section,

                            "article":
                                article,
                        },
                    }

                    chunks.append({
                        "id":
                            chunk_id,

                        "text":
                            part_text,

                        "metadata":
                            metadata,
                    })

                continue

            # ------------------------------------------------
            # Paragraph / Item Chunk
            # ------------------------------------------------

            fingerprint = text_fingerprint(
                part_text
            )

            if fingerprint in global_seen:
                continue

            global_seen.add(
                fingerprint
            )

            paragraph = part_data.get(
                "paragraph"
            )

            item = part_data.get(
                "item"
            )

            part_index = part_data.get(
                "part_index"
            )

            chunk_id = make_chunk_id(
                law_name=law_name,
                section=section,
                article=article,
                paragraph=paragraph,
                item=item,
                part_index=part_index,
            )

            metadata = {

                "law_name":
                    law_metadata["law_name"],

                "law_type":
                    law_metadata["law_type"],

                "law_number":
                    law_metadata["law_number"],

                "effective_date":
                    law_metadata["effective_date"],

                "section":
                    section,

                "chapter":
                    (
                        chapter["number"]
                        if chapter
                        else None
                    ),

                "chapter_title":
                    (
                        chapter["title"]
                        if chapter
                        else None
                    ),

                "subsection":
                    (
                        subsection["number"]
                        if subsection
                        else None
                    ),

                "subsection_title":
                    (
                        subsection["title"]
                        if subsection
                        else None
                    ),

                "article":
                    article,

                "article_title":
                    article_title,

                "paragraph":
                    paragraph,

                "item":
                    item,

                "chunk_type":
                    chunk_type,

                # 반드시 실제 Article Chunk를 가리킴
                "parent_id":
                    parent_article_id,

                "page_start":
                    page_start,

                "page_end":
                    page_end,

                "chunk_index":
                    (
                        part_index
                        if part_index is not None
                        else index
                    ),

                "chunk_count":
                    len(parts),

                "source": {
                    "file":
                        f"{document}.pdf",

                    "page_start":
                        page_start,

                    "page_end":
                        page_end,

                    "section":
                        section,

                    "article":
                        article,
                },
            }

            chunks.append({
                "id":
                    chunk_id,

                "text":
                    part_text,

                "metadata":
                    metadata,
            })

    return chunks


# ============================================================
# 19. Validation
# ============================================================

def validate_chunks(
    chunks,
    document,
    total_pages,
):

    print()
    print("=" * 70)
    print(f"VALIDATION : {document}")
    print("=" * 70)

    # --------------------------------------------------------
    # ID 중복
    # --------------------------------------------------------

    id_counter = Counter(
        chunk["id"]
        for chunk in chunks
    )

    duplicate_ids = [
        key
        for key, count in id_counter.items()
        if count > 1
    ]

    # --------------------------------------------------------
    # Chapter 누락
    # --------------------------------------------------------

    missing_chapter = [
        chunk["id"]
        for chunk in chunks
        if (
            chunk["metadata"]["section"] == "본문"
            and not chunk["metadata"]["chapter"]
        )
    ]

    # --------------------------------------------------------
    # 필수 Metadata
    # --------------------------------------------------------

    required_metadata = [
        "law_name",
        "law_type",
        "law_number",
        "effective_date",
        "section",
        "chapter",
        "chapter_title",
        "subsection",
        "subsection_title",
        "article",
        "article_title",
        "paragraph",
        "item",
        "chunk_type",
        "parent_id",
        "page_start",
        "page_end",
        "chunk_index",
        "chunk_count",
        "source",
    ]

    missing_metadata = []

    for chunk in chunks:

        for field in required_metadata:

            if field not in chunk["metadata"]:

                missing_metadata.append(
                    f"{chunk['id']} -> {field}"
                )

    # --------------------------------------------------------
    # 페이지
    # --------------------------------------------------------

    page_errors = []

    for chunk in chunks:

        start = chunk["metadata"]["page_start"]
        end = chunk["metadata"]["page_end"]

        if (
            start < 1
            or end < start
            or end > total_pages
        ):

            page_errors.append(
                chunk["id"]
            )

    # --------------------------------------------------------
    # 빈 Chunk
    # --------------------------------------------------------

    empty_chunks = [
        chunk["id"]
        for chunk in chunks
        if len(
            chunk["text"].strip()
        ) < MIN_CHUNK_CHARS
    ]

    # --------------------------------------------------------
    # Section
    # --------------------------------------------------------

    invalid_section = [
        chunk["id"]
        for chunk in chunks
        if chunk["metadata"]["section"]
        not in {"본문", "부칙"}
    ]

    # --------------------------------------------------------
    # Chunk Type
    # --------------------------------------------------------

    valid_chunk_types = {
        "article",
        "paragraph",
        "item",
    }

    invalid_chunk_type = [
        chunk["id"]
        for chunk in chunks
        if chunk["metadata"]["chunk_type"]
        not in valid_chunk_types
    ]

    # --------------------------------------------------------
    # Parent ID
    # --------------------------------------------------------

    chunk_ids = {
        chunk["id"]
        for chunk in chunks
    }

    invalid_parent = []

    for chunk in chunks:

        metadata = chunk["metadata"]

        chunk_type = metadata["chunk_type"]
        parent_id = metadata["parent_id"]

        if chunk_type == "article":

            if parent_id is not None:
                invalid_parent.append(
                    chunk["id"]
                )

        else:

            if (
                parent_id is None
                or parent_id not in chunk_ids
            ):

                invalid_parent.append(
                    chunk["id"]
                )

    # --------------------------------------------------------
    # 구조
    # --------------------------------------------------------

    structure_errors = []

    for chunk in chunks:

        metadata = chunk["metadata"]

        chunk_type = metadata["chunk_type"]

        paragraph = metadata["paragraph"]
        item = metadata["item"]

        if chunk_type == "article":

            if (
                paragraph is not None
                or item is not None
            ):

                structure_errors.append(
                    chunk["id"]
                )

        elif chunk_type == "paragraph":

            if paragraph is None:
                structure_errors.append(
                    chunk["id"]
                )

            if item is not None:
                structure_errors.append(
                    chunk["id"]
                )

        elif chunk_type == "item":

            if (
                paragraph is None
                or item is None
            ):

                structure_errors.append(
                    chunk["id"]
                )

    # --------------------------------------------------------
    # Parent Article 존재 여부
    # --------------------------------------------------------

    article_parent_errors = []

    article_parent_ids = {
        chunk["id"]
        for chunk in chunks
        if chunk["metadata"]["chunk_type"]
        == "article"
    }

    for chunk in chunks:

        metadata = chunk["metadata"]

        if metadata["chunk_type"] != "article":

            parent_id = metadata["parent_id"]

            if parent_id not in article_parent_ids:

                article_parent_errors.append(
                    chunk["id"]
                )

    # --------------------------------------------------------
    # 통계
    # --------------------------------------------------------

    main_count = sum(
        chunk["metadata"]["section"]
        == "본문"
        for chunk in chunks
    )

    supplementary_count = sum(
        chunk["metadata"]["section"]
        == "부칙"
        for chunk in chunks
    )

    article_chunks = sum(
        chunk["metadata"]["chunk_type"]
        == "article"
        for chunk in chunks
    )

    paragraph_chunks = sum(
        chunk["metadata"]["chunk_type"]
        == "paragraph"
        for chunk in chunks
    )

    item_chunks = sum(
        chunk["metadata"]["chunk_type"]
        == "item"
        for chunk in chunks
    )

    article_count = len({
        (
            chunk["metadata"]["section"],
            chunk["metadata"]["article"],
        )
        for chunk in chunks
    })

    # --------------------------------------------------------
    # 출력
    # --------------------------------------------------------

    print(
        f"Article 수              : "
        f"{article_count}"
    )

    print(
        f"Chunk 수                : "
        f"{len(chunks)}"
    )

    print(
        f"  ├─ Article Chunk      : "
        f"{article_chunks}"
    )

    print(
        f"  ├─ Paragraph Chunk    : "
        f"{paragraph_chunks}"
    )

    print(
        f"  └─ Item Chunk         : "
        f"{item_chunks}"
    )

    print(
        f"본문 Chunk              : "
        f"{main_count}"
    )

    print(
        f"부칙 Chunk              : "
        f"{supplementary_count}"
    )

    print(
        f"중복 ID                 : "
        f"{len(duplicate_ids)}"
    )

    print(
        f"Chapter 미지정          : "
        f"{len(missing_chapter)}"
    )

    print(
        f"필수 Metadata 누락      : "
        f"{len(missing_metadata)}"
    )

    print(
        f"페이지 오류              : "
        f"{len(page_errors)}"
    )

    print(
        f"빈 Chunk                : "
        f"{len(empty_chunks)}"
    )

    print(
        f"잘못된 Section          : "
        f"{len(invalid_section)}"
    )

    print(
        f"잘못된 Chunk Type       : "
        f"{len(invalid_chunk_type)}"
    )

    print(
        f"잘못된 Parent ID        : "
        f"{len(invalid_parent)}"
    )

    print(
        f"구조 오류               : "
        f"{len(structure_errors)}"
    )

    print(
        f"Article Parent 오류     : "
        f"{len(article_parent_errors)}"
    )

    # --------------------------------------------------------
    # 상세 출력
    # --------------------------------------------------------

    if duplicate_ids:

        print("\n중복 ID:")

        for item in duplicate_ids:
            print(f"  - {item}")

    if missing_chapter:

        print("\nChapter 미지정:")

        for item in missing_chapter[:20]:
            print(f"  - {item}")

    if missing_metadata:

        print("\n필수 Metadata 누락:")

        for item in missing_metadata[:20]:
            print(f"  - {item}")

    if page_errors:

        print("\n페이지 오류:")

        for item in page_errors[:20]:
            print(f"  - {item}")

    if invalid_parent:

        print("\n잘못된 Parent ID:")

        for item in invalid_parent[:20]:
            print(f"  - {item}")

    if structure_errors:

        print("\n구조 오류:")

        for item in structure_errors[:20]:
            print(f"  - {item}")

    if article_parent_errors:

        print("\nArticle Parent 오류:")

        for item in article_parent_errors[:20]:
            print(f"  - {item}")

    # --------------------------------------------------------
    # 최종 판정
    # --------------------------------------------------------

    passed = not any([
        duplicate_ids,
        missing_chapter,
        missing_metadata,
        page_errors,
        empty_chunks,
        invalid_section,
        invalid_chunk_type,
        invalid_parent,
        structure_errors,
        article_parent_errors,
    ])

    print()

    if passed:
        print("✓ 검증 통과")
    else:
        print("⚠ 검증 실패")

    return passed


# ============================================================
# 20. 부칙 Validation
# ============================================================

def validate_supplementary(
    parse_result,
    chunks,
):

    found_in_pdf = (
        parse_result[
            "supplementary_found"
        ]
    )

    pages = (
        parse_result[
            "supplementary_pages"
        ]
    )

    supplementary_chunks = [
        chunk
        for chunk in chunks
        if chunk["metadata"]["section"]
        == "부칙"
    ]

    print()
    print("-" * 70)
    print("부칙 검사")
    print("-" * 70)

    print(
        f"PDF 부칙 발견 : "
        f"{'YES' if found_in_pdf else 'NO'}"
    )

    print(
        f"부칙 페이지   : "
        f"{pages if pages else '없음'}"
    )

    print(
        f"부칙 Chunk     : "
        f"{len(supplementary_chunks)}"
    )

    if (
        found_in_pdf
        and not supplementary_chunks
    ):

        print(
            "⚠ PDF에는 부칙이 있는데 "
            "Chunk가 생성되지 않았습니다."
        )

        return False

    for chunk in supplementary_chunks:

        metadata = chunk["metadata"]

        print(
            f"  - "
            f"{metadata['article']} "
            f"{metadata['article_title'] or ''} "
            f"(p."
            f"{metadata['page_start']}-"
            f"{metadata['page_end']})"
        )

    return True


# ============================================================
# 21. Preview
# ============================================================

def print_preview(
    chunks,
    limit=5,
):

    print()
    print("=" * 70)
    print("CHUNK PREVIEW")
    print("=" * 70)

    for chunk in chunks[:limit]:

        metadata = chunk["metadata"]

        print()

        print(
            f"ID          : "
            f"{chunk['id']}"
        )

        print(
            f"Law         : "
            f"{metadata['law_name']}"
        )

        print(
            f"Law Type    : "
            f"{metadata['law_type']}"
        )

        print(
            f"Law Number  : "
            f"{metadata['law_number']}"
        )

        print(
            f"Effective   : "
            f"{metadata['effective_date']}"
        )

        print(
            f"Section     : "
            f"{metadata['section']}"
        )

        print(
            f"Chapter     : "
            f"{metadata['chapter']}"
        )

        print(
            f"Chapter Title: "
            f"{metadata['chapter_title']}"
        )

        print(
            f"Subsection  : "
            f"{metadata['subsection']}"
        )

        print(
            f"Subsection Title: "
            f"{metadata['subsection_title']}"
        )

        print(
            f"Article     : "
            f"{metadata['article']}"
        )

        print(
            f"Article Title: "
            f"{metadata['article_title']}"
        )

        print(
            f"Paragraph   : "
            f"{metadata['paragraph']}"
        )

        print(
            f"Item        : "
            f"{metadata['item']}"
        )

        print(
            f"Chunk Type  : "
            f"{metadata['chunk_type']}"
        )

        print(
            f"Parent ID   : "
            f"{metadata['parent_id']}"
        )

        print(
            f"Page        : "
            f"{metadata['page_start']} ~ "
            f"{metadata['page_end']}"
        )

        print(
            f"Part        : "
            f"{metadata['chunk_index']}/"
            f"{metadata['chunk_count']}"
        )

        print("-" * 70)

        preview = chunk["text"]

        if len(preview) > 800:
            preview = (
                preview[:800]
                + "..."
            )

        print(preview)


# ============================================================
# 22. JSON 저장
# ============================================================

def save_chunks(
    chunks,
    document,
):

    output_dir = (
        OUTPUT_DIR / document
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir / "chunks.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            chunks,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return output_path


# ============================================================
# 23. PDF 하나 처리
# ============================================================

def process_pdf(
    pdf_path: Path,
):

    document = get_document_name(
        pdf_path
    )

    print()
    print("#" * 70)

    print(
        f"PROCESSING : "
        f"{pdf_path.name}"
    )

    print("#" * 70)

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    print()
    print("PDF 읽는 중...")

    pages = extract_pages(
        pdf_path
    )

    # --------------------------------------------------------
    # 법령 Metadata
    # --------------------------------------------------------

    law_metadata = parse_law_metadata(
        document
    )

    print()
    print("법령 기본정보")

    print(
        f"  법령명     : "
        f"{law_metadata['law_name']}"
    )

    print(
        f"  법령종류   : "
        f"{law_metadata['law_type']}"
    )

    print(
        f"  법령번호   : "
        f"{law_metadata['law_number']}"
    )

    print(
        f"  시행일     : "
        f"{law_metadata['effective_date']}"
    )

    # --------------------------------------------------------
    # 구조 분석
    # --------------------------------------------------------

    print()
    print("법률 구조 분석 중...")

    parse_result = parse_document(
        pages=pages,
        document=document,
    )

    raw_articles = (
        parse_result["articles"]
    )

    print(
        f"초기 Article 발견 : "
        f"{len(raw_articles)}"
    )

    # --------------------------------------------------------
    # Article 병합
    # --------------------------------------------------------

    articles = merge_articles(
        raw_articles
    )

    print(
        f"병합 후 Article   : "
        f"{len(articles)}"
    )

    # --------------------------------------------------------
    # Chunk
    # --------------------------------------------------------

    print()
    print("Chunk 생성 중...")

    chunks = articles_to_chunks(
        articles=articles,
        document=document,
    )

    print(
        f"생성된 Chunk 수    : "
        f"{len(chunks)}"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validation_ok = validate_chunks(
        chunks=chunks,
        document=document,
        total_pages=len(pages),
    )

    supplementary_ok = (
        validate_supplementary(
            parse_result=parse_result,
            chunks=chunks,
        )
    )

    final_ok = (
        validation_ok
        and supplementary_ok
    )

    # --------------------------------------------------------
    # 저장
    # --------------------------------------------------------

    output_path = save_chunks(
        chunks=chunks,
        document=document,
    )

    print()

    print(
        f"저장 위치 : "
        f"{output_path}"
    )

    # --------------------------------------------------------
    # Preview
    # --------------------------------------------------------

    print_preview(
        chunks,
        limit=5,
    )

    return {
        "document":
            document,

        "law_name":
            law_metadata["law_name"],

        "law_type":
            law_metadata["law_type"],

        "law_number":
            law_metadata["law_number"],

        "effective_date":
            law_metadata["effective_date"],

        "raw_articles":
            len(raw_articles),

        "articles":
            len(articles),

        "chunks":
            len(chunks),

        "supplementary_found":
            parse_result[
                "supplementary_found"
            ],

        "supplementary_chunks":
            sum(
                chunk["metadata"]["section"]
                == "부칙"
                for chunk in chunks
            ),

        "validation":
            final_ok,

        "output":
            str(output_path),
    }


# ============================================================
# 24. Main
# ============================================================

def main():

    print("=" * 70)

    print(
        "SafeAgent - "
        "Legal RAG Document Ingestion"
    )

    print("=" * 70)

    print()

    print(
        f"PDF 폴더 : "
        f"{DOCUMENT_DIR}"
    )

    pdf_files = find_pdf_files()

    if not pdf_files:

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

    for index, pdf_path in enumerate(
        pdf_files,
        start=1,
    ):

        print()

        print(
            f"[{index}/{len(pdf_files)}] "
            f"{pdf_path.name}"
        )

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
                f"{type(e).__name__}: "
                f"{e}"
            )

            results.append({

                "document":
                    pdf_path.stem,

                "law_name":
                    None,

                "law_type":
                    None,

                "law_number":
                    None,

                "effective_date":
                    None,

                "raw_articles":
                    0,

                "articles":
                    0,

                "chunks":
                    0,

                "supplementary_found":
                    False,

                "supplementary_chunks":
                    0,

                "validation":
                    False,

                "output":
                    None,
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
            f"       Law         : "
            f"{result.get('law_name')}"
        )

        print(
            f"       Law Type    : "
            f"{result.get('law_type')}"
        )

        print(
            f"       Law Number  : "
            f"{result.get('law_number')}"
        )

        print(
            f"       Effective   : "
            f"{result.get('effective_date')}"
        )

        print(
            f"       Raw Article : "
            f"{result['raw_articles']}"
        )

        print(
            f"       Article     : "
            f"{result['articles']}"
        )

        print(
            f"       Chunk       : "
            f"{result['chunks']}"
        )

        print(
            f"       부칙 발견   : "
            f"{result['supplementary_found']}"
        )

        print(
            f"       부칙 Chunk  : "
            f"{result['supplementary_chunks']}"
        )

        print(
            f"       Output      : "
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