"""
legal_queriesV3.json 생성 스크립트

V2의 오류 항목을 수정하여 V3를 만든다.

수정 원칙:
  - 질문을 최대한 유지하고 정답 조문을 올바르게 교체한다
  - 정답 조문이 chunks에 실제로 존재하는지 먼저 검증한다
  - 수정이 어려운 경우 질문도 함께 교체한다

수정 대상 (V2 기준):
  ID 03  질문=적용범위        → 정답 제3조(근로조건의 기준) 불일치 → 질문 수정
  ID 10  질문=사업주 기본조치  → 정답 제4조의2(지방자치단체 책무) 불일치 → 정답 교체
  ID 12  질문=주요 용어 정의   → 정답 제2조(보험의 관장과 보험연도) 불일치 → 정답 교체
  ID 13  질문=어떤 사업 적용   → 정답 제3조(국가의 부담 및 지원) 불일치 → 정답 교체
  ID 14  질문=국가 부담 비용   → 정답 제4조(보험료) 불일치 → 정답 교체
  ID 15  질문=보험사업 운영    → 정답 제5조(정의) 불일치 → 질문 수정
  ID 18  질문=누구에게 적용    → 정답 제3조(국가 등의 책무) 불일치 → 정답 교체
  ID 19  질문=기본원칙        → 정답 제4조(기본계획의 수립시행) 불일치 → 질문 수정
  ID 20  질문=국가·지자체 역할 → 정답 제5조(연계운영) 불일치 → 정답 교체
"""

import json
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


# ============================================================
# 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

V2_PATH = BASE_DIR / "data" / "evaluation" / "legal_queriesV2.json"
V3_PATH = BASE_DIR / "data" / "evaluation" / "legal_queriesV3.json"

COLLECTION_NAME = "legal_documents_bge_m3"
QDRANT_HOST     = "localhost"
QDRANT_PORT     = 6333


# ============================================================
# chunk_id 존재 확인 + 조문 정보 조회
# ============================================================

def get_chunk_info(client, chunk_id: str) -> dict:
    """chunk_id로 Qdrant payload를 조회한다."""

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


def verify_chunk(client, chunk_id: str) -> bool:
    """chunk_id가 실제로 존재하는지 확인한다."""
    return bool(get_chunk_info(client, chunk_id))


# ============================================================
# 수정 명세
#
# 각 항목:
#   id           : V2의 ID
#   action       : "fix_answer" (정답 교체) | "fix_question" (질문 교체)
#   new_query    : action=fix_question 일 때 새 질문
#   new_law      : 새 정답 법률명
#   new_article  : 새 정답 조문 번호
#   new_chunk_id : 새 정답 chunk_id
#   reason       : 수정 이유
# ============================================================

FIXES = [

    # ----------------------------------------------------------
    # ID 03 — 근로기준법 제3조
    # 기존 질문: "모든 사업장에 동일하게 적용되나요?"
    # 기존 정답: 제3조(근로조건의 기준)
    # → 적용범위 질문인데 정답은 "최저기준" 조문
    # → 질문을 제3조 내용에 맞게 수정
    # ----------------------------------------------------------
    {
        "id":          3,
        "action":      "fix_question",
        "new_query":   "근로기준법에서 정하는 근로조건은 어떤 기준이며, 이 기준을 이유로 근로조건을 낮출 수 있나요?",
        "new_law":     "근로기준법",
        "new_article": "제3조",
        "new_chunk_id": "근로기준법_법률_제21533호_20270101_본문_제3조",
        "reason":      "질문이 적용범위를 묻지만 정답 조문은 근로조건의 기준(최저기준)임 → 조문 내용에 맞게 질문 수정",
    },

    # ----------------------------------------------------------
    # ID 10 — 산업안전보건법
    # 기존 질문: "사업주가 해야 할 기본적인 조치"
    # 기존 정답: 제4조의2(지방자치단체의 책무)
    # → 사업주 조치는 제5조(사업주 등의 의무)
    # ----------------------------------------------------------
    {
        "id":          10,
        "action":      "fix_answer",
        "new_query":   None,
        "new_law":     "산업안전보건법",
        "new_article": "제5조",
        "new_chunk_id": "산업안전보건법_법률_제21374호_20260801_본문_제5조",
        "reason":      "정답 조문 제4조의2는 지방자치단체의 책무임. 사업주 의무는 제5조(사업주 등의 의무)",
    },

    # ----------------------------------------------------------
    # ID 12 — 산업재해보상보험법
    # 기존 질문: "사용하는 주요 용어는 어떻게 정의되어 있나요?"
    # 기존 정답: 제2조(보험의 관장과 보험연도)
    # → 용어 정의는 제5조(정의)
    # ----------------------------------------------------------
    {
        "id":          12,
        "action":      "fix_answer",
        "new_query":   None,
        "new_law":     "산업재해보상보험법",
        "new_article": "제5조",
        "new_chunk_id": "산업재해보상보험법_법률_제21375호_20260701_본문_제5조",
        "reason":      "정답 조문 제2조는 보험의 관장과 보험연도임. 용어 정의는 제5조(정의)",
    },

    # ----------------------------------------------------------
    # ID 13 — 산업재해보상보험법
    # 기존 질문: "어떤 사업에 적용되나요?"
    # 기존 정답: 제3조(국가의 부담 및 지원)
    # → 적용범위는 제6조(적용 범위)
    # ----------------------------------------------------------
    {
        "id":          13,
        "action":      "fix_answer",
        "new_query":   None,
        "new_law":     "산업재해보상보험법",
        "new_article": "제6조",
        "new_chunk_id": "산업재해보상보험법_법률_제21375호_20260701_본문_제6조",
        "reason":      "정답 조문 제3조는 국가의 부담 및 지원임. 적용 범위는 제6조(적용 범위)",
    },

    # ----------------------------------------------------------
    # ID 14 — 산업재해보상보험법
    # 기존 질문: "국가가 부담해야 하는 비용은 무엇인가요?"
    # 기존 정답: 제4조(보험료) — 보험료 징수법 준용 조항
    # → 국가 부담은 제3조(국가의 부담 및 지원)
    # ----------------------------------------------------------
    {
        "id":          14,
        "action":      "fix_answer",
        "new_query":   None,
        "new_law":     "산업재해보상보험법",
        "new_article": "제3조",
        "new_chunk_id": "산업재해보상보험법_법률_제21375호_20260701_본문_제3조",
        "reason":      "정답 조문 제4조는 보험료(징수법 준용)임. 국가 부담 비용은 제3조(국가의 부담 및 지원)",
    },

    # ----------------------------------------------------------
    # ID 15 — 산업재해보상보험법
    # 기존 질문: "보험사업을 어떻게 운영하도록 규정하고 있나요?"
    # 기존 정답: 제5조(정의) — 용어 정의 조문
    # → ID 12가 이미 제5조(정의) 사용
    # → 보험사업 관장·보험연도는 제2조(보험의 관장과 보험연도)
    # → 질문을 제2조 내용에 맞게 수정
    # ----------------------------------------------------------
    {
        "id":          15,
        "action":      "fix_question",
        "new_query":   "산업재해보상보험 사업은 누가 관장하며, 보험연도는 어떻게 정해지나요?",
        "new_law":     "산업재해보상보험법",
        "new_article": "제2조",
        "new_chunk_id": "산업재해보상보험법_법률_제21375호_20260701_본문_제2조",
        "reason":      "정답 조문 제5조는 정의 조문이며 ID12와 중복. 보험사업 운영(관장) 내용은 제2조(보험의 관장과 보험연도) → 질문과 정답 모두 수정",
    },

    # ----------------------------------------------------------
    # ID 18 — 직업교육훈련 촉진법
    # 기존 질문: "누구에게 적용되나요?"
    # 기존 정답: 제3조(국가 등의 책무) — 국가·지자체 지원시책
    # → 이 법의 적용대상은 별도 조문 없으므로
    #   제3조 내용(국가의 지원 책무)에 맞게 질문 수정
    # ----------------------------------------------------------
    {
        "id":          18,
        "action":      "fix_question",
        "new_query":   "직업교육훈련 촉진을 위해 국가와 지방자치단체가 마련해야 할 지원시책에는 어떤 것들이 있나요?",
        "new_law":     "직업교육훈련 촉진법",
        "new_article": "제3조",
        "new_chunk_id": "직업교육훈련_촉진법_법률_제21065호_20251001_본문_제3조",
        "reason":      "정답 조문 제3조는 국가 등의 책무(지원시책)임. 누구에게 적용되는지와 불일치 → 질문 수정",
    },

    # ----------------------------------------------------------
    # ID 19 — 직업교육훈련 촉진법
    # 기존 질문: "어떤 기본원칙에 따라 실시되어야 하나요?"
    # 기존 정답: 제4조(직업교육훈련 기본계획의 수립·시행)
    # → 기본계획 수립 절차 조문이며 기본원칙과 다름
    # → 조문 내용에 맞게 질문 수정
    # ----------------------------------------------------------
    {
        "id":          19,
        "action":      "fix_question",
        "new_query":   "직업교육훈련 촉진을 위한 기본계획에는 어떤 내용이 포함되어야 하나요?",
        "new_law":     "직업교육훈련 촉진법",
        "new_article": "제4조",
        "new_chunk_id": "직업교육훈련_촉진법_법률_제21065호_20251001_본문_제4조",
        "reason":      "정답 조문 제4조는 기본계획 수립·시행임. 기본원칙과 불일치 → 질문 수정",
    },

    # ----------------------------------------------------------
    # ID 20 — 직업교육훈련 촉진법
    # 기존 질문: "국가와 지방자치단체는 어떤 역할을 해야 하나요?"
    # 기존 정답: 제5조(직업교육훈련기관의 연계운영)
    # → 국가·지자체 역할은 제3조(국가 등의 책무)
    # → 하지만 ID18이 이미 제3조로 수정됨
    #   → 제5조 내용(연계운영)에 맞게 질문 수정
    # ----------------------------------------------------------
    {
        "id":          20,
        "action":      "fix_question",
        "new_query":   "직업교육훈련기관은 직업교육훈련과정을 어떻게 연계하여 운영할 수 있나요?",
        "new_law":     "직업교육훈련 촉진법",
        "new_article": "제5조",
        "new_chunk_id": "직업교육훈련_촉진법_법률_제21065호_20251001_본문_제5조",
        "reason":      "정답 조문 제5조는 연계운영임. 국가·지자체 역할과 불일치 → 질문 수정. ID18이 이미 제3조 사용",
    },
]


# ============================================================
# V3 생성
# ============================================================

def build_v3(client):

    # --------------------------------------------------------
    # V2 로드
    # --------------------------------------------------------

    with open(V2_PATH, encoding="utf-8") as f:
        v2 = json.load(f)

    # fix_map: id → fix 명세
    fix_map = {fix["id"]: fix for fix in FIXES}

    # --------------------------------------------------------
    # 수정 전 새 chunk_id 존재 여부 전체 검증
    # --------------------------------------------------------

    print("수정 대상 chunk_id 존재 확인...")
    all_ok = True

    for fix in FIXES:
        cid   = fix["new_chunk_id"]
        info  = get_chunk_info(client, cid)

        if info:
            article = info.get("article", "")
            title   = info.get("article_title", "") or ""
            print(
                f"  [ID {fix['id']:02d}] ✓  "
                f"{cid.split('_본문_')[-1]}  "
                f"→ {article} {title}"
            )
        else:
            print(f"  [ID {fix['id']:02d}] ✗  {cid}  ← 존재하지 않음")
            all_ok = False

    if not all_ok:
        raise RuntimeError(
            "\n새 chunk_id 중 존재하지 않는 항목이 있습니다.\n"
            "수정 명세를 다시 확인하세요."
        )

    print()
    print("모든 새 chunk_id 확인 완료")

    # --------------------------------------------------------
    # V3 구성
    # --------------------------------------------------------

    v3 = []
    changed = []

    for item in v2:

        qid  = item["id"]
        fix  = fix_map.get(qid)

        if fix is None:
            # 수정 없음 — 그대로 복사
            v3.append(dict(item))
            continue

        # 수정 적용
        new_item = dict(item)

        if fix["action"] == "fix_question":
            new_item["query"] = fix["new_query"]

        new_item["expected_law"]      = fix["new_law"]
        new_item["expected_article"]  = fix["new_article"]
        new_item["expected_chunk_id"] = fix["new_chunk_id"]

        v3.append(new_item)
        changed.append(qid)

    # --------------------------------------------------------
    # 저장
    # --------------------------------------------------------

    V3_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(V3_PATH, "w", encoding="utf-8") as f:
        json.dump(v3, f, ensure_ascii=False, indent=2)

    return v3, changed


# ============================================================
# Main
# ============================================================

def main():

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    print("=" * 70)
    print("legal_queriesV3.json 생성")
    print("=" * 70)
    print()

    v3, changed = build_v3(client)

    # --------------------------------------------------------
    # 수정 결과 출력
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(f"수정 완료: {len(changed)}개 항목 변경")
    print("=" * 70)
    print()

    with open(V3_PATH.parent / "legal_queriesV2.json", encoding="utf-8") as f:
        v2_map = {item["id"]: item for item in json.load(f)}

    for item in v3:
        qid = item["id"]

        if qid not in changed:
            continue

        v2_item = v2_map[qid]
        fix     = next(f for f in FIXES if f["id"] == qid)

        print(f"[ID {qid:02d}]  {fix['action'].upper()}")
        print(f"  이유    : {fix['reason']}")

        if fix["action"] == "fix_question":
            print(f"  구 질문 : {v2_item['query']}")
            print(f"  신 질문 : {item['query']}")

        print(
            f"  구 정답 : {v2_item['expected_law']} "
            f"{v2_item['expected_article']} "
            f"({v2_item['expected_chunk_id'].split('_본문_')[-1]})"
        )
        print(
            f"  신 정답 : {item['expected_law']} "
            f"{item['expected_article']} "
            f"({item['expected_chunk_id'].split('_본문_')[-1]})"
        )
        print()

    print(f"저장 완료: {V3_PATH}")

    # --------------------------------------------------------
    # 중복 정답 확인 (같은 chunk_id가 두 번 쓰이면 경고)
    # --------------------------------------------------------

    chunk_ids = [item["expected_chunk_id"] for item in v3]
    seen = {}
    duplicates = []

    for item in v3:
        cid = item["expected_chunk_id"]
        if cid in seen:
            duplicates.append((seen[cid], item["id"], cid))
        seen[cid] = item["id"]

    if duplicates:
        print()
        print("⚠ 중복 chunk_id 발견:")
        for id1, id2, cid in duplicates:
            print(f"  ID {id1} & ID {id2} → {cid}")
    else:
        print()
        print("중복 chunk_id 없음 — 평가셋 V3 정상")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()
