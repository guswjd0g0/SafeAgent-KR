"""
frontend/app.py — SafeAgent Streamlit UI

실행:
    streamlit run frontend/app.py
"""

import sys
from pathlib import Path

import streamlit as st

# ============================================================
# 경로 설정 — rag_pipeline 임포트
# ============================================================
_RAG_DIR = Path(__file__).resolve().parents[1] / "app" / "rag"
if str(_RAG_DIR) not in sys.path:
    sys.path.insert(0, str(_RAG_DIR))

from rag_pipeline import init_pipeline, run_rag, LLM_MODEL  # noqa: E402


# ============================================================
# 페이지 설정
# ============================================================
st.set_page_config(
    page_title="SafeAgent",
    page_icon="⚖️",
    layout="wide",
)


# ============================================================
# 파이프라인 초기화 (앱 시작 시 1회)
# ============================================================
@st.cache_resource(show_spinner="RAG 파이프라인 초기화 중...")
def load_pipeline():
    init_pipeline()
    return True


load_pipeline()


# ============================================================
# 사이드바
# ============================================================
with st.sidebar:
    st.header("⚙️ 검색 설정")

    law_options = {
        "전체 법률": None,
        "근로기준법": "근로기준법",
        "산업안전보건법": "산업안전보건법",
        "산업재해보상보험법": "산업재해보상보험법",
        "직업교육훈련 촉진법": "직업교육훈련 촉진법",
    }

    selected_law_label = st.selectbox(
        "법률 필터",
        options=list(law_options.keys()),
        index=0,
    )
    selected_law = law_options[selected_law_label]

    top_k = st.slider("검색 조문 수 (Top-K)", min_value=1, max_value=10, value=5)

    st.divider()
    st.caption(f"🤖 LLM: `{LLM_MODEL}`")
    st.caption("📚 임베딩: `BAAI/bge-m3`")
    st.caption("🗄️ DB: Qdrant (514 chunks)")


# ============================================================
# 메인 UI
# ============================================================
st.title("⚖️ SafeAgent")
st.caption("노동법 전문 AI 보조원 — 근거 조문 기반 답변")

st.divider()

# 질문 입력
question = st.text_area(
    "질문을 입력하세요",
    placeholder="예: 근로자와 사용자는 어떻게 정의되나요?",
    height=100,
)

run_btn = st.button("🔍 검색 및 답변", type="primary", use_container_width=True)

# ============================================================
# 답변 실행
# ============================================================
if run_btn:
    if not question.strip():
        st.warning("질문을 입력해주세요.")
    else:
        # ── 검색 단계 ──────────────────────────────────────
        with st.spinner("관련 조문 검색 중..."):
            result = run_rag(
                question=question.strip(),
                law=selected_law,
                top_k=top_k,
                verbose=False,
            )

        # ── 검색 결과 ──────────────────────────────────────
        st.subheader("📄 검색된 조문")

        for i, doc in enumerate(result["docs"], start=1):
            with st.expander(
                f"[{i}] {doc['law']} {doc['article']}  —  score: {doc['score']:.4f}",
                expanded=(i == 1),
            ):
                st.markdown(doc["text"])

        st.divider()

        # ── LLM 답변 스트리밍 ──────────────────────────────
        st.subheader("💬 AI 답변")

        # 스트리밍: 이미 완성된 answer를 청크 단위로 흘려보냄
        answer_placeholder = st.empty()
        streamed = ""

        # ChatOllama는 invoke가 완성 후 반환이므로
        # 단어 단위로 점진적으로 표시하여 스트리밍 효과를 구현
        import time
        words = result["answer"].split(" ")
        for word in words:
            streamed += word + " "
            answer_placeholder.markdown(streamed + "▌")
            time.sleep(0.02)

        answer_placeholder.markdown(result["answer"])

        # ── 메타 정보 ──────────────────────────────────────
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("검색 시간", f"{result['retrieval_sec']}초")
        with col2:
            st.metric("LLM 시간", f"{result['llm_sec']}초")
        with col3:
            st.metric("참조 조문 수", f"{len(result['docs'])}개")

        st.caption(f"출처: {' | '.join(result['sources'])}")
