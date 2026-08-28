import time
from typing import List, Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

try:
    from app.rag.retriever import search_query
except ImportError:
    def search_query(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        return [
            {
                "law": "근로기준법",
                "article": "제2조(정의)",
                "text": "이 법에서 사용하는 용어의 뜻은 다음과 같다. 1. '근로자'란 직업의 종류와 관계없이 임금을 목적으로 사업이나 사업장에 근로를 제공하는 자를 말한다. 2. '사용자'란 사업주 또는 사업 경영 담당자, 그 밖에 사업주를 위하여 일하는 자를 말한다."
            }
        ]

def build_context(search_results: List[Dict[str, Any]]) -> str:
    """
    Qdrant에서 받아온 검색 결과 리스트를 LLM에 입력할 Context 문자열로 변환합니다.
    """
    context_blocks = []
    for idx, doc in enumerate(search_results, 1):
        # Dictionary 키 이름은 실제 Qdrant Payload 필드명에 맞게 조정해 주세요.
        law = doc.get("law", "관련 법률")
        article = doc.get("article", "")
        text = doc.get("text", "")
        
        block = f"[{idx}] {law} {article}\n{text}"
        context_blocks.append(block)
    
    return "\n\n".join(context_blocks)

llm = ChatOllama(
    model="gemma3:4b",
    temperature=0
)

prompt_template = PromptTemplate.from_template(
    """당신은 법률 전문 AI 보조원입니다. 
아래 제공된 [법률 조문]만을 바탕으로 사용자 질문에 정확하게 답변하세요.

[원칙]
1. 제공된 조문 내용에 직접적으로 언급된 사실만 근거로 작성하세요.
2. 조문에 없는 내용은 절대로 추측하거나 지어내지 마세요.
3. 답변 끝에는 반드시 참고한 법률명과 조문 번호를 근거로 명시하세요.

[법률 조문]
{context}

[사용자 질문]
{question}

[답변]
"""
)

def run_rag_pipeline(question: str, top_k: int = 3):
    print(f" 질문: {question}\n")
    
    # STEP A. 검색 (Retrieval)
    start_time = time.perf_counter()
    search_results = search_query(question, top_k=top_k)
    retrieval_time = time.perf_counter() - start_time
    print(f" 1단계: Qdrant 검색 완료 ({retrieval_time:.2f}초, 검색된 문서 수: {len(search_results)}개)")
    
    # STEP B. Context 생성
    context = build_context(search_results)
    
    # STEP C. LLM 전달 및 답변 생성
    formatted_prompt = prompt_template.format(context=context, question=question)
    
    llm_start_time = time.perf_counter()
    response = llm.invoke(formatted_prompt)
    llm_time = time.perf_counter() - llm_start_time
    
    print(f" 2단계: LLM 답변 생성 완료 ({llm_time:.2f}초)\n")
    print("=== [최종 RAG 응답] ===")
    print(response.content)
    print("=========================")

if __name__ == "__main__":
    test_question = "근로기준법에서 말하는 근로자와 사용자는 어떻게 정의되나요?"
    run_rag_pipeline(test_question)