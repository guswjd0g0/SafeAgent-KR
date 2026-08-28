from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

llm = ChatOllama(model="gemma3:4b", temperature=0)

dummy_context = """
[법률]
알파고법

[조문]
제1조(목적) 이 법은 인공지능 개발자의 주 4일 근무제를 보장하여 개발자의 휴식권을 확립하는 것을 목적으로 한다.
제2조(야근 금지) 인공지능 개발자는 오후 6시 이후의 모든 야근 및 주말 근무가 금지된다.
"""

question = "알파고법 제1조의 목적과 제2조의 야근 규칙에 대해 설명해줘."

prompt_template = PromptTemplate.from_template(
    """다음 제공된 법률 조문만을 바탕으로 질문에 답변하세요.
제공된 정보에 없는 내용은 절대 추측해서 작성하지 마세요.

[법률 조문]
{context}

[질문]
{question}

[답변]
"""
)

formatted_prompt = prompt_template.format(context=dummy_context, question=question)
response = llm.invoke(formatted_prompt)

print("=== Context 전달 테스트 결과 ===")
print(response.content)