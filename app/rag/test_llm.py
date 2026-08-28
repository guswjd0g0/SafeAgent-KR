import time
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="gemma3:4b",
    temperature=0
)

start_time = time.perf_counter()
response = llm.invoke("근로기준법이 무엇인지 한 문장으로 간단히 설명해줘.")
end_time = time.perf_counter()

latency = end_time - start_time

print(response.content)
print(f"\n[성능 측정] 총 응답 시간: {latency:.2f}초")