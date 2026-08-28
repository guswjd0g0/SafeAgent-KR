from langchain_ollama import OllamaEmbeddings

embedding = OllamaEmbeddings(
    model="nomic-embed-text"
)

print("Embedding 요청 시작")

vector = embedding.embed_query(
    "근로기준법 제1조의 목적은 무엇인가?"
)

print("Embedding 성공")
print("Vector dimension:", len(vector))
print("Vector 앞 5개:", vector[:5])