from ollama import chat

response = chat(
    model="qwen3.5:4b",
    messages=[
        {
            "role": "user",
            "content": "산업안전에서 안전모가 필요한 이유를 한 문장으로 설명해줘."
        }
    ]
)

print(response["message"]["content"])