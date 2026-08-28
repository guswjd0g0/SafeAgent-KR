from fastapi import FastAPI

app = FastAPI(
    title="SafeAgent API",
    description="Local Multimodal Industrial Safety Agent",
    version="0.1.0"
)


@app.get("/")
def root():
    return {
        "project": "SafeAgent",
        "status": "running"
    }