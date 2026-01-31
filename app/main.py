from fastapi import FastAPI

app = FastAPI(
    title="Calorie Tracker Demo",
    root_path="/calorie-tracker",
    docs_url="/docs",
    redoc_url="/redoc"
)


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "calorie-tracker",
        "message": "API is running"
    }
