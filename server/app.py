from fastapi import FastAPI
import uvicorn

from inference import run_inference

app = FastAPI()


# Health check
@app.get("/")
def root():
    return {"status": "ok"}


# Reset endpoint (required)
@app.post("/reset")
def reset():

    result = run_inference()

    return {
        "status": "reset successful",
        "result": result
    }


@app.get("/state")
def state():

    return {
        "status": "running"
    }


def main():

    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860
    )
