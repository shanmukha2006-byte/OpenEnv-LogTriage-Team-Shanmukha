from fastapi import FastAPI
import uvicorn

from inference import run_inference

app = FastAPI()


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/reset")
def reset():

    try:

        result = run_inference()

        return {
            "status": "reset successful",
            "result": result
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }


def main():

    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860
    )


if __name__ == "__main__":
    main()
