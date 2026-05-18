from fastapi import FastAPI

app = FastAPI(title="5sondoson AI Server")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"name": "5sondoson-ai", "version": "0.1.0"}
