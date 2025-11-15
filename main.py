from fastapi import FastAPI # type: ignore
from routers.perfumes import router as perfumes_router 

app = FastAPI(title="Inventory API")

@app.get("/")
def root():
    return {"message": "API is running"}

app.include_router(perfumes_router)
