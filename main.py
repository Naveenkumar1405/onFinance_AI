import uvicorn
from fastapi import FastAPI
from functions import get_answer as get_answer_from_dataset

app = FastAPI(docs_url='/')

@app.get("/answers/")
async def get_answer_endpoint(question: str):
    answer, evidence = get_answer_from_dataset(question)
    return {"answer": answer, "evidence": evidence}

if __name__ == "__main__":
    uvicorn.run("main:app", host="192.168.1.68", port=8080, reload=True, log_level="debug")
