from fastapi import FastAPI
from pydantic import BaseModel

from support_assistant.graph import ask_question, SupportResponse


app = FastAPI(
    title="Zepto Support Assistant",
    description="Policy-based customer support assistant",
    version="1.0.0",
)


class AskRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {
        "message": "Zepto Support Assistant is running"
    }


@app.post("/ask", response_model=SupportResponse)
def ask(request: AskRequest):

    return ask_question(request.query)