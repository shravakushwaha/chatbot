from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from RAG.rag import RAG_chatbot
import os

bot = RAG_chatbot()
app = FastAPI()

class UserQuery(BaseModel):
    user_query: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # Load the HTML template from the templates folder
    with open("templates/index.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/chat")
async def chat_endpoint(payload: UserQuery = Body(...)):
    try:
        response = await bot.get_response(payload.user_query)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
