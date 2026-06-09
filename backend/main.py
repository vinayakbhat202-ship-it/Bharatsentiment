# backend/main.py
import os
import requests
import psycopg2
import pandas as pd
import io
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()
app = FastAPI(title="BharatSentiment API")

# ── HuggingFace Inference API (no local model loading) ────
HF_API_URL = "https://api-inference.huggingface.co/models/vinny2005/bharatsentiment-muril"
HF_TOKEN   = os.getenv("HF_TOKEN")
vader      = SentimentIntensityAnalyzer()

def muril_predict(text):
    headers  = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(HF_API_URL, headers=headers, json={"inputs": text}, timeout=30)
    result   = response.json()
    if isinstance(result, list) and len(result) > 0:
        top = max(result[0], key=lambda x: x["score"])
        return {"label": top["label"], "score": round(top["score"], 3)}
    return {"label": "neutral", "score": 0.5}

def get_db():
    return psycopg2.connect(os.getenv("DATABASE_URL"))

# ── Schemas ───────────────────────────────────────────────
class TextInput(BaseModel):
    text: str

class ChatInput(BaseModel):
    question: str
    chat_history: list = []

# ── Endpoints ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "BharatSentiment API running ✅"}

@app.post("/analyse")
def analyse(input: TextInput):
    muril_result = muril_predict(input.text)
    vader_scores = vader.polarity_scores(input.text)
    vader_label  = (
        "positive" if vader_scores["compound"] >= 0.05
        else "negative" if vader_scores["compound"] <= -0.05
        else "neutral"
    )
    return {
        "text":  input.text,
        "muril": {"label": muril_result["label"], "confidence": muril_result["score"]},
        "vader": {"label": vader_label, "compound": round(vader_scores["compound"], 3)}
    }

@app.get("/brand/{name}/trend")
def brand_trend(name: str):
    conn = get_db()
    df   = pd.read_sql("""
        SELECT DATE(created_at) as date, sentiment, COUNT(*) as count
        FROM tweets
        WHERE LOWER(text) LIKE %s
        AND sentiment IS NOT NULL
        GROUP BY DATE(created_at), sentiment
        ORDER BY date
    """, conn, params=(f"%{name.lower()}%",))
    conn.close()
    return df.to_dict(orient="records")

@app.post("/bulk-analyse")
async def bulk_analyse(file: UploadFile = File(...)):
    contents = await file.read()
    df       = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    if "text" not in df.columns:
        return {"error": "CSV must have a 'text' column"}
    results = []
    for text in df["text"].dropna().head(50):
        result = muril_predict(str(text))
        results.append({
            "text":       text,
            "label":      result["label"],
            "confidence": result["score"]
        })
    return {"results": results, "total": len(results)}

@app.get("/stats")
def stats():
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM tweets")
    total = cur.fetchone()[0]
    cur.execute("""
        SELECT sentiment, COUNT(*) 
        FROM tweets 
        WHERE sentiment IS NOT NULL 
        GROUP BY sentiment
    """)
    breakdown = dict(cur.fetchall())
    conn.close()
    return {
        "total_tweets":        total,
        "sentiment_breakdown": breakdown
    }

@app.post("/chat")
def chat(input: ChatInput):
    from backend.rag import rag_chat
    result = rag_chat(input.question, input.chat_history)
    return result