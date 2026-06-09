
import os
import io
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
 
from backend.rag import rag_chat, index_tweets
 
load_dotenv()
 
app = FastAPI(title="BharatSentiment API")
 
# ── LOAD MODELS ───────────────────────────────────────────
classifier = pipeline(
    "text-classification",
    model="vinny2005/bharatsentiment-muril",
    tokenizer="vinny2005/bharatsentiment-muril"
)
vader = SentimentIntensityAnalyzer()
 
def get_db():
    return psycopg2.connect(os.getenv("DATABASE_URL"))
 
# ── DATA SCHEMAS ──────────────────────────────────────────
class TextInput(BaseModel):
    text: str
 
class ChatInput(BaseModel):
    question: str
    chat_history: list = []
 
# ── LIFECYCLE EVENTS ──────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    print("🔥 Warming up MuRIL model...")
    classifier("warmup")
    print("📦 Initializing Vector Store sync from Supabase...")
    index_tweets()
    print("✅ System Ready & Connected!")
 
# ── ENDPOINTS ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "BharatSentiment API running ✅"}
 
@app.post("/analyse")
def analyse(input: TextInput):
    muril_result = classifier(input.text)[0]
    vader_scores = vader.polarity_scores(input.text)
    vader_label = (
        "positive" if vader_scores["compound"] >= 0.05
        else "negative" if vader_scores["compound"] <= -0.05
        else "neutral"
    )
    return {
        "text": input.text,
        "muril": {
            "label": muril_result["label"],
            "confidence": round(muril_result["score"], 3)
        },
        "vader": {
            "label": vader_label,
            "compound": round(vader_scores["compound"], 3)
        }
    }
 
@app.get("/brand/{name}/trend")
def brand_trend(name: str):
    conn = get_db()
    df = pd.read_sql("""
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
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    if "text" not in df.columns:
        return {"error": "CSV must have a 'text' column"}
    results = []
    for text in df["text"].dropna().head(100):
        result = classifier(str(text))[0]
        results.append({
            "text": text,
            "label": result["label"],
            "confidence": round(result["score"], 3)
        })
    return {"results": results, "total": len(results)}
 
@app.get("/stats")
def stats():
    conn = get_db()
    cur = conn.cursor()
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
        "total_tweets": total,
        "sentiment_breakdown": breakdown
    }
 
@app.post("/chat")
def chat(input: ChatInput):
    return rag_chat(input.question, input.chat_history)
