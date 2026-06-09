# data/fetch_huggingface.py
import pandas as pd
from datasets import load_dataset
import psycopg2
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

def get_connection():
    return psycopg2.connect(os.getenv("DATABASE_URL"))

def fetch_and_store():
    conn = get_connection()
    cur = conn.cursor()
    all_rows = []

    # Dataset 1 — Real Hinglish tweets, labelled (SemEval 2020)
    print("Loading dataset 1: Abhishek4896/hindi-english-code-mixed-tweets-sentiment...")
    try:
        ds1 = load_dataset("Abhishek4896/hindi-english-code-mixed-tweets-sentiment", split="train")
        df1 = pd.DataFrame(ds1)
        print(f"  Columns: {df1.columns.tolist()}")
        print(f"  Shape: {df1.shape}")
        print(df1.head(3))

        label_map = {0: "negative", 1: "neutral", 2: "positive",
                     "0": "negative", "1": "neutral", "2": "positive",
                     "negative": "negative", "positive": "positive", "neutral": "neutral"}

        text_col = [c for c in df1.columns if "text" in c.lower() or "tweet" in c.lower() or "sentence" in c.lower()]
        label_col = [c for c in df1.columns if "label" in c.lower() or "sentiment" in c.lower()]

        text_col = text_col[0] if text_col else df1.columns[0]
        label_col = label_col[0] if label_col else df1.columns[1]

        for _, row in df1.iterrows():
            sentiment = label_map.get(str(row[label_col]).lower(), "neutral")
            all_rows.append({
                "text": str(row[text_col]),
                "platform": "twitter",
                "source": "semeval2020-hinglish",
                "sentiment": sentiment,
                "is_labelled": True
            })
        print(f"  ✅ Added {len(df1)} rows")
    except Exception as e:
        print(f"  ❌ Failed: {e}")

    # Dataset 2 — Large multiclass sentiment (English + some Hinglish)
    print("\nLoading dataset 2: Sp1786/multiclass-sentiment-analysis-dataset...")
    try:
        ds2 = load_dataset("Sp1786/multiclass-sentiment-analysis-dataset", split="train")
        df2 = pd.DataFrame(ds2)
        print(f"  Columns: {df2.columns.tolist()}")
        print(f"  Shape: {df2.shape}")
        print(df2.head(3))

        text_col = [c for c in df2.columns if "text" in c.lower() or "sentence" in c.lower()]
        label_col = [c for c in df2.columns if "label" in c.lower() or "sentiment" in c.lower()]
        text_col = text_col[0] if text_col else df2.columns[0]
        label_col = label_col[0] if label_col else df2.columns[1]

        label_map2 = {
            0: "negative", 1: "neutral", 2: "positive",
            "0": "negative", "1": "neutral", "2": "positive",
            "negative": "negative", "positive": "positive", "neutral": "neutral"
        }

        hindi_markers = ['hai', 'nahi', 'kya', 'yaar', 'bhai', 'acha', 'bahut',
                         'bohot', 'mast', 'bakwaas', 'sahi', 'nhi', 'tha', 'thi',
                         'hoga', 'kar', 'raha', 'wala', 'bilkul', 'ekdum']
        count = 0
        for _, row in df2.iterrows():
            text = str(row[text_col])
            has_hindi = any(m in text.lower() for m in hindi_markers)
            if has_hindi:
                sentiment = label_map2.get(row[label_col], "neutral")
                all_rows.append({
                    "text": text,
                    "platform": "social_media",
                    "source": "multiclass-sentiment",
                    "sentiment": str(sentiment),
                    "is_labelled": True
                })
                count += 1
            if count >= 2000:
                break
        print(f"  ✅ Added {count} Hinglish rows")
    except Exception as e:
        print(f"  ❌ Failed: {e}")

    # Dataset 3 — Already loaded Hinglish conversations (unlabelled)
    # These are already in DB from previous run, skip to avoid duplicates

    # Insert into PostgreSQL
    print(f"\nInserting {len(all_rows)} labelled rows into database...")
    inserted = 0
    for row in all_rows:
        try:
            cur.execute("""
                INSERT INTO tweets (text, platform, source, sentiment, is_labelled, scraped_at)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                row["text"],
                row["platform"],
                row["source"],
                row.get("sentiment"),
                row["is_labelled"],
                datetime.now()
            ))
            inserted += 1
        except Exception as e:
            pass

    conn.commit()
    cur.close()
    conn.close()

    print(f"\n✅ Done! Inserted {inserted} labelled rows.")
    df_all = pd.DataFrame(all_rows)
    if len(df_all) > 0 and "sentiment" in df_all.columns:
        print("\nSentiment distribution:")
        print(df_all["sentiment"].value_counts())

if __name__ == "__main__":
    fetch_and_store()

