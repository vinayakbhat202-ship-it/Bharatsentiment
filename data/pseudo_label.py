# data/pseudo_label.py
import os, time, psycopg2, psycopg2.extras
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_connection():
    """Always get a fresh connection."""
    return psycopg2.connect(os.getenv("DATABASE_URL"))

def get_sentiment(text):
    prompt = f"""You are a sentiment classifier for Hinglish (Hindi+English mixed) social media text.
Classify the sentiment as exactly one word: positive, negative, or neutral.
Text: {text}
Reply with only one word."""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0
    )
    label = response.choices[0].message.content.strip().lower()
    if label not in ("positive", "negative", "neutral"):
        label = "neutral"
    return label

def fetch_unlabelled():
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT id, cleaned_text FROM tweets
        WHERE sentiment IS NULL
        AND cleaned_text IS NOT NULL
        AND cleaned_text != ''
        LIMIT 2000
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

def save_label(row_id, label):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE tweets
        SET sentiment = %s, is_labelled = TRUE
        WHERE id = %s
    """, (label, row_id))
    conn.commit()
    cur.close()
    conn.close()

def main():
    rows = fetch_unlabelled()
    print(f"Found {len(rows)} unlabelled rows to process")

    for i, row in enumerate(rows):
        success = False
        attempts = 0
        while not success and attempts < 3:
            try:
                label = get_sentiment(row["cleaned_text"])
                save_label(row["id"], label)
                print(f"  [{i+1}/{len(rows)}] id={row['id']} → {label}")
                success = True
                time.sleep(0.3)  # small delay to avoid rate limits

            except Exception as e:
                err = str(e)
                if "rate_limit" in err.lower() or "429" in err:
                    print(f"  Rate limit hit — waiting 15s...")
                    time.sleep(15)
                else:
                    print(f"  Error on row {row['id']}: {e}")
                    attempts += 1
                    time.sleep(2)

        if not success:
            print(f"  ⚠️  Skipping row {row['id']} after 3 failed attempts")

    print("\n✅ Pseudo-labelling complete!")

if __name__ == "__main__":
    main()