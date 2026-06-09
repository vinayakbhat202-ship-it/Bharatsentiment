# data/preprocess.py
import re
import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

# Common Hinglish/Hindi words for code-mix ratio detection
HINDI_WORDS = {
    'bakwaas','bekar','mast','zabardast','yaar','bhai','ekdum','bilkul',
    'nahi','nhi','haan','theek','accha','sahi','faltu','kya','hai','tha',
    'thi','ho','kar','raha','rahi','wala','wali','bohot','bahut','thoda',
    'zyada','abhi','phir','matlab','sach','acha','hoga','karega','karenge',
    'karo','karna','dono','sabse','kitna','kyun','isliye','lekin','aur',
    'paisa','vasool','khushi','aansu','match','coffee','thandi','taste',
    'exam','clear','full','kal','yeh','ye','toh','to','bhi','se','pe',
    'mein','mai','ne','ko','ka','ki','ke','log','kuch','sab','bas','ab'
}

EMOJI_MAP = {
    '🔥': 'ekdum mast',
    '💯': 'bilkul sahi',
    '😂': 'bahut funny',
    '😡': 'bohot gussa',
    '👎': 'bilkul bekar',
    '🙄': 'seriously kya',
    '❤️': 'bohot pasand',
    '😍': 'ekdum zabardast',
    '😢': 'bahut dukh',
    '👍': 'bahut accha',
    '😤': 'bohot gussa',
    '🤮': 'ekdum bekar',
}

def compute_code_mix_ratio(text):
    """What fraction of words are Hindi/Hinglish?"""
    words = text.lower().split()
    if not words:
        return 0.0
    hindi_count = sum(1 for w in words if w in HINDI_WORDS)
    return round(hindi_count / len(words), 3)

def clean_text(text):
    """Full Hinglish-aware cleaning pipeline"""
    if not isinstance(text, str):
        return ""
    # 1. Replace emojis with Hindi equivalents
    for emoji, replacement in EMOJI_MAP.items():
        text = text.replace(emoji, f' {replacement} ')
    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # 3. Remove @mentions
    text = re.sub(r'@\w+', '', text)
    # 4. Remove hashtag symbol but keep the word
    text = re.sub(r'#(\w+)', r'\1', text)
    # 5. Remove special characters but keep Hindi-relevant punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    # 6. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # 7. Lowercase
    text = text.lower()
    return text

def preprocess_all():
    """Clean all tweets in DB and update cleaned_text + code_mix_ratio"""
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()

    # Fetch all unprocessed tweets
    cur.execute("SELECT id, text FROM tweets WHERE cleaned_text IS NULL")
    rows = cur.fetchall()
    print(f"Processing {len(rows)} tweets...")

    updated = 0
    skipped = 0
    for tweet_id, text in rows:
        cleaned = clean_text(text)
        if len(cleaned) < 5:  # skip very short texts
            skipped += 1
            continue
        ratio = compute_code_mix_ratio(cleaned)
        cur.execute("""
            UPDATE tweets
            SET cleaned_text = %s, code_mix_ratio = %s
            WHERE id = %s
        """, (cleaned, ratio, tweet_id))
        updated += 1

    conn.commit()
    cur.close()
    conn.close()

    print(f"✅ Updated {updated} rows")
    print(f"⏭️  Skipped {skipped} rows (too short)")

    # Show stats
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()
    cur.execute("""
        SELECT
            COUNT(*) as total,
            ROUND(AVG(code_mix_ratio)::numeric, 3) as avg_hindi_ratio,
            COUNT(CASE WHEN is_labelled THEN 1 END) as labelled,
            COUNT(CASE WHEN NOT is_labelled THEN 1 END) as unlabelled
        FROM tweets
        WHERE cleaned_text IS NOT NULL
    """)
    stats = cur.fetchone()
    cur.close()
    conn.close()

    print(f"\n📊 Dataset stats:")
    print(f"   Total processed : {stats[0]}")
    print(f"   Avg Hindi ratio : {stats[1]}")
    print(f"   Labelled        : {stats[2]}")
    print(f"   Unlabelled      : {stats[3]}")

if __name__ == "__main__":
    preprocess_all()
    