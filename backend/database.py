# backend/database.py
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    return psycopg2.connect(os.getenv("DATABASE_URL"))

def create_tables():
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tweets (
            id SERIAL PRIMARY KEY,
            text TEXT NOT NULL,
            cleaned_text TEXT,
            platform VARCHAR(20),
            source VARCHAR(100),
            created_at TIMESTAMP,
            scraped_at TIMESTAMP DEFAULT NOW(),
            sentiment VARCHAR(20),
            confidence FLOAT,
            code_mix_ratio FLOAT,
            is_labelled BOOLEAN DEFAULT FALSE
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS brand_mentions (
            id SERIAL PRIMARY KEY,
            tweet_id INTEGER REFERENCES tweets(id),
            brand VARCHAR(100),
            date DATE,
            sentiment VARCHAR(20),
            confidence FLOAT
        );
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_brand_date 
        ON brand_mentions(brand, date);
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_sentiment 
        ON tweets(sentiment);
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_created 
        ON tweets(created_at);
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("Tables created successfully!")

if __name__ == "__main__":
    create_tables()
    