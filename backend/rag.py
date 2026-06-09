import os
import psycopg2
from dotenv import load_dotenv
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

# Instantiate shared resource singletons
encoder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
chroma_client = PersistentClient(path="./backend/chroma_db")
collection = chroma_client.get_or_create_collection(name="hinglish_tweets")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def index_tweets():
    """Pulls rows from Supabase PostgreSQL and updates ChromaDB if empty."""
    # Optimization: Prevent re-embedding data arrays on hot-reloads if already indexed
    if collection.count() > 0:
        print(f"⏩ ChromaDB already populated ({collection.count()} vectors). Skipping pipeline ingestion.")
        return

    print("🔌 Extracting corpus entries from Supabase PostgreSQL cluster...")
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cursor = conn.cursor()
    cursor.execute("SELECT id::text, text, sentiment FROM tweets WHERE text IS NOT NULL;")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("⚠️ Supabase dataset returned 0 valid text nodes.")
        return

    print(f"🧬 Generating text embeddings for {len(rows)} nodes...")
    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    metadatas = [{"sentiment": r[2] if r[2] else "neutral"} for r in rows]
    
    # Batch compute embeddings efficiently
    embeddings = encoder.encode(texts, batch_size=64, show_progress_bar=True).tolist()

    print("💾 Committing vectors to local storage blocks...")
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )
    print("🏁 Vector Store synchronization complete.")

def rag_chat(question: str, chat_history: list) -> dict:
    """Queries context metrics and synthesizes response generation via Groq."""
    # 1. Similarity Vector Lookup
    query_vector = encoder.encode(question).tolist()
    results = collection.query(query_embeddings=[query_vector], n_results=5)
    
    # 2. Context Aggregation
    retrieved_tweets = []
    context_blocks = []
    if results["documents"] and results["documents"][0]:
        for i in range(len(results["documents"][0])):
            text_content = results["documents"][0][i]
            sentiment_tag = results["metadatas"][0][i].get("sentiment", "neutral")
            
            retrieved_tweets.append({
                "text": text_content,
                "sentiment": sentiment_tag,
                "distance": round(float(results["distances"][0][i]), 4) if results["distances"] else 0.0
            })
            context_blocks.append(f"[{sentiment_tag.upper()}] {text_content}")

    context_str = "\n".join(context_blocks)

    # 3. Prompt Construction
    system_prompt = (
        "You are an AI core engine analyzing customer sentiment data patterns. "
        "Review the retrieved context chunks from Hinglish user feeds to answer the user inquiry. "
        "Be direct, objective, and reference explicit sample instances if useful.\n\n"
        f"Retrieved Social Context Matrix:\n{context_str}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    for turn in chat_history[-6:]:  # Slice context window to balance buffer constraints
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": question})

    # 4. Compute Generation via Groq API
    completion = groq_client.chat.completions.create(
        messages=messages,
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=350
    )

    return {
        "answer": completion.choices[0].message.content,
        "relevant_tweets": retrieved_tweets
    }