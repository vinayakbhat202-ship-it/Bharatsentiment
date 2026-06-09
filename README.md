# BharatSentiment 🇮🇳

> Hinglish (Hindi+English) Sentiment Analysis for Indian social media

## What it does
Analyses sentiment of code-mixed Hinglish text from Twitter/social media.
Detects **positive / negative / neutral** sentiment using a fine-tuned MuRIL model.

## Tech Stack
| Layer | Tool |
|---|---|
| Core Model | MuRIL (google/muril-base-cased) fine-tuned |
| Pseudo-labelling | Groq llama-3.3-70b-versatile |
| RAG Chatbot | ChromaDB + Groq |
| Backend | FastAPI (Railway) |
| Frontend | Streamlit Cloud |
| Database | Supabase PostgreSQL |
| Analytics | Power BI Desktop |

## Dataset
- 4,498 Hinglish tweets
- Sources: HuggingFace datasets + pseudo-labelled with LLM
- Labels: positive / negative / neutral

## Results
| Model | Accuracy | F1 (macro) |
|---|---|---|
| VADER (baseline) | 64% | 0.644 |
| TF-IDF + LR (baseline) | 69% | 0.680 |
| MuRIL fine-tuned | ~76% | 0.759 |

*(fill after training)*

## Run Locally
```bash
git clone https://github.com/YOUR_USERNAME/bharatsentiment
cd bharatsentiment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your keys
uvicorn backend.main:app --reload
streamlit run streamlit_app/app.py
```

## Live Demo
- API: https://bharatsentiment.up.railway.app
- Frontend: https://bharatsentiment.streamlit.app