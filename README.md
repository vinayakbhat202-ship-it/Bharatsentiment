# BharatSentiment 🇮🇳
> Hinglish (Hindi+English) Sentiment Analysis for Indian social media

## Live Demo
- 🖥️ App: https://bharatsentiment.streamlit.app
- ⚡ API: https://vinny2005-bharatsentiment.hf.space/docs

## Results
| Model | F1 (macro) |
|---|---|
| VADER baseline | 0.644 |
| TF-IDF + LR | 0.680 |
| **MuRIL fine-tuned** | **0.759** |

## Tech Stack
| Layer | Tool |
|---|---|
| Core Model | MuRIL fine-tuned (google/muril-base-cased) |
| Pseudo-labelling | Groq llama-3.3-70b-versatile |
| RAG Chatbot | ChromaDB + Groq |
| Backend | FastAPI on HuggingFace Spaces |
| Frontend | Streamlit Cloud |
| Database | Supabase PostgreSQL |

## Dataset
- 3,315 Hinglish tweets
- Sources: HuggingFace datasets + Groq pseudo-labelling
- Labels: positive / negative / neutral

## Run Locally
```bash
git clone https://github.com/vinayakbhat202-ship-it/Bharatsentiment
cd Bharatsentiment
source venv/bin/activate
venv/bin/python3 -m uvicorn backend.main:app --reload --reload-exclude venv
streamlit run streamlit_app/app.py
```

## Model
🤗 [vinny2005/bharatsentiment-muril](https://huggingface.co/vinny2005/bharatsentiment-muril)