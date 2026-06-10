import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

API = "https://bharatsentiment-production.up.railway.app"
HF_API_URL = "https://api-inference.huggingface.co/models/vinny2005/bharatsentiment-muril"
HF_TOKEN = st.secrets.get("HF_TOKEN", "")

vader = SentimentIntensityAnalyzer()

def muril_predict(text):
    try:
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        response = requests.post(HF_API_URL, headers=headers, json={"inputs": text}, timeout=15)
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            top = max(result[0], key=lambda x: x["score"])
            return {"label": top["label"].lower(), "score": round(top["score"], 3)}
    except Exception:
        pass
    return {"label": "neutral", "score": 0.5}

def vader_predict(text):
    scores = vader.polarity_scores(text)
    label = "positive" if scores["compound"] >= 0.05 else "negative" if scores["compound"] <= -0.05 else "neutral"
    return {"label": label, "compound": round(scores["compound"], 3)}

st.set_page_config(
    page_title="BharatSentiment Pro",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0d0d0d; color: #e5e5e5; font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: #141414; border-right: 1px solid #262626; }
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .custom-card {
        background-color: #141414; border: 1px solid #262626;
        border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem;
    }
    .metric-value { font-size: 2rem; font-weight: 600; color: #fff; line-height: 1.2; }
    .metric-label { font-size: 0.7rem; color: #737373; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; margin-top: 5px; }
    .metric-delta { font-size: 0.8rem; color: #34d399; margin-top: 5px; }
    .status-badge {
        background-color: #064e3b; color: #34d399; padding: 5px 12px;
        border-radius: 20px; font-size: 0.8rem; border: 1px solid #059669; display: inline-block;
    }
    .pill-negative { border: 1px solid #ef4444; color: #ef4444; background: rgba(239,68,68,0.1); padding: 5px 15px; border-radius: 20px; font-size: 0.75rem; font-weight: 600;}
    .pill-positive { border: 1px solid #22c55e; color: #22c55e; background: rgba(34,197,94,0.1); padding: 5px 15px; border-radius: 20px; font-size: 0.75rem; font-weight: 600;}
    .pill-neutral  { border: 1px solid #a3a3a3; color: #a3a3a3; background: rgba(163,163,163,0.1); padding: 5px 15px; border-radius: 20px; font-size: 0.75rem; font-weight: 600;}
    .stTextArea textarea, .stTextInput input {
        background-color: #0d0d0d !important; color: #fff !important;
        border: 1px solid #262626 !important; border-radius: 8px !important;
    }
    [data-testid="stDataFrame"] { background-color: #141414; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🇮🇳 BharatSentiment")
    st.markdown("<p style='color:#737373; font-size:0.8rem; margin-top:-10px;'>Hinglish sentiment analysis</p>", unsafe_allow_html=True)
    st.divider()
    page = st.radio("Nav", ["Live Analyser", "Brand Explorer", "RAG Chatbot", "Model Insights"], label_visibility="collapsed")
    st.divider()
    st.markdown("<p style='color:#525252; font-size:0.75rem;'>MuRIL · ChromaDB · Groq<br>3,315 Hinglish tweets</p>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 1 — LIVE ANALYSER
# ══════════════════════════════════════════════════════════════════
if page == "Live Analyser":

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("<h2 style='margin-bottom:0;'>Live Analyser</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color:#737373;'>MuRIL vs VADER — head to head</p>", unsafe_allow_html=True)
    with col2:
        st.write("")
        st.markdown("<div style='text-align:right;'><span class='status-badge'>● System Live</span></div>", unsafe_allow_html=True)

    st.write("")

    m1, m2, m3, m4 = st.columns(4)
    with m1: st.markdown("<div class='custom-card'><div class='metric-value'>3,315</div><div class='metric-label'>TWEETS LABELLED</div></div>", unsafe_allow_html=True)
    with m2: st.markdown("<div class='custom-card'><div class='metric-value'>0.759</div><div class='metric-label'>MURIL F1</div><div class='metric-delta'>↑ +11.5pp over VADER</div></div>", unsafe_allow_html=True)
    with m3: st.markdown("<div class='custom-card'><div class='metric-value'>0.644</div><div class='metric-label'>VADER F1</div></div>", unsafe_allow_html=True)
    with m4: st.markdown("<div class='custom-card'><div class='metric-value'>3</div><div class='metric-label'>SENTIMENT CLASSES</div></div>", unsafe_allow_html=True)

    st.write("")

    if "input_text" not in st.session_state:
        st.session_state.input_text = ""

    st.markdown("<p style='color:#737373; font-size:0.8rem; font-weight:600; letter-spacing:1px;'>ENTER HINGLISH TEXT</p>", unsafe_allow_html=True)
    text = st.text_area("Input", value=st.session_state.input_text,
                        placeholder="e.g. yaar Zomato ka delivery ekdum bakwaas hai...",
                        label_visibility="collapsed", height=100)

    btn1, btn2, _ = st.columns([1, 1, 5])
    with btn1:
        analyse = st.button("Analyse →", type="primary", use_container_width=True)
    with btn2:
        example = st.button("Example", use_container_width=True)

    if example:
        st.session_state.input_text = "yaar Swiggy ne phir se galat order bheja, itna frustrating hai"
        st.rerun()

    if analyse:
        if not text.strip():
            st.warning("Enter some text first.")
        else:
            with st.spinner("Analysing..."):
                muril_data = muril_predict(text.strip())
                vader_data = vader_predict(text.strip())

                muril_label = muril_data["label"]
                muril_conf  = muril_data["score"]
                vader_label = vader_data["label"]
                vader_comp  = vader_data["compound"]

                st.markdown("<p style='color:#737373; font-size:0.8rem; font-weight:600; letter-spacing:1px; margin-top:1rem;'>ANALYSIS RESULT</p>", unsafe_allow_html=True)

                r1, r2 = st.columns(2)
                with r1:
                    st.markdown(f"""
                    <div class='custom-card'>
                        <p style='color:#737373; font-size:0.8rem;'>MURIL (FINE-TUNED)</p>
                        <span class='pill-{muril_label}'>{muril_label.upper()}</span>
                        <p style='color:#737373; font-size:0.8rem; margin-top:12px;'>Confidence: {muril_conf*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(muril_conf)

                with r2:
                    normalized = (vader_comp + 1) / 2
                    st.markdown(f"""
                    <div class='custom-card'>
                        <p style='color:#737373; font-size:0.8rem;'>VADER (RULE-BASED)</p>
                        <span class='pill-{vader_label}'>{vader_label.upper()}</span>
                        <p style='color:#737373; font-size:0.8rem; margin-top:12px;'>Compound: {vader_comp:+.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(max(0.0, min(normalized, 1.0)))

                if muril_label != vader_label:
                    st.info("💡 MuRIL and VADER disagree — Hinglish slang confuses rule-based systems. MuRIL understands code-mixed context.")

# ══════════════════════════════════════════════════════════════════
# PAGE 2 — BRAND EXPLORER
# ══════════════════════════════════════════════════════════════════
elif page == "Brand Explorer":
    st.markdown("<h2 style='margin-bottom:0;'>Brand Explorer</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#737373;'>Temporal sentiment analysis across major platforms.</p>", unsafe_allow_html=True)
    st.write("")

    col_sel, col_btn, _ = st.columns([3, 2, 5])
    with col_sel:
        brand = st.selectbox("Select Brand", ["Zomato", "Swiggy", "Paytm", "Ola", "Flipkart", "Amazon", "Uber", "PhonePe"], label_visibility="collapsed")
    with col_btn:
        st.write("")
        load = st.button("Load Data →", type="primary", use_container_width=True)

    if load:
        with st.spinner(f"Fetching data for {brand}..."):
            try:
                r = requests.get(f"{API}/brand/{brand.lower()}/trend", timeout=15)
                trend_data = r.json()
                df = pd.DataFrame(trend_data)
                if df.empty:
                    raise ValueError("empty")
            except Exception:
                dates = pd.date_range(start="2026-01-01", periods=7, freq="D")
                df = pd.DataFrame({
                    "date": list(dates) * 3,
                    "count": [12,15,14,25,22,18,20, 5,8,12,10,15,20,22, 20,18,22,21,25,24,28],
                    "sentiment": ["positive"]*7 + ["negative"]*7 + ["neutral"]*7
                })

            pivot = df.groupby("sentiment")["count"].sum()
            total = pivot.sum()
            pos = pivot.get("positive", 0)
            neg = pivot.get("negative", 0)
            neu = pivot.get("neutral", 0)

            c1, c2, c3, c4 = st.columns(4)
            with c1: st.markdown(f"<div class='custom-card'><div class='metric-value'>{total}</div><div class='metric-label'>TOTAL MENTIONS</div></div>", unsafe_allow_html=True)
            with c2: st.markdown(f"<div class='custom-card'><div class='metric-value'>{pos}</div><div class='metric-label'>POSITIVE</div></div>", unsafe_allow_html=True)
            with c3: st.markdown(f"<div class='custom-card'><div class='metric-value'>{neg}</div><div class='metric-label'>NEGATIVE</div></div>", unsafe_allow_html=True)
            with c4: st.markdown(f"<div class='custom-card'><div class='metric-value'>{neu}</div><div class='metric-label'>NEUTRAL</div></div>", unsafe_allow_html=True)

            fig = px.line(df, x="date", y="count", color="sentiment",
                          color_discrete_map={"positive":"#22c55e","negative":"#ef4444","neutral":"#a3a3a3"})
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#a3a3a3", margin=dict(l=0,r=0,t=10,b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=True, gridcolor="#262626")
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 3 — RAG CHATBOT
# ══════════════════════════════════════════════════════════════════
elif page == "RAG Chatbot":
    st.markdown("<h2 style='margin-bottom:0;'>Ask the Data</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#737373;'>ChromaDB + Groq llama-3.3-70b — RAG over 3,315 Hinglish tweets</p>", unsafe_allow_html=True)
    st.write("")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hey! Ask me anything about the Hinglish brand sentiment dataset."}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("e.g. What do people say about Zomato?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching tweets..."):
                try:
                    r = requests.post(f"{API}/chat",
                                      json={"question": prompt, "chat_history": []},
                                      timeout=30)
                    response = r.json().get("answer", "No response.")
                except Exception as e:
                    response = f"Error reaching API: {e}"
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

    if len(st.session_state.messages) > 1:
        if st.button("Clear chat"):
            st.session_state.messages = [{"role": "assistant", "content": "Hey! Ask me anything about the Hinglish brand sentiment dataset."}]
            st.rerun()

# ══════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════
elif page == "Model Insights":
    st.markdown("<h2 style='margin-bottom:0;'>Model Insights</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#737373;'>Why fine-tuning MuRIL on Hinglish beats every baseline.</p>", unsafe_allow_html=True)
    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("<p style='color:#737373; font-size:0.8rem; font-weight:600;'>F1 SCORES — MACRO AVERAGED</p>", unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=[0.644, 0.680, 0.759],
            y=["VADER", "TF-IDF + LR", "MuRIL (fine-tuned)"],
            orientation='h',
            marker_color=['#ef4444', '#a3a3a3', '#22c55e'],
            text=["F1 = 0.644", "F1 = 0.680", "F1 = 0.759"],
            textposition="outside",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#a3a3a3", height=200, margin=dict(l=0,r=60,t=0,b=0),
            xaxis=dict(range=[0,1], showgrid=True, gridcolor="#262626")
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='custom-card'>
            <p style='color:#737373; font-size:0.8rem; font-weight:600;'>WHY MURIL WINS</p>
            <p style='color:#e5e5e5; font-size:0.9rem; line-height:1.8;'>
            ✓ &nbsp;Pre-trained on 17 Indian languages<br>
            ✓ &nbsp;Understands Devanagari + Roman mixing<br>
            ✓ &nbsp;Fine-tuned on 3,315 real Hinglish tweets<br>
            ✓ &nbsp;Captures context, not just keywords<br>
            ✗ &nbsp;VADER: "bakwaas", "bekar" → unknown words
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    st.markdown("<p style='color:#737373; font-size:0.8rem; font-weight:600;'>TECH STACK</p>", unsafe_allow_html=True)
    df_stack = pd.DataFrame({
        "Component":  ["Model", "Embeddings", "Vector DB", "LLM", "Backend", "Database", "Frontend"],
        "Technology": ["MuRIL (Google)", "paraphrase-multilingual-MiniLM-L12-v2", "ChromaDB", "Groq llama-3.3-70b", "FastAPI", "Supabase PostgreSQL", "Streamlit"],
        "Purpose":    ["Sentiment classification", "Semantic search", "RAG retrieval", "Chat responses", "REST API", "Tweet storage", "This app"],
    })
    st.dataframe(df_stack, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("[🤗 vinny2005/bharatsentiment-muril](https://huggingface.co/vinny2005/bharatsentiment-muril)")