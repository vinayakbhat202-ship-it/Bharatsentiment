# models/baseline.py
import pandas as pd
import psycopg2
import os
import joblib
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()

def load_data():
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    df = pd.read_sql("""
        SELECT cleaned_text, sentiment FROM tweets
        WHERE is_labelled = TRUE
        AND cleaned_text IS NOT NULL
        AND sentiment IN ('positive', 'negative', 'neutral')
    """, conn)
    conn.close()
    print(f"Loaded {len(df)} labelled rows")
    print(df['sentiment'].value_counts())
    return df

def train_baseline(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], df['sentiment'],
        test_size=0.2, random_state=42, stratify=df['sentiment']
    )

    # Character n-grams — handles transliteration variation
    vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(2, 5),
        max_features=50000,
        sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(X_train_vec, y_train)
    preds = lr.predict(X_test_vec)

    baseline_f1 = f1_score(y_test, preds, average='macro')
    print(f"\n📊 TF-IDF + LR Baseline F1: {baseline_f1:.3f}")
    print(classification_report(y_test, preds))

    # VADER comparison
    analyzer = SentimentIntensityAnalyzer()
    def vader_predict(text):
        score = analyzer.polarity_scores(str(text))['compound']
        if score > 0.05: return 'positive'
        elif score < -0.05: return 'negative'
        return 'neutral'

    vader_preds = [vader_predict(t) for t in X_test]
    vader_f1 = f1_score(y_test, vader_preds, average='macro')
    print(f"📊 VADER F1 on same test set: {vader_f1:.3f}")
    print(f"\n🏆 Improvement over VADER: +{(baseline_f1 - vader_f1):.3f}")

    # Save model
    os.makedirs('models/saved', exist_ok=True)
    joblib.dump(vectorizer, 'models/saved/tfidf_vectorizer.pkl')
    joblib.dump(lr, 'models/saved/baseline_lr.pkl')

    # Save test set for MuRIL comparison later
    test_df = pd.DataFrame({'text': X_test, 'true_label': y_test})
    test_df.to_csv('data/processed/test_set.csv', index=False)
    print(f"\n✅ Model saved to models/saved/")
    print(f"✅ Test set saved to data/processed/test_set.csv")

    return baseline_f1, vader_f1

if __name__ == "__main__":
    df = load_data()
    train_baseline(df)