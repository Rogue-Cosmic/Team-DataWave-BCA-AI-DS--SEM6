import flask
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import scipy.sparse as sp
import re
import time
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load AI components
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    model = pickle.load(open("isolation_forest_yelp.pkl", "rb"))
except Exception as e:
    print(f"Error loading models: {e}")

def explain_anomaly(review_text, decision_score, sentiment_score):
    reasons = []
    word_count = len(review_text.split())
    if word_count < 10: reasons.append("Unusually short review length, a common trait of bot-generated spam.")
    elif word_count > 300: reasons.append("Unusually long review length compared to typical bounds.")
    if sentiment_score > 0.9: reasons.append(f"Extreme positive polarization (Score: {sentiment_score:.2f}). Fake reviews are often wildly exaggerated.")
    elif sentiment_score < -0.9: reasons.append(f"Extreme negative polarization (Score: {sentiment_score:.2f}). Fake reviews often show heavy exaggeration.")
    upper_chars = sum(1 for c in review_text if c.isupper())
    if len(review_text) > 0 and (upper_chars / len(review_text)) > 0.2: reasons.append("Excessive use of capital letters (>20%), common in promotional/spam content.")
    if review_text.count('!') > 4: reasons.append("Excessive use of exclamation marks detected.")
    if not reasons: reasons.append("Subtle statistical mismatch in terminology or structural density against normal baseline.")
    return reasons

def get_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    return sid.polarity_scores(text)['compound']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/analyze/text', methods=['POST'])
def analyze_text():
    data = request.json
    review = data.get("review", "")
    if not review: return jsonify({"error": "No review provided"}), 400
    
    vec = vectorizer.transform([review])
    review_length = [[len(review.split())]]
    X_live = sp.hstack([vec, review_length, [[3]]])
    prediction = int(model.predict(X_live)[0])
    d_score = float(model.decision_function(X_live)[0])
    s_score = get_sentiment(review)
    
    reasons = []
    if prediction == -1:
        reasons = explain_anomaly(review, d_score, s_score)
        
    return jsonify({
        "prediction": prediction,
        "is_fake": prediction == -1,
        "sentiment_score": s_score,
        "anomaly_score": d_score,
        "reasons": reasons
    })

@app.route('/api/analyze/url', methods=['POST'])
def analyze_url():
    data = request.json
    url = data.get("url", "")
    if "yelp.com" not in url.lower():
        return jsonify({"error": "Invalid Yelp URL"}), 400

    try:
        chrome_options = Options()
        chrome_options.add_argument("--start-maximized")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option("useAutomationExtension", False)

        driver = webdriver.Chrome(options=chrome_options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        driver.get(url)
        wait = WebDriverWait(driver, 20)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "p")))

        for _ in range(5):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)

        elements = driver.find_elements(By.TAG_NAME, "p")
        raw_reviews = [el.text.strip() for el in elements if len(el.text.strip()) > 80]
        driver.quit()
        
        if not raw_reviews:
            return jsonify({"error": "No reviews extracted."}), 404
            
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        
        results = []
        clean_docs = []
        for r in raw_reviews:
            text = r.lower()
            text = re.sub(r"http\S+", "", text)
            text = re.sub(r"[^a-z\s]", "", text)
            tokens = text.split()
            cleaned_tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
            clean_docs.append(" ".join(cleaned_tokens))
            
        X_text = vectorizer.transform(clean_docs)
        review_lengths = [[len(r.split())] for r in clean_docs]
        rating_placeholders = [[3] for _ in clean_docs]
        X_live = sp.hstack([X_text, review_lengths, rating_placeholders])
        
        predictions = model.predict(X_live)
        d_scores = model.decision_function(X_live)
        
        suspicious_count = 0
        for idx, (review, pred, d_score) in enumerate(zip(raw_reviews, predictions, d_scores)):
            s_score = get_sentiment(review)
            is_fake = pred == -1
            if is_fake: suspicious_count += 1
            reasons = explain_anomaly(review, d_score, s_score) if is_fake else []
            results.append({
                "text": review,
                "is_fake": bool(is_fake),
                "sentiment_score": float(s_score),
                "anomaly_score": float(d_score),
                "reasons": reasons
            })
            
        susp_percentage = (suspicious_count / len(results)) * 100 if results else 0
        
        return jsonify({
            "total_reviews": len(results),
            "suspicious_percentage": susp_percentage,
            "reviews": results
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/insights', methods=['GET'])
def get_insights():
    try:
        df = pd.read_csv("yelp_reviews_clean.csv")
        avg_rating = float(df['rating'].mean())
        avg_words = float(df['review_length'].mean())
        rating_counts = df['rating'].value_counts().sort_index().to_dict()
        
        return jsonify({
            "total_reviews": len(df),
            "average_rating": avg_rating,
            "average_word_count": avg_words,
            "rating_distribution": rating_counts
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
