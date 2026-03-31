import streamlit as st
import pickle
import numpy as np
import scipy.sparse as sp
import re
import time
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("yelp_reviews_clean.csv")
        return df
    except Exception as e:
        return None

def explain_anomaly(review_text, decision_score, sentiment_score):
    reasons = []
    word_count = len(review_text.split())
    
    if word_count < 10:
        reasons.append("Unusually short review length, a common trait of bot-generated spam.")
    elif word_count > 300:
        reasons.append("Unusually long review length compared to typical bounds.")
        
    if sentiment_score > 0.9:
        reasons.append(f"Extreme positive polarization (Score: {sentiment_score:.2f}). Fake reviews are often wildly exaggerated.")
    elif sentiment_score < -0.9:
        reasons.append(f"Extreme negative polarization (Score: {sentiment_score:.2f}). Fake reviews often show heavy exaggeration.")
        
    upper_chars = sum(1 for c in review_text if c.isupper())
    if len(review_text) > 0 and (upper_chars / len(review_text)) > 0.2:
        reasons.append("Excessive use of capital letters (>20%), common in promotional/spam content.")
    
    if review_text.count('!') > 4:
        reasons.append("Excessive use of exclamation marks detected.")
        
    if not reasons:
        reasons.append("Subtle statistical mismatch in terminology or structural density against normal baseline.")
        
    return reasons

def get_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    return sid.polarity_scores(text)['compound']

@st.cache_resource
def load_models():
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    model = pickle.load(open("isolation_forest_yelp.pkl", "rb"))
    return vectorizer, model

vectorizer, model = load_models()

# Page config
st.set_page_config(page_title="Yelp AI Analyzer", layout="wide")

# Load CSS
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Sidebar
st.sidebar.title("🔍 Yelp AI Dashboard")
page = st.sidebar.radio("Navigate", ["Home", "Analyze Review", "Insights"])

# ---------- HOME ----------
if page == "Home":
    st.title("🍽️ Yelp Review Intelligence")
    st.markdown("### AI-powered Sentiment & Anomaly Detection")

    st.markdown("""
    <div class="glass">
    🚀 Analyze restaurant reviews using Machine Learning  
    📊 Detect fake/spam reviews  
    💡 Get insights instantly  
    </div>
    """, unsafe_allow_html=True)

# ---------- ANALYZE ----------
elif page == "Analyze Review":
    st.title("🧠 Analyze Review")

    input_mode = st.radio("Choose Input Type:", ["Single Review Text", "Yelp Business Link"])
    
    if input_mode == "Single Review Text":
        user_input = st.text_area("Enter a Yelp Review")

        if st.button("Analyze"):
            if user_input:
                vec = vectorizer.transform([user_input])
                review_length = [[len(user_input.split())]]
                rating_placeholder = [[3]]
                
                X_live = sp.hstack([vec, review_length, rating_placeholder])
                prediction = model.predict(X_live)
                decision_score = model.decision_function(X_live)[0]
                sentiment_score = get_sentiment(user_input)

                st.markdown('<div class="glass">', unsafe_allow_html=True)
                
                # Show Sentiment Segment
                if sentiment_score > 0.05:
                    sent_label = "🟢 Positive"
                elif sentiment_score < -0.05:
                    sent_label = "🔴 Negative"
                else:
                    sent_label = "⚪ Neutral"
                st.write(f"**Sentiment Score:** {sentiment_score:.2f} ({sent_label})")

                if prediction[0] == -1:
                    st.error("⚠️ Anomalous / Fake Review Detected")
                    st.markdown("**Why this prediction?**")
                    reasons = explain_anomaly(user_input, decision_score, sentiment_score)
                    for reason in reasons:
                        st.markdown(f"- {reason}")
                    st.markdown(f"*(Anomaly Score Strength: {abs(decision_score):.3f})*")
                else:
                    st.success("✅ Genuine Review")

                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("Please enter a review")
                
    elif input_mode == "Yelp Business Link":
        url_input = st.text_input("Enter Yelp Business URL")
        
        if st.button("Analyze Business"):
            if url_input:
                if "yelp.com" not in url_input.lower():
                    st.warning("Please enter a valid Yelp URL (containing yelp.com).")
                else:
                    with st.spinner("Fetching reviews and analyzing... This may take a minute."):
                        try:
                            from selenium import webdriver
                            from selenium.webdriver.common.by import By
                            from selenium.webdriver.chrome.options import Options
                            from selenium.webdriver.support.ui import WebDriverWait
                            from selenium.webdriver.support import expected_conditions as EC
                            
                            chrome_options = Options()
                            chrome_options.add_argument("--start-maximized")
                            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
                            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
                            chrome_options.add_experimental_option("useAutomationExtension", False)

                            driver = webdriver.Chrome(options=chrome_options)
                            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
                            
                            driver.get(url_input)
                            
                            wait = WebDriverWait(driver, 20)
                            wait.until(EC.presence_of_element_located((By.TAG_NAME, "p")))

                            for _ in range(5):
                                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                                time.sleep(2)

                            elements = driver.find_elements(By.TAG_NAME, "p")
                            reviews = [el.text.strip() for el in elements if len(el.text.strip()) > 80]
                            driver.quit()
                            
                            if not reviews:
                                st.warning("No reviews found or extracted.")
                            else:
                                import nltk
                                from nltk.corpus import stopwords
                                from nltk.stem import WordNetLemmatizer
                                nltk.download('stopwords', quiet=True)
                                nltk.download('wordnet', quiet=True)
                                
                                stop_words = set(stopwords.words('english'))
                                lemmatizer = WordNetLemmatizer()
                                
                                clean_reviews = []
                                for r in reviews:
                                    text = r.lower()
                                    # Use backslashes correctly for python strings
                                    text = re.sub(r"http\S+", "", text)
                                    text = re.sub(r"[^a-z\s]", "", text)
                                    tokens = text.split()
                                    cleaned_tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
                                    clean_reviews.append(" ".join(cleaned_tokens))
                                    
                                X_text = vectorizer.transform(clean_reviews)
                                review_lengths = [[len(r.split())] for r in clean_reviews]
                                rating_placeholders = [[3] for _ in clean_reviews]
                                
                                X_live = sp.hstack([X_text, review_lengths, rating_placeholders])
                                predictions = model.predict(X_live)
                                
                                suspicious = [1 if p == -1 else 0 for p in predictions]
                                suspicious_percentage = (sum(suspicious) / len(suspicious)) * 100
                                
                                st.markdown('<div class="glass">', unsafe_allow_html=True)
                                st.write(f"**Reviews collected:** {len(reviews)}")
                                st.write(f"**Suspicious review percentage:** {round(suspicious_percentage, 2)}%")
                                
                                if suspicious_percentage > 20:
                                    st.error("⚠️ Business may have suspicious review activity")
                                else:
                                    st.success("✅ Reviews appear mostly genuine")
                                    
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                                st.markdown('---')
                                st.markdown('### 📊 Live Data Visualization Dashboard')
                                
                                all_sentiments = [get_sentiment(r) for r in reviews]
                                all_anomaly_scores = model.decision_function(X_live)
                                
                                live_df = pd.DataFrame({
                                    "Review Length": [len(r.split()) for r in reviews],
                                    "Sentiment": all_sentiments,
                                    "Anomaly Score": all_anomaly_scores,
                                    "Classification": ["Suspicious/Fake" if p == -1 else "Genuine" for p in predictions]
                                })
                                
                                st.markdown("#### Sentiment vs. Anomaly Score")
                                st.caption("Lower Anomaly Score indicates higher suspicion.")
                                st.scatter_chart(live_df, x="Sentiment", y="Anomaly Score", color="Classification", use_container_width=True)
                                
                                if sum(suspicious) > 0:
                                    with st.expander("View Suspicious Reviews & Analysis"):
                                        for review, pred in zip(reviews, predictions):
                                            if pred == -1:
                                                r_len = [[len(review.split())]]
                                                r_vec = vectorizer.transform([review])
                                                r_live = sp.hstack([r_vec, r_len, [[3]]])
                                                d_score = model.decision_function(r_live)[0]
                                                s_score = get_sentiment(review)
                                                
                                                st.markdown(f"**Review:** {review}")
                                                
                                                if s_score > 0.05:
                                                    sent_label = "🟢 Positive"
                                                elif s_score < -0.05:
                                                    sent_label = "🔴 Negative"
                                                else:
                                                    sent_label = "⚪ Neutral"
                                                    
                                                st.markdown(f"**Sentiment Score:** {s_score:.2f} ({sent_label})")
                                                st.markdown("**Why Flagged?**")
                                                reasons = explain_anomaly(review, d_score, s_score)
                                                for reason in reasons:
                                                    st.markdown(f"- {reason}")
                                                st.markdown(f"*(Anomaly Strength: {abs(d_score):.3f})*")
                                                st.markdown("---")
                                                
                        except Exception as e:
                            st.error(f"Error analyzing URL: {e}")
            else:
                st.warning("Please enter a Yelp Business URL")

# ---------- INSIGHTS ----------
elif page == "Insights":
    st.title("📊 Insights Dashboard")
    
    st.markdown("""
        <div class="glass">
        <strong>Data Insight Summary:</strong> This dashboard provides a macro-view of the historical Yelp Reviews dataset (<code>yelp_reviews_clean.csv</code>) 
        used to train our Machine Learning models. It illustrates the real statistical distributions of thousands of reviews, 
        helping you understand the baseline of what normal reviews typically look like across ratings and lengths.
        </div>
        <br>
    """, unsafe_allow_html=True)
    
    df = load_data()
    
    if df is not None:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Database Reviews", f"{len(df):,}")

        with col2:
            avg_rating = df['rating'].mean()
            st.metric("Average Dataset Rating", f"{avg_rating:.2f} ⭐")

        with col3:
            avg_words = int(df['review_length'].mean())
            st.metric("Avg. Word Count", f"{avg_words} words")

        st.markdown("### 📈 Yelp Rating Distribution")
        rating_counts = df['rating'].value_counts().sort_index()
        st.bar_chart(rating_counts, use_container_width=True)
        
    else:
        st.warning("Could not load yelp_reviews_clean.csv data.")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Reviews", "N/A")

        with col2:
            st.metric("Average Rating", "N/A")

        with col3:
            st.metric("Avg. Word Count", "N/A")

        st.markdown("### 📈 Visualization")
        chart_data = np.random.randn(50, 3)
        st.line_chart(chart_data)