import streamlit as st
import pickle
import scipy.sparse as sp
import re
import time
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- NLTK setup ---
for resource in [('sentiment/vader_lexicon.zip', 'vader_lexicon'),
                 ('corpora/stopwords', 'stopwords'),
                 ('corpora/wordnet', 'wordnet')]:
    try:
        nltk.data.find(resource[0])
    except LookupError:
        nltk.download(resource[1], quiet=True)


@st.cache_data
def load_data():
    try:
        df = pd.read_csv("yelp_reviews_clean.csv")
        return df
    except Exception:
        return None


@st.cache_resource
def load_models():
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    model = pickle.load(open("isolation_forest_yelp.pkl", "rb"))
    return vectorizer, model


def create_driver():
    """
    Exact same driver setup as live_yelp_analysis.ipynb.
    No headless — runs visibly on Windows/Mac to avoid Yelp bot detection.
    """
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager

    chrome_options = Options()

    # NO headless — this is what makes the notebook work
    chrome_options.add_argument("--start-maximized")

    # Anti-detection (identical to notebook)
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options
    )

    # Remove selenium flag (identical to notebook)
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    return driver


def fetch_yelp_reviews(url):
    """
    Exact same scraping logic as live_yelp_analysis.ipynb.
    """
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    driver = create_driver()
    reviews = []

    try:
        driver.get(url)

        # Wait for page to load
        wait = WebDriverWait(driver, 20)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "p")))

        # Scroll to load reviews (identical to notebook)
        for _ in range(5):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)

        # Extract reviews (identical to notebook)
        elements = driver.find_elements(By.TAG_NAME, "p")
        for el in elements:
            text = el.text.strip()
            if len(text) > 80:
                reviews.append(text)
    finally:
        driver.quit()

    return reviews


def explain_anomaly(review_text, sentiment_score):
    reasons = []
    word_count = len(review_text.split())
    if word_count < 10:
        reasons.append("Unusually short review (possible spam).")
    elif word_count > 300:
        reasons.append("Unusually long review.")
    if sentiment_score > 0.9:
        reasons.append("Extreme positive sentiment — possibly fake praise.")
    elif sentiment_score < -0.9:
        reasons.append("Extreme negative sentiment — possibly fake complaint.")
    upper_chars = sum(1 for c in review_text if c.isupper())
    if len(review_text) > 0 and (upper_chars / len(review_text)) > 0.2:
        reasons.append("Excessive capital letters (>20%).")
    if review_text.count('!') > 4:
        reasons.append("Excessive exclamation marks.")
    if not reasons:
        reasons.append("Statistical anomaly detected by model.")
    return reasons


def get_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    return sid.polarity_scores(text)['compound']


# --- App setup ---
vectorizer, model = load_models()

st.set_page_config(page_title="Yelp AI Analyzer", layout="wide")
st.sidebar.title("🔍 Yelp AI Dashboard")
page = st.sidebar.radio("Navigate", ["Home", "Analyze Business", "Insights"])

# ── HOME ──────────────────────────────────────────────────────────────────────
if page == "Home":
    st.title("🍽️ Yelp Review Intelligence")
    st.markdown("AI-powered Sentiment & Fake Review Detection using Isolation Forest.")
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model", "Isolation Forest")
    with col2:
        st.metric("Sentiment", "VADER NLP")
    with col3:
        st.metric("Scraping", "Selenium")

# ── ANALYZE BUSINESS ──────────────────────────────────────────────────────────
elif page == "Analyze Business":
    st.title("🧠 Analyze Yelp Business")
    st.markdown("Enter a Yelp business URL to scrape and analyze its reviews.")
    st.info("💡 A Chrome window will open briefly while scraping — this is normal.")

    url_input = st.text_input(
        "Yelp Business URL",
        placeholder="https://www.yelp.com/biz/..."
    )

    if st.button("Scrape & Analyze"):
        if not url_input:
            st.warning("Please enter a URL.")
        elif "yelp.com" not in url_input.lower():
            st.warning("Please enter a valid Yelp URL.")
        else:
            with st.spinner("Opening Chrome and scraping reviews (~30s)..."):
                raw_reviews = []
                error_msg = None

                try:
                    raw_reviews = fetch_yelp_reviews(url_input)
                except Exception as e:
                    error_msg = str(e)

            if error_msg:
                st.error(f"Scraping failed: {error_msg}")
            elif not raw_reviews:
                st.warning(
                    "No reviews extracted. "
                    "Yelp may have blocked the request — try again in a few minutes."
                )
            else:
                # ── Preprocess ────────────────────────────────────────────────
                from nltk.corpus import stopwords
                from nltk.stem import WordNetLemmatizer

                stop_words = set(stopwords.words('english'))
                lemmatizer = WordNetLemmatizer()

                clean_docs = []
                for r in raw_reviews:
                    text = r.lower()
                    text = re.sub(r"http\S+", "", text)
                    text = re.sub(r"[^a-z\s]", "", text)
                    tokens = text.split()
                    cleaned = [
                        lemmatizer.lemmatize(w)
                        for w in tokens
                        if w not in stop_words
                    ]
                    clean_docs.append(" ".join(cleaned))

                # ── Features ──────────────────────────────────────────────────
                X_text = vectorizer.transform(clean_docs)
                review_lengths = [[len(r.split())] for r in clean_docs]
                rating_placeholders = [[3] for _ in clean_docs]
                X_live = sp.hstack([X_text, review_lengths, rating_placeholders])

                # ── Predict ───────────────────────────────────────────────────
                predictions = model.predict(X_live)
                d_scores = model.decision_function(X_live)

                suspicious = [1 if p == -1 else 0 for p in predictions]
                percent = (sum(suspicious) / len(suspicious)) * 100

                # ── Summary ───────────────────────────────────────────────────
                st.markdown("---")
                col1, col2 = st.columns(2)
                col1.metric("Reviews Scraped", len(raw_reviews))
                col2.metric("Suspicious Activity", f"{round(percent, 1)}%")

                if percent > 20:
                    st.error("⚠️ High suspicious review activity detected!")
                else:
                    st.success("✅ Reviews appear mostly genuine.")

                # ── Flagged reviews ───────────────────────────────────────────
                st.markdown("### 🚩 Flagged Reviews")
                flagged = False
                for review, pred, d_score in zip(raw_reviews, predictions, d_scores):
                    if pred == -1:
                        flagged = True
                        s_score = get_sentiment(review)
                        reasons = explain_anomaly(review, s_score)
                        with st.expander(f"🔴 {review[:80]}..."):
                            st.write(f"**Sentiment Score:** {s_score:.2f}")
                            st.write(f"**Anomaly Score:** {d_score:.3f}")
                            st.write("**Reasons:**")
                            for r in reasons:
                                st.write(f"- {r}")
                if not flagged:
                    st.info("No flagged reviews found.")

# ── INSIGHTS ──────────────────────────────────────────────────────────────────
elif page == "Insights":
    st.title("📊 Dataset Insights")
    st.markdown("Overview of the Yelp training dataset used to calibrate the model.")

    df = load_data()
    if df is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Reviews", f"{len(df):,}")
        col2.metric("Avg Rating", f"{df['rating'].mean():.2f} ⭐")
        col3.metric("Avg Word Count", int(df['review_length'].mean()))

        st.markdown("### Rating Distribution")
        st.bar_chart(df['rating'].value_counts().sort_index())
    else:
        st.error(
            "Could not load `yelp_reviews_clean.csv`. "
            "Make sure it exists in the project root."
        )
