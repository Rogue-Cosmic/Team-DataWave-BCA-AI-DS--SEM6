#!/usr/bin/env bash
# exit on error
set -o errexit

# Install Chromium + chromedriver (needed for Selenium on Render)
apt-get update -qq
apt-get install -y -qq chromium-browser chromium-chromedriver \
  || apt-get install -y -qq chromium chromium-driver \
  || echo "WARNING: Could not install chromium via apt — scraping will fail on server"

# Install Python dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')
"
