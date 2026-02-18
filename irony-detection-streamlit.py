# Install dependencies
# pip install streamlit langdetect emoticon-fix scikit-learn nltk

"""
Replaced word_tokenize with TweetTokenizer because TweetTokenizer
works without downloading punkt, making the app cloud-deployable
"""

import streamlit as st
import joblib
import unicodedata
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import TweetTokenizer
from emoticon_fix import emoticon_fix

# --- Preprocessing setup ---
tknzr = TweetTokenizer()
snow_stemmer = SnowballStemmer(language='english')

# --- Preprocessing functions ---
def EmotToText(text):
    return emoticon_fix(text)

def performTkn(text):
    return " ".join(tknzr.tokenize(text))

def performAccent(text):
    return unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8')

def peformStopWordRemoval(text):
    tokens = tknzr.tokenize(text)  # Use TweetTokenizer
    tokens = [w for w in tokens if w.lower() not in stopwords.words('english')]
    return " ".join(tokens)

def peformStemming(text):
    tokens = tknzr.tokenize(text)  # Use TweetTokenizer
    tokens = [snow_stemmer.stem(w) for w in tokens]
    return " ".join(tokens)

def preProcessingModule(text):
    text = EmotToText(text)
    text = peformStemming(text)
    text = peformStopWordRemoval(text)
    text = performAccent(text)
    text = performTkn(text)
    return text

# --- Load trained model ---
@st.cache_resource
def load_model():
    return joblib.load("irony_model.pkl")

model = load_model()

# --- Streamlit UI ---
st.title("🔹Irony Detection🔹")
st.write("Enter any sentence or tweet to check if it is ironic or not.")

user_input = st.text_area("Your Text Here:")

if st.button("Check Irony"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        processed_text = preProcessingModule(user_input)
        prediction = model.predict([processed_text])[0]
        st.success(f"Prediction: **{prediction}**")
