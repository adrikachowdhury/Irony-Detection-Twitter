# Install dependencies
# pip install streamlit langdetect emoticon-fix scikit-learn nltk

import streamlit as st
import joblib
import unicodedata
from nltk.tokenize import word_tokenize, TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk
from nltk.data import find
from emoticon_fix import emoticon_fix

# --- Download NLTK resources if missing ---
try:
    find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# --- Preprocessing functions ---
tknzr = TweetTokenizer()
snow_stemmer = SnowballStemmer(language='english')

def EmotToText(text):
    return emoticon_fix(text)

def performTkn(text):
    return " ".join(tknzr.tokenize(text))

def performAccent(text):
    return unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8')

def peformStopWordRemoval(text):
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w.lower() not in stopwords.words()]
    return " ".join(tokens)

def peformStemming(text):
    tokens = word_tokenize(text)
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
st.title("Irony Detection 🔹 IroniQ")
st.write("Enter any sentence or tweet to check if it is ironic or not.")

user_input = st.text_area("Your Text Here:")

if st.button("Check Irony"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        processed_text = preProcessingModule(user_input)
        prediction = model.predict([processed_text])[0]
        st.success(f"Prediction: **{prediction}**")
