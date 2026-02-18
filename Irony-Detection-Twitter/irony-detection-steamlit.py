# Install dependencies
!pip install streamlit langdetect emoticon-fix scikit-learn nltk

import streamlit as st
import joblib
from nltk.tokenize import TweetTokenizer
from emoticon_fix import emoticon_fix
import unicodedata
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from langdetect import detect
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# --- Preprocessing functions ---
tknzr = TweetTokenizer()
snow_stemmer = SnowballStemmer(language='english')

def EmotToText(text):
    return emoticon_fix(text)

def performTkn(text):
    token = [tkn for tkn in tknzr.tokenize(text)]
    return " ".join(token)

def performAccent(text):
    return unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8')

def peformStopWordRemoval(text):
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if word.lower() not in stopwords.words()]
    return " ".join(tokens_without_sw)

def peformStemming(text):
    wordTok = word_tokenize(text)
    return " ".join([snow_stemmer.stem(word) for word in wordTok])

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
    return joblib.load('irony_model.pkl')

model = load_model()

# --- Streamlit UI ---
st.title("Irony Detection")
st.write("Enter a sentence or tweet to check if it is ironic or not.")

user_input = st.text_area("Your Text Here:")

if st.button("Check Irony"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        processed_text = preProcessingModule(user_input)
        prediction = model.predict([processed_text])[0]
        st.success(f"Prediction: **{prediction}**")