# 🌀 IronyScope- Irony Detection from Twitter Dataset

Live Demo: [IronyScope- Irony Detector](https://ironyscope.streamlit.app/)
<img src="https://github.com/user-attachments/assets/fa501fb0-beaf-4524-a95b-c1ba7fa634fa" alt="Demo GIF" width="600"/>


This compact NLP project, developed for the **_Introduction to Data Science_** course, detects irony in Twitter data using a streamlined machine learning pipeline. It leverages the small version of the SemEval-2018 Task-3 dataset, a well-known benchmark for irony detection, to preprocess tweets with NLTK (including tokenization, stemming, stopword removal, emoji, and emoticon conversion), extract TF-IDF features, and train an MLPClassifier for binary classification (ironic: 1 vs. non-ironic: 0). The model's performance is evaluated using macro-average F1-score, accuracy, and a confusion matrix, demonstrating proficiency in text processing, feature engineering, and classification techniques. The macro-average metrics evaluate model performance equally across ironic and non-ironic classes. Designed as an accessible showcase of NLP skills, this project highlights practical data science applications for real-world social media data analysis. This project is also deployed in Streamlit for practice purpose.

---

## 🚀 Features
- **Preprocessing**: Tokenization, stemming, stopword removal, emoji, and emoticon conversion.
- **Model**: MLPClassifier with TF-IDF features.
- **Evaluation**: Accuracy and macro-average precision, recall, F1-Score and confusion matrix with a detailed classification report.

## 📊 Dataset
- **SemEval-2018 Task-3 (Small version)** [C. Van Hee et al., 2018 – ACL Anthology](https://aclanthology.org/S18-1005.pdf)

## 🧠 Model Architecture
- **CountVectorizer:** Transforms the text data into token count matrices using unigrams, bigrams, and trigrams (1–3 n-grams).
- **TF-IDF Transformer:** Converts token counts to TF-IDF scores with L1 normalization and smoothed IDF values.
- **MLPClassifier:** A feedforward neural network with one hidden layer of 100 units, using ReLU activation and the Adam optimizer.

## 🛠️ Tools and Libraries
- Python
- scikit-learn
- NLTK
- pandas
- numpy
- langdetect
- matplotlib
- seaborn
- emoji
- emoticon_fix

## ⚙️ Setup
1️⃣ Clone the Repository
```bash
git clone https://github.com/adrikachowdhury/Irony-Detection-Twitter.git
cd Irony-Detection-Twitter
```

2️⃣ Install Dependencies

pip install -r `requirements.txt`

3️⃣ Run the Streamlit App Locally

streamlit run `irony-detection-streamlit.py`
After the link opens in your browser, enter any text in the input box and click Check Irony, and you should be able to see the predicted label.

## 🔑Notes
- The trained model (irony_model.pkl) is already included, so no training is required.
- NLTK data (punkt, stopwords) will be downloaded automatically if missing.
- Ensure your Python version is >=3.9 for full compatibility.

## 💫 Credits
Feel free to explore the documentation, and please give **credit to the owner** when using content from this repo! 
Many thanks!🙌
