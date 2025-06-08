# 🌀 Irony Detection from Twitter Dataset
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This compact NLP project, developed for the **_Introduction to Data Science_** course, detects irony in Twitter data using a streamlined machine learning pipeline. It leverages the small version of the SemEval-2018 Task-3 dataset, a well-known benchmark for irony detection, to preprocess tweets with NLTK (including tokenization, stemming, stopword removal, emoji, and emoticon conversion), extract TF-IDF features, and train an MLPClassifier for binary classification (ironic: 1 vs. non-ironic: 0). The model’s performance is evaluated using macro-average F1-score, accuracy, and a confusion matrix, demonstrating proficiency in text processing, feature engineering, and classification techniques. The macro-average metrics evaluate model performance equally across ironic and non-ironic classes. Designed as an accessible showcase of NLP skills, this project highlights practical data science applications for real-world social media data analysis.

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
1. Download/clone the repository from GitHub.
2. Install dependencies available on requirements.txt
3. Install NLTK data
4. Open IronyDetection.ipynb in a notebook (Google Colab/Jupyter) and run all cells to train and evaluate the model through hyperparameter tuning.
