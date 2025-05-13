# Irony Detection from Twitter Dataset
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This compact NLP project, developed for the **_Introduction to Data Science_** course, detects irony in Twitter data using a streamlined machine learning pipeline. It leverages the small version of the SemEval-2018 Task-3 [C. Van Hee et al., 2018](https://aclanthology.org/S18-1005.pdf) dataset, a well-known benchmark for irony detection, to preprocess tweets with NLTK (including tokenization, stemming, stopword removal, emoji, and emoticon conversion), extract TF-IDF features, and train an MLPClassifier for binary classification (ironic: 1 vs. non-ironic: 0). The model’s performance is evaluated using macro-average F1-score, accuracy, and a confusion matrix, demonstrating proficiency in text processing, feature engineering, and classification techniques. The macro-average metrics evaluate model performance equally across ironic and non-ironic classes. Designed as an accessible showcase of NLP skills, this project highlights practical data science applications for real-world social media data analysis.

## Features
- **Preprocessing**: Tokenization, stemming, stopword removal, emoji, and emoticon conversion.
- **Model**: MLPClassifier with TF-IDF features.
- **Evaluation**: Macro-average F1-score and confusion matrix.

## Tools
- Python, scikit-learn, NLTK, pandas, numpy, langdetect, matplotlib, seaborn, emoji, emoticon

## Setup
1. Download the repository from GitHub.
2. Install dependencies available on requirements.txt
3. Install NLTK data
4. Open IronyDetection.ipynb in a notebook (Google Colab/Jupyter) and run all cells to train and evaluate the model.
