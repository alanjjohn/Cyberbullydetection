import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load trained model and vectorizer
with open("sentiment_model.pkl", "rb") as model_file:
    clf = pickle.load(model_file)
with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def get_vader_scores(text):
    """Get VADER sentiment scores."""
    scores = analyzer.polarity_scores(text)
    return scores['neg'], scores['neu'], scores['pos'], scores['compound']


text = "@Blackman38Tide: @WhaleLookyHere @HowdyDowdy11 queer"" gaywad"


vader_features = np.array(get_vader_scores(text)).reshape(1, -1)
text_features = vectorizer.transform([text]).toarray()
input_features = np.hstack((text_features, vader_features))
prediction = clf.predict(input_features)[0]
print(prediction)