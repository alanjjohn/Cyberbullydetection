import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load Twitter dataset
data = pd.read_csv("DataSet/labeled_data.csv")  # Ensure dataset has 'text' and 'label' columns

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def get_vader_scores(text):
    """Get VADER sentiment scores."""
    scores = analyzer.polarity_scores(text)
    return scores['neg'], scores['neu'], scores['pos'], scores['compound']

# Apply VADER analysis
data[['neg', 'neu', 'pos', 'compound']] = data['text'].apply(lambda x: pd.Series(get_vader_scores(str(x))))

# Convert text data to numerical features using CountVectorizer
vectorizer = CountVectorizer(stop_words='english', max_features=5000)
x_text_features = vectorizer.fit_transform(data['text']).toarray()

# Combine VADER features with text features
X = np.hstack((x_text_features, data[['neg', 'neu', 'pos', 'compound']].values))
y = data['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Save model and vectorizer
with open("sentiment_model.pkl", "wb") as model_file:
    pickle.dump(clf, model_file)
with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

# Predict on test data
y_pred = clf.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

# Predict function
def predict_sentiment(text):
    """Predict sentiment for new tweets."""
    vader_features = np.array(get_vader_scores(text)).reshape(1, -1)
    text_features = vectorizer.transform([text]).toarray()
    input_features = np.hstack((text_features, vader_features))
    return clf.predict(input_features)[0]

# Example usage
print(predict_sentiment("I love this product!"))
