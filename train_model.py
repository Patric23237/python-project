
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load the datasets
fake = pd.read_csv("dataset/Fake.csv")
real = pd.read_csv("dataset/True.csv")

# Add labels
fake["label"] = 0
real["label"] = 1

# Combine datasets
data = pd.concat([fake, real], ignore_index=True)

# Combine title and article text
data["content"] = (
    data["title"].fillna("") + " " +
    data["text"].fillna("")
)

# Remove empty articles
data = data[data["content"].str.strip() != ""]

# Separate input and output
X = data["content"]
y = data["label"]

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create machine learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000)),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train the model
print("Training model...")
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

# Save trained model
joblib.dump(model, "fake_news_model.pkl")

print("Model saved successfully!")
