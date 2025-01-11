# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
df = pd.read_csv("SMSSpamCollection", sep='\t', header=None, names=["Label", "Message"])

# Display the first few rows of the data
df.head()

# Convert labels to binary (ham=0, spam=1)
df['Label'] = df['Label'].map({'ham': 0, 'spam': 1})

# Split the data into features (X) and target (y)
X = df['Message']
y = df['Label']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the text data to numeric using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initialize the model
nb_model = MultinomialNB()

# Train the model
nb_model.fit(X_train_tfidf, y_train)

# Predict on the test set
y_pred = nb_model.predict(X_test_tfidf)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Save the trained model and the vectorizer
joblib.dump(nb_model, 'spam_classifier_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

# Function to predict spam or ham
def predict_spam(message):
    # Load the model and vectorizer
    model = joblib.load('spam_classifier_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    
    # Transform the message to match the vectorizer
    message_tfidf = vectorizer.transform([message])
    
    # Predict using the trained model
    prediction = model.predict(message_tfidf)
    
    if prediction == 1:
        return "Spam"
    else:
        return "Ham"

# Test the prediction function
print(predict_spam("Free cash prize, click here!"))
print(predict_spam("Hey, how are you?"))
