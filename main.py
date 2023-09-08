import nltk
import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Download NLTK stopwords if not already downloaded
nltk.download("stopwords")
nltk.download("punkt")

# Define a list of sample spam and ham emails (you can replace these with your own data)
spam_emails = [
    "Free luxury cruise! Call now!",
    "Buy cheap watches and pills!",
    "You've won a million dollars!",
]

ham_emails = [
    "Let's have lunch tomorrow?",
    "Meeting at 3 PM today.",
    "Please find the attached report.",
]

# Create a list of labeled email data (1 for spam, 0 for ham)
labeled_emails = [(email, 1) for email in spam_emails] + [(email, 0) for email in ham_emails]

# Shuffle the labeled data
random.shuffle(labeled_emails)

# Define a function to preprocess emails
def preprocess(email):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(email.lower())
    return [word for word in words if word.isalnum() and word not in stop_words]

# Create a feature set using Bag of Words (BoW)
vectorizer = CountVectorizer(analyzer=preprocess)
X = vectorizer.fit_transform([email for email, _ in labeled_emails])
y = [label for _, label in labeled_emails]

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the classifier
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy:", accuracy_score(y_test, y_pred))
