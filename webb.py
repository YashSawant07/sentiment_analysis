import re
import os
import nltk
import streamlit as st
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pickle
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer


stemmer = PorterStemmer()

import nltk
nltk.download('stopwords')

# Define the NLTK data directory
nltk_data_dir = 'nltk_data'
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

# Download the stopwords corpus if not already available
try:
    STOPWORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_dir)
    STOPWORDS = set(stopwords.words('english'))

# Define the stemming function
def steaming(content):
    review = re.sub('[^a-zA-Z]', ' ', content)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
    review = ' '.join(review)
    return review


model = pickle.load(open(r"logisticmodel.pkl", 'rb'))

encoder = pickle.load(open(r"encoderde.pkl", 'rb'))

vectorizer = pickle.load(open(r"countVectorizerde.pkl", 'rb'))

# Streamlit app layout
st.markdown("<h1 style='text-align: center;'>Sentiment Analysis</h1>", unsafe_allow_html=True)
st.subheader("Unlock the emotions behind every message and gain deeper insights")
st.markdown("---")
# Streamlit sidebar layout
st.sidebar.title('Sentiment Categories')
category = st.sidebar.selectbox('Choose a category:', ['Emotions','Depression', 'Positive or Negative'])

# Main layout for the selected category
if category == 'Depression':
# Text input field
    user_input = st.text_area("Enter text to analyze")

# Button for sentiment analysis
    if st.button('Analyze'):
        if user_input:
            # Split input text into individual sentences based on new lines
            sentences = user_input.splitlines()

            # Preprocess and analyze each sentence
            sentiments = []
            for sentence in sentences:
                if sentence.strip():  # Skip empty lines
                    processed_input = steaming(sentence)
                    input_vect = vectorizer.transform([processed_input])
                    prediction = model.predict(input_vect)
                    predicted_sentiment = encoder.inverse_transform(prediction)[0]
                    sentiments.append(predicted_sentiment)

            # Count the occurrence of each sentiment
            sentiment_counts = Counter(sentiments)

            # Display the sentiment counts
            st.write("Sentiment Analysis Results:")
            for sentiment, count in sentiment_counts.items():
                st.write(f"{sentiment}: {count} statement(s)")
        else:
            st.write("Please enter some sentences.")
            
            
            

model_pn = pickle.load(open(r"logisticmodel.pkl", 'rb'))

vectorizer_pn = pickle.load(open(r"countVectorizer.pkl", 'rb'))

# Main layout for the selected category
if category == 'Positive or Negative':
# Text input field
    user_input = st.text_area("Enter text to analyze")

# Button for sentiment analysis
    if st.button('Analyze'):
        if user_input:
            # Split input text into individual sentences based on new lines
            sentences = user_input.splitlines()

            # Preprocess and analyze each sentence
            sentiments = []
            for sentence in sentences:
                if sentence.strip():  # Skip empty lines
                    processed_input = steaming(sentence)
                    input_vect = vectorizer_pn.transform([processed_input])
                    prediction = model_pn.predict(input_vect)
                    if prediction[0] == 1:
                        predicted_sentiment = "Positive"
                    elif prediction[0] == 0:
                        predicted_sentiment = "Negative"
                    sentiments.append(predicted_sentiment)

            # Count the occurrence of each sentiment
            sentiment_counts = Counter(sentiments)

            # Display the sentiment counts
            st.write("Sentiment Analysis Results:")
            for sentiment, count in sentiment_counts.items():
                st.write(f"{sentiment}: {count} statement(s)")
        else:
            st.write("Please enter some sentences.")

model_Em = pickle.load(open(r"logisticmodelEm.pkl", 'rb'))

vectorizer_Em = pickle.load(open(r"countVectorizerEm.pkl", 'rb')) 

emotion_mapping = {
    0: "Sadness",
    1: "Joy",
    2: "Love",
    3: "Anger",
    4: "Fear",
    5: "Surprise"
}           
# Main layout for the selected category
if category == 'Emotions':
# Text input field
    user_input = st.text_area("Enter text to analyze")

# Button for sentiment analysis
    if st.button('Analyze'):
        if user_input:
            # Split input text into individual sentences based on new lines
            sentences = user_input.splitlines()

            # Preprocess and analyze each sentence
            sentiments = []
            for sentence in sentences:
                if sentence.strip():  # Skip empty lines
                    processed_input = steaming(sentence)
                    input_vect = vectorizer_Em.transform([processed_input])
                    prediction = model_Em.predict(input_vect)[0]
                    predicted_sentiment = emotion_mapping[prediction]
                    sentiments.append(predicted_sentiment)

            # Count the occurrence of each sentiment
            sentiment_counts = Counter(sentiments)

            # Display the sentiment counts
            st.write("Sentiment Analysis Results:")
            for sentiment, count in sentiment_counts.items():
                st.write(f"{sentiment}: {count} statement(s)")
        else:
            st.write("Please enter some sentences.")
            
