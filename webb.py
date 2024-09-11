import re
import streamlit as st
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pickle
from collections import Counter

# Initialize the PorterStemmer and stopwords
stemmer = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))

# Define the stemming function
def steaming(content):
    review = re.sub('[^a-zA-Z]', ' ', content)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
    review = ' '.join(review)
    return review


model = pickle.load(open("D:\projects\ML Project\Sentiment analysis\logisticmodel.pkl", 'rb'))

encoder = pickle.load(open("D:\projects\ML Project\Sentiment analysis\encoder.pkl", 'rb'))

vectorizer = pickle.load(open("D:\projects\ML Project\Sentiment analysis\countVectorizer.pkl", 'rb'))

# Streamlit app layout
st.title('Sentiment Analysis Web App')
# Streamlit sidebar layout
st.sidebar.title('Sentiment Categories')
category = st.sidebar.selectbox('Choose a category:', ['Depression', 'Positive or Negative', 'Emotions'])

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