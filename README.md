# Sentiment Analysis Web Application

This project is a **Sentiment Analysis Web Application** built using `Streamlit`, `NLTK`, and `Scikit-learn`. It classifies text into various sentiment categories, including **Depression, Positive/Negative, and Emotions (Joy, Anger, Sadness, etc.)**.

## Features
- Detects depression-related sentiments in text.
- Classifies text as Positive or Negative.
- Identifies emotions like Joy, Sadness, Anger, etc.
- Uses `TF-IDF` vectorization and logistic regression models.

## Requirements
Ensure you have the following dependencies installed before running the script:

```bash
pip install streamlit nltk scikit-learn numpy pickle5
```

## Project Structure
```
|-- models/
|   |-- logisticmodelde.pkl        # Model for Depression Analysis
|   |-- logisticmodel.pkl          # Model for Positive/Negative Analysis
|   |-- logisticmodelEm.pkl        # Model for Emotion Analysis
|-- vectorizers/
|   |-- countVectorizerde.pkl      # Vectorizer for Depression Analysis
|   |-- countVectorizer.pkl        # Vectorizer for Positive/Negative Analysis
|   |-- countVectorizerEm.pkl      # Vectorizer for Emotion Analysis
|-- sentiment_analysis.py           # Main Streamlit script
```

## How to Run
1. Ensure all models and vectorizers are placed in the appropriate directories.
2. Run the script using:

```bash
streamlit run sentiment_analysis.py
```

## How It Works
1. Users can enter text into the application.
2. Depending on the selected category, the text is preprocessed using:
   - Stopword removal
   - Stemming
   - TF-IDF Vectorization
3. The processed text is then passed through a pre-trained logistic regression model and LSTM model.
4. Machine learning framework used for this projects are TensorFlow and sci-kit learn
5. The output displays the detected sentiment category.

## Example Output
- **Depression Analysis:** Counts of detected depression-related statements.
- **Positive/Negative Analysis:** Number of positive and negative statements.
- **Emotion Analysis:** Categorized statements based on emotions.


