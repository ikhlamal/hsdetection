import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from senti_classifier import senti_classifier

# Load the model and vectorizer
with open('model.pkl', 'rb') as file:
    classifier = pickle.load(file)

with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Function to predict the sentiment
def predict_sentiment(tweet):
    text_vectorized = vectorizer.transform([tweet])
    pos_score, neg_score = senti_classifier.polarity_scores(text_vectorized)
    return pos_score, neg_score

def main():
    st.set_page_config(page_title="Hate Speech Detection", page_icon="🗣️")
    st.title("Hate Speech Detection")
    tweet = st.text_area("Masukkan tweet:")
    
    if st.button("Prediksi"):
        if tweet.strip() == "":
            st.warning("Masukkan tweet terlebih dahulu!")
        else:
            prediction = predict_sentiment(tweet)
            st.success(prediction)

if __name__ == '__main__':
    main()
