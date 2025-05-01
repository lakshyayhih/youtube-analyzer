import spacy
from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import nltk
from transformers import pipeline

import re
from youtube_transcript_api import YouTubeTranscriptApi
from crf import (
    sent2feats,
    load_conll03_data,
    train_seq 
)

def extract_video_id(url):
    """
    Extract the video ID from a YouTube URL.
    Supports both full and short YouTube links.
    """
    regex = (
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    )
    match = re.search(regex, url)
    if match:
        return match.group(1)
    raise ValueError("Invalid YouTube URL")

def get_transcript(video_id):
    """
    Fetch the transcript using the YouTubeTranscriptApi.
    Returns the full transcript as a single string.
    """
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        full_transcript = " ".join([entry["text"] for entry in transcript_list])
        return full_transcript
    except Exception as e:
        raise RuntimeError(f"Could not fetch transcript: {e}")


nltk.download('punkt')
summarizer = pipeline("summarization")
# Load spaCy model for NER
import spacy

# Safe spaCy model loader
def load_spacy_model(model_name="en_core_web_sm"):
    try:
        return spacy.load(model_name)
    except OSError:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", model_name])
        return spacy.load(model_name)

nlp = load_spacy_model()

from sklearn_crfsuite import CRF

def get_named_entities(text):
    """
    Trains a CRF model using CoNLL03 data and predicts entities from the input text.
    Returns a list of (word, predicted_label) tuples."""
    
    # Step 1: Split transcript into dev sentences
    dev_sentences = split_transcript_string_into_sentences(text, max_words=15)

    # Step 2: Convert dev sentences to features
    X_dev = [sent2feats(sentence) for sentence in dev_sentences]

    # Step 3: Load training data
    train_path = r'C:\Users\PRP\.cache\kagglehub\datasets\alaakhaled\conll003-englishversion\versions\1\train.txt'
    train_sentences, y_train = load_conll03_data(train_path)
    X_train = [sent2feats(sentence) for sentence in train_sentences]

    # Step 4: Train CRF and predict
    crf = CRF(algorithm='lbfgs', c1=0.1, c2=10, max_iterations=10)
    crf.fit(X_train, y_train)
    y_pred = crf.predict(X_dev)
    
    # Step 5: Combine words and 
    named_entities = []
    seen = set()
    for sentence, labels in zip(dev_sentences, y_pred):
        for word, label in zip(sentence, labels):
            if label != 'O':  # Only keep named entities
             if (word) not in seen:  # Check if the pair is already in the set
                named_entities.append((word, label))  # Add it to the list
                seen.add((word))  # Add the pair to the set for uniqueness  
    return named_entities

    
def split_transcript_string_into_sentences(text, max_words=15):
    seen = set()
    sentences = []

    words = text.strip().split()

    for i in range(0, len(words), max_words):
        sentence = words[i:i + max_words]  # Just raw words
        sentence_tuple = tuple(sentence)
        if sentence_tuple and sentence_tuple not in seen:
            seen.add(sentence_tuple)
            sentences.append(sentence)  # Keep as list of strings

    return sentences



# Function for summarizing text
from transformers import pipeline
import nltk
import math
import streamlit as st

# Initialize the summarizer once
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")  # Use a faster model like DistilBART if needed

summarizer = load_summarizer()

def summarize_text(text, max_length=200, max_chunk_size=512):
    # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(text)
    
    # Create chunks of sentences for summarization
    chunk = ""
    summary_chunks = []
    chunk_size = 0  # Track the number of tokens in the current chunk

    for sentence in sentences:
        # Estimate the number of tokens in the sentence (rough estimation)
        estimated_tokens = len(sentence.split())
        
        # If adding this sentence exceeds the max chunk size, summarize the current chunk
        if chunk_size + estimated_tokens > max_chunk_size:
            # Summarize the current chunk and reset for the next one
            summary = summarizer(chunk.strip(), max_length=max_length, min_length=50, do_sample=False)[0]['summary_text']
            summary_chunks.append(summary)
            chunk = sentence  # Start a new chunk
            chunk_size = estimated_tokens
        else:
            # Add sentence to the current chunk
            chunk += " " + sentence
            chunk_size += estimated_tokens

    # Summarize the last chunk
    if chunk:
        summary = summarizer(chunk.strip(), max_length=max_length, min_length=50, do_sample=False)[0]['summary_text']
        summary_chunks.append(summary)
    
    # Join all summaries into one
    return " ".join(summary_chunks)





# Function for analyzing sentiment (using TextBlob)
def analyze_sentiment(text):
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment > 0:
        return "Positive"
    elif sentiment < 0:
        return "Negative"
    else:
        return "Neutral"

# Function for sentiment over time (split text into sentences)
def sentiment_over_time(text):
    sentences = nltk.sent_tokenize(text)
    sentiments = [TextBlob(sentence).sentiment.polarity for sentence in sentences]
    return sentiments

# Function to generate a word cloud from text
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400).generate(text)
    return wordcloud

# Function to extract top keywords (using CountVectorizer and LDA)
def extract_keywords(text, top_n=10):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform([text])
    lda = LatentDirichletAllocation(n_components=1, random_state=42)
    lda.fit(X)
    keywords = [vectorizer.get_feature_names_out()[i] for i in lda.components_[0].argsort()[:-top_n-1:-1]]
    return keywords

# Function to split transcript into a timeline (for visualization)
import pandas as pd

def create_timeline(text, chunk_size=50):
    """
    Splits transcript into chunks of N words to simulate a timeline.
    Useful for transcripts that lack actual timestamps.
    """
    words = text.split()
    timeline = []

    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        timestamp = f"{i}-{i + len(chunk_words)}"
        timeline.append([timestamp, chunk_text])

    df = pd.DataFrame(timeline, columns=["Timestamp", "Text"])
    return df

