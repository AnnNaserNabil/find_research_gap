import streamlit as st
import fitz  # PyMuPDF for extracting text
import os
import re
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sentence_transformers import SentenceTransformer, util

# Download necessary NLTK data
nltk.download("punkt")

# Initialize the BERT model
model = SentenceTransformer("allenai/scibert_scivocab_uncased")

# Function to extract text from PDFs
def extract_text_from_pdfs(uploaded_files):
    all_papers = []
    
    for uploaded_file in uploaded_files:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = "".join([page.get_text() for page in doc])
        all_papers.append({"filename": uploaded_file.name, "text": text})
    
    return pd.DataFrame(all_papers)

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.lower()  # Convert to lowercase
    sentences = sent_tokenize(text)  # Sentence tokenization
    return " ".join(sentences)

# Function for topic modeling with LDA
def extract_topics(corpus, num_topics=5):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(corpus)
    
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)
    
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
        topics.append(f"Topic {topic_idx + 1}: {' '.join(top_words)}")
    
    return topics

# Function to find research gaps
def find_research_gaps(corpus):
    reference_sentences = [
        "Future research should focus on...",
        "This study has some limitations...",
        "Further exploration is needed in...",
    ]
    
    reference_embeddings = model.encode(reference_sentences, convert_to_tensor=True)
    gaps = []
    
    for idx, text in enumerate(corpus):
        sentences = sent_tokenize(text)
        sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
        
        similarities = util.pytorch_cos_sim(sentence_embeddings, reference_embeddings)
        max_similarities = similarities.max(dim=1)[0]
        
        for i, score in enumerate(max_similarities):
            if score > 0.7:  # Threshold for identifying research gaps
                gaps.append({"paper": idx, "sentence": sentences[i], "score": score.item()})
    
    return pd.DataFrame(gaps)

# Streamlit UI
st.title("Research Gap Finder")
st.write("Upload research papers (PDFs), and this app will analyze them to identify research gaps.")

uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    st.write("Extracting text from PDFs...")
    papers_df = extract_text_from_pdfs(uploaded_files)
    
    st.write("Preprocessing text...")
    papers_df["clean_text"] = papers_df["text"].apply(preprocess_text)
    
    st.write("Performing topic modeling...")
    topics = extract_topics(papers_df["clean_text"])
    st.write("### Identified Topics:")
    for topic in topics:
        st.write(topic)
    
    st.write("Finding research gaps...")
    research_gaps_df = find_research_gaps(papers_df["clean_text"])
    
    if not research_gaps_df.empty:
        st.write("### Potential Research Gaps:")
        st.dataframe(research_gaps_df)
    else:
        st.write("No significant research gaps found.")

    # Option to download results
    csv = research_gaps_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Research Gaps as CSV", csv, "research_gaps.csv", "text/csv")
