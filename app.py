import streamlit as st
from transformers import pipeline
import PyPDF2

# Load the Hugging Face model
model_name = "DanL/scientific-challenges-and-directions"
model = pipeline("text-generation", model=model_name)

# Function to extract text from uploaded PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

# Streamlit App
st.title("Research Gap Finder from PDF")

# Upload PDF file
uploaded_file = st.file_uploader("Upload your research paper (PDF)", type=["pdf"])

# Process uploaded file
if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        extracted_text = extract_text_from_pdf(uploaded_file)
    
    # Display extracted text
    st.subheader("Extracted Text Preview:")
    st.text_area("PDF Content", extracted_text[:1000], height=250)  # Showing first 1000 characters for preview

    # Button to find research gap
    if st.button("Find Research Gap"):
        with st.spinner("Analyzing research gaps..."):
            result = model(extracted_text[:1024], max_length=200)  # Limiting input to avoid exceeding model limit
        st.subheader("Identified Research Gaps:")
        st.write(result[0]['generated_text'])

else:
    st.info("Please upload a PDF file to begin.")