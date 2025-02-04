import streamlit as st
from transformers import pipeline

# Load the Hugging Face model
model_name = "DanL/scientific-challenges-and-directions"
model = pipeline("text-generation", model=model_name)

# Title of the app
st.title("Research Gap Finder")

# Input field for the user to describe their research area
research_description = st.text_area("Describe your research area:")

# Button to generate results
if st.button("Find Research Gap"):
    if research_description:
        # Use the model to generate suggestions for research gaps
        result = model(research_description, max_length=200)
        st.write("Research Gaps:")
        st.write(result[0]['generated_text'])
    else:
        st.error("Please enter a description of your research area.")