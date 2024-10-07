import streamlit as st
from backend import backend  # Import backend logic for processing user inputs

st.set_page_config(page_title="Navatar-Helper", page_icon=":robot_face:")

st.title("Navatar-Helper Chatbot")
st.write("Ask any NEET-related question and get answers based on official documents.")

# Input field for user to type questions
user_input = st.text_input("Enter your question:")

if st.button("Submit"):
    # Process the input using backend logic
    response = backend.handle_question(user_input)
    st.write("Chatbot Response:", response)
