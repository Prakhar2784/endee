import streamlit as st
from rag import generate_answer

st.set_page_config(page_title="Smart College Notes AI", layout="wide")

st.title("🎓 Smart College Notes - Semantic Search + RAG")
st.markdown("Ask questions from your uploaded college notes using AI.")

query = st.text_input("Enter your question:")

if st.button("Ask"):
    if query:
        with st.spinner("Generating answer..."):
            answer = generate_answer(query)
        st.success("Answer Generated")
        st.write(answer)
    else:
        st.warning("Please enter a question.")