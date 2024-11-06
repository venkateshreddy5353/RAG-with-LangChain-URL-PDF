import streamlit as st
import utils

st.set_page_config(page_title="RAG", page_icon="🧠", layout="wide")
st.title("Retrieval-Augmented Generation (RAG) with LangChain : URL")
st.divider()


col_input , col_rag , col_normal = st.columns([3,5,5])
with col_input:
    target_url = st.text_input("URL",placeholder="https://en.wikipedia.org/wiki/Large_language_model")
    st.divider()
    prompt = st.text_input("Prompt", placeholder="What is the LLM?",
                           key="url_prompt")
    st.divider()
    sumbit_btn = st.button(label="Submit",key="url_btn")

    if sumbit_btn:
        with col_rag:
            with st.spinner("Processing..."):
                st.success("Response: Answering with RAG...")
                response = utils.rag_with_url(target_url,prompt)
                st.markdown(response)
                st.divider()
            with col_normal:
                with st.spinner("Processing..."):
                    st.info("Response: Answering without RAG...")
                    response = utils.rag_with_pdf(prompt)
                    st.markdown(response)
                    st.divider()
            

st.title("Retrieval-Augmented Generation (RAG) with LangChain : PDF ")
st.divider()

col_input , col_rag , col_normal = st.columns([3,5,5])
with col_input:
    selected_file = st.file_uploader("PDF File", type=["pdf"])
    st.divider()
    prompt = st.text_input("Prompt",key="pdf_prompt")
    st.divider()
    sumbit_btn = st.button(label="Submit",key="pdf_btn")

if sumbit_btn:
    with col_rag:
        with st.spinner("Processing..."):
            st.success("Response: Answering with RAG...")
            response,relevant_documents = utils.rag_with_pdf(file_path=f"./data/{selected_file.name}",
                                                                  prompt=prompt)
            st.markdown(response)
            st.divider()
            st.info("Documents")
            for doc in relevant_documents:
                st.caption(doc.page_content)
                st.markdown(f"Source: {doc.metadata}")
                st.divider()

            with col_normal:
                with st.spinner("Processing..."):
                    st.info("Response: Answering without RAG...")
                    response = utils.ask_gemini(prompt)
                    st.markdown(response)
                    st.divider()
            