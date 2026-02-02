from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
# from langchain_community.llms import Ollama

import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

@st.cache_resource(show_spinner="Processing...")
def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_texts(chunks, embeddings)

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")
    
    # Upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        
        # Split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # Embeddings with Gemini
        # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        knowledge_base = create_vectorstore(chunks)
        
        # Ask question
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            # docs = knowledge_base.similarity_search(user_question)
            
            # Gemini LLM
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
            # llm = Ollama(model="llama3.2")
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=knowledge_base.as_retriever(search_kwargs={"k": 3})
            )
            
            response = chain.run(user_question)
            st.subheader("Answer:")
            st.write(response)
        # Footer GitHub Link (Bottom Right Corner)
    st.markdown(
        """
        <style>
        .github-corner {
            position: fixed;
            bottom: 10px;
            right: 15px;
            font-size: 14px;
        }
        .github-corner a {
            text-decoration: none;
            color: #6c63ff;
            font-weight: bold;
        }
        </style>
        <div class="github-corner">
            <a href="https://github.com/vijaypatange21/Ask_your_pdf" target="_blank">
                🔗 View on GitHub
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()
