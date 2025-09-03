from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


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
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        # Ask question
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            # Gemini LLM
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
            chain = load_qa_chain(llm, chain_type="stuff")
            
            response = chain.run(input_documents=docs, question=user_question)
            st.subheader("Answer:")
            st.write(response)

if __name__ == '__main__':
    main()
