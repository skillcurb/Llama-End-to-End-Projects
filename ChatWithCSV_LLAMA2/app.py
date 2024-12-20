from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
import os
import streamlit as st
import pandas as pd

import tempfile
# Function to get response from LLaMA2 or similar model
def get_model_response(data,query):
    DB_FAISS_PATH = "vectorstore/db_faiss"

    # Split the text into Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(data)

    print(len(text_chunks))

    # Load a model for generating embeddings from HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name='moka-ai/m3e-base')

    # COnverting the text Chunks into embeddings and saving the embeddings into FAISS Knowledge Base
    docsearch = FAISS.from_documents(text_chunks, embeddings)

    docsearch.save_local(DB_FAISS_PATH)

    llm = CTransformers(model="models/llama-2-7b-chat.ggmlv3.q4_0.bin",
                        model_type="llama",
                        max_new_tokens=512,
                        temperature=0.1)

    qa = ConversationalRetrievalChain.from_llm(llm, retriever=docsearch.as_retriever())

    result = qa(query)

    return "result"




# Load CSV file

def load_csv(file_path):
    return pd.read_csv(file_path)

# Main app
def main():
    st.title("Chat with CSV File")

    # File uploader
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getvalue())
    if uploaded_file is not None:
        df = load_csv(uploaded_file)
        st.write("CSV File Contents:")
        st.dataframe(df)
        user_input = st.text_input("Your Message:")
    #    try:
        csv_loader = CSVLoader(f.name)
        data = csv_loader.load()
        if user_input:
            response = get_model_response(data,user_input)
            st.text_area(response)
    #        print(data)
            # Chat interface

    #    except Exception as e:
     #      print(f"Error processing CSV file: {e}")



if __name__ == "__main__":
    main()
