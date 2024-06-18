import os
import json
import models
import requests
from openai import OpenAI
from dotenv import load_dotenv
from constants import Constants
import prompts
import streamlit as st

from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

load_dotenv()
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

def main():
    st.title("Search Engine")
    to_read_file_path = "data/data_model/data_model.json"
    file_path = "data/data_model/corpus.json"

    # Check if the file exists
    if not os.path.isfile(file_path):
        # If the file does not exist, create it and write some content
        with open(file_path, 'w') as file:
            corpus = models.generate_summary_from_data_model(to_read_file_path)
            with open(file_path, 'w') as file:
                json.dump(corpus, file, indent=4)
        print(f'Created the file at location {file_path}')
    else:
        # If the file exists, read its contents
        with open(file_path, 'r') as file:
            corpus = json.load(file)
        print(f'Loaded the data at located {file_path}')

    documents = [Document(text=doc['summary']) for doc in corpus]

    # create index
    index = VectorStoreIndex.from_documents(documents)
    
    if 'query' not in st.session_state:
        st.session_state.query = ''
    
    st.session_state.query = st.text_input("Enter your query:", st.session_state.query)

    if st.session_state.query:
        if st.session_state.query.lower() == "exit":
            st.write("Exiting...")
            # Optionally reset the query or handle the exit condition
            st.session_state.query = ''
        else:
            st.write(f"You entered: {st.session_state.query}")
            # configure retriever
            retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=5,
            )

            # configure response synthesizer
            response_synthesizer = get_response_synthesizer()

            # assemble query engine
            query_engine = RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=response_synthesizer,
                node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
            )

            response = query_engine.query(st.session_state.query)
                # print(response)
            st.write(response.response)
            st.write(response.source_nodes)
            st.write([doc.text for doc in response.source_nodes])
            st.session_state.query = ''
    else:
        st.write("Please enter a query")
    pass

if __name__ == "__main__":
    main()