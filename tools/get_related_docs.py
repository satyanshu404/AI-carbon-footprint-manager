import sys
import os

# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from constants import Constants
from tools.utils import ReadFiles
from langchain.agents import tool
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core import Document
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

load_dotenv()


@tool
def reterive_chunks(query:str, file_paths: list[str]) -> list[str] | str:
    '''Retrive the relevant chunks from the files at the based on the query and returns the chunks as a list of strings which are important to the query.'''
    try:
        print("Calling reterive_chunks...")
        print("Retrieving chunks...")
        RF = ReadFiles()
        documents = [Document(text=RF.read_file(file_path)) for file_path in file_paths]
        index = VectorStoreIndex.from_documents(documents)
        retriever = VectorIndexRetriever(
                    index=index,
                    similarity_top_k=Constants.SIMILARITY_TOP_K,
                )

        response_synthesizer = get_response_synthesizer()

        # assemble query engine
        print("Calling query engine...")
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=Constants.SIMILARITY_CUTOFF)],
        )
        
        response = query_engine.query(query)
        print(f"Got {len(response.source_nodes)} chunks.")
        return [doc.text for doc in response.source_nodes]
    except Exception as e:
        return f"Got an error while reteriving the chunks: {e}"