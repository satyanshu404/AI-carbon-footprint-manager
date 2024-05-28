import os
import json
import models
import requests
from openai import OpenAI
from dotenv import load_dotenv
from constants import Constants
import prompts
import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core import SimpleDirectoryReader
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

load_dotenv()


def main():
    pass

if __name__ == "__main__":
    main()