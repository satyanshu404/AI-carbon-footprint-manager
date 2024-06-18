import os
import json
from openai import OpenAI
import pandas as pd
from bs4 import BeautifulSoup
from constants import Constants
import requests
import os
import re
import pdfplumber
from docx import Document
import ast

from dotenv import load_dotenv

import streamlit as st
from langchain.text_splitter import MarkdownTextSplitter

from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core import SimpleDirectoryReader
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor


load_dotenv()
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

def call_gpt(messages: list[dict[str, str]], 
             functions:list[dict] = None, 
             tools:list[dict] = None,
             response_format:str = None,
             temperature:float = Constants().TEMPERATURE,
             model: str = Constants().MODEL_NAME) -> dict[str, str]:
    data = None
    error = None
    try:
        client = OpenAI()
        chat_completion = client.chat.completions.create(
        messages=messages,
        functions=functions,
        tools=tools,
        temperature=temperature,
        model=model,
        )
        data = chat_completion.choices[0].message
    except Exception as e:
        error = f"Error in call_gpt: {e}"
        print(error)
        
    return {"data": data,
            "error": error}
    
def extract_text(file_path: str) -> dict[str, str]:
    data = None
    error = None
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(content, 'html.parser')

        text_content = soup.get_text()

        data = text_content
    except Exception as e:
        error = f"Error in extract_text: {e}"
        print(error)

    return {"data": data,
            "error": error}

def scrape_data(url:str) -> dict[str, str]:
    data = None
    error = None
    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        data = soup.get_text()
    except Exception as e:
        error = f"Error in scrape_data: {e}"
        print(error)
        
    return {"data": data    ,
            "error": error}

def chunk_document(document:str, 
                   chunk_size:int=Constants().CHUNK_SIZE, 
                   chunk_overlap:int=Constants().CHUNK_OVERLAP) -> dict[str, list[str] | str]:
    '''
    This function take the document as input and return the list of chunked document.
    The document is the chunked on the basis of Markdown. It uses the Markdown Text splitter from the lang chain.
    '''
    data = None
    error = None
    try:
        splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_text(document)

        data = chunks
    except Exception as e:
        error = f"Error in chunk_document: {e}"
        print(error)
        
    return {"data": data,
            "error": error} 
    
def extract_data_model(document:str) -> dict[str, str]:
    data = None
    error = None
    try:
        messages: str = [{"role": "system", "content": "You are a helpful assistant and give response only in json format."}]
        messages.append({"role": "user", "content": "From the given below the documentation, extract the complete json object which is being used to store the product's carbon footprint and convert that object into a generic json object, whcih can used to store the carbon footprint of any product."})
        messages.append({"role": "user", "content": "Return only json object in simple plain text, nothing else since it will be directly used in code files."})
        messages.append({"role": "user", "content": "and for values of each key, put what data type it can hold based on the given data model. For ex: 'carbon_footprint': int."})
        #   messages.append({"role": "user", "content": "And If Id or something like this present in the json object then do not include it the final json result."})
        messages.append({"role": "user", "content": document})
        res: dict = call_gpt(messages)
        data = res
    except Exception as e:
        error = f"Error in extract_data_model: {e}"
        print(error)
       
    return {"data": data,
            "error": error}

def get_chunked_data(chunks: list[str]) -> dict[str, str]:
    try:
        messages: list[dict[str, str]]=[
        {"role": "system", "content": "You are a helpful assistant and give response in 'YES' or 'NO'."},
        {"role": "system", "content": f''' You are given a chunk of a Policy Document. Find wether the complete json object is present in the chunk or not, which is being used to store the carbon foot print of the product. If the json object is present in the chunk, respond with 'YES' otherwise respond with 'NO'.'''}]

        final_chunk: str =  ""
        for chunk in chunks:
            messages.append({"role": "user", "content": chunk})
            response = call_gpt(messages)

            if response["data"]:
                if response["data"].content == "YES":
                    final_chunk += f"{chunk}\n\n"
        return {"data": final_chunk,
                "error": None}
    except Exception as e:
        print("Error in get_chunked_data: ", str(e))
        return {"data": None,
                "error": str(e)}
    
def create_validation_script(data_model:dict) -> dict[str, str]:
    try:
        message=[
        {"role": "user", "content": "From the given below the data model, create a python script in one go for validating the json object using Pydantic libaray. Assume that the all the necessary libraries are installed. just wirte the python script."},
        {"role": "user", "content": "Just return the python script, nothing else. The script must have a callable function named as 'validate_data_model' which will take the json object as input and return the validated json object."}, 
        {"role": "user", "content": data_model}]
        response = call_gpt(message)
        return response
    except Exception as e:
        print("Error in create_validation_script: ", str(e))
        return {"data": None,
                "error": str(e)}

def extract_data_model_from_url(doc_url:str) -> dict[str, str]:
    data = None
    error = None
    try:
        extracted_text = extract_text(doc_url)
        if extracted_text["error"]:
            error = extracted_text["error"]
            return {"data": data,
                    "error": error}
        chunks = chunk_document(extracted_text["data"])
        if chunks["error"]:
            error = chunks["error"]
            return {"data": data,
                    "error": error}
        relevant_chunks = get_chunked_data(chunks["data"])
        if relevant_chunks["error"]:
            error = relevant_chunks["error"]
            return {"data": data,
                    "error": error}
        if relevant_chunks['error']:
            error = relevant_chunks['error']
            return {"data": data,
                    "error": error}
        data_model = extract_data_model(relevant_chunks['data'])
        if data_model['error']:
            error = data_model['error']
            return {"data": data,
                    "error": error}
        data = data_model['data']
    except Exception as e:
        error = f"Error in extract_data_model: {e}"
        print(error)
        
    return {"data": data,
            "error": error}

def save_uploaded_file(uploaded_file):
    try:
        file_path = os.path.join(Constants().RETRIEVER_FILE_LOCAITON, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        print("Error in save_uploaded_file: ", str(e))
        return None

def upload_files() -> list[str]:
    '''
    This function take files input from the user and returns the list of file paths.
    '''
    uploaded_files = st.file_uploader("Upload Documents", type=None, accept_multiple_files=True)
    # stores the file paths
    file_paths:list[str] = []

    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            # Save the uploaded file
            file_path = save_uploaded_file(uploaded_file)
            if file_path:
                file_paths.append(file_path)
    return file_paths

def extract_text(file_path):
    _, file_extension = os.path.splitext(file_path)

    if file_extension == '.txt':
        return extract_text_from_txt(file_path)
    elif file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension == '.docx':
        return extract_text_from_docx(file_path)
    elif file_extension == '.csv':
        return extract_text_from_csv(file_path)
    elif file_extension in ['.xls', '.xlsx']:
        return extract_text_from_excel(file_path)
    elif file_extension == '.html':
        return extract_text_from_html(file_path)
    else:
        print(ValueError(f"Unsupported file type: {file_extension}", f'\nSkipping the file at location: {file_path}'))

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
        return text

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = ''
    for para in doc.paragraphs:
        text += para.text + '\n'
    return text

def extract_text_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df.to_string(index=False)

def extract_text_from_excel(file_path):
    df = pd.read_excel(file_path)
    return df.to_string(index=False)

def extract_text_from_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
        return soup.get_text()

def extract_text_from_json(file_path:str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def clean_response(text:str, pattern:str):
    """
    Extracts a Python list from text surrounded by backticks (`) and the specified pattern.
    
    Args:
    - text (str): Input text containing the Python list surrounded by backticks and pattern.
    - pattern (str): The pattern (e.g., "python") to match and remove along with the backticks.
    
    Returns:
    - list: The extracted Python list, or None if extraction fails.
    """
    # Construct regex pattern to match backticks and the specified pattern
    pattern_regex = rf'^```(?:{pattern})?\n|\n```$'
    
    # Remove surrounding ``` and the specified pattern
    cleaned_text = re.sub(pattern_regex, '', text, flags=re.MULTILINE)
    
    # Safely parse the Python list
    try:
        python_list = ast.literal_eval(cleaned_text.strip())
        return python_list
    except (ValueError, SyntaxError) as e:
        print(f"Error extracting Python list: {e}")
        return None

def find_product_from_documents(file_paths:list[str], data_model: dict[str, str]) -> dict[str, str]:
    '''
    This function takes the file paths and data model as input and returns the name of the product for which data model is to be created.
    '''
    try:
        text = f"{'='*50}\n".join([extract_text(file_path) for file_path in file_paths])
        chunks = chunk_document(text)
        products = []
        for chunk in chunks['data']:
            messages: list[dict[str, str]]=[
            {"role": "system", "content": "You are a advanced assistant having the capability to understand the instruction properly and context of the given data and provide the response accordingly and you always give respone only as python list."},
            {"role": "user", "content": f'''

            Please extract the names of products for which carbon footprint data can be extracted from the document. The document contains information about various products and their associated carbon footprints.

            Data Model: {data_model}

            Document: {chunk}

            Our objective is to create a JSON object following the specified data model. This model is a standard format used in our documentation.

            The PACT Initiative aims to facilitate decarbonization through a global network for exchanging verified product emissions data. This transparency helps businesses accurately measure, account for, and disclose their carbon emissions, supporting efforts to achieve net zero emissions by 2050.

            Please return a Python list containing only the names of products for which carbon or emission data are explicitly provided in the document.
            If there isn't any product name mentioned in the document, return an empty list.
            Do not include any other than prodcut names in the final list avoid including city names or other things. 
        '''}]
            response = call_gpt(messages, temperature=0.1)
            # if response['data']:
            # # cross check the products with the data model
            #     messages.append({'role': 'assistant', 'content': str(response['data'])})
            #     messages.append({'role': 'user', 'content': 'Cross check the above your product names and exclude those product names from the list which are irrelevant.'})
            #     response = call_gpt(messages)
            print(response['data'].content)
            if response['data']:
                product = clean_response(response['data'].content, 'python')
                products.extend(product)
            else:
                print(response['error'])
            
        return set(products)
    except Exception as e:  
        error = f"Error in find_product_from_documents: {e}"
        print(error)
        
    return None

def generate_summary_from_data_model(file_path:str) -> list[dict[str, str]]:
    try:
        json_objects = extract_text_from_json(file_path)
        if not json_objects:
            return None
        docs = []
        for json_object in json_objects:
            messages = [
                {"role": "system", "content": "You are a helpful assistant. You have the capability to think and to understand the given instructions."},
                {"role": "user", "content": f'''
                Objective: Generate a natural language summary of the given JSON object, incorporating all key-value pairs where the value is not null.
                Input:
                A JSON object containing various fields with corresponding values.
                                
                Output:
                A text summary in natural language that includes:
                Descriptions of each field and its corresponding value.
                Only those fields where the value is not null.
                                
                Guidelines:
                Examine each field in the JSON object.
                If the value of a field is null, exclude that field from the summary.
                For fields with non-null values, describe them in a coherent sentence.
                Ensure that the summary is clear, concise, and reads like a natural language description.
                                
                JSON Object: {json_object}

                Give only the summary as the response.
                 
            '''}]

            summary_response = call_gpt(messages)
            if summary_response['data']:
                docs.append({"summary": summary_response['data'].content, "json_object": json_object})
        return docs
    except Exception as e:
        print("Error in generate_summary: ", str(e))
        return []
    
