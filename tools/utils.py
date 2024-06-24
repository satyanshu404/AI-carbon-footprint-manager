import sys
import os

# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import pandas as pd
import PyPDF2
import docx
import json
from bs4 import BeautifulSoup
from openai import OpenAI
import base64
import mimetypes
import requests
import constants
import subprocess
import importlib.util
import constants 
from groq import Groq
from langchain.agents import tool
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core import Document
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

from dotenv import load_dotenv


load_dotenv()


class ReadFiles:
    '''Reads the content of a file based on its extension and returns the content as a string or a dataframe.'''
    def __init__(self):
        pass

    def get_extension(self, file_path:str) -> str:
        '''Returns the extension of a file.'''
        return os.path.splitext(file_path)[1].lower()
    
    def read_file(self, file_path:str) -> str:
        '''Reads the content of a file based on its extension and returns the content as a string or a dataframe.'''
        print(self.get_extension(file_path))
        if self.get_extension(file_path) == '.txt':
            return self.read_txt(file_path)
        elif self.get_extension(file_path) == '.pdf':
            return self.read_pdf(file_path)
        elif self.get_extension(file_path) == '.docx':
            return self.read_docx(file_path)
        elif self.get_extension(file_path) in ['.csv', '.xlsx', '.xls']:
            return self.read_excel_or_csv(file_path)
        elif self.get_extension(file_path) == '.html':
            return self.read_html(file_path)
        elif self.get_extension(file_path) == '.json':
            return self.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {self.get_extension(file_path)}")
    
    def read_txt(self, file_path:str) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def read_pdf(self, file_path: str) -> str:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
        return text
    
    def read_docx(self, file_path:str) -> str:
        doc = docx.Document(file_path)
        return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    
    def read_excel_or_csv(self, file_path:str) -> str:
        if self.get_extension(file_path) == '.csv':
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        return df.to_string(index=False)
    
    def read_json(self, file_path:str) -> str:
        with open(file_path, 'r') as file:
            return json.dumps(json.load(file), indent=4)
    
    def read_html(self, file_path:str) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
            return soup.get_text()
        
class ReadImage:
    def __init__(self, model_name:str = constants.Constants.MODEL_NAME):
        self.model_name:str = model_name
        self.client = OpenAI()
    
    def image_to_base64(self, image_path:str):
        '''Converts an image to base64 string'''
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type or not mime_type.startswith('image'):
            raise ValueError("The file type is not recognized as an image")
        
        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        image_base64 = f"data:{mime_type};base64,{encoded_string}"
        return image_base64

    def read_image(self, image_path:str) -> str:
        '''Reads an image using OPENAI'''
        try:
            base64_string = self.image_to_base64(image_path)
            messages=[
                    {"role": "system", "content": "You are a helpful assistant that responds in Markdown."},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Carefully analyze the image and extract its content, then present the extracted information in JSON format., nothing else"},
                        {"type": "image_url", 
                         "image_url": {
                            "url": base64_string}
                        }
                ]}]
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"A error occurred while reading the image: {str(e)}"

class GoogleSearch:
    def __init__(self):
        self.key = os.getenv('GOOGLE_API_KEY')
        self.cx = os.getenv('GOOGLE_SEARCH_ENGINE_ID')
        self.url = constants.GoogleSearchConstants.GOOGLE_SEARCH_URL
        self.save_file_path = constants.GoogleSearchConstants.GOOGLE_SEARCH_RESULTS_PATH

    def build_payloads(self, 
                       query: str, 
                       start:int=1, 
                       num:int=10,
                       **params)->dict:
        ''' Build the payloads for the Google Search API'''

        payload = {
            'key': self.key,
            'cx': self.cx,
            'q': query,
            'start': start,
            'num': num,
        }
        payload.update(params)
        return payload

    def make_request(self, payload: dict) -> dict:
        '''Make a request to the Google Search API'''
        response = requests.get(self.url, params=payload)
        return response.json()

    def clean_file_name(self, file_name:str)->str:
        '''Clean the file name'''
        # remove special characters
        file_name = file_name.lower()
        return re.sub(r'[^\w\s-]', '', file_name).strip().replace(' ', '_')

    def search(self,
               query:str,
               restult_count:int=10)->str:
        '''Search Google and return status of the search'''
        items = []
        reminder = restult_count%10
        if reminder > 0:
            pages = restult_count//10 + 1
        else:
            pages = restult_count//10 

        try:
            for i in range(pages):
                if pages == i+1 and reminder > 0:
                    payload = self.build_payloads(query, start=(i+1)*10, num=reminder)
                else:
                    payload = self.build_payloads(query, start=(i+1)*10)
                # print(payload)
                response = self.make_request(payload)
                # print(response)
                items.extend(response['items'])

            query_string_cleaned = self.clean_file_name(query)
            df = pd.json_normalize(items)
            search_urls = [url for url in df['link']]
            # save_file_path = '{0}/Search_resutls_{1}.csv'.format(self.save_file_path, query_string_cleaned)
            # df.to_csv(save_file_path, index=False)
            return search_urls
        except Exception as e:
            return f'''Error occured during search: {e}'''

class GroqModel:
    '''Groq model class'''
    def __init__(self, model_name:str = constants.GroqModelConstants.MODEL_NAME):
        self.client = Groq()
        self.model = model_name
    def get_completion(self, messages, **kwargs) -> str:
        '''Get completion from the model'''
        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            **kwargs
        )
        if chat_completion:
            return chat_completion.choices[0].message.content
       
class CodeExecuter:
    def __init__(self):
        pass
    
    def generate_code(self, prompt:str, filename:str, function_name:str):
        '''Generate code from the messages'''
        groq = GroqModel()
        messages = [
            {'role': 'system', 'content': 'You are a Python tutor. Just give the python code in text, nothing else, no notations, not any asterisk, nothing. Just the code.'},
            {'role': 'system', 'content': f'The pyhton function must have the name {function_name}'},
            {'role': 'user', 'content': prompt}]
        code =  groq.get_completion(messages)

        with open(filename, 'w+') as file:
            file.write(code)

    def load_function_from_file(self, filename:str, function_name:str):
        '''Load the function from a file'''
        spec = importlib.util.spec_from_file_location(function_name, filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, function_name)
    
    def execute_code(self, prompt:str, function_name:str, **kwargs):
        '''Execute the generated function'''
        try:
            code_filename = f"{constants.CodeExecuterConstants.BASEFILEPATH}/generated_code.py"
            self.generate_code(prompt, code_filename, function_name)
            generated_function = self.load_function_from_file(code_filename, function_name)
            result = generated_function(**kwargs)
            return result
        except Exception as e:
            return f"An error occurred while executing the code: {str(e)}"
        
class Retriever:
    def __init__(self):
        self.RF = ReadFiles()

    def create_documents(self, file_paths: list[str]) -> list[Document]:
        '''Create the documents from the file paths'''
        return [Document(text=self.RF.read_file(file_path)) for file_path in file_paths]
    
    def create_reteriver(self, documents: list[Document]) -> VectorStoreIndex:
        '''Create the reteriver'''
        index = VectorStoreIndex.from_documents(documents)
        retriever = VectorIndexRetriever(
                    index=index,
                    similarity_top_k=constants.RetrieverConstants.SIMILARITY_TOP_K,
                )
        return retriever
    
    def query_engine(self, query:str, retriever:VectorIndexRetriever) -> RetrieverQueryEngine:
        '''create the query engine and call the engine for the query'''
        response_synthesizer = get_response_synthesizer()
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=constants.RetrieverConstants.SIMILARITY_CUTOFF)],
        )
        response = query_engine.query(query)
        return response

    def reterive(self, query:str, file_paths: list[str]) -> list[str] | str:
        '''Retrive the relevant chunks from the files at the based on the query and returns the chunks as a list of strings which are important to the query.'''
        try:
            print("Retrieving chunks...")
            documents = self.create_documents(file_paths)
            retriever = self.create_reteriver(documents)
            response = self.query_engine(query, retriever)

            print(f"Got {len(response.source_nodes)} chunks.")
            return response, [doc.text for doc in response.source_nodes]
        except Exception as e:
            return f"Got an error while reteriving the chunks: {e}"

class WebScraper:
    def __init__(self):
        self.header = {'User-Agent': constants.WebScraperConstants.USERAGENT}

        if not os.path.exists(constants.WebScraperConstants.SCRAPER_FILE_LOCATION):
                os.makedirs(constants.WebScraperConstants.SCRAPER_FILE_LOCATION)

    def make_request(self, url:str):
        '''Make a request to the url'''
        response = requests.get(url, headers=self.header)
        return response
    
    def scrape(self, url:str, query:str = None) -> tuple[str, list[str]] | str:
        '''Scrape the content aof the url and return the relevant content based on the query'''
        try:
            response = self.make_request(url)

            if len(url.split('/')[-1]) == 0:                
                file_path = os.path.join(constants.WebScraperConstants.SCRAPER_FILE_LOCATION, url.split('/')[-2])
            else:
                file_path = os.path.join(constants.WebScraperConstants.SCRAPER_FILE_LOCATION, url.split('/')[-1])

            if not file_path.endswith(('.pdf','.xlsx','.xls','.csv','.docx','.txt')):
                file_path += '.html'

            with open(file_path, 'wb') as file:
                file.write(response.content)
            if query:
                return Retriever().reterive(query, [file_path])
            
            return ReadFiles().read_file(file_path)
        except Exception as e:
            return f"An error occurred while scraping the content: {str(e)}"

    