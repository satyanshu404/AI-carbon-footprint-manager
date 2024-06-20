import sys
import os

# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import PyPDF2
import docx
import json
from bs4 import BeautifulSoup
from openai import OpenAI
import base64
from enum import Enum
from io import BytesIO, StringIO
from typing import Union
from constants import Constants
import models
from PIL import Image
from dotenv import load_dotenv
from langchain.agents import tool

load_dotenv()

class ReadFiles():
    '''Reads the content of a file based on its extension and returns the content as a string or a dataframe.'''
    def __init__(self):
        pass

    def get_extension(self, file_path:str) -> str:
        '''Returns the extension of a file.'''
        return os.path.splitext(file_path)[1].lower()
    
    def read_file(self, file_path:str) -> str:
        '''Reads the content of a file based on its extension and returns the content as a string or a dataframe.'''
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
    
    def read_pdf(self, file_path:str) -> str:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfFileReader(file)
            text = ''
            for page in range(reader.numPages):
                text += reader.getPage(page).extractText()
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
        
class ReadImage():
    def __init__(self, model_name:str = Constants().MODEL_NAME):
        self.model_name:str = model_name
        self.client = OpenAI()
    
    def encode_image(self, image):
        '''Encodes an image to base64 format'''
        return base64.b64encode(image).decode("utf-8")
    
    def read_image(self, image_path:str) -> str:
        '''Reads an image'''
        try:
            image = Image.open(image_path)
            encoded_image = self.encode_image(image)
            messages=[
                    {"role": "system", "content": "You are a helpful assistant that responds in Markdown."},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Carefully analyze the image and extract its content, then present the extracted information in JSON format., nothing else"},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{encoded_image}"}
                        }
                ]}]
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"A error occurred while reading the image: {str(e)}"


