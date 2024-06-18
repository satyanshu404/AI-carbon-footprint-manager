import os
import pandas as pd
import PyPDF2
import docx
import json
from bs4 import BeautifulSoup

class ReadFiles():
    '''Reads the content of a file based on its extension and returns the content as a string or a dataframe.'''
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_extension = os.path.splitext(file_path)[1].lower()
    
    def read_file(self):
        if self.file_extension == '.txt':
            return self.read_txt()
        elif self.file_extension == '.pdf':
            return self.read_pdf()
        elif self.file_extension == '.docx':
            return self.read_docx()
        elif self.file_extension in ['.csv', '.xlsx', '.xls']:
            return self.read_excel_or_csv()
        elif self.file_extension == '.html':
            return self.read_html()
        elif self.file_extension == '.json':
            return self.read_json()
        else:
            raise ValueError(f"Unsupported file type: {self.file_extension}")
    
    def read_txt(self):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def read_pdf(self):
        with open(self.file_path, 'rb') as file:
            reader = PyPDF2.PdfFileReader(file)
            text = ''
            for page in range(reader.numPages):
                text += reader.getPage(page).extractText()
            return text
    
    def read_docx(self):
        doc = docx.Document(self.file_path)
        return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    
    def read_excel_or_csv(self):
        if self.file_extension == '.csv':
            df = pd.read_csv(self.file_path)
        else:
            df = pd.read_excel(self.file_path)
        return df.to_string(index=False)
    
    def read_json(self) -> str:
        with open(self.file_path, 'r') as file:
            return json.dumps(json.load(file), indent=4)
    
    def read_html(self) -> str:
        with open(self.file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
            return soup.get_text()
        
