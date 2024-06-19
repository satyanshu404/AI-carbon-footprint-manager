import sys
import os

# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from dotenv import load_dotenv
from constants import Constants
from tools.utils import ReadFiles
import prompts.prompts as prompts
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

load_dotenv()

@tool
def create_json_summary(file_paths: list[str], save_dir_path: str) -> str:
    '''Create summaries for the data model for better understanding of the json.
        Arguments:
            file_paths: List of file paths of the json objects (data models).
            save_dir_path: Path to save the summaries must be in txt format. and the name of the file must be intutive. '''
    try:
        print("Creating summaries...")
        documents = []
        documents.extend([ReadFiles(file_path).read_json() for file_path in file_paths])
        llm = ChatOpenAI(model=Constants.MODEL_NAME, temperature=Constants.TEMPERATURE)
        prompt_template = prompts.get_json_summary_prompt_template()
        output_parser = StrOutputParser()
        
        chain = prompt_template | llm | output_parser
        
        # summaries = [chain.invoke({"json_object": document}) for document in documents[0]]
        summaries = [print(f"Processing document {index + 1}/{len(json.loads(documents[0]))}") or chain.invoke({"json_object": document}) for index, document in enumerate(json.loads(documents[0]))]
        print(f"\n\nSummaries created successfully and saved at location '{save_dir_path}'.")

        # save the file
        with open(save_dir_path, "w") as f:
            f.write("\n".join(summaries))
        return f"successfully created summaries of the json object and saved at location '{save_dir_path}'."
    except Exception as e:
        return f"Error in creating summaries: {e}"
    

if __name__ == "__main__":
    create_json_summary.invoke({"file_paths": ['data/data_model/data_model.json'], "save_dir_path": 'data/data_model/test.txt'})
    
