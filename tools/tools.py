import os
import sys

# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
import tools.utils as utils
import constants
from langchain.agents import tool
import logging

load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@tool
def read_files(file_paths:list[str]) -> list[str] | str:
    '''Reads the content files from the provided file paths and returns the content as a list of strings.'''
    try:
        logging.log(logging.INFO, "Reading files...")
        read_files = utils.ReadFiles()
        return [read_files.read_file(file_path) for file_path in file_paths]
    except Exception as e:
        logging.log(logging.ERROR, f"An error occurred while reading the files: {str(e)}")
        return f"An error occurred while reading the files: {str(e)}"
    
@tool
def google_search(query: str) -> str:
    '''Searches the web for the provided query and returns the links of the related web pages.'''
    try:
        logging.log(logging.INFO, "Searching the web...")
        urls =  utils.GoogleSearch().search(query, restult_count=constants.GoogleSearchConstants.NUMBEROFSEARCHS)
        return [utils.WebScraper().scrape(url, query) for url in urls]
    except Exception as e:
        logging.log(logging.ERROR, f"An error occurred while searching the web: {str(e)}")
        return f"An error occurred while searching the web: {str(e)}"
    
@tool
def get_image_content(image_path: str) -> str:
    '''Reads the content of the image at the provided path and returns the content as a string.'''
    try:
        logging.log(logging.INFO, "Reading image content...")
        return utils.ReadImage().read_image(image_path)
    except Exception as e:
        logging.log(logging.ERROR, f"An error occurred while reading the image content: {str(e)}")
        return f"An error occurred while reading the image content: {str(e)}"
    
@tool
def code_generator_and_executer(prompts:str, function_name: str, *args):
    '''Create and Executes the code provided in the prompts and returns the result.
    Arguments:
    prompts: str - A detailed prompt with informations and insturctions for generating the code.
    function_name: str - The name of the function to execute.
    *args: Any - The arguments to pass to the function for calculations.
    '''
    try:
        logging.log(logging.INFO, "Executing the code...")
        return utils.CodeExecuter().execute_code(prompts, function_name, *args)
    except Exception as e:
        logging.log(logging.ERROR, f"An error occurred while executing the code: {str(e)}")
        return f"An error occurred while executing the code: {str(e)}"
    
@tool
def reterive_data(query: str, file_paths: list[str]) -> str:
    '''Retrieves the data related to the provided query and returns the data.'''
    try:
        logging.log(logging.INFO, "Retrieving data...")
        return utils.Retriever().reterive(query, file_paths)
    except Exception as e:
        logging.log(logging.ERROR, f"An error occurred while retrieving the data: {str(e)}")
        return f"An error occurred while retrieving the data: {str(e)}"