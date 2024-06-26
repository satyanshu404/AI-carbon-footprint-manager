import os
import sys

# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
import utils as utils
import constants
from prompts import prompts
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
def code_generator_and_executer(prompts:str, function_name: str, **kwargs):
    '''Create and Executes the code provided in the prompts and returns the result.
    Arguments:
    prompts: str - A detailed prompt with informations and insturctions for generating the code.
    function_name: str - The name of the function to execute.
    **kwargs - The necessary arguments to pass to the function.
    '''
    try:
        logging.log(logging.INFO, "Executing the code...")
        return utils.CodeExecuter().execute_code(prompts, function_name, **kwargs)
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

@tool
def ai_assistant(prompt:str) -> str:
    '''Executes the AI assistant and returns the response.'''
    try:
        logging.log(logging.INFO, "Executing the AI assistant...")
        return utils.GptModel().get_completion(prompt)
    except Exception as e:
        logging.log(logging.ERROR, f"An error occurred from ai assistant side: {str(e)}")
        return f"An error occurred from ai assistant side: {str(e)}"
    
@tool
def get_product_names(data_model_type:str, file_paths:list[str]) -> str:
    '''Returns the list of product names.'''
    try:
        logging.log(logging.INFO, "Getting product names...")

        read_files = utils.ReadFiles()
        text_splitter = utils.TextSplitter()
        assistant = utils.GptModel()

        data_model_type = data_model_type.lower()
        data_model:str = utils.ReadFiles().read_txt(constants.DataModelGeneratorConstants().DATA_MODEL_PATH[data_model_type])

        products_names = []
        for file_path in file_paths:
            content = read_files.read_file(file_path)
            chunks = text_splitter.split_text(content)
            for chunk in chunks:
                prompt = prompts.get_prompt_for_getting_product().format(data_model, chunk)
                response = assistant.get_completion(prompt)
                formated_response:list = utils.FormatResponse().format_response(response, 'python')
                products_names.extend(formated_response)
        return list(set(products_names))
    except Exception as e:
        logging.log(logging.ERROR, f"An error occurred while getting the product names: {str(e)}")
        return f"An error occurred while getting the product names: {str(e)}"