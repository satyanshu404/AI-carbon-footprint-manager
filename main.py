import os
import json
import models
import requests
from openai import OpenAI
from dotenv import load_dotenv
from constants import Constants
import prompts.prompts as prompts
import pprint

load_dotenv()


os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

def main():
    message: list[dict]= [{"role": "system", "content": "You are a helpful assistant and give results only in json format."}]
    message.append({"role": "user", "content": prompts.general_instruction()})

    with open(Constants().TOOL_SPECIFICATION_JSON_PATH, 'r') as file:
        callable_functions = json.load(file)
    
    documentation_urls = "data/PACT Initiative Developer Documentation.html"
    doc_urls = ["data\planet-positive.pdf"]

    message.append({"role": "user", "content": f"Documentation file path: {documentation_urls}\n Document file paths: {doc_urls}"})
    i = 0
    while i < 4:
        print("Iteration: ", i, "="*50)
        print(message)
        print("-"*50)

        response = models.call_gpt(message, functions=callable_functions)
        print(response["data"])
        if response["error"]:
            print(response['error'])
            break

        function_name = response['data'].function_call.name
        arguments = json.loads(response['data'].function_call.arguments)

        

        if function_name == "extract_data_model_from_url":
            message.append(response['data'])
            function_response = models.extract_data_model_from_url(**arguments)
        
            message.append({"role": "function", 
                            "name": function_name,
                            "content": function_response["data"]['data'].content})
        if function_name == "read_pdf":
            message.append(response['data'])
            function_response = models.scrape_data(**arguments)
            message.append({"role": "function", 
                        "name": function_name,
                        "content": function_response['data']})

        i += 1




if __name__ == "__main__":
    main()
