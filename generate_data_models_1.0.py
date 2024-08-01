import os
import json
from dotenv import load_dotenv
import constants
from tools import utils, tools, reteriver
import prompts.prompts as prompts
import streamlit as st
import logging

load_dotenv(override=True)
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DataModelGenerator:
    def __init__(self):
        logging.log(logging.INFO, "Initializing the data model generator application...")
        self.data_model_list = constants.DataModelGeneratorConstants().DATA_MODEL_LIST 
        
        self.repo_path = constants.DataModelGeneratorConstants.REPO_PATH
        self.save_file_dir = constants.DataModelGeneratorConstants.SAVE_FILE_DIR

        self.reteriver = reteriver.RetrieverRouter()

        # temporary file for downloading the data 
        if not os.path.exists(self.repo_path):
            os.makedirs(self.repo_path)
            logging.log(logging.INFO, f"Created repository path: {self.repo_path}")

    def save_file(self, file_path: str, content: str):
        with open(file_path, "wb") as f:
            f.write(content)

    def read_input(self):
        # upload files
        uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)

        file_paths = []
        # save the files
        if uploaded_files:
            for file in uploaded_files:
                file_path = os.path.join(self.repo_path, file.name)
                self.save_file(file_path, file.getbuffer())
                file_paths.append(file_path)
                
            st.success(f"{len(uploaded_files)} files uploaded successfully!")
        
        self.data_model_type = st.selectbox(
            'Choose data model:',
            self.data_model_list
        ).lower()

        st.write(f"Selected data model: {self.data_model_type}")
        # manaully add the data model path at location constants.DataModelGeneratorConstants().DATA_MODEL_PATH
        self.data_model_output_schema = utils.ReadFiles().read_txt(constants.DataModelGeneratorConstants().DATA_MODEL_PATH[self.data_model_type])
        return file_paths
    
    def load_product_names(self, file_paths: list[str]):
        # get product names 
        product_names = tools.get_product_names.invoke({"data_model_type": self.data_model_type, 
                                                        "file_paths": file_paths})
        return product_names
    
    def create_json_objects(self, product_names: list[str], file_paths: list[str]) -> list[dict]:
        json_objects: list[dict] = []
        for product_name in product_names:
            logging.log(logging.INFO, f"Generating data model for {product_name}...")
            message = prompts.get_prompt_for_generating_prompt_for_data_model().format(product_name, self.data_model_output_schema)
            # query = f"data related to {product_name}"
            query = tools.ai_assistant.invoke({"prompt": message})
            data = self.reteriver.route(query, file_paths)
            prompt = prompts.get_prompt_for_datamodel_generation().format(product_name, data, self.data_model_output_schema)
            response = tools.ai_assistant.invoke({"prompt": prompt})
            # json_objects.append(response)
            formated_response = response.split("```")[1].split("json")[-1].strip()
            json_objects.append(json.loads(formated_response))
        return json_objects

    def get_file_name(self, file_paths: list[str]) -> str:
        message = f'''
        Overview:
        Your task to come up with a file name to store the data model. We use a common format for saving the files company_name_data_model.json
        so based on the give files paths, extract the company name and provide the file name.
        Input:
        File Paths: {file_paths}
        Output:
        a file name in the format company_name_data_model.json

        Note: Always return the file name, nothing else, no text, no explanations just the file path.

        ### Guidlines:
        - Extract the company name from the file paths
        - Use the extracted company name to create the file name in the format company_name_data_model.json
        - Return only the file name, noting else
        - the file extension must be .json
'''
        response = tools.ai_assistant.invoke({"prompt": message})
        return response
    
    def save_data_models(self, json_objects: list[dict], file_name: str):
        # save the data models
        file_path = os.path.join(self.save_file_dir, file_name)
        with open(file_path, "w") as f:
            json.dump(json_objects, indent=4, fp=f)
        logging.log(logging.INFO, f"Data models saved successfully at {file_path}")
    

    def generate(self):
        logging.log(logging.INFO, "Running the data model generator application...")
        st.set_page_config(layout="wide")
        st.title("Data Model Generator Application")
        st.write("---")


        try: 
            file_paths: list[str] = self.read_input()
            if st.button("Run Agent"):
                with st.spinner("Processing..."):
                    product_names = self.load_product_names(file_paths) 
                    st.write("---")
                    st.header("Results:")
                    st.subheader("Product Names:")
                    st.write(product_names)
                    json_objects = self.create_json_objects(product_names, file_paths)
                    st.write("---")

                    st.subheader("Data Models:")
                    st.write(json_objects)
                    st.write("---")
                    # save the data models
                    file_name = self.get_file_name(file_paths)
                    self.save_data_models(json_objects, file_name)
                    # tools.save_as_json.invoke({"content": str(json_objects), "file_path": file_name, "directory": self.save_file_dir})
                    logging.log(logging.INFO, "Data models saved successfully!")
        except:
            logging.log(logging.ERROR, "An error occurred while running the data model generator application...")
            st.write("An error occurred while running the data model generator application...")


if __name__ == "__main__":
    data_model_generator = DataModelGenerator()
    data_model_generator.generate()