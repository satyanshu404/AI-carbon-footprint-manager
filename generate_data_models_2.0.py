import streamlit as st
import os
import re
import logging
import json
from dotenv import load_dotenv
import constants
from langchain_openai import ChatOpenAI
from tools import tools, utils
from tools.get_all_files_of_directory import files_in_directory
from prompts import prompts
from langchain.tools.render import render_text_description
from langchain.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.prompts import MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain.schema.agent import AgentFinish
from langchain.schema.runnable import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# load environment variables
load_dotenv(override=True)
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataModelGenerator:
    def __init__(self):
        self.tools = [tools.get_product_names, tools.ai_assistant, tools.retriever_router, tools.save_as_json]
        self.data_model_list = constants.DataModelGeneratorConstants().DATA_MODEL_LIST 
        self.save_file_dir = constants.DataModelGeneratorConstants.SAVE_FILE_DIR
        self.repo_path = constants.DataModelGeneratorConstants.REPO_PATH

        # temporary file for downloading the data 
        if not os.path.exists(self.repo_path):
            os.makedirs(self.repo_path)
            logging.log(logging.INFO, f"Created repository path: {self.repo_path}")
    
    def get_prompt(self, tools_list: str, tool_names: str):
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompts.get_react_prompt().format(tools_list, tool_names)),
            ("user", prompts.get_data_model_generator_prompt().format(self.data_model_output_schema)),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "File paths: {input}"),
            ("user", f"data_model_type: {self.data_model_type}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        return prompt
    
    def get_tool_description(self, tools: list):
        tools_list=render_text_description(list(tools))
        tool_names=", ".join([t.name for t in tools])
        return tools_list, tool_names
    
    def get_functions(self, tools: list):
        return [convert_to_openai_function(f) for f in tools]
    
    def search_engine_agent(self):
        tools_list, tool_names = self.get_tool_description(self.tools)
        functions = self.get_functions(self.tools)
        prompt = self.get_prompt(tools_list, tool_names)
        
        llm = ChatOpenAI(model=constants.Constants.MODEL_NAME).bind(functions=functions)
        output_parser = OpenAIFunctionsAgentOutputParser()
        chain = prompt | llm | output_parser
        agent_chain = RunnablePassthrough.assign(agent_scratchpad= lambda x: format_to_openai_functions(x["intermediate_steps"])) | chain
        memory = ConversationBufferMemory(return_messages=True,memory_key="chat_history")

        agent_executor = AgentExecutor(agent=agent_chain, 
                                       tools=self.tools, 
                                       verbose=True, 
                                       memory=memory, 
                                       return_intermediate_steps=True, 
                                       handle_parsing_errors=True, 
                                       max_execution_time=constants.DataModelGeneratorAgentConstants.MAX_EXECUTION_TIME, 
                                       max_iterations=constants.DataModelGeneratorAgentConstants.MAX_ITERATIONS)
        
        return agent_executor
    
    def save_file(self, file_path: str, content: str):
        with open(file_path, "wb") as f:
            f.write(content)

    def display_results(self, file_path: str):
        pattern = r'[^\\/"\s]+\.json'
        match = re.search(pattern, file_path)
        if match:
            file_path = match.group(0)
        else:
            raise ValueError(f"Invalid file path {file_path}")
        
        full_file_path = os.path.join(self.save_file_dir, file_path)
        with open(full_file_path, "r") as f:
            content = json.load(f)
        st.subheader("Results:")
        st.write(content)
        st.write("---")
    
    def invoke_agent(self):
        try:
            logging.log(logging.INFO, "Running the data model generator agent...")
            st.set_page_config(layout="wide")
            st.title("Data Model Generator Agent")
            st.write("---")

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

            self.data_model_output_schema = utils.ReadFiles().read_txt(constants.DataModelGeneratorConstants().DATA_MODEL_PATH[self.data_model_type])

            agent_executor = self.search_engine_agent()
            st_callback = StreamlitCallbackHandler(st.container())

            if st.button("Run Agent"):
                with st.spinner("Processing..."):
                    response = agent_executor.invoke(
                        {"input": file_paths}, 
                        callback_handler=st_callback)
                    st.write("---")
                    self.display_results(response["output"])
        except Exception as e:
            logging.log(logging.ERROR, f"An error occurred: {str(e)}")
            st.write(f"An error occurred: {str(e)}")
        
if __name__ == "__main__":
    data_model_generator = DataModelGenerator()
    data_model_generator.invoke_agent()