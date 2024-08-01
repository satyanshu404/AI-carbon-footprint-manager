import streamlit as st
import logging
import json
import os
from dotenv import load_dotenv
import constants
from langchain_openai import ChatOpenAI
from tools import tools
from tools.get_all_files_of_directory import files_in_directory
from prompts.prompts import *
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

class DecarbonizationProtocolAgent:
    def __init__(self):
        self.tools = [files_in_directory, tools.retriever_router, tools.save_as_json]
        self.tasks_list = constants.DecarbonizationProtocolConstants().TASK_LIST
        self.docs_repo_path = constants.DecarbonizationProtocolConstants.DOCS_REPO_PATH
        self.data_repo_path = constants.DecarbonizationProtocolConstants.DATA_REPO_PATH

        self.save_file_dir = constants.DecarbonizationProtocolConstants.SAVE_FILE_DIR
        self.save_file_name = constants.DecarbonizationProtocolConstants.FILE_NAME

        os.makedirs(self.save_file_dir, exist_ok=True)

    def get_prompt(self, tools_list: str, tool_names: str):
        prompt = ChatPromptTemplate.from_messages([
            ("system", get_react_prompt().format(tools_list, tool_names)),
            ("user", self.prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "file paths: {input}"),
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
                                       max_execution_time=constants.DecarbonizationProtocolConstants.MAX_EXECUTION_TIME, 
                                       max_iterations=constants.DecarbonizationProtocolConstants.MAX_ITERATIONS)
        
        return agent_executor
    
    def delete_previous_file(self):
        if os.path.exists(os.path.join(self.save_file_dir, self.save_file_name)):
            logging.log(logging.INFO, f"Deleting the file: {os.path.join(self.save_file_dir, self.save_file_name)}")
            os.remove(os.path.join(self.save_file_dir, self.save_file_name))
    
    def save_file(self, file_path: str, content: str):
        with open(file_path, "wb") as f:
            f.write(content)

    def upload_files(self, alt_text: str, repo_path: str):
        uploaded_files = st.file_uploader(alt_text, accept_multiple_files=True)
        file_paths = []
        if uploaded_files:
            for file in uploaded_files:
                file_path = os.path.join(repo_path, file.name)
                self.save_file(file_path, file.getbuffer())
                file_paths.append(file_path)
                
            st.success(f"{len(uploaded_files)} files uploaded successfully!")
        return file_paths
    
    def get_query(self):        
        self.delete_previous_file()
        if self.task_type == 'automatic':
            data_file_paths= self.upload_files("Upload Company files", self.data_repo_path)
            documentation_file_paths = self.upload_files("Upload Documentation files", self.docs_repo_path)
            query = f"documentation_file_paths: {documentation_file_paths}\ndata_file_paths:  {data_file_paths}"
            self.prompt = get_prompt_for_automatic_decarbonization_protocol().format(self.save_file_dir, self.save_file_name)
        elif self.task_type == 'custom':
            data_file_paths= self.upload_files("Upload Company files", self.data_repo_path)
            
            if 'query' not in st.session_state:
                st.session_state.query = ''

            st.session_state.query = st.text_input("Question:", st.session_state.query)
            query = f"query: {st.session_state.query}\ndata_file_paths:  {data_file_paths}"
            self.prompt = get_prompt_for_custom_decarbonization_protocol().format(self.save_file_dir, self.save_file_name)
        else:
            # Create two columns
            documentation_file_paths = self.upload_files("Upload Documentation files", self.docs_repo_path)

            col1, col2 = st.columns(2)

            with col1:
                data_file_paths_1= self.upload_files("Upload files for Company A", self.data_repo_path)
            
            with col2:
                data_file_paths_2= self.upload_files("Upload files for Company B", self.data_repo_path)

            query = f"documentation_file_paths: {documentation_file_paths}\ndata_file_paths:  {[data_file_paths_1, data_file_paths_2]}"
            self.prompt = get_prompt_for_side_by_side_arena_decarbonization_protocol().format(self.save_file_dir, self.save_file_name)

        return query
    
    def read_results_for_battle(self):
        full_file_path = os.path.join(self.save_file_dir, self.save_file_name)
        
        with open(full_file_path, 'r') as f:
            data = json.load(f)

        st.title("Result Comparisions")
        st.write("---")
        for idx, item in enumerate(data):
            st.subheader(f"Question {idx+1}:")
            
            st.code(f"{item['question']}", language='text')

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Answer A")
                st.write({'Answer': item['answer A'],
                        'Source': item['source A']})

            with col2:
                st.subheader("Answer B")
                st.write({'Answer': item['answer B'],
                        'Source': item['source B']})

            st.write("---")

    def read_results(self):
        full_file_path = os.path.join(self.save_file_dir, self.save_file_name)
        
        with open(full_file_path, 'r') as f:
            data = json.load(f)
        
        st.title("Results")
        st.write("---")
        for idx, item in enumerate(data):
            st.subheader(f"Question {idx+1}:")
            
            st.code(f"{item['question']}", language='text')

            st.subheader("Answer")
            st.write({'Answer': item['answer'],
                    'Source': item['source']})

            st.write("---")
    
    def invoke_agent(self):
        try:
            logging.log(logging.INFO, "Running the Decarboninzation Protocol agent...")
            st.set_page_config(layout="wide")
            st.title("Decarboninzation Protocol Agent")
            st.write("---")

            self.task_type = st.selectbox(
                'Choose the type of Task:',
                self.tasks_list
            ).lower()

            query = self.get_query()    

            agent_executor = self.search_engine_agent()
            st_callback = StreamlitCallbackHandler(st.container())

            if st.button("Run Agent"):
                with st.spinner("Processing..."):
                    agent_executor.invoke(
                            {"input": query}, 
                            callback_handler=st_callback)
                    if self.task_type.lower() == self.tasks_list[-1].lower():
                        self.read_results_for_battle()
                    else:
                        self.read_results()  

        except Exception as e:
            logging.log(logging.ERROR, f"An error occurred: {str(e)}")
            st.write(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    agent = DecarbonizationProtocolAgent()
    agent.invoke_agent()