import streamlit as st
import logging
import json
import os
from dotenv import load_dotenv
import constants
from langchain_openai import ChatOpenAI
from tools import tools
from tools.get_all_files_of_directory import files_in_directory
from prompts.prompts import get_react_prompt, get_prompt_for_automatic_decarbonization_protocol, get_prompt_for_custom_decarbonization_protocol
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
        self.tools = [files_in_directory, tools.retrieve_data_using_llm]
        self.tasks_list = constants.DecarbonizationProtocolConstants().TASK_LIST
        self.docs_repo_path = constants.DecarbonizationProtocolConstants.DOCS_REPO_PATH
        self.data_repo_path = constants.DecarbonizationProtocolConstants.DATA_REPO_PATH
        # self.prompt = get_prompt_for_automatic_decarbonization_protocol()

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
        data_file_paths= self.upload_files("Upload Company files", self.data_repo_path)

        if self.data_model_type == 'automatic':
            documentation_file_paths = self.upload_files("Upload Documentation files", self.docs_repo_path)
            query = f"documentation_file_paths: {documentation_file_paths}\ndata_file_paths:  {data_file_paths}"
            self.prompt = get_prompt_for_automatic_decarbonization_protocol()
        else:
            if 'query' not in st.session_state:
                st.session_state.query = ''

            st.session_state.query = st.text_input("Question:", st.session_state.query)
            query = f"query: {st.session_state.query}\ndata_file_paths:  {data_file_paths}"
            self.prompt = get_prompt_for_custom_decarbonization_protocol()

        return query

    def invoke_agent(self):
        try:
            logging.log(logging.INFO, "Running the Decarboninzation Protocol agent...")

            st.title("Decarboninzation Protocol Agent")

            self.data_model_type = st.selectbox(
                'Choose the type of Task:',
                self.tasks_list
            ).lower()

            query = self.get_query()    
            # data_file_paths= self.upload_files("Upload Company files", self.data_repo_path) 

            agent_executor = self.search_engine_agent()
            st_callback = StreamlitCallbackHandler(st.container())


            if st.button("Run Agent"):
                with st.spinner("Processing..."):
                    response = agent_executor.invoke(
                            {"input": query}, 
                            callback_handler=st_callback)
                    if response['output']:
                        st.write(response['output'])                    

                
        except Exception as e:
            logging.log(logging.ERROR, f"An error occurred: {str(e)}")
            st.write(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    agent = DecarbonizationProtocolAgent()
    agent.invoke_agent()