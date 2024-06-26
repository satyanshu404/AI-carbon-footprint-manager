import streamlit as st
import logging
import json
from dotenv import load_dotenv
import constants
from langchain_openai import ChatOpenAI
from tools import tools, utils
from tools.create_corpus import create_json_summary
from tools.get_all_files_of_directory import files_in_directory
from prompts.prompts import get_react_prompt, get_search_engine_prompt
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
load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SearchEngineAgent:
    def __init__(self):
        self.tools = [files_in_directory, create_json_summary, tools.google_search, tools.reterive_data]

        self.output_schema:str = utils.ReadFiles().read_txt(constants.SearchEngineConstants.FILE_PATH)
    
    def get_prompt(self, tools_list: str, tool_names: str):
        prompt = ChatPromptTemplate.from_messages([
            ("system", get_react_prompt().format(tools_list, tool_names)),
            ("user", get_search_engine_prompt().format(self.output_schema)),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "Company Name: {input}"),
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

        agent_executor = AgentExecutor(agent=agent_chain, tools=self.tools, verbose=True, memory=memory, return_intermediate_steps=True)
        
        return agent_executor
    
    def invoke_agent(self):
        try:
            logging.log(logging.INFO, "Running the search engine agent...")
            agent_executor = self.search_engine_agent()
            st_callback = StreamlitCallbackHandler(st.container())

            st.title("Search Agent")
            if 'query' not in st.session_state:
                st.session_state.query = ''

            st.session_state.query = query = st.text_input("Enter your query:", st.session_state.query)

            if st.session_state.query.lower() == "exit":
                st.write("Exiting...")
                # Optionally reset the query or handle the exit condition
                st.session_state.query = ''

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
    agent = SearchEngineAgent()
    agent.invoke_agent()