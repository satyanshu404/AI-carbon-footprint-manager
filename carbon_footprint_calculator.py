import os
import json
from langchain.agents import tool
import pandas as pd
from PIL import Image
import io
import prompts.prompts as prompts
import streamlit as st
from io import BytesIO
from dotenv import load_dotenv
from constants import Constants, SaveDirectoryConstants
from langchain_openai import ChatOpenAI
from tools.get_related_docs import reterive_chunks
from tools.create_corpus import create_json_summary
from tools.get_all_files_of_directory import files_in_directory
from tools import tools
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


load_dotenv()

# steps:
# 1. get the image and load it for gpt 
# 2. tools: [google_search, get_file_directory, reterive_chunks]
# 3. call agent

class CarbonFootprintCalculator:
    def __init__(self):
        self.tools = [files_in_directory, reterive_chunks, tools.get_image_content, tools.read_files, tools.google_search, tools.code_generator_and_executer]
        self.allowed_file_types = ["jpg", "jpeg", "png"]

    def get_prompt(self, tools_list: str, tool_names: str):
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompts.get_react_prompt().format(tools_list, tool_names)),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        return prompt
    
    def get_tool_description(self, tools: list):
        tools_list=render_text_description(list(tools))
        tool_names=", ".join([t.name for t in tools])
        return tools_list, tool_names
    
    def get_functions(self, tools: list):
        return [convert_to_openai_function(f) for f in tools]

    
    def calculator_agent(self):
        tools_list, tool_names = self.get_tool_description(self.tools)
        functions = self.get_functions(self.tools)
        prompt = self.get_prompt(tools_list, tool_names)

        llm = ChatOpenAI(model=Constants.MODEL_NAME).bind(functions=functions)
        output_parser = OpenAIFunctionsAgentOutputParser()
        chain = prompt | llm | output_parser
        agent_chain = RunnablePassthrough.assign(agent_scratchpad= lambda x: format_to_openai_functions(x["intermediate_steps"])) | chain
        memory = ConversationBufferMemory(return_messages=True,memory_key="chat_history")

        agent_executor = AgentExecutor(agent=agent_chain, tools=self.tools, verbose=True, memory=memory, return_intermediate_steps=True)
 
        return agent_executor
    
    def invoke_agent(self):
        try:
            agent_executor = self.calculator_agent()
            st_callback = StreamlitCallbackHandler(st.container())

            st.title("Carbon Footprint Calculator Agent")
            file = st.file_uploader("Upload file", type=self.allowed_file_types)
            show_file = st.empty()
            if not file:
                show_file.info("Please upload a file of type: " + ", ".join(self.allowed_file_types))
                return
            
            if file is not None:
                # Display the uploaded image
                image = Image.open(file)
                st.image(image, caption='Uploaded Image.', use_column_width=True)
                
                # Save the image
                with open(SaveDirectoryConstants().CALCULATOR_IMAGE_PATH, "wb") as f:
                    f.write(file.getbuffer())

            if isinstance(file, BytesIO):
                response = agent_executor.invoke(
                    {"input": prompts.get_calculator_prompt().format(SaveDirectoryConstants().CALCULATOR_IMAGE_PATH)},
                    callbacks=[st_callback])
                
                if response['output']:
                    st.write(response['output'])
        except Exception as e:
            st.write(f"An error occurred: {e}")


if __name__ == "__main__":
    calculator = CarbonFootprintCalculator()
    calculator.invoke_agent()
    
    # print(calculator.get_prompt(["google_search", "get_file_directory", "reterive_chunks"], "Google Search, Get File Directory, Reterive Chunks"))
