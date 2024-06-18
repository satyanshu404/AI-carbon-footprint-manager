import streamlit as st
from dotenv import load_dotenv
from constants import Constants
from langchain_openai import ChatOpenAI
from tools.get_related_docs import reterive_chunks
from tools.create_corpus import create_json_summary
from tools.get_all_files_of_directory import files_in_directory
from prompts import get_react_prompt, get_search_engine_prompt
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

def main():
    st.title("Search Agent")
    # define
    tools = [files_in_directory, create_json_summary, reterive_chunks]
    functions = [convert_to_openai_function(f) for f in tools]

    # get the tools list
    tools_list=render_text_description(list(tools))
    tool_names=", ".join([t.name for t in tools])


    prompt = ChatPromptTemplate.from_messages([
        ("system", get_react_prompt().format(tools_list, tool_names)),
        ("user", get_search_engine_prompt()),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "Question: {input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    llm = ChatOpenAI(model=Constants.MODEL_NAME).bind(functions=functions)
    output_parser = OpenAIFunctionsAgentOutputParser()
    chain = prompt | llm | output_parser
    agent_chain = RunnablePassthrough.assign(agent_scratchpad= lambda x: format_to_openai_functions(x["intermediate_steps"])) | chain
    memory = ConversationBufferMemory(return_messages=True,memory_key="chat_history")

    agent_executor = AgentExecutor(agent=agent_chain, tools=tools, verbose=True, memory=memory, return_intermediate_steps=True)
 
    # Define the Streamlit callback handler
    st_callback = StreamlitCallbackHandler(st.container())

    # query = input("Query: ")

    # query = input("Query: ")
    if 'query' not in st.session_state:
        st.session_state.query = ''
    
    st.session_state.query = query = st.text_input("Enter your query:", st.session_state.query)


    if st.session_state.query.lower() == "exit":
        st.write("Exiting...")
        # Optionally reset the query or handle the exit condition
        st.session_state.query = ''
    else:
        st.write(f"You entered: {st.session_state.query}")

    # agent_executor.invoke({"input": query})

        if st.button("Run Agent"):
            with st.spinner("Processing..."):
                response = agent_executor.invoke(
                    {"input": query},
                    callbacks=[st_callback]
                )

                if response['output']:
                    st.write(response['output'])



if __name__ == "__main__":
    main()