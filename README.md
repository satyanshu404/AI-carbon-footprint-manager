<div align="center">

# Carbon Footprint Management using AI Agents

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-purple.svg)](https://www.python.org/downloads/release/python-3110/)
[![Langchain](https://img.shields.io/badge/Langchain-0.2.11-blue)](https://python.langchain.com/v0.2/docs/introduction/)
[![Llama Index](https://img.shields.io/badge/Llama%20Index-0.10.59-orange)](https://docs.llamaindex.ai/en/stable/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.34.0-red)](https://streamlit.io/)
![License](https://img.shields.io/badge/License-Hitachi-green.svg)

&nbsp;
</div>

## Overview
This project is part of the AI Research at Hitachi. The goal of this project is to design an automated agentic application that can perform the following components of carbon footprint management:
1. <b>Automated Data Model Template Extraction</b>
    - Develop agents to extract data model templates from provided documentation automatically
2. <b>Company Data Model Creation</b>
    - Create data models specific to each company using the extracted templates
3. <b>Search Engine Development</b>
    - Build a search engine that accepts company names (or user’s queries) as input and returns product data model as output
    - Respond to two formats:
        - Custom data model (compact/simple)
        - General data model
4. <b>Carbon Footprint Calculator</b>
    - Design a calculator agent based on the provided data models to compute the companies' carbon footprint
5. <b>Decarbonization Progress Tracking Agent</b>
    - Create an agent to monitor and report on a company's progress in following decarbonization protocols
    - The agent should support three functionalities:
        - Automatic full tracking
        - Custom User-Based Tracking
        - Side by Side comparison
<br>
<hr>
<br>
<img src="data\readme_images\flow-chart.png" alt="Flow-chart" style="border-radius: 15px; width: 100%; max-width: auto; height: auto;">


## Installation
1. Clone the repository
    ```bash
    git clone https://github.com/satyanshu404/AI-carbon-footprint-manager.git
    ```
2. Change the working directory
    ```bash
    cd AI-carbon-footprint-manager
    ```
3. Install the required packages
    ```bash
    pip install -r requirements.txt
    ```
4. Set up API keys
    - Create a `.env` file in the root directory and add the following keys to the `.env` file

        ```bash
        OPENAI_API_KEY=<your_openai_api_key>
        GOOGLE_API_KEY=<your_google_api_key>
        GOOGLE_SEARCH_ENGINE_ID=<your_google_search_engine_id>
        ```
5. Run the Streamlit app for specific components
    - To run the Data Model Template Extraction component
        ```python
        streamlit run extract_data_model.py
        ```
    - To run the Company Data Model Creation component
        ```bash
        streamlit run generate_data_models_1.0.py
        ```
        or
        ```bash
        streamlit run generate_data_models_2.0.py
        ```
    - To run the Search Engine component
        ```bash
        streamlit run search_engine_2.0.py
        ```
    - To run the Carbon Footprint Calculator component

        ```bash
        streamlit run calculate_carbon_footprint.py
        ```
    - To run the Decarbonization Progress Tracking Agent component

        ```bash
        streamlit run decarbonization_protocol.py
        ```

## Task Descriptions
1. <b>Automated Data Model Template Extraction</b>
    - This agents can extract data model templates from provided documentation autonomously from any kind of documentation.
    - The agent extracts the data model templates from the documentation and store them in a JSON structured format.
2. <b>Company Data Model Creation</b>
    - This agent can create data models for different products and services of a company using the extracted data model templates which is essential for storing the carbon footprint data.
    - The agent first extracts the product and service names from the companies data and then creates data models for each product and service.
3. <b>Search Engine</b>
    - This agent act as a search engine that accepts company names (or user’s queries) as input and returns product data model as output.
    - The agent responds can be from two formats:
        - Custom data model <i>(compact)</i> <br>
        ```templates/custom_output_schema_compact.txt``` 
        - Custom data model <i>(simple)</i> <br>
        ```templates/custom_output_schema_simple.txt```
        - General data model 
4. <b>Carbon Footprint Calculator (INCOMPLETE)</b>
    - This agent can calculate the carbon footprint of a company based on the data models created for the company.
    - The agent uses the data models to calculate the carbon footprint of the company.
5. <b>Decarbonization Progress Tracking Agent</b>
    - This agent can monitor and report on a company's progress in following decarbonization protocols.
    - The agent supports three functionalities:
        - Automatic full tracking
          - Agent finds the 15-20 most important questions to ask the company to assess their decarbonization progress and then it finds the answers to those questions from the company's data.
        - Custom User-Based Tracking
            - Agent allows users to ask questions, and the agent will retrieve and present relevant data.
        - Side by Side comparison
            - Agent allows users to compare the progress of two companies side by side.
    
## Features
1. <b>Self-Route Technique for RAG</b>
    - We implemented a self-routing technique that intelligently directs queries to either a Retrieval-Augmented Generation (RAG) system or to a long context large language model (LLM), optimizing query handling and minimizing costs. [[paper](https://arxiv.org/pdf/2407.16833)]

2. <b>Autonomous Agents</b>
    - We developed autonomous agents that can perform various tasks such as data model extraction, data model creation, search engine development, carbon footprint calculation, and decarbonization progress tracking.
    - The agents are based on ReACT framework [[paper](https://arxiv.org/pdf/2210.03629)], which gives the model to reason, act, and communicate with the tools.
    - The agents are designed to work on a wide range of formats and can be easily customized to suit the needs of different companies.

## Optimizations that can be done
1. <b>Carbon Footprint Calculator</b>
    - Complete the development of the Carbon Footprint Calculator agent.
    - The agent will be able to calculate the carbon footprint of a company based on the data models created for the company.
2. <b> Cost Reduction</b>
    - The performance of the agents can be improved by reducing the number of API calls and optimizing the code.
3. <b>RAG Optimization</b>
    - The RAG system can be optimized to handle more complex queries and provide more accurate answers.
    - The RAG system can be trained on a domain specific dataset to improve its performance.
4. <b>Memory Enhancement for Agents</b>
    - The agents can be enhanced to remember the previous queries and responses to improve the performance of the agents.
    - It can significantly reduce the number of API calls and improve the response time of the agents and can save costs.
5. <b> Database Integration</b>
    - The agents can be integrated with a database to store the data models and other information.
    - The database can be used to store the data models, queries, responses, and other information to improve the performance of the agents.
<hr>
