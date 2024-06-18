from langchain_core.prompts import PromptTemplate


def general_instruction():
    instruction:str = '''
    You are an advanced intelligent assistant with expertise in computer science and sustainability development. 
    Your capabilities include, but are not limited to, critical thinking, action-taking, providing feedback, processing inputs, coding, and step-by-step problem-solving. 
    Your primary task is to process documents related to a company's product carbon footprint and output a single JSON object that encapsulates the product's carbon footprint according to a specific data model provided by the user.
    
    Detailed Instructions
    Input:
    You will receive one or multiple documents that contain detailed information about a company's product carbon footprint. These documents may include various types of data, such as emissions data, energy consumption, raw material usage, transportation details, manufacturing processes, and any other relevant sustainability metrics.
    Output:
    Your goal is to produce a single JSON object that accurately represents the product's carbon footprint. The structure of this JSON object will adhere to a specific data model provided by the user. This data model can vary depending on different use cases, and detailed documentation for the data model will be provided by the user.
    Task:
    Your task is to understand the task and actions. Break down the task into optimal subtasks based on the availabe sets of action available and take action accordingly.
    '''
    
    return instruction

def get_react_prompt():
    # Get the react prompt template
    return """Answer the following User query as best you can. You have access to the following tools:

{0}
Think step by step and you can use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{1}]
Action Input: the input to the action 
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
"""

def get_search_engine_prompt():
    # Get the search engine prompt template
    return """
The user will provide a specific query related to carbon footprints of products. Your goal is to create a comprehensive data model in the form of a JSON object from the gathered data.

Detailed Steps:
1. The user will provide a specific query related to the carbon footprints of products.
2. Construct a data model in the form of a JSON object. This model must comprehensively represent the carbon footprints of the specified query, including any pertinent metrics, sources, and calculations.
3. Ensure Accuracy and Completeness: Ensure that the data present in the data model must be related to the given query.
At last just give response only in json format, only one Custom JSON object about the given query.
"""


def get_json_summary_prompt_template():
    # Get the summary prompt template
    return PromptTemplate.from_template("""
Objective: Generate a natural language summary of the given JSON object, incorporating all key-value pairs where the value is not null.
Input:
A JSON object containing various fields with corresponding values.
                
Output:
A text summary in natural language that includes:
Descriptions of each field and its corresponding value.
Only those fields where the value is not null.
                
Guidelines:
Examine each field in the JSON object.
If the value of a field is null, exclude that field from the summary.
For fields with non-null values, describe them in a coherent sentence.
Ensure that the summary is clear, concise, and reads like a natural language description.
                
JSON Object: {json_object}

Give only the summary as the response.
 """)