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
First Unserstand the question, it if complex break it down and think step by step then act.

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
Objective: The user will provide a query, in most cases it will be company name, and your task is to collect the products of the company and their carbon footprints.
Input: Name of the company (mostly, not in all cases)
Output: To create list of object containing the products of the company and their carbon footprints in full details in the given JSON template.
Json Object template: 
[{json_output_schema}]
The response should be in this JSON template only. 

Guidelines:
    - Understand the problem carefully and restrict yourself towards the collection and creation of data model only.
    - Focus only the data related to the given query, do not include any irrelevant data.
    - You are also provided with the search tool, use it if necessary, however first check for the data in the directory.
    - If the data is not available in the directory, then use the search tool to get the data.
    - Ensure Accuracy and Completeness: Ensure that the data present in the data model must be related to the given query.
    - Do not make any assumptions, always search for the data if it is not available, data accuracy is crucial.
    - Do mention the references or links to the sources used for verification in the JSON output.
    - At last just give output in the JSON format only, without any additional text.
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

def get_calculator_prompt():
    # Get the calculator prompt template
    return """
Objective: To calculate the carbon emissions of the product(s). 
For that Always think and act, if the data is not available then search for the data.

Input: An image path
{0}.
Output: Total carbon emitted by each product (Numerical Value) with actual unit.

Calculations for the carbon emission may be difficilt, so always first think and then act.
If needed break down the task into subtasks and act accordingly.
The information about the product carbon footprint may not be present, so break down the task, search about from which material product is made, then search for its carbon footprint.
You are also provided with the a tool which can write code and execute it, so call that tool if you need to do mathematical calulations. Beacuase data accuracy is crucial.


Guidelines:
    - Restrict yourself to the calculation of carbon emissions of the products only, noting else, If the content is not useful for the calculation skip it and search again.
    - Search for data if unknown; do not make assumptions. Data accuracy is crucial.
    - Always first check if the files exists in the directory or not for data, if it is not available, then use tools to get the data.
    - Utilize the provided tools for the entire process.
    - Return results only in JSON format, without any additional text.
    - Cross-verify facts and figures from reliable sources if needed.
    - Do not infer or add any information that is not explicitly stated or provided.
    - If the data in the image is unclear or ambiguous, search for the data to make a calculation.
    - Include references or links to sources used for verification in the JSON output.
"""