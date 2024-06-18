import sys
import os
# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import requests
import pandas as pd
from dotenv import load_dotenv
from constants import Constants


load_dotenv()

def build_payloads(query: str, 
                   start:int=1, 
                   num:int=10,
                   **params)->dict:
    ''' Build the payloads for the Google Search API'''

    payload = {
        'key': os.getenv('GOOGLE_API_KEY'),
        'cx': os.getenv('GOOGLE_SEARCH_ENGINE_ID'),
        'q': query,
        'start': start,
        'num': num,
    }
    payload.update(params)
    return payload

def make_request(payload: dict) -> dict:
    '''Make a request to the Google Search API'''
    response = requests.get(Constants.GOOGLE_SEARCH_URL, params=payload)
    return response.json()

def clean_file_name(file_name:str)->str:
    '''Clean the file name'''
    # remove special characters
    file_name = file_name.lower()
    return re.sub(r'[^\w\s-]', '', file_name).strip().replace(' ', '_')

def google_search(query:str,
                  restult_count:int=10)->list:
    '''Search Google and return the results'''
    items = []
    reminder = restult_count%10
    if reminder > 0:
        pages = restult_count//10 + 1
    else:
        pages = restult_count//10 
    for i in range(pages):
        if pages == i+1 and reminder > 0:
            payload = build_payloads(query, start=(i+1)*10, num=reminder)
        else:
            payload = build_payloads(query, start=(i+1)*10)
        # print(payload)
        response = make_request(payload)
        # print(response)
        items.extend(response['items'])

    query_string_cleaned = clean_file_name(query)
    df = pd.json_normalize(items)
    df.to_csv("{0}/Search_resutls_{1}.csv".format(Constants.GOOGLE_SEARCH_RESULTS_PATH, query_string_cleaned), index=False)