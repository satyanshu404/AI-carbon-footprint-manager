import os
import json
import models
import requests
import base64
from enum import Enum
from io import BytesIO, StringIO
from typing import Union

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from constants import Constants
import prompts
import streamlit as st


load_dotenv()

def encode_image(image):
    return base64.b64encode(image).decode("utf-8")

def main():
    file = st.file_uploader("Upload file", type=["png", "jpg"])
    show_file = st.empty()
 
    if not file:
        show_file.info("Please upload a file of type: " + ", ".join(["csv", "png", "jpg"]))
        return
 
    content = file.getvalue()
 
    if isinstance(file, BytesIO):
        show_file.image(file)
        base64_image = encode_image(content)
        messages=[
            {"role": "system", "content": "You are a helpful assistant that responds in Markdown. Help me with my math tasks!"},
            {"role": "user", "content": [
                {"type": "text", "text": "Calculate the carbon emmisons."},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"}
                }
        ]}]

        openai = OpenAI()
        response = models.call_gpt(messages)
        st.write(response['data'].content)

    else:
        show_file.write("File not supported")



if __name__ == "__main__":
    main()