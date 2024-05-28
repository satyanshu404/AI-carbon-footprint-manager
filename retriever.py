import os
import json
import models
import requests
from openai import OpenAI
from dotenv import load_dotenv
from constants import Constants
import prompts
import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core import SimpleDirectoryReader
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor


load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

def main():
    st.title("Retriever")

    # stores the file paths
    # file_paths:list[str] = models.upload_files()
    # text:str = models.from_paths_to_text(file_paths)

    file_paths:list[str] = ['data/reterival_data/planet-positive.pdf',
                            'data/reterival_data/Untitled spreadsheet.xlsx']
    
    data_model = {
      "id": "<unique-identifier>",
      "specVersion": "<specification-version>",
      "version": "<numeric-version>",
      "created": "<timestamp>",
      "status": "<status>",
      "validityPeriodStart": "<start-date>",
      "validityPeriodEnd": "<end-date>",
      "companyName": "<company-name>",
      "companyIds": [
        "<company-identifier>"
      ],
      "productDescription": "<description>",
      "productIds": [
        "<product-identifier>"
      ],
      "productCategoryCpc": "<category-code>",
      "productNameCompany": "<product-name>",
      "pcf": {
        "declaredUnit": "<unit>",
        "unitaryProductAmount": "<quantity>",
        "pCfExcludingBiogenic": "<amount>",
        "pCfIncludingBiogenic": "<amount>",
        "fossilGhgEmissions": "<emissions-amount>",
        "fossilCarbonContent": "<content-amount>",
        "biogenicCarbonContent": "<content-amount>",
        "dLucGhgEmissions": "<emissions-amount>",
        "landManagementGhgEmissions": "<emissions-amount>",
        "otherBiogenicGhgEmissions": "<emissions-amount>",
        "iLucGhgEmissions": "<emissions-amount>",
        "biogenicCarbonWithdrawal": "<amount>",
        "aircraftGhgEmissions": "<emissions-amount>",
        "characterizationFactors": "<factors>",
        "ipccCharacterizationFactorsSources": [
          "<source>"
        ],
        "crossSectoralStandardsUsed": [
          "<standard>"
        ],
        "productOrSectorSpecificRules": [
          {
            "operator": "<operator-type>",
            "ruleNames": [
              "<rule-name>"
            ],
            "otherOperatorName": "<name>"
          }
        ],
        "biogenicAccountingMethodology": "<methodology>",
        "boundaryProcessesDescription": "<description>",
        "referencePeriodStart": "<start-date>",
        "referencePeriodEnd": "<end-date>",
        "geographicScope": {
          "geographyRegionOrSubregion": "<region>"
        },
        "secondaryEmissionFactorSources": [
          {
            "name": "<source-name>",
            "version": "<version>"
          }
        ],
        "exemptedEmissionsPercent": "<percentage>",
        "exemptedEmissionsDescription": "<description>",
        "packagingEmissionsIncluded": "<boolean>",
        "allocationRulesDescription": "<description>",
        "uncertaintyAssessmentDescription": "<description>",
        "primaryDataShare": "<percentage>",
        "dqi": {
          "coveragePercent": "<percentage>",
          "technologicalDQR": "<rating>",
          "temporalDQR": "<rating>",
          "geographicalDQR": "<rating>",
          "completenessDQR": "<rating>",
          "reliabilityDQR": "<rating>"
        },
        "assurance": {
          "assured": "<boolean>",
          "providerName": "<name>"
        }
      }
    }
    
    product_names = list(models.find_product_from_documents(file_paths, data_model))
    st.write(product_names)

    documents = SimpleDirectoryReader("data/reterival_data").load_data()

    # build index
    index = VectorStoreIndex.from_documents(documents)

    # configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=5,
    )

    # configure response synthesizer
    response_synthesizer = get_response_synthesizer()

    # assemble query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
    )

    # query
    json_objects: list[dict] = []
    for id, product_name in enumerate(product_names):
        response = query_engine.query(product_name)
        # print(response)
        st.write(product_name)
        doc = ""
        for idx, node in enumerate(response.source_nodes):
            doc += f"Document {idx+1}\n"
            doc += node.text
            doc += f'\n\n {"="*50}'
        messages = [{'role': 'system', 'content': 'You are a helpful assistant and give respone only as json. You have the capability to understand the context of the given data and provide the response accordingly.'},
                    {'role': 'user', 'content': f'''
                     You are provided with the company's environmental related data, like how much energy is consumed, how much a product release carbon and many more and You are also provided with the product name and an data model.
                     Product name: {product_name}
                     Documents: {doc}
                     Data model: {data_model}
                     Based on the the provided data and product name create a similar data_model which store all the carbon footprint data of the product. If you do not find the values for any field from the document, assign its value as null, do not make it up also do not delete that field. 
                     You are strictly advised to provide the response in json format.
        '''}]
        data_model = models.call_gpt(messages, temperature=0)['data'].content.strip()
        data_model = data_model.split("```")[1].split("json")[-1].strip()
        print(f"{id+1}th Data Model successfully created.")
        json_objects.append(json.loads(data_model))
    st.write(json_objects)
    file_name = "data/data_model.json"

    # Open a file in write mode
    with open(file_name, 'w') as json_file:
        # Write the JSON data to the file
        json.dump(json_objects, json_file, indent=4)


if __name__ == "__main__":
    main()