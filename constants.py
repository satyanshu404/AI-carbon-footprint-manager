from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Constants:
    MODEL_NAME: str = "gpt-4o"
    CHUNK_SIZE: int = 5000
    CHUNK_OVERLAP: int = 1000
    PRINT_CHUNKS: bool = True
    TEMPERATURE: float = 1.0
    CREATIVE_TEMPERATURE: float = 1.5
    FACTUAL_TEMPERATURE: float = 0.1
    TOOL_SPECIFICATION_JSON_PATH: str = "tool_specs.json"
    RETRIEVER_FILE_LOCAITON:str = "data/reterival_data"
    SIMILARITY_TOP_K: int = 3
    SIMILARITY_CUTOFF: float = 0.7
    GOOGLE_SEARCH_URL: str = "https://www.googleapis.com/customsearch/v1"
    GOOGLE_SEARCH_RESULTS_PATH: str = "data/google_search_results"