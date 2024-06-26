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

@dataclass
class GoogleSearchConstants:
    GOOGLE_SEARCH_URL: str = "https://www.googleapis.com/customsearch/v1"
    GOOGLE_SEARCH_RESULTS_PATH: str = "data/google_search_results"
    NUMBEROFSEARCHS: int = 5

@dataclass
class SaveDirectoryConstants:
    CALCULATOR_IMAGE_PATH: str = "images/saved_image.png"

@dataclass
class FormatJsonConstants:
    SAVE_DIRECTORY: str = "templates/{0}.txt"

@dataclass
class GroqModelConstants:
    MODEL_NAME: str = "llama3-70b-8192"

@dataclass
class CodeExecuterConstants:
    BASEFILEPATH: str = "data/code_executer"

@dataclass
class RetrieverConstants:
    RETRIEVER_FILE_LOCAITON:str = "data/reterival_data"
    SIMILARITY_TOP_K: int = 3
    SIMILARITY_CUTOFF: float = 0.8

@dataclass
class WebScraperConstants:
    SCRAPER_FILE_LOCATION: str = "data/scraper_data"
    USERAGENT: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"

@dataclass
class SearchEngineConstants:
    FILE_PATH: str = "templates/custom_output_schema_simple.txt"
    MAX_TOKEN_LIMIT: int = 200_000

@dataclass
class DataModelGeneratorConstants:
    DATA_MODEL_PATH: dict = field(default_factory=lambda: {
        'pact': 'templates/pact_data_model.txt',
    })
    REPO_PATH: str = "data/tmp_reterival_data"

@dataclass
class TextSplitterConstants:
    CHUNK_SIZE: int = 5000
    CHUNK_OVERLAP: int = 1000
