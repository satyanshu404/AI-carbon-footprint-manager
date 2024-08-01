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
class DataModelGeneratorAgentConstants:
    MAX_EXECUTION_TIME: int = 1000
    MAX_ITERATIONS: int = 100

@dataclass
class DecarbonizationProtocolConstants:
    MAX_EXECUTION_TIME: int = 10_000
    MAX_ITERATIONS: int = 100
    DATA_REPO_PATH: str = "data/Decarbonization Protocol/Company Data"
    DOCS_REPO_PATH: str = "data/Decarbonization Protocol/Documenations"
    SAVE_FILE_DIR: str = "data/Decarbonization Protocol/Temporary files"
    FILE_NAME: str = "temp.json"
    TASK_LIST: List[str] = field(default_factory=lambda: ['Automatic','Custom', 'Side by Side Arena'])

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
    # MODEL_NAME: str = "llama3-70b-8192"
    MODEL_NAME: str = "gemma2-9b-it"

@dataclass
class CodeExecuterConstants:
    BASEFILEPATH: str = "data/code_executer"

@dataclass
class RetrieverConstants:
    RETRIEVER_FILE_LOCAITON:str = "data/reterival_data"
    SIMILARITY_TOP_K: int = 3
    SIMILARITY_CUTOFF: float = 0.7

@dataclass
class LLMRetrieverConstants:
    CHUNK_SIZE: int = 125_000
    CHUNK_OVERLAP: int = 0
    MODEL_NAME: str = "gemma2-9b-it"

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
    DATA_MODEL_LIST: List[str] = field(default_factory=lambda: ['PACT'])
    REPO_PATH: str = "data/tmp_reterival_data"
    SAVE_FILE_DIR: str = "data/data_model"

@dataclass
class TextSplitterConstants:
    CHUNK_SIZE: int = 5000
    CHUNK_OVERLAP: int = 1000

@dataclass
class DataModelExtractorConstants:
    SAVE_DIR: str = "data/data_model_templates"
    MAX_EXECUTION_TIME: int = 1000
    MAX_ITERATIONS: int = 50

@dataclass
class HybridRetrieverConstants:
    # llm parameters
    MODEL_NAME: str = "gpt-4o"
    TEMPERATURE: float = 0.1
    # embedding parameters
    EMBEDDING_MODEL_NAME: str = "text-embedding-3-small"
    DIMENSIONS: int = 1536
    # transformation parameters
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100
    NODES: int = 5
    KEYWORDS: int = 10
    SUMMARIES: List[str] = field(default_factory=lambda: ["prev", "self"])
    # index parameters
    DEFAULT_DIR: str = 'Database/Indexes'
    INDEXES: List[str] = field(default_factory=lambda: ["VectorStoreIndex", "SimpleKeywordTableIndex"])
    # retriever parameters
    SIMILARITY_TOP_K: int = 2

@dataclass
class VectorRetrieverConstants:
    # llm parameters
    MODEL_NAME: str = "gpt-4o"
    TEMPERATURE: float = 0.1
    # embedding parameters
    EMBEDDING_MODEL_NAME: str = "text-embedding-3-small"
    DIMENSIONS: int = 1536
    # transformation parameters
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = CHUNK_SIZE // 10
    NODES: int = 5
    KEYWORDS: int = CHUNK_SIZE // 100
    SUMMARIES: List[str] = field(default_factory=lambda: ["prev", "self"])
    # index parameters
    DEFAULT_DIR: str = 'Database/Indexes'
    INDEX: str = "VectorStoreIndex"
    # retriever parameters
    SIMILARITY_TOP_K: int = 2
