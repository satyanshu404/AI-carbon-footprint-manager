from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Constants:
    MODEL_NAME: str = "gpt-4o"
    CHUNK_SIZE: int = 5000
    CHUNK_OVERLAP: int = 1000
    PRINT_CHUNKS: bool = True
    TEMPERATURE: float = 1.0
    TOOL_SPECIFICATION_JSON_PATH: str = "tool_specs.json"
    RETRIEVER_FILE_LOCAITON:str = "data/reterival_data"
    