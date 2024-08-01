import sys
import os

# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tools.utils as utils
import constants
from prompts import prompts
from typing import Optional
import nest_asyncio
nest_asyncio.apply()

from llama_index.core import SimpleDirectoryReader
import pandas as pd
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import (
    SummaryExtractor,
    TitleExtractor,
    KeywordExtractor,
)
from llama_index.core import StorageContext
from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleKeywordTableIndex, VectorStoreIndex
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)
from typing import List
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import StorageContext, load_indices_from_storage, load_index_from_storage

from dotenv import load_dotenv
import logging
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv(override=True)

class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and hybrid search."""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        keyword_retriever: KeywordTableSimpleRetriever,
        mode: str = "AND",
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        keyword_ids = {n.node.node_id for n in keyword_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in keyword_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(keyword_ids)
        else:
            retrieve_ids = vector_ids.union(keyword_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes

class LLMRetriever:
    def __init__(self):
        self.model = utils.GptModel()
        self.RF = utils.ReadFiles()
        self.text_splitter = utils.TextSplitter(chunk_overlap=constants.LLMRetrieverConstants.CHUNK_OVERLAP,
                                          chunk_size=constants.LLMRetrieverConstants.CHUNK_SIZE)

    def make_prompt(self, query:str, context: str) -> list[dict]:
        '''Make the prompt for the LLM'''
        return prompts.get_system_prompt_for_llm_retriever().format(context, query)
    
    def read_files(self, file_paths: list[str]) -> list[str]:
        '''Read the files and return the documents'''
        documents:list = [self.RF.read_file(file_path) for file_path in file_paths]
        return documents

    def retriever(self, query:str, documents: list) -> Optional[list[str]]:
        '''Retrieve the relevant chunks from the documents based on the query, context and instruction.'''
        data = []
        for document in documents:
            chunks: list[str] = self.text_splitter.split_text(document)
            print(f"Got {len(chunks)} chunks.")
            messages: list[str] = [self.make_prompt(query, chunk) for chunk in chunks]  
            responses: list[str] = [self.model.get_completion(message) for message in messages]
            data.extend(responses)
        return data
    
    def retrieve(self, query:str, file_paths: list[str]) -> Optional[list[str]]:
        '''Retrieve the relevant chunks from the documents based on the query, context and instruction.'''
        documents = self.read_files(file_paths)
        return self.retriever(query, documents)
    
    def retrieve_and_find(self, query:str, content: list[str]) -> Optional[list[str]]:
        '''Finds answers to the query from the documents'''
        messages = self.make_prompt(query, content)
        data = self.model.get_completion(messages)

        prompt = prompts.get_prompt_for_finding_response_to_query().format(query, data)
        return self.model.get_completion(prompt)

class HybridRetriever:
    '''Advanced Retriever class that performs Hybrid search'''
    def __init__(self):
        logging.log(logging.INFO, "Initializing Advanced Retriever...")
        self.llm = OpenAI(temperature=constants.HybridRetrieverConstants.TEMPERATURE,
                          model=constants.HybridRetrieverConstants.MODEL_NAME)
        self.embed_model = OpenAIEmbedding(model=constants.HybridRetrieverConstants.EMBEDDING_MODEL_NAME,
                                           dimensions=constants.HybridRetrieverConstants.DIMENSIONS)
    
    def load_transformations(self):
        '''Load the transformations for the pipeline'''
        # define the transformations
        transformations = [
            SentenceSplitter(chunk_size=constants.HybridRetrieverConstants.CHUNK_SIZE, 
                            chunk_overlap=constants.HybridRetrieverConstants.CHUNK_OVERLAP),
            TitleExtractor(nodes= constants.HybridRetrieverConstants.NODES,
                        #    llm=self.llm
                           ),
            SummaryExtractor(summaries=constants.HybridRetrieverConstants().SUMMARIES,
                            # llm=self.llm
                            ),
            KeywordExtractor(keywords=constants.HybridRetrieverConstants.KEYWORDS,
                            # llm=self.llm
                            ),
        ]
        return transformations

    def load_and_create_nodes_with_metadata(self, file_path: str) -> List[NodeWithScore]:
        '''Load the documents and create metadata for the documents'''
        logging.log(logging.INFO, f"Creating metadata for...")
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

        transformations = self.load_transformations()
        
        pipeline = IngestionPipeline(transformations=transformations)
        nodes = pipeline.run(documents=documents)
        return nodes
    
    def save_indexes(self, file_name: str, indexes: list) -> None:
        '''Save the indexes to the storage'''
        for index in indexes:
            # save the indexes
            index.storage_context.persist(persist_dir=f"{constants.HybridRetrieverConstants.DEFAULT_DIR}/{index.__class__.__name__}/{os.path.basename(file_name)}")
        logging.log(logging.INFO, "Indexes saved successfully...")
                
    def load_indexes(self, file_name: str, index_names) -> list:
        '''Load indexes from the storage'''
        logging.log(logging.INFO, "Loading indexes...")
        persist_dir = f"{constants.HybridRetrieverConstants.DEFAULT_DIR}/{index_names[-1]}/{os.path.basename(file_name)}"
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        indexes = load_indices_from_storage(storage_context)
        return indexes
    
    def create_hybrid_index(self, nodes: List[NodeWithScore]) -> tuple:
        '''Create the hybrid index'''
        # initialize storage context (by default it's in-memory)
        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(nodes, overwrite=True)

        # create the indexes
        vector_index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=self.embed_model)
        keyword_index = SimpleKeywordTableIndex(nodes, storage_context=storage_context, 
                                                llm=self.llm
                                                )
        return vector_index, keyword_index
    
    def hybrid_query_engine(self, vector_index: VectorStoreIndex, keyword_index: SimpleKeywordTableIndex) -> RetrieverQueryEngine:
        '''Create the query engine for the hybrid search'''
        logging.log(logging.INFO, "Creating Hybrid Query Engine...")
        # create the retriever
        vector_retriever = VectorIndexRetriever(vector_index, 
                                                similarity_top_k=constants.HybridRetrieverConstants.SIMILARITY_TOP_K,)
        keyword_retriever = KeywordTableSimpleRetriever(keyword_index)
        # define custom retriever
        custom_retriever = CustomRetriever(vector_retriever, keyword_retriever)

        response_synthesizer = get_response_synthesizer()

        # create the query engine
        query_engine = RetrieverQueryEngine(
            retriever=custom_retriever,
            response_synthesizer=response_synthesizer,
        )
        return query_engine
    
    def hybrid_search(self, query: str, file_paths: List[str]) -> tuple[list, list]:
        '''Perform hybrid search on the indexes'''
        logging.log(logging.INFO, "Performing Hybrid search...")
        responses = []
        source_nodes = []
        for file_path in file_paths:
            nodes = self.load_indexes(file_path, constants.HybridRetrieverConstants().INDEXES)
            vector_index, keyword_index = nodes
            query_engine = self.hybrid_query_engine(vector_index, keyword_index)
            response = query_engine.query(query)
            responses.append(response)
            source_nodes.append(response.source_nodes)
        return responses, source_nodes
        
    def check_and_create_index(self, file_paths: List[str], indexes: List[str]) -> None:
        '''Check if indexes exist and create them if they don't'''
        logging.log(logging.INFO, "Checking and creating indexes...")
        for file_path in file_paths:
            for index_name in indexes:
                persist_dir = f"{constants.HybridRetrieverConstants.DEFAULT_DIR}/{index_name}/{os.path.basename(file_path)}"
                if not os.path.exists(persist_dir):
                    logging.log(logging.INFO, f"Index for {file_path} not found. Creating index...")
                    nodes = self.load_and_create_nodes_with_metadata(file_path)
                    vector_index, keyword_index = self.create_hybrid_index(nodes)
                    self.save_indexes(file_path, [vector_index, keyword_index])
        return 
        
    def retrieve(self, query: str, file_paths: List[str]):
        try:
            self.check_and_create_index(file_paths, constants.HybridRetrieverConstants().INDEXES)
            response, source_nodes = self.hybrid_search(query, file_paths)
            print([node.text  for nodes in source_nodes for node in nodes if node])
            return response, [node.text  for nodes in source_nodes for node in nodes if node]
        except Exception as e:
            logging.log(logging.ERROR, f"Error occured: {e}")
            return None, None

class VectorRetriever:
    def __init__(self):
        logging.log(logging.INFO, "Initializing Advanced Retriever...")
        self.RF = utils.ReadFiles()
        self.llm = OpenAI(temperature=constants.VectorRetrieverConstants.TEMPERATURE,
                          model=constants.VectorRetrieverConstants.MODEL_NAME)
        self.embed_model = OpenAIEmbedding(model=constants.VectorRetrieverConstants.EMBEDDING_MODEL_NAME,
                                           dimensions=constants.VectorRetrieverConstants.DIMENSIONS)
    
    def load_transformations(self):
        '''Load the transformations for the pipeline'''
        # define the transformations
        transformations = [
            SentenceSplitter(chunk_size=constants.VectorRetrieverConstants.CHUNK_SIZE, 
                            chunk_overlap=constants.VectorRetrieverConstants.CHUNK_OVERLAP),
            TitleExtractor(nodes= constants.VectorRetrieverConstants.NODES,
                           llm=self.llm
                           ),
            SummaryExtractor(summaries=constants.VectorRetrieverConstants().SUMMARIES,
                            llm=self.llm
                            )
        ]
        return transformations

    def create_documents(self, file_paths: list[str]) -> list[Document]:
        '''Create the documents from the file paths'''
        return [Document(text=self.RF.read_file(file_path)) for file_path in file_paths]
    
    def load_and_create_nodes_with_metadata(self, file_path: str) -> List[NodeWithScore]:
        '''Load the documents and create metadata for the documents'''
        logging.log(logging.INFO, f"Creating metadata for...")
        documents = self.create_documents([file_path])

        transformations = self.load_transformations()
        
        pipeline = IngestionPipeline(transformations=transformations)
        nodes = pipeline.run(documents=documents)
        return nodes
    
    def save_indexes(self, file_name: str, index: VectorStoreIndex) -> None:
        '''Save the indexes to the storage'''
        # save the indexes
        index.storage_context.persist(persist_dir=f"{constants.VectorRetrieverConstants.DEFAULT_DIR}/{index.__class__.__name__}/{os.path.basename(file_name)}")
        logging.log(logging.INFO, "Indexes saved successfully...")

    def load_indexes(self, file_name: str, index_name) -> list:
        '''Load indexes from the storage'''
        logging.log(logging.INFO, "Loading indexes...")
        persist_dir = f"{constants.VectorRetrieverConstants.DEFAULT_DIR}/{index_name}/{os.path.basename(file_name)}"
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        return load_index_from_storage(storage_context)
    
    def create_index(self, node: VectorStoreIndex) -> VectorStoreIndex:
        '''Create the vector index'''
        # initialize storage context (by default it's in-memory)
        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(node)

        # create the indexes
        vector_index = VectorStoreIndex(node, storage_context=storage_context, embed_model=self.embed_model)
        return vector_index
    
    def query_engine(self, vector_index: VectorStoreIndex) -> RetrieverQueryEngine:
        '''Create the query engine for the vector search'''
        logging.log(logging.INFO, "Creating Query Engine...")
        # create the retriever
        vector_retriever = VectorIndexRetriever(vector_index, 
                                                similarity_top_k=constants.VectorRetrieverConstants.SIMILARITY_TOP_K,
                                                embed_model=self.embed_model)

        response_synthesizer = get_response_synthesizer()

        # create the query engine
        query_engine = RetrieverQueryEngine(
            retriever=vector_retriever,
            response_synthesizer=response_synthesizer,
        )
        return query_engine
    
    def search(self, query: str, file_paths: List[str]) -> tuple[list, list]:
        '''Perform vector search on the indexes'''
        logging.log(logging.INFO, "Performing search...")
        responses = []
        source_nodes = []
        for file_path in file_paths:
            nodes = self.load_indexes(file_path, constants.VectorRetrieverConstants.INDEX)
            vector_index = nodes
            query_engine = self.query_engine(vector_index)
            response = query_engine.query(query)
            responses.append(response.response)
            source_nodes.extend([res.text for res in response.source_nodes])
        return responses, source_nodes
    
    def check_and_create_index(self, file_paths: List[str], index) -> None:
        '''Check if indexes exist and create them if they don't'''
        logging.log(logging.INFO, "Checking and creating indexes...")
        for file_path in file_paths:
            persist_dir = f"{constants.VectorRetrieverConstants.DEFAULT_DIR}/{index}/{os.path.basename(file_path)}"
            if not os.path.exists(persist_dir):
                logging.log(logging.INFO, f"Index for {file_path} not found. Creating index...")
                node = self.load_and_create_nodes_with_metadata(file_path)
                vector_index = self.create_index(node)
                self.save_indexes(file_path, vector_index)
        return 
    
    def retrieve(self, query: str, file_paths: List[str]) -> tuple[list, list]:
        '''Retrieve the response from the vector search'''
        # try:
        self.check_and_create_index(file_paths, constants.VectorRetrieverConstants.INDEX)
        responses, source_nodes = self.search(query, file_paths)
        return responses, source_nodes

class RetrieverRouter:
    '''Router class that routes the query to the LLM or Vector search'''
    def __init__(self):
        self.llm_retriever = LLMRetriever()
        self.retriever = VectorRetriever()
        self.gpt_model = utils.GptModel()

    
    def check_response(self, query: str, response: Optional[list[str]]) -> bool:
        '''Check if the response is valid'''
        prompt = prompts.get_prompt_for_response_comparison().format(query, response)
        feedback = self.gpt_model.get_completion(prompt)
        if feedback.lower() == "yes":
            return True
        
        return False

    def route(self, query: str, file_paths: List[str]):
        '''Route the query to the appropriate search method
        1. Vector search
        2. LLM Retrieval with reterived chunks by Vector search
        3. LLM Retrieval with full documents           
        '''
        try:
            # if search response if good, return it
            _, responses_texts = self.retriever.retrieve(query, file_paths)

            # if search response is not good, try LLM search with reterived chunks
            response = self.llm_retriever.retrieve_and_find(query, responses_texts)
            if self.check_response(query, response):
                logging.log(logging.INFO, "LLM search response with chunks are good. Returning response...")
                return response
            else:
                logging.log(logging.INFO, "LLM search response with chunks are not good. Going for full LLM retrieval...")
            
            # if responses are not good, do full llm retrieval
            logging.log(logging.INFO, "Performing full LLM retrieval...")
            return self.llm_retriever.retrieve(query, file_paths)
        except Exception as e:
            logging.log(logging.ERROR, f"Error occured in router retrieval: {e}")
            return f"Error occured in router retrieval: {e}"