import os
import logging
import time
import pickle
import numpy as np

from statistics import mean
from pathlib import Path
from typing import List, Dict, Any
from PyPDF2 import PdfFileReader, PdfFileWriter
from urllib.parse import urljoin
from sklearn.metrics.pairwise import cosine_similarity

# import langchain
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, OpenAI, ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs.graph_document import Node, Relationship, Document

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

import docker
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
_ = load_dotenv(find_dotenv())  # read local .env file

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# Global variables and constants
LLM_NAME = 'gpt-4o-mini'

# Abstract Factory for LLM and Vector Store Management
class LLMFactory:
    @staticmethod
    def create_llm():
        return OpenAI(temperature=0)

    @staticmethod
    def create_embeddings():
        return OpenAIEmbeddings()

    @staticmethod
    def create_vector_store(documents: List[Any], vector_db_dir: Path):
        embeddings = LLMFactory.create_embeddings()
        return Chroma.from_documents(documents, embeddings, persist_directory=str(vector_db_dir))

class PodcastRAGPipeline:
    def __init__(self, transcript_dir: str, vector_db_dir: str):
        """
        Initialize the PodcastRAGPipeline
        :param transcript_dir: Directory containing the downloaded transcripts.
        :param vector_db_dir: Directory to store the Chroma vector database.
        """
        self.transcript_dir = Path(transcript_dir)
        self.vector_db_dir = Path(vector_db_dir)

    def load_documents(self) -> List[Any]:
        """
        Load all PDF documents from the transcript directory.
        """
        documents = []
        for pdf_file in self.transcript_dir.glob("*.pdf"):
            loader = PyPDFLoader(str(pdf_file))
            documents.extend(loader.load())
        return documents

    def create_vector_db(self, documents: List[Any]) -> Any:
        """
        Create a Chroma vector database from the loaded documents.
        :param documents: List of documents loaded from PDFs.
        """
        try:
            # Create or load Chroma vector store then add documents to the vector store
            vector_store = LLMFactory.create_vector_store(documents, self.vector_db_dir)
            logging.info("Chroma vector database created successfully.")
            return vector_store
        except AttributeError as e:
            logging.error(f"Error encountered: {e}.")
            raise

    def run_qa_pipeline(self, query: str) -> Any:
        """
        Run a RetrievalQA pipeline on the Chroma vector database.
        :param query: Query to ask the RetrievalQA pipeline.
        """
        vector_store = Chroma(persist_directory=str(self.vector_db_dir), embedding_function=LLMFactory.create_embeddings())
        llm = LLMFactory.create_llm()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())
        return qa_chain.invoke(query)

class PodcastGraphRAGPipeline:
    def __init__(self, graph_db_uri: str, graph_db_user: str, graph_db_password: str, vector_db_dir: str):
        """
        Initialize the PodcastGraphRAGPipeline
        :param graph_db_uri: URI for the Neo4j graph database.
        :param graph_db_user: Username for the Neo4j graph database.
        :param graph_db_password: Password for the Neo4j graph database.
        """
        self.llm = ChatOpenAI(
            temperature=0,
            model_name=LLM_NAME,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.graph_db_uri = graph_db_uri
        self.graph_db_user = graph_db_user
        self.graph_db_password = graph_db_password
        self.driver = GraphDatabase.driver(self.graph_db_uri, auth=(self.graph_db_user, self.graph_db_password))
        self.vector_db_dir = Path(vector_db_dir)
        self.embeddings_model = LLMFactory.create_embeddings()
        self.graph_documents_file = "graph_documents.pkl" # Local file to store graph_documents to save computation and cost

        # Create a full-text index for faster searches
        # with self.driver.session() as session:
        #     session.run("CALL db.index.fulltext.createNodeIndex('entityIndex', ['Entity'], ['name', 'description'])")

    def close(self):
        """Close the connection to the graph database."""
        self.driver.close()

    def create_graph_from_documents(self, documents: List[Any]):
        """
        Create a graph from the documents using LLMGraphTransformer.
        :param documents: List of documents to extract entities and relationships from.
        """
        # Load graph_documents from a local file if available
        if os.path.exists(self.graph_documents_file):
            with open(self.graph_documents_file, "rb") as file:
                graph_documents = pickle.load(file)
                logging.info("Loaded graph_documents from local file/cache.")
                print(f"Loaded graph_documents from {self.graph_documents_file}")
        else:
            # If not available, create graph_documents using LLM and save it locally
            graph_transformer = LLMGraphTransformer(llm=self.llm)
            graph_documents = graph_transformer.convert_to_graph_documents(documents)
            print(f'Graph Documents: {len(graph_documents)}\n {graph_documents}')

            with open(self.graph_documents_file, "wb") as file:
                pickle.dump(graph_documents, file)
                logging.info("Graph documents cached locally.")
                print(f"Saved graph_documents to {self.graph_documents_file}.")
        
        # Store graph data in Neo4j
        with self.driver.session() as session:
            try:
                # Process nodes
                for doc in graph_documents:
                    for node in doc.nodes:
                        if isinstance(node, Node):
                            name = str(node.id).lower() # Convert to lowercase
                            embedding = self.embeddings_model.embed_query(name)  # Generate embedding
                            node_data = {
                                "name": name,
                                "type": str(node.type),
                                "embedding": embedding, # Convert numpy array to list for storage
                            }
                            session.run(
                                """
                                MERGE (n:Entity {name: $name})
                                ON CREATE SET n.type = $type, n.embedding = $embedding
                                ON MATCH SET n.type = $type, n.embedding = $embedding
                                """,
                                **node_data,
                            )
                            logging.info(f"Created/Updated node: {node_data}")

                # Process relationships (unchanged)
                for doc in graph_documents:
                    for rel in doc.relationships:
                        if isinstance(rel, Relationship):
                            rel_data = {
                                "start": str(rel.source.id).lower(),
                                "end": str(rel.target.id).lower(),
                                "type": str(rel.type),
                            }
                            session.run(
                                """
                                MATCH (a:Entity {name: $start}), (b:Entity {name: $end})  
                                MERGE (a)-[r:RELATED {type: $type}]->(b)
                                RETURN a.name, b.name, r.type
                                """,
                                **rel_data,
                            )
                            logging.info(f"Created relationship: {rel_data}")

            except Exception as e:
                logging.error(f"Error during graph creation: {str(e)}")
                raise

            # Optional: Verify the graph creation and print out the number of nodes and relationships in the graph
            nodes_count = session.run("MATCH (n:Entity) RETURN count(n) as count").single()["count"]
            rels_count = session.run("MATCH ()-[r:RELATED]->() RETURN count(r) as count").single()["count"]
            logging.info(f"Graph creation completed. Nodes: {nodes_count}, Relationships: {rels_count}")

    
    def run_qa_pipeline(self, query: str) -> Dict[str, Any]:
        """
        Run a sophisticated graph-based query against the GraphRAG pipeline to retrieve an answer.
        :param query: Query to ask the GraphRAG pipeline.
        """
        with self.driver.session() as session:
            try:
                # Step 1: Find similar entities using embeddings
                similar_entities = self._find_similar_entities(query, session)
                if not similar_entities:
                    logging.info("No similar entities found.")
                    return {"result": "No relevant information found."}

                # Step 2: Extract relevant entity names
                relevant_entity_names = [entity["name"] for entity in similar_entities]

                # Step 3: Fetch relationships involving relevant entities
                relationships_query = """
                MATCH (a:Entity)-[r:RELATED]->(b:Entity)
                WHERE a.name IN $entities OR b.name IN $entities
                RETURN a.name AS start, b.name AS end, r.type AS relationship
                """
                relationships = session.run(
                    relationships_query,
                    parameters={"entities": relevant_entity_names},
                ).data()

                # Step 4: Construct context from entities and relationships
                context = self._construct_context(similar_entities, relationships)

                # Step 5: Use LLM to generate an answer based on the constructed context
                return self._use_llm_for_answer(context, query)

            except Neo4jError as e:
                logging.error(f"Neo4j error during QA pipeline: {str(e)}")
                return {"result": "Error in processing the query"}
    def _get_embeddings_from_neo4j(self, session):
        """
        Retrieve precomputed embeddings from Neo4j.
        """
        result = session.run("MATCH (n:Entity) WHERE n.embedding IS NOT NULL RETURN n.name AS name, n.embedding AS embedding").data()
        names = [record["name"] for record in result]
        embeddings = [np.array(record["embedding"]) for record in result if record["embedding"]]
        return names, embeddings

    def _find_similar_entities(self, query: str, session: Any) -> List[Dict[str, Any]]:
        """
        Find similar entities based on embeddings similarity.
        :param query: User query for which similar entities are to be found.
        :param session: Neo4j session.
        :return: List of similar entities.
        """
        try: 
            # Compute the query embedding using embed_query
            query_embedding = self.embeddings_model.embed_query(query)

            # Retrieve stored embeddings from Neo4j
            # names, embeddings = self._get_embeddings_from_neo4j(session)
            names, embeddings = zip(*[(record["name"], np.array(record["embedding"])) for record in session.run(
            "MATCH (n:Entity) WHERE n.embedding IS NOT NULL RETURN n.name AS name, n.embedding AS embedding").data()])

            # Compute cosine similarity
            similarities = cosine_similarity([query_embedding], embeddings)[0]
            top_n_indices = np.argsort(similarities)[::-1][:10] # Get the top n (=10) most similar entities

            # Return the top n(=10) most similar entities
            top_n_entities = [{"name": names[i], "similarity": similarities[i]} for i in top_n_indices]
            return top_n_entities
        
        except Exception as e:
            logging.error(f"Error finding similar entities: {str(e)}")
            return []

    def _construct_context(self, entities: List[Dict[str, Any]], relationships: List[Any]) -> str:
        context = "Relevant Entities:\n"
        for entity in entities:
            context += f"- {entity['name']} (similarity: {entity['similarity']:.2f})\n"
        context += "\nRelationships:\n"
        for rel in relationships:
            context += f"- {rel['start']} -[{rel['relationship']}]-> {rel['end']}\n"
        return context

    def _use_llm_for_answer(self, context: str, query: str) -> Any:
        """
        Use the LLM to generate an answer given the context and query.
        :param context: The context to provide to the LLM.
        :param query: The query to answer.
        :return: The generated answer.
        """
        try:
            # Prepare a retriever (example using Chroma as a vector store)
            vector_store = Chroma(persist_directory=str(self.vector_db_dir), embedding_function=self.embeddings_model)
            retriever = vector_store.as_retriever()

            # Construct the QA chain with the retriever
            llm = LLMFactory.create_llm() # Ensure the LLM is initialized properly
            qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
            prompt = f"Context: {context}\nQuestion: {query}"
            
            # Run the QA chain with the provided prompt (context and query)
            return qa_chain.invoke(prompt)
        except Exception as e:
            logging.error(f"Error in LLM QA pipeline: {str(e)}")
            return {"result": "Error during answer generation"}

# Test Cases to Measure and Compare Performance Between Vanilla RAG and GraphRAG Applied to Podcast Transcripts
class PerformanceComparison:
    def __init__(self, vanilla_rag_pipeline: PodcastRAGPipeline, graph_rag_pipeline: PodcastGraphRAGPipeline, test_queries: List[str]):
        """
        Initialize the PerformanceComparison class.
        :param vanilla_rag_pipeline: Vanilla RAG pipeline instance
        :param graph_rag_pipeline: GraphRAG pipeline instance
        :param test_queries: List of test queries to evaluate
        """
        self.vanilla_rag_pipeline = vanilla_rag_pipeline
        self.graph_rag_pipeline = graph_rag_pipeline
        self.test_queries = test_queries

    def evaluate_latency(self):
        """
        Measure and compare latency between Vanilla RAG and GraphRAG for the given queries.
        """
        vanilla_latencies, graph_latencies = [], []

        for query in self.test_queries:
            # Measure Vanilla RAG latency
            start_time = time.time()
            self.vanilla_rag_pipeline.run_qa_pipeline(query)
            vanilla_latencies.append(time.time() - start_time)

            # Measure GraphRAG latency
            start_time = time.time()
            self.graph_rag_pipeline.run_qa_pipeline(query)
            graph_latencies.append(time.time() - start_time)

        logging.info(f"Vanilla RAG Latency: {mean(vanilla_latencies):.2f}s, GraphRAG Latency: {mean(graph_latencies):.2f}s")

    def evaluate_recall_quality(self):
        """
        Measure recall quality between Vanilla RAG and GraphRAG based on retrieved context.
        """
        for query in self.test_queries:
            vanilla_response = self.vanilla_rag_pipeline.run_qa_pipeline(query)
            graph_response = self.graph_rag_pipeline.run_qa_pipeline(query)
            logging.info(f"Query: {query}")
            logging.info(f"Vanilla RAG Response: {vanilla_response['result']}")
            logging.info(f"GraphRAG Response: {graph_response['result']}")

    def evaluate_response_quality(self):
        """
        Evaluate the response quality, including factual correctness and contextual coherence.
        """
        results = []
        for query in self.test_queries:
            vanilla_response = self.vanilla_rag_pipeline.run_qa_pipeline(query)['result']
            graph_response = self.graph_rag_pipeline.run_qa_pipeline(query)['result']
            results.append({"query": query, "vanilla_response": vanilla_response, "graph_response": graph_response})

            # Factual correctness and coherence evaluation would require manual review or automated methods for fact-checking.
            logging.info(f"Query: {query}")
            logging.info(f"Vanilla RAG Response: {vanilla_response}")
            logging.info(f"GraphRAG Response: {graph_response}")
            # Use tools like GPT-3-based evaluators to automatically score correctness and coherence.

        return results

if __name__ == "__main__":
    BASE_URL = "https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/"
    TARGET_DIRECTORY = "./data/podcast_transcripts"

    # Step 1: Implement Vanilla RAG using LangChain and Chroma
    # Load the pipelines (assume they are initialized and properly set up)
    VECTOR_DB_DIRECTORY = "./vector_db"
    rag_pipeline = PodcastRAGPipeline(TARGET_DIRECTORY, VECTOR_DB_DIRECTORY)
    
    documents = rag_pipeline.load_documents()
    # Print document statistics
    total_docs = len(documents)
    total_tokens = sum(len(doc.page_content.split()) for doc in documents)
    logging.info(f"Total documents: {total_docs}")
    logging.info(f"Total tokens: {total_tokens}")
    
    rag_pipeline.create_vector_db(documents) 

    # Step 2: Implement GraphRAG using LLMGraphTransformer and Neo4j
    GRAPH_DB_URI = "bolt://localhost:7687"
    GRAPH_DB_USER = "neo4j"
    GRAPH_DB_PASSWORD = "*******" # Relace with your password
    
    graph_rag_pipeline = PodcastGraphRAGPipeline(GRAPH_DB_URI, GRAPH_DB_USER, GRAPH_DB_PASSWORD, VECTOR_DB_DIRECTORY)
    graph_rag_pipeline.create_graph_from_documents(documents)
    
    # Step 3: Evaluate the performance of the pipelines
    # Define test queries
    test_queries = [
        "Who are the official host(s) of SDS Podcast?",
        "Who was the guest(s) in Episode 835? What was his/her role at the target organisation? What were the main points discussed?",
        "What is You.com about and what are the business pain points it endeavours to address? How does it address them? How deep is You.com's moat compared to other competitors?",
        "What are the key takeaways regarding the Transformers architecture across all episodes (particularly from both Episode 747 and 759)?",
        "What are the primary advantages and limitations of the Transformers architecture? How do they compare and contrast with other mainstream architectures?",
        "Was there any mention of quantum computing in all those episodes involved? If yes, which one(s), mentioned by whom and under what context?",       
    ]

    # Initialize PerformanceComparison
    performance_comparator = PerformanceComparison(
        vanilla_rag_pipeline=rag_pipeline,
        graph_rag_pipeline=graph_rag_pipeline,
        test_queries=test_queries,
    )

    # Evaluate latency
    performance_comparator.evaluate_latency()

    # Evaluate response quality
    _ = performance_comparator.evaluate_response_quality()
    
    # Close the graph database connection
    graph_rag_pipeline.close()