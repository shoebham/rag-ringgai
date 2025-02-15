import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from main import Query
from openai import OpenAI
import os
import weaviate
from weaviate.classes.init import Auth
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from weaviate.classes.query import MetadataQuery
from weaviate.classes.query import Filter

from weaviate.classes.config import Property, DataType
from weaviate.classes.config import Configure

import logging
import hashlib
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm = OpenAI()
model = "gpt-4o-mini"
# Best practice: store your credentials in environment variables
wcd_url = os.environ["WCD_URL"]
wcd_api_key = os.environ["WCD_API_KEY"]

db = weaviate.connect_to_weaviate_cloud(
    cluster_url=wcd_url,                                    
    auth_credentials=Auth.api_key(wcd_api_key),            
)
# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
)
processed_documents = set()

def get_file_hash(file_path: str) -> str:
    """Generate hash from file content"""
    with open(file_path, 'rb') as f:
        content = f.read()
        return hashlib.md5(content).hexdigest()

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return llm.embeddings.create(input = [text], model=model).data[0].embedding


def process_new_document(file_path: str):
    try:
        logger.info(f"Starting to process new document: {file_path}")
        file_stats = os.stat(file_path)
        doc_identifier = f"{file_path}_{file_stats.st_mtime}"
        
        if doc_identifier in processed_documents:
            logger.info(f"Document already processed: {file_path}")
            return f"Document already processed: {file_path}"
        
        logger.info("Starting document ingestion")
        num_chunks = ingest_documents(file_path)
        processed_documents.add(doc_identifier)
        logger.info(f"Successfully processed document: {file_path}. Created {num_chunks} chunks")
        return f"Successfully processed document: {file_path}. Created {num_chunks} chunks."
    
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise Exception(f"File not found: {file_path}")
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise Exception(f"Error processing document: {str(e)}")
    

def ingest_documents(file_path: str):
    try:
        logger.info(f"Starting document ingestion for: {file_path}")
        _, ext = os.path.splitext(file_path)
        logger.info(f"Detected file extension: {ext}")
        
        if ext == ".txt":
            loader = TextLoader(file_path)
        elif ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
        else:
            logger.error(f"Unsupported file type: {ext}")
            raise ValueError(f"Unsupported file type: {ext}")
        
        logger.info("Loading document")
        documents = loader.load()
        logger.info("Document loaded successfully, storing in database")
        return store_in_db(documents=documents, file_path=file_path)
    
    except Exception as e:
        logger.error(f"Error ingesting document: {str(e)}")
        raise Exception(f"Error ingesting document: {str(e)}")


def store_in_db(documents, file_path):
    try:
        logger.info("Starting to store documents in database")
        chunks = text_splitter.split_documents(documents=documents)
        logger.info(f"Split documents into {len(chunks)} chunks")
        
        # Get just the filename instead of full path
        filename = os.path.basename(file_path)
        
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"Processing chunk {i}/{len(chunks)}")
            embedding = get_embedding(chunk.page_content)
            metadata_str = json.dumps(chunk.metadata) if chunk.metadata else "{}"
            
            properties = {
                "content": chunk.page_content,
                "source": filename,  # Store just the filename
                "metadata": metadata_str
            }
            
            db.collections.get("Documents").data.insert(
                properties=properties,
                vector=embedding
            )
        
        logger.info(f"Successfully stored {len(chunks)} chunks in database")
        return len(chunks)
    except Exception as e:
        logger.error(f"Error storing documents in database: {str(e)}")
        raise Exception(f"Error storing documents in database: {str(e)}")

def setup_weaviate_schema():
    # Create the schema if it doesn't exist
    if not db.collections.exists("Documents"):
        db.collections.create(
            name="Documents",
            properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="metadata", data_type=DataType.TEXT)
            ],
            vectorizer_config=None
            # vectorizer_config=[
            #     Configure.NamedVectors.text2vec_openai(
            #     name="title_vector",
            #     source_properties=["title"],
            #     model="text-embedding-3-small",
            #     dimensions=1024
            #     )
            # ]   
        )

# Also add proper client cleanup
import atexit

def cleanup():
    db.close()

atexit.register(cleanup)

def query(query:Query):
    completion = llm.chat.completions.create(
        model=model,
        messages=[
            {"role": "developer", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": query.question
            }
        ]
    )
    return completion.choices[0].message.content 


def query_doc(query: Query, file_path: str, limit: int = 1):
    try:
        logger.info(f"Starting query_doc for file: {file_path}")
        
        doc_identifier = get_file_hash(file_path)
        filename = os.path.basename(file_path)
        logger.info(f"Document hash: {doc_identifier}")
        
        if doc_identifier not in processed_documents:
            logger.info("Document not previously processed, starting processing...")
            process_new_document(file_path)
            processed_documents.add(doc_identifier)
        else:
            logger.info("Document was previously processed, skipping processing step")
        
        logger.info(f"Generating embedding for query: {query.question}")
        query_embedding = get_embedding(query.question)
        
        logger.info("Searching Weaviate for similar content")
        result = db.collections.get("Documents").query.near_vector(
            near_vector=query_embedding,
            limit=limit,
            return_metadata=MetadataQuery(distance=True),
            
            filters=Filter.by_property("source").equal(filename)
                            
        )
        
        contexts = []
        metadata_list = []
        if not result.objects is None and len(result.objects) > 0:
            for obj in result.objects:
                content = obj.properties["content"]
                metadata = {
                    "source": filename,  # Use the clean filename
                    "snippet": content[:200] + "..." if len(content) > 200 else content
                }
                contexts.append(content)
                metadata_list.append(metadata)
            
            try:
                logger.info("Generating response from OpenAI")
                response = llm.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Answer the question based on the provided context."},
                        {"role": "user", "content": f"Context: {' '.join(contexts)}\n\nQuestion: {query.question}"}
                    ]
                )
                
                return {
                    "answer": response.choices[0].message.content,
                    "metadata": {
                        "sources": metadata_list,
                        "total_sources": len(metadata_list)
                    }
                }
            except Exception as e:
                logger.error(f"Error generating response from OpenAI: {str(e)}")
                raise Exception(f"Error generating response from OpenAI: {str(e)}")
        
        return {
            "answer": "No relevant information found in the document.",
            "metadata": {
                "sources": [],
                "total_sources": 0
            }
        }
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise Exception(f"Error processing query: {str(e)}")



setup_weaviate_schema()
