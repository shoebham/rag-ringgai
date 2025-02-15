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

from weaviate.classes.config import Property, DataType
from weaviate.classes.config import Configure

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


def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return llm.embeddings.create(input = [text], model=model).data[0].embedding


def process_new_document(file_path: str):
    try:
        file_stats = os.stat(file_path)
        doc_identifier = f"{file_path}_{file_stats.st_mtime}"
        
        if doc_identifier in processed_documents:
            return f"Document already processed: {file_path}"
        
        num_chunks = ingest_documents(file_path)
        processed_documents.add(doc_identifier)
        return f"Successfully processed document: {file_path}. Created {num_chunks} chunks."
    
    except FileNotFoundError:
        raise Exception(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error processing document: {str(e)}")
    

def ingest_documents(file_path: str):
    try:
        _, ext = os.path.splitext(file_path)
        
        if ext == ".txt":
            loader = TextLoader(file_path)
        elif ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        documents = loader.load()
        return store_in_db(documents=documents, file_path=file_path)
    
    except Exception as e:
        raise Exception(f"Error ingesting document: {str(e)}")


def store_in_db(documents, file_path):
    try:
        chunks = text_splitter.split_documents(documents=documents)
        
        for chunk in chunks:
            
            embedding = get_embedding(chunk.page_content)
            metadata_str = json.dumps(chunk.metadata) if chunk.metadata else "{}"

            properties = {
                "content": chunk.page_content,
                "source": file_path,
                "metadata": metadata_str
            }
            
            db.collections.get("Documents").data.insert(
                properties=properties,
                vector=embedding
            )
        
        return len(chunks)
    except Exception as e:
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


def query_doc(query: Query, file_path: str):
    try:
        # Check if document needs processing
        file_stats = os.stat(file_path)
        doc_identifier = f"{file_path}_{file_stats.st_mtime}"
        
        if doc_identifier not in processed_documents:
            process_new_document(file_path)
        
        # Generate embedding for the query
        query_embedding = get_embedding(query.question)
        
        # Search Weaviate for similar content
        result = db.collections.get("Documents").query.near_vector(
            near_vector=query_embedding,
            limit=3,
            return_metadata=MetadataQuery(distance=True)
        )
        contexts=[]
        if not result.objects is None and len(result.objects) > 0:
            for obj in result.objects:
                content = obj.properties["content"]
                metadata = json.loads(obj.properties["metadata"] if obj.properties else {})
                contexts.append(content)

            
            # Generate response using OpenAI
            try:
                response = llm.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Answer the question based on the provided context."},
                        {"role": "user", "content": f"Context: {' '.join(contexts)}\n\nQuestion: {query.question}"}
                    ]
                )
                return response.choices[0].message.content
            except Exception as e:
                raise Exception(f"Error generating response from OpenAI: {str(e)}")
        
        return "No relevant information found in the document."
    
    except FileNotFoundError:
        raise Exception(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error processing query: {str(e)}")



setup_weaviate_schema()
