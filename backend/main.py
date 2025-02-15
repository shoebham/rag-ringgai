from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
from typing import Dict, Optional
from dotenv import load_dotenv
import backend.rag as rag
import tempfile
import os
from fastapi import HTTPException
import shutil

load_dotenv()

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://*.streamlit.app"],  # Your Streamlit app domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Create a directory for storing files if it doesn't exist
UPLOAD_DIR = "files"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

class Query(BaseModel):
    question: str
    context: Optional[Dict] = None


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
    
@app.get("/")
def root():
    return {"Hello": "World"}

@app.post("/upload")
async def upload_doc(file:UploadFile):
    return {"filename":file.filename}

@app.post("/qa")
async def ask_question(query:Query):
    return rag.query(query=query) 

@app.post("/qa-doc")
async def ask_question_doc(message: str, file: UploadFile):
    try:
        # Save file to the upload directory
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        query_obj = Query(question=message)
        response = rag.query_doc(query_obj, file_path)
        
        # Clean up: remove the file after processing
        os.remove(file_path)
        
        if response is None:
            raise HTTPException(status_code=404, detail="No relevant information found")
            
        return response
    except Exception as e:
        # Ensure file cleanup in case of errors
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)