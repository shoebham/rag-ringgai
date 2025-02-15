from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
from typing import Dict, Optional
from dotenv import load_dotenv
import rag
import tempfile
import os
from fastapi import HTTPException

load_dotenv()

app = FastAPI()

class Query(BaseModel):
    question: str
    context: Optional[Dict] = None



@app.get("/")
def root():
    return {"Hello": "World"}

@app.post("/upload")
async def upload_doc(file:UploadFile):
    return {"filename":file.filename}

@app.post("/qa")
async def ask_question(query):
    return rag.query(query=query) 

@app.post("/qa-doc")
async def ask_question_doc(message: str, file: UploadFile):
    try:
        # Create a temporary file to store the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file.flush()
            
            # Create query object
            query_obj = Query(question=message)
            
            # Process the query with the document
            response = rag.query_doc(query_obj, tmp_file.name)
            
        # Clean up the temporary file
        os.unlink(tmp_file.name)
        
        if response is None:
            raise HTTPException(status_code=404, detail="No relevant information found")
            
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)