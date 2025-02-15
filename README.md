# Document RAG (Retrieval Augmented Generation)

Website: https://rag-ringgai-ynt5mrasz6oadoegaxkwaf.streamlit.app/

A basic RAG application. This application allows users to upload documents and ask questions about their content, leveraging RAG architecture for accurate and context-aware responses.

## Architecture

- **Frontend**: Streamlit-based interactive UI
- **Backend**: FastAPI server
- **Vector Database**: Weaviate for efficient document storage and retrieval
- **LLM Integration**: OpenAI for generating responses

## Features

- Document upload and processing (PDF, TXT, DOCX)
- Interactive chat interface
- Metadata visibility for transparency

## API Endpoints

### Document Q&A
- **Endpoint**: `/qa-doc`
- **Method**: POST
- **Purpose**: Ask questions about uploaded documents
- **Parameters**:
  - `message` (query): The question about the document
  - `file` (form-data): Document file (PDF/TXT/DOCX)
- **Returns**: Answer with relevant metadata

### General Q&A
- **Endpoint**: `/qa`
- **Method**: POST
- **Purpose**: Ask general questions (without document context)
- **Parameters**:
  - `question` (JSON body): The question to ask
- **Returns**: Generated answer

## Setup and Running

1. **Backend Setup**:
   ```bash
   cd backend
   pip install -r requirements.txt
   python main.py
   ```

2. **Frontend Setup**:
   ```bash
   cd frontend
   pip install -r requirements.txt
   streamlit run ui.py
   ```

3. **Environment Variables**:
   Create a `.env` file with:
   ```
   OPENAI_API_KEY=your_api_key
   WEAVIATE_URL=your_weaviate_url
   WEAVIATE_API_KEY=your_weaviate_api_key
   ```

## Usage

1. Start the backend server (runs on port 8000)
2. Launch the Streamlit UI
3. Upload a document through the UI
4. Start asking questions about your document
5. View detailed metadata for each response

## Requirements

- Python 3.8+
- OpenAI API key
- Weaviate instance
- Required Python packages (see requirements.txt)


