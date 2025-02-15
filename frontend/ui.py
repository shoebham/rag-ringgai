import streamlit as st
import requests
import json
import os

# Configure the API endpoint
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Document Chat",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Document Chat Assistant")

# Main content area
uploaded_file = st.file_uploader(
    "Upload your document (PDF, TXT, DOCX)", 
    type=["pdf", "txt", "docx"],
    help="Upload a document to start asking questions about it"
)

if uploaded_file:
    st.success(f"Document uploaded: {uploaded_file.name}")
    
    # Create a chat-like interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "metadata" in message:
                with st.expander("View Metadata"):
                    st.json(message["metadata"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your document"):
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Prepare the file
                    files = {
                        "file": (uploaded_file.name, uploaded_file, uploaded_file.type)
                    }
                    # Send the message as a query parameter
                    response = requests.post(
                        f"{API_URL}/qa-doc?message={prompt}",
                        files=files
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.write(result["answer"])
                        
                        # Store message and metadata
                        message_data = {
                            "role": "assistant",
                            "content": result["answer"]
                        }
                        
                        # Add metadata if available
                        if "metadata" in result:
                            with st.expander("View Metadata"):
                                st.json(result["metadata"])
                            message_data["metadata"] = result["metadata"]
                        
                        st.session_state.messages.append(message_data)
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Error connecting to the server: {str(e)}")
else:
    st.info("👆 Please upload a document to start chatting about it!")

# Sidebar information
st.sidebar.title("About")
st.sidebar.info(
    "This Document Chat Assistant helps you analyze and understand your documents. "
    "Simply upload a document and start asking questions about its content. "
    "The system will provide answers based on the document's content along with relevant sources."
)

# Clear chat button in sidebar
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

if __name__ == "__main__":
    # Run this with: streamlit run ui.py
    pass 