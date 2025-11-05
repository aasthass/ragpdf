📄 RAG-Based Q&A Agent with Reflection (LangGraph-Style Workflow)
This project implements a Retrieval-Augmented Generation (RAG) based Q&A Agent with a four-step LangGraph-inspired workflow:
Plan → Retrieve → Answer → Reflect.
The agent can ingest PDF documents, store them in a vector database, retrieve relevant chunks based on a question, generate an answer using an LLM, and evaluate its own output for relevance.
A Streamlit UI is included for uploading PDFs and chatting with the agent interactively.
🎯 Objective
To demonstrate understanding of:
AI agent workflow design
Retrieval-Augmented Generation (RAG) pipelines
LangGraph-style execution nodes
Vector search using ChromaDB
LLM-based reasoning and self-evaluation
🧠 Agent Workflow
Plan
The agent interprets the user query and determines that retrieval is needed.
Retrieve
Relevant text chunks are retrieved from a vector database (ChromaDB) based on similarity search using OpenAI embeddings.
Answer
The retrieved context is passed to an OpenAI model (gpt-4o-mini), which generates a concise and cited answer.
Reflect
A secondary self-evaluation step checks whether the generated answer adequately addresses the question.
The reflection step returns:
A rating (1–10)
A justification sentence
📂 Project Structure
project/
│
├── app.py                 # Main Streamlit app with agent pipeline
├── .env                   # API keys (not committed to Git)
├── requirements.txt       # Python dependencies
└── data/                  # Folder to store local PDFs (optional)
🔧 Setup Instructions
Clone the repository (if submitted via Git)
git clone repo
Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
Install required dependencies
pip install -r requirements.txt
Add your API keys in a .env file
AZURE_COGNITIVE_ENDPOINT=your_endpoint_here
AZURE_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
Run the Streamlit application
streamlit run app.py
📌 How to Use
Upload one or more PDFs from the UI.
Click Ingest to extract text using Azure Form Recognizer and store embeddings in ChromaDB.
Ask any question in the chat input field.
The system will:
Plan the query
Retrieve relevant passages
Generate an answer using the context
Reflect on answer quality
The answer will appear along with reflection logs in the sidebar.
🔍 Example Query
“Explain the role of renewable energy in sustainable development.”
The system retrieves relevant passages, responds concisely, and provides a self-assessed quality rating.
✅ Features
Fully local vector storage using ChromaDB
OpenAI embeddings and language generation
Azure Form Recognizer for PDF extraction (handles scanned PDFs)
Answer quality scoring using LLM-based reflection
Interactive Streamlit UI
Supports batch PDF ingestion (./data/*.pdf)
