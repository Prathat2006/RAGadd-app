import os
import tempfile
import shutil
import uuid
import requests
from flask import send_from_directory
from flask import Flask, request, jsonify, render_template, session
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredWordDocumentLoader,
    CSVLoader, UnstructuredExcelLoader, UnstructuredPowerPointLoader,
    UnstructuredFileLoader, JSONLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from typing import List, Dict, TypedDict, Optional
from dotenv import load_dotenv
from pydantic import BaseModel
from llminit import LLMManager
from configobj import ConfigObj
from offline_validation import is_ollama_installed, is_lmstudio_installed
from api_validation import validate_groq_api_key, validate_openrouter_api_key





cfg = ConfigObj('config.ini')
app = Flask(__name__, template_folder='templates')
app.secret_key = os.urandom(24)  # Secure random key for sessions
# order = cfg['offline-order']['order']
manager=LLMManager()
llm_instance=manager.setup_llm_with_fallback()


EMBEDDING_MODEL = cfg['embedding_model']['embedding_model']

DOCUMENT_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
    ".doc": UnstructuredWordDocumentLoader,
    ".csv": CSVLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".xls": UnstructuredExcelLoader,
    ".ppt": UnstructuredPowerPointLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".html": UnstructuredFileLoader,
    ".htm": UnstructuredFileLoader,
    ".css": TextLoader,
    ".js": TextLoader,
    ".json": JSONLoader,
    ".py": TextLoader
}

# Agent state
class AgentState(TypedDict):
    input: str
    
    chat_history: List[Dict[str, str]]
    retrieved_docs: Optional[List[str]]
    response: Optional[str]

    
class validate_key(BaseModel):
    status : bool = True

# Initializing models
def init_models():
    return {
        "embeddings": HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    }

# temp file handling for Windows
def create_temp_file(uploaded_file, suffix):
    """Safely create a temporary file on Windows"""
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, f"file{suffix}")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())
    return temp_path, temp_dir



# Process uploaded files
def process_documents(files, models):
    documents = []
    temp_dirs = []
    
    for file in files:
        filename = file.filename
        ext = os.path.splitext(filename)[1].lower()
        loader_type = DOCUMENT_MAP.get(ext)
        
        if not loader_type:
            continue
            
        try:
            temp_path, temp_dir = create_temp_file(file, ext)
            temp_dirs.append(temp_dir)
            loader = loader_type(temp_path)
            docs = loader.load()
            if docs:
                documents.extend(docs)
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
    
    return documents, temp_dirs

# vector store
def create_vector_store(documents, embeddings):
    if not documents:
        return None
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return FAISS.from_documents(chunks, embeddings)

# agent components
def create_agent_components(vector_store):
    if not vector_store:
        return None, None
    
    # Retrieval component
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # Prompt templates
    retrieval_prompt = ChatPromptTemplate.from_template(
        "You are a helpful document assistant called RAGadd AI . Given the following context:\n\n{context}\n\n"
        "Your purpose is to help users understand and interact with their uploaded documents.\n"
        "Answer this question based on the context: {question}\n"
        "**Accuracy First**: Extract relevant details, summarize clearly, and maintain factual correctness.\n"
        "Use Markdown formatting in your response for:\n"
        "- Headings (## Heading)\n"
        "- Bold (**text**)\n"
        "- Italic (*text*)\n"
        "- Lists (- item)\n"
        "- Code blocks (```code```)\n"
        "- Tables (| Header | ... |)\n"
        "Format any code snippets, tables, or important concepts appropriately.\n"
        "Structure complex answers with sections, tables, or code snippets when needed.\n"
        "**Polite & Professional**: Always respond in a clear, concise, and professional but very easy to understand tone.\n"
        "Your main goal is to provide the **most accurate, easy to understand, well-structured, and context-grounded response possible** to the user's query.\n"
    )
    
    # Tools
    def retrieve_docs(state: AgentState):
        docs = retriever.get_relevant_documents(state["input"])
        return {"retrieved_docs": [d.page_content for d in docs]}
    
    def generate_response(state: AgentState):
        context = "\n\n".join(state["retrieved_docs"])
        prompt = retrieval_prompt.format(context=context, question=state["input"])
        order = state.get("mode", "default")  # Default to "default" if not set


        response=manager.invoke_with_fallback(llm_instance, order, prompt)

        return {"response": response}
        # try:
        # #     response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload)
        # #     if response.status_code == 200:
        # # print(response)
        #     if response.status_code == 200:
        #         # print({"response": response.json()['choices'][0]['message']['content'].strip()})
        #         # return {"response": response.json()['choices'][0]['message']['content'].strip()}

        #     else:
        #         error_detail = f"API Error: {response.status_code}"
        #         try:
        #             error_data = response.json()
        #             if 'error' in error_data:
        #                 error_detail += f" - {error_data['error'].get('message', 'Unknown error')}"
        #         except:
        #             error_detail += f" - {response.text[:200]}"
        #         return {"response": f"Error generating response: {error_detail}"}
        # except Exception as e:
        #     return {"response": f"Network Error: {str(e)}"}
    
    return retrieve_docs, generate_response

# LangGraph workflow
def build_workflow(retrieve_docs, generate_response):
    if not retrieve_docs or not generate_response:
        return None
    
    # Graph nodes
    def retrieve_node(state: AgentState):
        return retrieve_docs(state)
    
    def generate_node(state: AgentState):
        return generate_response(state)
    
    # Conditional edges
    def route_nodes(state: AgentState):
        if state["retrieved_docs"]:
            return "generate"
        return "retrieve"
    
    # Graph
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    
    workflow.set_entry_point("retrieve")
    workflow.add_conditional_edges(
        "retrieve",
        route_nodes,
        {
            "retrieve": "retrieve",
            "generate": "generate"
        }
    )
    workflow.add_edge("generate", END)
    
    return workflow.compile()

# Initializing models
models = init_models()

# User session data
user_sessions = {}

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/')
def home():
    return render_template('backup.html')

@app.route('/is-api-key-set', methods=['GET'])
def is_api_key_set():
    load_dotenv(override=True)
    # openrouter_status = bool(os.getenv('OPENROUTER_API_KEY'))
    # groq_status = bool(os.getenv('GROQ_API_KEY'))


    openrouter_status = bool(os.getenv('OPENROUTER_API_KEY'))
    if openrouter_status == True:
       openrouter_status= validate_openrouter_api_key(os.getenv('OPENROUTER_API_KEY'))
    #    print(openrouter_status)
    # print("openrouter_status:", openrouter_status)
    
    groq_status = bool(os.getenv('GROQ_API_KEY'))
    if groq_status == True:
        groq_status=validate_groq_api_key(os.getenv('GROQ_API_KEY'))
        # print(groq_status)
        
    # print("groq_status:", groq_status)

    return jsonify({
        "openrouter_api_key": openrouter_status,
        "groq_api_key": groq_status
    })
   

@app.route('/offline')
def offline():
    return jsonify({
        "ollama_installed": is_ollama_installed(),
        "lmstudio_installed": is_lmstudio_installed()
    })

@app.route('/upload', methods=['POST'])
def upload_documents():
    # For checking if API key is activated
    # if 'openrouter_api_key' not in session:
    #     return jsonify({'error': 'API key not activated. Please activate your API key first.'}), 403
    
    # To generate new session ID
    session_id = str(uuid.uuid4())
    
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('files[]')
    if not files or all(file.filename == '' for file in files):
        return jsonify({'error': 'No files selected'}), 400
    
    try:
        documents, temp_dirs = process_documents(files, models)
        if not documents:
            return jsonify({'error': 'No valid content found in documents'}), 400
        
        vector_store = create_vector_store(documents, models["embeddings"])
        if not vector_store:
            return jsonify({'error': 'Failed to create vector store from documents'}), 400
            
        retrieve_docs, generate_response = create_agent_components(vector_store)
        if not retrieve_docs or not generate_response:
            return jsonify({'error': 'Failed to create agent components'}), 400
            
        workflow = build_workflow(retrieve_docs, generate_response)
        if not workflow:
            return jsonify({'error': 'Failed to build workflow'}), 400
        
        # Store session data
        user_sessions[session_id] = {
            'vector_store': vector_store,
            'workflow': workflow,
            'messages': [],
            'temp_dirs': temp_dirs
        }
        
        # For cleaning up temp dirs after processing
        for temp_dir in temp_dirs:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass
        
        response = jsonify({
            'success': True,
            'message': f'Processed {len(documents)} document pages!',
            'session_id': session_id
        })
        
        # Set session ID as HTTP-only cookie
        response.set_cookie('session_id', session_id, httponly=True, samesite='Lax')
        return response
        
    except Exception as e:
        return jsonify({'error': f'Error processing documents: {str(e)}'}), 500



@app.route('/chat', methods=['POST'])
def chat():
    session_id = request.cookies.get('session_id')
    if not session_id:
        return jsonify({'error': 'Session not found. Please upload documents first.'}), 400
    
    session_data = user_sessions.get(session_id)
    if not session_data or not session_data.get('workflow'):
        return jsonify({'error': 'Session expired or invalid. Please upload documents again.'}), 400
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    message = data.get('message')
    mode = data.get('mode') 
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    if not mode:
        return jsonify({'error': 'No mode provided'}), 400
    
   
    session_data['mode'] = mode  

    session_data['messages'].append({'role': 'user', 'content': message.strip()})
    
    agent_state = {
        "input": message.strip(),
        "chat_history": session_data['messages'][:-1],
        "retrieved_docs": None,
        "response": None,
        "mode": mode  
    }
    
    try:
        result = session_data['workflow'].invoke(agent_state)
        response = result["response"]
        session_data['messages'].append({'role': 'assistant', 'content': response})
        
        return jsonify({
            'response': response,
            'session_id': session_id
        })
    except Exception as e:
        return jsonify({'error': f'Error generating response: {str(e)}'}), 500




if __name__ == '__main__':
    app.run(debug=True, port=5000)