from flask import Flask, render_template, request, jsonify
from datetime import datetime
import os
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ------------ Backend Setup ------------
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
DB_FAISS_PATH = "vector_store/db_emb"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

llm = InferenceClient(model=MODEL_NAME, token=HF_TOKEN)

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(
        DB_FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

db = load_vector_store()

def format_prompt(question, context):
    return f"""<s>[INST] You are a medical assistant. Use ONLY the following context to answer.
If unsure, say you don't know. Keep answers technical but clear.

Context:
{context}

Question: {question} [/INST]"""

# ------------ Frontend Routes ------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_query = request.json['message']
    response_data = {
        'answer': '',
        'sources': [],
        'timestamp': datetime.now().strftime("%H:%M")
    }

    try:
        docs = db.similarity_search(user_query, k=3)
        context = "\n".join([d.page_content for d in docs])
        
        response = llm.text_generation(
            format_prompt(user_query, context),
            max_new_tokens=512,
            temperature=0.5,
            return_full_text=False
        )

        response_data['answer'] = response.strip()
        response_data['sources'] = [{
            'content': doc.page_content,
            'source': doc.metadata.get('source', 'Unknown source')
        } for doc in docs]

    except Exception as e:
        response_data['error'] = f"System error: {str(e)}"

    return jsonify(response_data)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)