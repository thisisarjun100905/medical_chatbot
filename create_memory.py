import os
from huggingface_hub import InferenceClient
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
DB_FAISS_PATH = "vector_store/db_emb"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize Hugging Face Inference Client
def load_llm():
    return InferenceClient(
        model=MODEL_NAME,
        token=HF_TOKEN
    )

# Custom prompt template with Mistral instruction formatting
def format_prompt(question, context):
    return f"""<s>[INST] You are a medical assistant. Use ONLY the following context to answer.
If you don't know the answer, say you don't know. Keep answers technical but clear.

Context:
{context}

Question: {question} [/INST]"""

# Load FAISS Vector Database
def load_vector_store():
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        return FAISS.load_local(
            DB_FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"Error loading FAISS database: {e}")
        exit()

# Main execution
if __name__ == "__main__":
    # Initialize components
    llm = load_llm()
    db = load_vector_store()
    
    # Get user query
    user_query = input("\nWrite your medical query here: ").strip()
    if not user_query:
        print("No query entered. Exiting...")
        exit()

    try:
        # Retrieve relevant context
        docs = db.similarity_search(user_query, k=3)
        context = "\n".join([d.page_content for d in docs])
        
        # Generate response
        response = llm.text_generation(
            format_prompt(user_query, context),
            max_new_tokens=512,
            temperature=0.5,
            return_full_text=False
        )
        
        # Display results
        print("\nRESULT:", response.strip())
        print("\nSOURCE DOCUMENTS:")
        for i, doc in enumerate(docs, 1):
            print(f"\nDocument {i}:")
            print(doc.page_content)
            print(f"Source: {doc.metadata.get('source', 'unknown')}")
            
    except Exception as e:
        print(f"Error during query execution: {e}")