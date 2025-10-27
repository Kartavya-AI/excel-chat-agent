import os
import pandas as pd
import chardet
from chromadb import PersistentClient
from chromadb.api.types import EmbeddingFunction, Documents
from crewai.tools import tool
import google.generativeai as genai
from typing import List

# === Configure Google Generative AI ===
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# === Custom Embedding Function Using Gemini ===
class GeminiEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, task_type="retrieval_document"):
        self.model_name = "models/text-embedding-004"  # Updated to a more stable model
        self.task_type = task_type

    def __call__(self, input: Documents) -> List[List[float]]:
        """
        Embed a list of documents using Gemini's embedding API.
        
        Args:
            input: List of documents (strings) to embed
            
        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        if isinstance(input, str):
            input = [input]
        
        embeddings = []
        for text in input:
            try:
                response = genai.embed_content(
                    model=self.model_name,
                    content=str(text),
                    task_type=self.task_type,
                    title="Semantic embedding for Excel row"
                )
                embeddings.append(response["embedding"])
            except Exception as e:
                print(f"⚠️ Embedding failed for text: {e}")
                # Fallback to zero vector - adjust dimension based on your model
                embeddings.append([0.0] * 768)  # Standard embedding dimension
        
        return embeddings

# === ChromaDB Setup ===
DB_PATH = os.getenv("CHROMA_DB_PATH", "/tmp/chroma_db")
collection_name = "excel_data"

os.makedirs(DB_PATH, exist_ok=True)
embedding_fn = GeminiEmbeddingFunction(task_type="retrieval_document")

# Delete existing collection if any
try:
    client = PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
    )
    print(f"✅ ChromaDB collection '{collection_name}' is ready at {DB_PATH}")
except Exception as e:
    print(f"❌ Failed to initialize ChromaDB at {DB_PATH}: {e}")
    collection = None

# === CSV Encoding Detection ===
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        rawdata = f.read(100000)
        result = chardet.detect(rawdata)
        return result['encoding'] or 'utf-8'

# === Text Cleaning Helper ===
def clean_text(text):
    try:
        if pd.isna(text) or text is None:
            return ""
        return str(text).encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    except:
        return ""

# === Excel/CSV Data Loader ===
def load_excel_or_csv(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    try:
        if ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path, engine="openpyxl", nrows=50)
            #return df.fillna("").to_dict(orient="records")
        elif ext == ".csv":
            encoding = detect_encoding(file_path)
            df = pd.read_csv(file_path, encoding=encoding, nrows=50)
            #return df.fillna("").to_dict(orient="records")
        return df.fillna("").to_dict(orient="records")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to process file '{file_path}': {str(e)}")

# === Store Rows into ChromaDB ===
def generate_and_store(file_path, batch_size=10):  # Reduced batch size for stability
    try:
        rows = load_excel_or_csv(file_path)
        if not rows:
            print("⚠️ No data found in the file.")
            return

        batch_docs, batch_ids, batch_metadata = [], [], []

        for i, row in enumerate(rows):
            # Create a readable content string from the row
            content_parts = []
            for k, v in row.items():
                clean_key = clean_text(k)
                clean_value = clean_text(v)
                if clean_key and clean_value:  # Only include non-empty values
                    content_parts.append(f"{clean_key}: {clean_value}")
            
            content = " | ".join(content_parts)
            if not content.strip():  # Skip empty rows
                continue
                
            batch_docs.append(content)
            batch_ids.append(f"row-{i}")
            batch_metadata.append({k: clean_text(v) for k, v in row.items()})

            # Process batch when it reaches the specified size
            if len(batch_docs) >= batch_size:
                try:
                    collection.add(
                        documents=batch_docs, 
                        ids=batch_ids, 
                        metadatas=batch_metadata
                    )
                    print(f"✅ Processed batch of {len(batch_docs)} rows")
                except Exception as batch_error:
                    print(f"⚠️ Batch processing error: {batch_error}")
                finally:
                    batch_docs, batch_ids, batch_metadata = [], [], []

        # Process remaining documents
        if batch_docs:
            try:
                collection.add(
                    documents=batch_docs, 
                    ids=batch_ids, 
                    metadatas=batch_metadata
                )
                print(f"✅ Processed final batch of {len(batch_docs)} rows")
            except Exception as final_error:
                print(f"⚠️ Final batch processing error: {final_error}")

        print(f"✅ Completed storing data into ChromaDB collection '{collection.name}'.")
        
    except Exception as e:
        print(f"❌ Failed to generate and store embeddings: {str(e)}")
        raise

# === Semantic Search Tool ===
#@tool("search_excel_data")
def search_excel_data(query: str, top_k: int = 3) -> str:
    """
    Perform a semantic search over the embedded Excel/CSV data using Gemini embeddings and ChromaDB.
    
    Args:
        query: The search query string
        top_k: Number of top results to return (default: 3)
        
    Returns:
        String containing the search results
    """
    try:
        # Create a query-specific embedding function
        query_embedder = GeminiEmbeddingFunction(task_type="retrieval_query")
        query_embedding = query_embedder([query])[0]  # Pass as list and get first result

        # Perform the search
        results = collection.query(
            query_embeddings=[query_embedding], 
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        
        if not documents:
            return "No relevant documents found."
        
        # Format results with relevance scores
        formatted_results = []
        for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
            formatted_results.append(f"Result {i+1} (relevance: {1-distance:.3f}):\n{doc}\n")
        
        return "\n".join(formatted_results)
        
    except Exception as e:
        error_msg = f"❌ Search failed: {str(e)}"
        print(error_msg)
        return error_msg

# === Utility function to check collection status ===
def get_collection_info():
    """Get information about the current collection"""
    try:
        count = collection.count()
        return f"Collection '{collection_name}' contains {count} documents."
    except Exception as e:
        return f"Error getting collection info: {str(e)}"

# Test function
if __name__ == "__main__":
    print("🔧 Excel Tool initialized successfully!")
    print(get_collection_info())
