import os
import pandas as pd
import chardet
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from crewai.tools import tool

# === ChromaDB Setup ===
client = PersistentClient(path="../../../knowledge/chroma_db")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="hkunlp/instructor-xl"
)

collection_name = "excel_data"

# Delete collection if it exists
try:
    client.delete_collection(name=collection_name)
    print(f"🧹 Deleted existing collection '{collection_name}'")
except Exception as e:
    print(f"⚠️ No existing collection to delete or failed: {e}")

# Create fresh collection
collection = client.create_collection(
    name=collection_name,
    embedding_function=embedding_fn
)

# === Encoding Detection for CSV ===
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        rawdata = f.read(100000)
        result = chardet.detect(rawdata)
        return result['encoding'] or 'utf-8'

# === Data Cleaning ===
def clean_text(text):
    try:
        return str(text).encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    except:
        return ""

# === Data Loading (only first 1000 rows) ===
def load_excel_or_csv(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    try:
        if ext == ".xlsx":
            return pd.read_excel(file_path, engine="openpyxl", nrows=1000).fillna("").to_dict(orient="records")
        elif ext == ".csv":
            encoding = detect_encoding(file_path)
            return pd.read_csv(file_path, encoding=encoding, nrows=1000).fillna("").to_dict(orient="records")
        else:
            raise ValueError("❌ Unsupported file type. Use .xlsx or .csv")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to process file '{file_path}': {str(e)}")

# === Embedding and Storing ===
def generate_and_store(file_path, batch_size=512):
    try:
        rows = load_excel_or_csv(file_path)
        if not rows:
            print("⚠️ No data found in the file.")
            return

        batch_docs, batch_ids, batch_metadata = [], [], []

        for i, row in enumerate(rows):
            # Embed row with keys (columns) for better context
            content = " | ".join(f"{clean_text(k)}: {clean_text(v)}" for k, v in row.items())
            batch_docs.append(content)
            batch_ids.append(f"row-{i}")
            batch_metadata.append({k: clean_text(v) for k, v in row.items()})

            if len(batch_docs) >= batch_size:
                collection.add(documents=batch_docs, ids=batch_ids, metadatas=batch_metadata)
                batch_docs, batch_ids, batch_metadata = [], [], []

        if batch_docs:
            collection.add(documents=batch_docs, ids=batch_ids, metadatas=batch_metadata)

        print(f"✅ Stored {len(rows)} rows into ChromaDB collection '{collection.name}'.")
    except Exception as e:
        print(f"❌ Failed to generate and store embeddings: {str(e)}")
# === CrewAI Tool Function ===
@tool("search_excel_data")
def search_excel_data(query: str, top_k: int = 3) -> str:
    """
    Perform a semantic search over the embedded Excel/CSV data using ChromaDB.
    """
    try:
        results = collection.query(query_texts=[query], n_results=top_k)
        documents = results.get("documents", [[]])[0]
        return "\n".join(documents) if documents else "No relevant documents found."
    except Exception as e:
        print(f"❌ Search failed: {str(e)}")
        return "❌ Search failed: " + str(e)
