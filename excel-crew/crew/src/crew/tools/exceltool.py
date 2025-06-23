import pandas as pd
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

class EmbeddingTool:
    def __init__(self, collection_name="excel_data", db_path="../../../knowledge/chroma_db"):
        self.client = PersistentClient(path=db_path)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="hkunlp/instructor-xl"
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )

    def load_excel(self, file_path):
        df = pd.read_excel(file_path)
        return df.fillna("").astype(str).to_dict(orient="records")

    def generate_and_store(self, file_path):
        rows = self.load_excel(file_path)
        for i, row in enumerate(rows):
            content = " | ".join(row.values())
            self.collection.add(
                documents=[content],
                ids=[f"row-{i}"],
                metadatas=[row]
            )
        print(f"✅ Stored {len(rows)} rows into ChromaDB collection '{self.collection.name}'.")

    def search(self, query, top_k=3):
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        return results
