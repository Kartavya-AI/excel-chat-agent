import sys
from tools.exceltool import EmbeddingTool
from crew import excelcrew
import os
import tempfile

def main(excel_path: str, user_question: str):
    try:
        # === Step 1: Embed Excel ===
        print("🔄 Loading and embedding Excel data...")
        embedder = EmbeddingTool()
        embedder.generate_and_store(excel_path)

        # === Step 2: Search relevant context ===
        print("🔍 Retrieving relevant content for your question...")
        search_result = embedder.search(query=user_question, top_k=5)
        context_docs = "\n".join(search_result["documents"][0])

        # === Step 3: Prepare input ===
        inputs = {
            "question": user_question,
            "search_results": context_docs
        }

        print(f"\n📦 Inputs:\n{inputs}\n")

        # === Step 4: Run CrewAI ===
        crew = excelcrew()
        print("🚀 Running agents...")
        result = crew.crew().kickoff(inputs=inputs)

        print("\n✅ Final Answer:\n")
        for r in result:
            print(str(r))
        print()

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("❗ Usage: python main.py <excel_path> \"<your question>\"")
        sys.exit(1)

    excel_file_path = sys.argv[1]
    user_question = sys.argv[2]
    main(excel_file_path, user_question)
