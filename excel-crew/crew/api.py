from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from tools.exceltool import EmbeddingTool
from crew import excelcrew
import tempfile
import os

app = FastAPI()

@app.post("/analyze-excel")
async def analyze_excel(file: UploadFile = File(...), question: str = Form(...)):
    try:
        # Save the uploaded Excel file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        print("🔄 Loading and embedding Excel data...")
        embedder = EmbeddingTool()
        embedder.generate_and_store(tmp_path)

        print("🔍 Retrieving relevant content for your question...")
        search_result = embedder.search(query=question, top_k=5)
        context_docs = "\n".join(search_result["documents"][0])

        inputs = {
            "question": question,
            "search_results": context_docs
        }

        print("🚀 Running agents...")
        crew = excelcrew()
        result = crew.crew().kickoff(inputs=inputs)

        # Clean up temporary file
        os.remove(tmp_path)

        return JSONResponse(content={"answer": [str(r) for r in result]})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
