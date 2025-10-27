from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from src.crew.tools.exceltool import generate_and_store,search_excel_data
from src.crew.crew import excelcrew
import tempfile
import os

app = FastAPI()

@app.post("/analyze-excel")
async def analyze_excel(file: UploadFile = File(...), question: str = Form(...)):
    try:
        # Save the uploaded Excel file to a temporary location
        suffix = os.path.splitext(file.filename)[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        print("🔄 Loading and embedding Excel data...")
        generate_and_store(tmp_path)

        print("🔍 Retrieving relevant content for your question...")
        if hasattr(search_excel_data, "func"):
            search_result = search_excel_data.func(query=question, top_k=5)
        else:
            search_result = search_excel_data(query=question, top_k=5)

        # ✅ search_result is a string, not dict
        context_docs = str(search_result)

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
