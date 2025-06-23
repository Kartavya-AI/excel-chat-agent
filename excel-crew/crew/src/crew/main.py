import streamlit as st
from tools.exceltool import generate_and_store
from crew import excelcrew
import tempfile
import os

st.set_page_config(page_title="Excel AI Analyst", layout="centered")
st.title("📊 Excel Chat Analyst")

# === Session Setup ===
if "file_processed" not in st.session_state:
    st.session_state.file_processed = False
if "context_docs" not in st.session_state:
    st.session_state.context_docs = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "embedder" not in st.session_state:
    st.session_state.embedder = None

# === Upload File ===
uploaded_file = st.file_uploader("Upload Excel (.xlsx or .csv)", type=["xlsx", "csv"])

if uploaded_file and not st.session_state.file_processed:
    with st.spinner("📊 Processing Excel and creating embeddings..."):
        suffix = ".csv" if uploaded_file.name.endswith(".csv") else ".xlsx"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        try:
            
            generate_and_store(temp_path)

            st.session_state.temp_path = temp_path
            st.session_state.file_processed = True

            st.success("✅ Excel/CSV file embedded and ready to chat!")

        except Exception as e:
            st.error(f"❌ Failed to process file: {e}")
            os.remove(temp_path)

# === Chat UI ===
if st.session_state.file_processed:
    st.markdown("### 🤖 Ask a question about your data")

    user_question = st.chat_input("Ask a question...")

    if user_question:
        with st.spinner("🤔 Thinking..."):
            try:
                inputs = {
                    "question": user_question,
                }

                crew = excelcrew()
                result = crew.crew().kickoff(inputs=inputs)
                response = result

                st.session_state.chat_history.append(("user", user_question))
                st.session_state.chat_history.append(("ai", response))

            except Exception as e:
                response = f"❌ Error: {e}"
                st.session_state.chat_history.append(("ai", response))

# === Display Chat History ===
for sender, msg in st.session_state.chat_history:
    with st.chat_message(sender):
        st.markdown(msg)
