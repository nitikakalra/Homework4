import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import fitz

from google import genai

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ----------------------------
# PAGE CONFIG
# ----------------------------

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF Question Answering Chatbot")

st.write("Ask questions from the PDF.")

# ----------------------------
# GEMINI CLIENT
# ----------------------------

client = genai.Client(
    api_key=""   
)

# ----------------------------
# EMBEDDING MODEL
# ----------------------------

@st.cache_resource
def load_vector_db():

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    pdf_path = "data/thinkpython2.pdf"

    if os.path.exists("vector_db/index.faiss"):

        db = FAISS.load_local(
            "vector_db",
            embedding_model,
            allow_dangerous_deserialization=True
        )

    else:

        pdf = fitz.open(pdf_path)

        text = ""

        for page in pdf:
            text += page.get_text()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        chunks = splitter.split_text(text)

        db = FAISS.from_texts(
            chunks,
            embedding_model
        )

        db.save_local("vector_db")

    return db


with st.spinner("Loading Vector Database..."):
    vector_db = load_vector_db()

st.success("Vector Database Ready!")

# ----------------------------
# USER INPUT
# ----------------------------

question = st.text_input(
    "Ask your question",
    placeholder="Example: What is Python?"
)

# ----------------------------
# BUTTON
# ----------------------------

if st.button("Ask"):

    if question.strip() == "":
        st.warning("Please enter a question.")
        st.stop()

    try:

        docs = vector_db.similarity_search(
            question,
            k=3
        )

        context = ""

        for doc in docs:
            context += doc.page_content + "\n\n"

        prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the context below.

If the answer is not available, reply exactly:

I could not find the answer in the PDF.

Context:
{context}

Question:
{question}

Answer:
"""

        with st.spinner("Generating Answer..."):

            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt
            )

        st.subheader("Answer")

        st.write(response.text)

        with st.expander("Retrieved Chunks"):

            for i, doc in enumerate(docs, start=1):

                st.markdown(f"### Chunk {i}")

                st.write(doc.page_content)

    except Exception as e:

        st.error("Error")

        st.exception(e)