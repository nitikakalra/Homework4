import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import fitz
from google import genai

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# -------------------------------------------------
# GEMINI
# -------------------------------------------------

client = genai.Client(
    api_key=""      
)

# -------------------------------------------
# PDF
# -------------------------------------------------

pdf_path = "data/thinkpython2.pdf"

# -------------------------------------------------
# EMBEDDING MODEL
# -------------------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------------------------
# VECTOR DATABASE
# -------------------------------------------------

if os.path.exists("vector_db/index.faiss"):

    print("Loading Existing Vector Database...")

    vector_db = FAISS.load_local(
        "vector_db",
        embedding_model,
        allow_dangerous_deserialization=True
    )

    print("Vector Database Loaded Successfully!")

else:

    print("Creating New Vector Database...")

    pdf = fitz.open(pdf_path)

    text = ""

    for page in pdf:
        text += page.get_text()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    vector_db = FAISS.from_texts(
        chunks,
        embedding_model
    )

    vector_db.save_local("vector_db")

    print("Vector Database Saved Successfully!")

# -------------------------------------------------
# CHAT
# -------------------------------------------------

while True:

    question = input("\nAsk Question (type exit to quit): ")

    if question.lower() == "exit":
        print("\nGoodbye!")
        break

    docs = vector_db.similarity_search(
        question,
        k=3
    )

    print("\nTop Retrieved Chunks\n")

    context = ""

    for i, doc in enumerate(docs, start=1):

        print("=" * 80)
        print(f"Chunk {i}")
        print("=" * 80)
        print(doc.page_content)
        print()

        context += doc.page_content + "\n\n"

    prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the context below.

If the answer is not present in the context, reply exactly:

I could not find the answer in the PDF.

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    print("\nAnswer:\n")
    print(response.text)
