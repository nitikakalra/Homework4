# 📚 RAG Chatbot using Gemini, LangChain, FAISS and Streamlit

## 📌 Project Overview

This project is a Retrieval-Augmented Generation (RAG) chatbot that answers questions from a PDF document.

Instead of answering from general knowledge, the chatbot first searches the PDF for the most relevant information and then uses Google's Gemini model to generate an answer based only on that information.

---

## 🚀 Features

- Read text from a PDF
- Split text into smaller chunks
- Create embeddings using HuggingFace
- Store embeddings in a FAISS vector database
- Retrieve the most relevant chunks
- Generate answers using Gemini
- Simple Streamlit web interface
- Terminal version also included

---

## 🛠️ Technologies Used

- Python
- Google Gemini API
- LangChain
- FAISS
- HuggingFace Embeddings
- Sentence Transformers
- PyMuPDF (fitz)
- Streamlit

---

## 📂 Project Structure

```
Homework 4
│
├── app.py
├── web_app.py
├── README.md
│
├── data
│   └── thinkpython2.pdf
│
├── vector_db
│   ├── index.faiss
│   └── index.pkl
│
└── models
```

---

## ⚙️ Installation

### Step 1 : Clone the Repository

```bash
git clone <repository_link>
```

or download the project folder.

---

### Step 2 : Open the Project Folder

```bash
cd Homework 4
```

---

### Step 3 : Install Dependencies

```bash
pip install google-genai
pip install streamlit
pip install langchain
pip install langchain-community
pip install langchain-huggingface
pip install sentence-transformers
pip install faiss-cpu
pip install pymupdf
```

or

```bash
pip install -r requirements.txt
```

if a requirements file is available.

---

### Step 4 : Add Your Gemini API Key

Open

```
app.py
```

and

```
web_app.py
```

Replace

```python
api_key=""
```

with

```python
api_key="YOUR_GEMINI_API_KEY"
```

---

## ▶️ Running the Project

### Run Terminal Version

```bash
python app.py
```

Example

```
Ask Question:
What is Python?
```

Type

```
exit
```

to close the chatbot.

---

### Run Streamlit Web Application

```bash
streamlit run web_app.py
```

Open your browser and visit

```
http://localhost:8501
```

Enter your question and click **Ask Question**.

---

## 🔄 Project Workflow

```
PDF
   │
   ▼
Extract Text
   │
   ▼
Split into Chunks
   │
   ▼
Generate Embeddings
   │
   ▼
Store in FAISS
   │
   ▼
User Question
   │
   ▼
Similarity Search
   │
   ▼
Retrieve Top Chunks
   │
   ▼
Gemini Model
   │
   ▼
Generate Answer
   │
   ▼
Display Result
```

---

## 📖 How It Works

1. The PDF is loaded using PyMuPDF.
2. Text is extracted from every page.
3. The text is divided into smaller chunks.
4. Each chunk is converted into an embedding.
5. Embeddings are stored inside a FAISS vector database.
6. The user asks a question.
7. FAISS retrieves the most relevant chunks.
8. The retrieved chunks are sent to Gemini.
9. Gemini generates an answer based only on the retrieved context.
10. The answer is displayed in the terminal or Streamlit application.

---

## 📷 Sample Question

```
What is Python?
```

### Sample Answer

```
Python is an object-oriented programming language.
```

---

## 🎯 Learning Outcomes

From this project, we learned:

- How Retrieval-Augmented Generation (RAG) works
- How to read PDF files using PyMuPDF
- How to split large documents into chunks
- How embeddings represent text
- How FAISS performs similarity search
- How Gemini generates context-aware answers
- How to build a simple chatbot using Streamlit

---

## 👩‍💻 Author

**Nitika**

