
# Uniform Manual AI Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that provides accurate answers from the Uniform Manual by combining semantic search, reranking, and large language models.

## Features

- **Semantic Search** – Uses ChromaDB with `text-embedding-3-small` for vector-based retrieval.
- **Dual-Query Retrieval** – Executes two separate searches (current and historical context) to prevent topic pollution.
- **BGE Reranker** – Employs `BAAI/bge-reranker-v2-m3` Cross-Encoder for precision re-ranking.
- **Conversation Memory** – Maintains the last three exchanges to handle follow-up questions.
- **Follow-up Detection** – Detects user follow-up queries using pattern matching, word count, and keyword overlap.
- **Gemini LLM** – Generates concise answers using `gemini-3.1-flash-lite`.

## Tech Stack

- Python
- ChromaDB (vector database)
- text-embedding-3-small (embedding model)
- BAAI/bge-reranker-v2-m3 (reranker)
- Google Gemini (LLM)
- 
## How to Run

### 1. Clone this repository

```bash
git clone https://github.com/stupidFLOWERch/Uniform-Manual-Chatbot.git
cd Uniform-Manual-Chatbot
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Import the knowledge base

Run the following command:

```bash
python loader.py
```

This will create the `chroma_db/` directory containing embedded chunks generated from `uniform_manual.json`.

## Disclaimer

* The knowledge base is built from **The Boys' Brigade in Malaysia Uniform Manual (May 2020)**.
* This chatbot is **not** an official product of **The Boys' Brigade in Malaysia**.
* The information provided may not reflect the latest revisions to the Uniform Manual.
* For the most up-to-date information, please refer to the official website: https://www.bbmalaysia.org/


