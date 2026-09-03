# RAG Study Assistant 

A RAG (Retrieval-Augmented Generation) chatbot that answers questions about your course PDFs, with grounded answers and source citations (document + page number).

Built to demonstrate practical RAG / LLM engineering skills: PDF ingestion, chunking, embeddings, vector search, prompt engineering, and generation — end to end, with a Streamlit interface.

## Features

-  Upload one or several course PDFs
-  Semantic search across all indexed documents (multi-course knowledge base)
-  Grounded answers generated with Groq (`openai/gpt-oss-20b`)
-  Source citation — every answer points back to the exact document and page
-  Anti-hallucination guardrail — the assistant says when the answer isn't in the course content, instead of making things up
-  Deduplication — re-indexing the same PDF never creates duplicate chunks (content-hash based IDs)

## Tech stack

| Layer | Tool |
|---|---|
| PDF extraction | PyMuPDF (`fitz`) |
| Chunking | LangChain (`RecursiveCharacterTextSplitter`) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector store | ChromaDB (persistent, local) |
| LLM generation | Groq API (`openai/gpt-oss-20b`) |
| UI | Streamlit |

## Project structure

```
rag-study-assistant/
├── app.py                      # Streamlit app (entry point)
├── reset_db.py                 # utility script to wipe the vector store
├── data/                       # (optional) local PDFs for notebook testing
├── notebooks/
│   └── exploration.ipynb       # exploration / development notebook
├── src/
│   ├── extraction_and_chunking.py   # extract_and_chunk(file_path)
│   ├── dep_index.py                 # index_chunks(chunks, metadatas, collection, model)
│   ├── retrieve_chunks.py           # retrieve_chunks(question, collection, model, n_results=5)
│   ├── build_promt.py               # build_prompt(question, chunks, metadatas)
│   └── generate_response.py         # generate_answer(prompt, groq_client)
├── vectorstore/                # ChromaDB persistent storage (gitignored)
├── .env                        # GROQ_API_KEY (gitignored)
├── .gitignore
└── requirements.txt
```

## How the pipeline works

1. **Extraction + chunking** (`extract_and_chunk`) — the PDF is read page by page with PyMuPDF, and each page's text is split into ~500-character chunks. Every chunk keeps track of its source filename and page number.
2. **Deduplication + indexing** (`index_chunks`) — each chunk gets a stable ID from an MD5 hash of its own text. Identical content always produces the same ID, so re-indexing the same document never creates duplicates (`collection.upsert` overwrites instead of adding).
3. **Retrieval** (`retrieve_chunks`) — the question is embedded with the same model as the chunks, and ChromaDB returns the most semantically similar chunks across *all* indexed documents.
4. **Prompt construction** (`build_prompt`) — the retrieved chunks are tagged with their source/page and assembled into a prompt that instructs the LLM to answer only from the given context, and to say so if the answer isn't there.
5. **Generation** (`generate_answer`) — the prompt is sent to Groq, which returns a grounded, cited answer.

## Setup

```bash
git clone https://github.com/Mouad-Karma/rag-study-assistant.git
cd rag-study-assistant

conda create -n rag-assistant python=3.11
conda activate rag-assistant
pip install -r requirements.txt
```

Create a `.env` file at the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

## Running the app

```bash
streamlit run app.py
```

Then in the browser:
1. Upload a course PDF in the sidebar
2. Click **"Indexer ce document"** and wait for the confirmation message
3. Ask a question in the main text box — the answer will appear with an expandable section showing the exact source passages used

You can upload and index several PDFs — the assistant searches across all of them at once, so you can ask questions spanning multiple courses.

## Resetting the vector store

If you want to wipe all indexed documents and start fresh (useful after testing, or if the database gets into a confusing mixed state):

```bash
python3 reset_db.py
```

This deletes the `mon_cours` ChromaDB collection. The app will recreate an empty one automatically the next time it starts. Note: this only clears the *index* — your original PDFs on disk are untouched, so you can just re-upload and re-index them.

**Tip:** if you experiment in the notebook, consider using a separate collection name there (e.g. `mon_cours_test`) so notebook experiments never pollute the collection the Streamlit app uses.

## Known limitations

- Retrieval quality depends on chunk boundaries — a chunk that's cut mid-definition can occasionally cause the assistant to miss information that technically exists in the document. Increasing `n_results` in `retrieve_chunks` helps but doesn't fully solve this.
- ChromaDB's local persistent client is well suited to single-user, local use. It isn't designed for concurrent multi-user writes at scale.
- Source citation relies on the LLM reading the `[Source: ..., page ...]` tags in the prompt correctly — it isn't a programmatically guaranteed link, so citations should be treated as generally reliable rather than 100% verified.

## Roadmap

- [ ] Retrieval evaluation (e.g. with Ragas)
- [ ] Quiz generation from indexed content
- [ ] Agentic version with LangGraph (RAG v2)
- [ ] MLflow integration for experiment tracking
- [ ] Optional Docker packaging