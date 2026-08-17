import os
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_and_chunk(file_path):
    filename = os.path.basename(file_path)
    doc = fitz.open(file_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    metadatas = []

    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        page_chunks = splitter.split_text(page_text)
        for chunk in page_chunks:
            chunks.append(chunk)
            metadatas.append({"source": filename, "page": page_num + 1})

    doc.close()
    return chunks, metadatas   