import hashlib

def index_chunks(chunks, metadatas, collection, model):
    
    seen_ids = set()
    unique_chunks = []
    unique_metadatas = []
    unique_ids = []

    for chunk, meta in zip(chunks, metadatas):
        chunk_id = hashlib.md5(chunk.encode()).hexdigest()
        if chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            unique_chunks.append(chunk)
            unique_metadatas.append(meta)
            unique_ids.append(chunk_id)

    embeddings = model.encode(unique_chunks).tolist()

    collection.upsert(
        documents=unique_chunks,
        embeddings=embeddings,
        ids=unique_ids,
        metadatas=unique_metadatas
    )

    return len(unique_chunks)