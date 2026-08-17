def retrieve_chunks(question, collection, model, n_results=5):
    query_embedding = model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )

    retrieved_chunks = results["documents"][0]
    retrieved_metadatas = results["metadatas"][0]

    return retrieved_chunks, retrieved_metadatas