def retrieve_chunks(question, collection, model, n_results=5):
    query_embedding = model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )

    retrieved_chunks = results["documents"][0]
    retrieved_metadatas = results["metadatas"][0]

    return retrieved_chunks, retrieved_metadatas


def retrieve_document_context(question, collection, model, source, n_results=5):
    records = collection.get(where={"source": source}, include=["documents", "metadatas"])
    documents = records["documents"]
    metadatas = records["metadatas"]
    if not documents:
        return [], []

    # Short documents fit in context, so overview questions can cover every page.
    if sum(len(document) for document in documents) <= 24000:
        ordered = sorted(zip(documents, metadatas), key=lambda item: item[1]["page"])
        return [item[0] for item in ordered], [item[1] for item in ordered]

    results = collection.query(
        query_embeddings=model.encode([question]).tolist(),
        where={"source": source},
        n_results=min(n_results, len(documents)),
    )
    return results["documents"][0], results["metadatas"][0]
