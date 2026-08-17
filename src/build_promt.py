def build_prompt(question, chunks, metadatas):
    context_parts = []
    for chunk, metadata in zip(chunks, metadatas):
        source_info = f"[Source: {metadata['source']}, page {metadata['page']}]"
        context_parts.append(f"{source_info}\n{chunk}")

    context = "\n\n".join(context_parts)

    prompt = f"""Tu es un assistant qui répond aux questions en te basant UNIQUEMENT sur le contexte fourni ci-dessous, extrait d'un cours.

Contexte :
{context}

Question : {question}

Réponds de manière claire et concise, en te basant uniquement sur le contexte. Indique la page source à la fin de ta réponse (ex: "Source : page X"). Si l'information n'est pas dans le contexte, dis-le clairement plutôt que d'inventer."""

    return prompt