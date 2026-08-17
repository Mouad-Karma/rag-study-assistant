def generate_answer(prompt, groq_client):
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",   # ← changé, ancien: llama-3.1-8b-instant
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content