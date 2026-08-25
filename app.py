import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
from dotenv import load_dotenv
import sys
import tempfile

sys.path.append("src")
from extraction_and_chunking import extract_and_chunk
from dep_index import index_chunks
from retrieve_chunks import retrieve_chunks
from build_promt import build_prompt
from generate_response import generate_answer

load_dotenv(".env")

st.set_page_config(page_title="RAG Study Assistant", page_icon="📚")


# === SETUP — chargé une seule fois grâce au cache Streamlit ===
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_collection():
    client_db = chromadb.PersistentClient(path="vectorstore")
    return client_db.get_or_create_collection(name="mon_cours")


@st.cache_resource
def load_groq_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


model = load_model()
collection = load_collection()
groq_client = load_groq_client()

st.title("📚 RAG Study Assistant")
st.caption("Pose des questions sur tes cours, réponses ancrées avec citation de la source.")

# === SIDEBAR — upload de PDF ===
with st.sidebar:
    st.header("Ajouter un document")
    uploaded_file = st.file_uploader("Upload un PDF de cours", type="pdf")

    if uploaded_file is not None:
        if st.button("Indexer ce document"):
            with st.spinner("Extraction et indexation en cours..."):
                # on sauvegarde le fichier uploadé temporairement pour que fitz puisse l'ouvrir
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                chunks, metadatas = extract_and_chunk(tmp_path)
                # on remplace le nom temporaire par le vrai nom du fichier uploadé
                for meta in metadatas:
                    meta["source"] = uploaded_file.name

                nb_indexed = index_chunks(chunks, metadatas, collection, model)
                os.remove(tmp_path)

            st.success(f"{nb_indexed} chunks indexés depuis {uploaded_file.name}")

    st.divider()
    st.caption(f"Total dans la base : {collection.count()} chunks")

# === ZONE PRINCIPALE — question / réponse ===
question = st.text_input("Pose ta question sur le cours :")

if question:
    with st.spinner("Recherche et génération de la réponse..."):
        retrieved_chunks, retrieved_metadatas = retrieve_chunks(question, collection, model, n_results=5)

        if not retrieved_chunks:
            st.warning("Aucun contenu indexé pour l'instant — upload un PDF d'abord.")
        else:
            prompt = build_prompt(question, retrieved_chunks, retrieved_metadatas)
            answer = generate_answer(prompt, groq_client)

            st.markdown("### Réponse")
            st.write(answer)

            with st.expander("Voir les passages sources utilisés"):
                for chunk, meta in zip(retrieved_chunks, retrieved_metadatas):
                    st.markdown(f"**{meta['source']} — page {meta['page']}**")
                    st.text(chunk)
                    st.divider()