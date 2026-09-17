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
from generate_quiz import generate_quiz

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

# === ZONE PRINCIPALE — question / réponse ou quiz ===
mode = st.radio("Choisis un mode :", ("Poser une question", "Générer un quiz"), horizontal=True)

if mode == "Poser une question":
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

else:
    st.subheader("Générer un quiz")
    st.caption("Le quiz est créé uniquement à partir des passages pertinents de tes documents indexés.")

    with st.form("quiz_settings"):
        topic = st.text_input("Sujet du quiz")
        num_questions = st.slider("Nombre de questions", min_value=1, max_value=10, value=5)
        generate = st.form_submit_button("Générer le quiz")

    if generate:
        if collection.count() == 0:
            st.warning("Aucun contenu indexé pour l'instant — upload un PDF d'abord.")
        elif not topic.strip():
            st.warning("Indique un sujet pour le quiz.")
        else:
            with st.spinner("Recherche des passages et génération du quiz..."):
                try:
                    n_results = min(10, collection.count())
                    retrieved_chunks, retrieved_metadatas = retrieve_chunks(
                        topic, collection, model, n_results=n_results
                    )
                    quiz = generate_quiz(
                        topic,
                        retrieved_chunks,
                        retrieved_metadatas,
                        groq_client,
                        num_questions=num_questions,
                    )
                    for key in list(st.session_state):
                        if key.startswith("quiz_answer_"):
                            del st.session_state[key]
                    st.session_state.quiz = quiz
                    st.session_state.quiz_submitted = False
                except ValueError as error:
                    st.error(f"Impossible de générer le quiz : {error}")
                except Exception:
                    st.error("Impossible de générer le quiz pour le moment. Réessaie plus tard.")

    quiz = st.session_state.get("quiz")
    if quiz:
        with st.form("quiz_answers_form"):
            answers = []
            for index, quiz_question in enumerate(quiz["questions"], start=1):
                st.markdown(f"**Question {index}. {quiz_question['question']}**")
                answer = st.radio(
                    "Choisis une réponse",
                    options=[None, 0, 1, 2, 3],
                    format_func=lambda option, choices=quiz_question["options"]: (
                        "-- Sélectionne une réponse --" if option is None else choices[option]
                    ),
                    key=f"quiz_answer_{index}",
                )
                answers.append(answer)

            submit_answers = st.form_submit_button("Voir mon résultat")

        if submit_answers:
            st.session_state.quiz_results = answers
            st.session_state.quiz_submitted = True

        if st.session_state.get("quiz_submitted"):
            answers = st.session_state.quiz_results
            score = sum(
                answer == quiz_question["correct_answer"]
                for answer, quiz_question in zip(answers, quiz["questions"])
            )
            st.success(f"Résultat : {score}/{len(quiz['questions'])}")

            for index, (answer, quiz_question) in enumerate(zip(answers, quiz["questions"]), start=1):
                correct_option = quiz_question["options"][quiz_question["correct_answer"]]
                if answer == quiz_question["correct_answer"]:
                    st.success(f"Question {index} : bonne réponse.")
                else:
                    st.error(f"Question {index} : la bonne réponse était « {correct_option} ».")
                st.write(quiz_question["explanation"])
                st.caption(f"Source : {quiz_question['source']} — page {quiz_question['page']}")
