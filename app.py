import hashlib
import os
from pathlib import Path
import sys
import tempfile

import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import streamlit as st

sys.path.append("src")
from extraction_and_chunking import extract_and_chunk
from dep_index import index_chunks
from retrieve_chunks import retrieve_chunks, retrieve_document_context
from build_promt import build_prompt
from generate_response import generate_answer
from generate_quiz import generate_quiz

load_dotenv(".env")

st.set_page_config(page_title="Study Studio · RAG Study Assistant", page_icon="📚", layout="wide")
st.html((Path(__file__).parent / "src" / "styles.css").read_text())


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

st.markdown("<div class='eyebrow'>VOTRE ESPACE POUR APPRENDRE</div>", unsafe_allow_html=True)
st.title("Moins de recherche. Plus de déclics.")
st.write("Explorez vos documents, posez vos questions et testez ce que vous avez retenu.")

# === SIDEBAR — upload de PDF ===
with st.sidebar:
    st.markdown("## 📚 Study Studio")
    st.caption("Vos documents, vos idées, votre rythme.")
    st.divider()
    st.subheader("Votre bibliothèque")
    uploaded_file = st.file_uploader("Ajouter un PDF", type="pdf")

    if uploaded_file is not None:
        pdf_bytes = uploaded_file.getvalue()
        document_key = (uploaded_file.name, hashlib.sha256(pdf_bytes).hexdigest())
        indexed_uploads = st.session_state.setdefault("indexed_uploads", {})
        if document_key not in indexed_uploads:
            tmp_path = None
            try:
                if not uploaded_file.name.lower().endswith(".pdf") or not pdf_bytes:
                    raise ValueError("Choisis un fichier PDF non vide.")
                with st.spinner("Préparation de votre document…"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_path = tmp_file.name
                        tmp_file.write(pdf_bytes)

                    chunks, metadatas = extract_and_chunk(tmp_path)
                    if not chunks:
                        raise ValueError("Ce PDF ne contient aucun texte exploitable.")
                    for meta in metadatas:
                        meta["source"] = uploaded_file.name

                    nb_indexed = index_chunks(chunks, metadatas, collection, model)
                    indexed_uploads[document_key] = nb_indexed
            except ValueError as error:
                st.error(f"Impossible d'indexer ce PDF : {error}")
            except Exception:
                st.error("Impossible d'indexer ce PDF. Vérifie le fichier puis réessaie.")
            finally:
                if tmp_path is not None:
                    os.remove(tmp_path)

        if document_key in indexed_uploads:
            nb_indexed = indexed_uploads[document_key]
            st.success("Document prêt. À vous de jouer !")

    st.divider()
    st.caption("Les documents restent disponibles après leur import. Retirez le fichier sélectionné pour explorer toute votre bibliothèque.")

# Keep conversations and quizzes separate for each document scope.
active_source = None
if uploaded_file is not None and document_key in indexed_uploads:
    active_source = uploaded_file.name
scope = active_source or "__library__"
st.caption(f"📄 {active_source}" if active_source else "📚 Toute votre bibliothèque")
mode = st.segmented_control(
    "Votre espace", ("Discussion", "Quiz interactif"), default="Discussion",
    required=True, width="stretch", key="study_mode", label_visibility="collapsed",
)


def get_context(question, n_results=5):
    if active_source:
        return retrieve_document_context(question, collection, model, active_source, n_results=n_results)
    return retrieve_chunks(question, collection, model, n_results=min(n_results, collection.count()))


def show_sources(chunks, metadatas):
    with st.expander("Consulter les passages sources"):
        for chunk, meta in zip(chunks, metadatas):
            st.caption(f"{meta['source']} · Page {meta['page']}")
            st.text(chunk)
            st.divider()


if mode == "Discussion":
    conversations = st.session_state.setdefault("conversations", {})
    messages = conversations.setdefault(scope, [])
    heading, action = st.columns([4, 1])
    heading.subheader("Une question, une nouvelle perspective.")
    if action.button("Effacer la discussion", disabled=not messages, use_container_width=True):
        conversations[scope] = []
        st.rerun()
    if not messages:
        with st.container(border=True):
            st.markdown("### Faites parler vos documents")
            st.write("Demandez une explication, un résumé ou des exemples. Chaque réponse s’appuie sur vos sources.")
            st.caption("Essayez : « Résume les idées principales » ou « Explique ce sujet simplement ».")
    for message in messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant":
                show_sources(message["chunks"], message["metadatas"])
    question = st.chat_input("Que souhaitez-vous comprendre ?")
    if question:
        if collection.count() == 0:
            st.info("Ajoutez votre premier PDF dans la bibliothèque pour commencer.")
        else:
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                try:
                    with st.spinner("Je parcours vos sources…"):
                        chunks, metadatas = get_context(question)
                        if not chunks:
                            st.warning("Aucun passage disponible pour ce document.")
                        else:
                            answer = generate_answer(build_prompt(question, chunks, metadatas), groq_client)
                            st.write(answer)
                            show_sources(chunks, metadatas)
                            messages.extend([
                                {"role": "user", "content": question},
                                {"role": "assistant", "content": answer, "chunks": chunks, "metadatas": metadatas},
                            ])
                except Exception:
                    st.error("La réponse n’a pas pu être générée. Réessayez dans un instant.")
else:
    st.subheader("Transformez vos lectures en acquis.")
    st.caption("Un sujet, quelques questions, et un retour immédiat pour progresser.")
    quizzes = st.session_state.setdefault("quizzes", {})
    session = quizzes.get(scope)
    with st.expander("Créer un nouveau quiz") if session else st.container(border=True):
        st.markdown("### Préparez votre session")
        with st.form("quiz_settings", border=False):
            topic = st.text_input("Que voulez-vous réviser ?", placeholder="Ex. : les concepts clés, les projets présentés…")
            num_questions = st.slider("Nombre de questions", min_value=1, max_value=10, value=5)
            generate = st.form_submit_button("Créer mon quiz →", type="primary", use_container_width=True)
        if generate:
            if collection.count() == 0:
                st.info("Ajoutez un PDF pour créer votre premier quiz.")
            elif not topic.strip():
                st.warning("Indiquez un sujet à réviser.")
            else:
                try:
                    with st.spinner("Votre quiz prend forme…"):
                        chunks, metadatas = get_context(topic, n_results=10)
                        quiz = generate_quiz(topic, chunks, metadatas, groq_client, num_questions=num_questions)
                    quizzes[scope] = {"quiz": quiz, "topic": topic, "answers": [], "position": 0,
                                      "attempt": st.session_state.get("quiz_attempt", 0) + 1}
                    st.session_state.quiz_attempt = quizzes[scope]["attempt"]
                    st.rerun()
                except ValueError:
                    st.error("Impossible de créer un quiz à partir de ce sujet. Essayez un sujet plus précis.")
                except Exception:
                    st.error("Le quiz n’a pas pu être créé. Réessayez dans un instant.")
    if session:
        questions = session["quiz"]["questions"]
        position = session["position"]
        total = len(questions)
        st.markdown(f"### {session['topic']}")
        st.progress(len(session["answers"]) / total, text=f"{len(session['answers'])} / {total} questions validées")
        if position < total:
            current = questions[position]
            checked = len(session["answers"]) > position
            with st.container(border=True):
                st.caption(f"QUESTION {position + 1} SUR {total}")
                st.markdown(f"### {current['question']}")
                choice = st.radio("Votre réponse", options=range(4), index=None,
                                  format_func=lambda option: f"{chr(65 + option)}. {current['options'][option]}",
                                  key=f"choice_{session['attempt']}_{position}", disabled=checked)
                if not checked:
                    if st.button("Valider ma réponse", type="primary", disabled=choice is None):
                        session["answers"].append(choice)
                        st.rerun()
                else:
                    correct = session["answers"][position] == current["correct_answer"]
                    if correct:
                        st.success("Bien joué, c’est la bonne réponse !")
                    else:
                        st.error(f"La bonne réponse : {current['options'][current['correct_answer']]}")
                    st.write(current["explanation"])
                    st.caption(f"{current['source']} · Page {current['page']}")
                    if st.button("Voir mon bilan →" if position == total - 1 else "Question suivante →", type="primary"):
                        session["position"] += 1
                        st.rerun()
        else:
            score = sum(answer == question["correct_answer"] for answer, question in zip(session["answers"], questions))
            with st.container(border=True):
                st.markdown("### Votre session est terminée")
                first, second = st.columns(2)
                first.metric("Bonnes réponses", f"{score} / {total}")
                second.metric("Réussite", f"{round(score / total * 100)} %")
                st.write("Excellent travail !" if score == total else "Chaque essai vous fait progresser. Consultez les explications et réessayez à votre rythme.")
                if st.button("Rejouer ce quiz", type="primary"):
                    session["answers"] = []
                    session["position"] = 0
                    st.session_state.quiz_attempt += 1
                    session["attempt"] = st.session_state.quiz_attempt
                    st.rerun()
            for number, (answer, question) in enumerate(zip(session["answers"], questions), start=1):
                correct = answer == question["correct_answer"]
                with st.expander(f"{'✓' if correct else '○'} {number}. {question['question']}"):
                    st.write(f"Votre réponse : {question['options'][answer]}")
                    st.write(f"Bonne réponse : {question['options'][question['correct_answer']]}")
                    st.write(question["explanation"])
                    st.caption(f"{question['source']} · Page {question['page']}")
