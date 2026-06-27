import streamlit as st
import pandas as pd
import faiss

from sentence_transformers import SentenceTransformer

 
# Configuration de la page
 
st.set_page_config(
    page_title="Medical Chatbot",
    page_icon="🩺",
    layout="wide"
)

 
# Chargement des données
 
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_index():
    return faiss.read_index("models/medical_index.faiss")


@st.cache_data
def load_dataset():
    return pd.read_csv("data/processed/medical_qa.csv")


model = load_model()
index = load_index()
df = load_dataset()

 
# Interface
st.title("🩺 Medical Chatbot")

st.info(
    " Ce chatbot fournit des informations issues du dataset MedQuAD. "
    "Il ne remplace pas l'avis d'un professionnel de santé."
)

st.markdown(
    """
Posez une question médicale en anglais.
Le chatbot recherchera la réponse la plus proche dans la base MedQuAD.
"""
)

question = st.text_input(
    "Votre question",
    placeholder="Example: What are the symptoms of diabetes?"
)

# Recherche

if st.button("Rechercher"):

    if question.strip() == "":
        st.warning("Veuillez saisir une question.")
    else:

        embedding = model.encode([question])

        distances, indices = index.search(embedding, k=5)

        idx = indices[0][0]

        answer = df.iloc[idx]["answer"]

        similarity = 1 / (1 + distances[0][0])

        st.success("Réponse trouvée")

        st.subheader("Top 5 résultats")

        for i in range(5):

            idx = indices[0][i]

            similarity = 1 / (1 + distances[0][i])

            with st.expander(f"Résultat {i+1} - {similarity:.2%}"):

                st.markdown("### Question")

                st.write(df.iloc[idx]["question"])

                st.markdown("### Réponse")

                st.write(df.iloc[idx]["answer"])

        st.write(answer)

        st.subheader("Score de similarité")

        st.progress(float(similarity))

        st.write(f"{similarity:.2%}")

        with st.expander("Question trouvée dans MedQuAD"):
            st.write(df.iloc[idx]["question"])