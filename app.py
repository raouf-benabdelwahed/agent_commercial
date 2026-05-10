import os
import json
from datetime import datetime

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from rag import SimpleRAG, load_docs
from tools import compute_quote, create_lead
from llm import llm_generate

# Local : charge .env (en déploiement, Streamlit Cloud utilise Secrets)
load_dotenv()

st.set_page_config(page_title="Agent Commercial", layout="wide")

st.markdown("# Workshop Hackathon : Agent Commercial")
st.write(
    "Votre mission : construire un agent commercial qui accompagne un client du besoin à la proposition, "
    "en utilisant RAG pour le catalogue/FAQ, des outils pour les devis, et une validation humaine avant envoi."
)

LOG_PATH = "logs.jsonl"


def log_event(event: dict):
    event["ts"] = datetime.utcnow().isoformat()
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


@st.cache_resource
def init_rag():
    rag = SimpleRAG()
    rag.build(load_docs())
    return rag


rag = init_rag()

# -------------------------------
# Paramètres
# -------------------------------
with st.sidebar:
    st.header("Paramètres")
    top_k = st.slider("Top-k passages (RAG)", 1, 5, 3)
    threshold = st.slider("Seuil confiance (score)", 0.0, 1.0, 0.20, 0.01)
    review_amount = st.number_input("Seuil validation humaine (TND)", value=20000.0)

# -------------------------------
# Onglets (conformes au cahier des charges)
# -------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Entrée : Chat client",
        "RAG : Catalogue + FAQ",
        "Outils : Devis automatique",
        "Outil : Créer Lead",
        "Validation humaine + Logs",
    ]
)

# 1) Entrée : Chat client
with tab1:
    st.subheader("Entrée : Chat client")

    user_need = st.text_area(
        "Le client décrit son besoin en langage naturel",
        value=st.session_state.get(
            "draft_need",
            "Je cherche un CRM pour une PME de 10 commerciaux, avec formation."
        ),
        key="draft_need",
        height=140,
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("Valider le besoin"):
            st.session_state["validated_need"] = st.session_state["draft_need"]
            st.session_state["rag_question"] = st.session_state["validated_need"]  # <-- AJOUT
            st.success("Besoin validé.")

    
    # Affichage de l'état
    if st.session_state.get("validated_need"):
        st.caption("Besoin validé (utilisé dans les autres onglets) :")
        st.info(st.session_state["validated_need"])
    else:
        st.warning("Aucun besoin validé. Validez le besoin pour continuer proprement.")
# 2) RAG + LLM (réponse avec citations)
with tab2:
    st.subheader("RAG : Catalogue + FAQ (réponse générée par LLM + citations)")
    question = st.text_input(
    "Question (automatique depuis l'entrée validée)",
    value=st.session_state.get("rag_question", "Quels sont les délais de mise en place ?")
)
    if st.button("Répondre (LLM + RAG)"):
        sources = rag.retrieve(question, top_k=top_k)
        best = sources[0]["score"] if sources else 0.0

        if best < threshold:
            answer = (
                "Je ne sais pas à partir des documents disponibles. "
                "Pouvez-vous fournir plus d’informations ?"
            )
            st.warning(answer)
            log_event(
                {
                    "type": "rag",
                    "question": question,
                    "best_score": best,
                    "answer": answer,
                    "sources": sources,
                }
            )
        else:
            context = "\n\n".join([f"[{s['doc']}] {s['text']}" for s in sources])

            prompt = (
                "Tu es un assistant commercial. Réponds en français.\n"
                "Règles :\n"
                "- Utilise uniquement les SOURCES ci-dessous.\n"
                "- Si l'information n'est pas dans les sources, réponds : Je ne sais pas.\n"
                "- Donne une réponse courte (2 à 5 phrases).\n"
                "- Ajoute une section 'Citations' listant les documents utilisés (ex: [faq.md], [catalogue.md]).\n\n"
                f"QUESTION: {question}\n\n"
                f"SOURCES:\n{context}\n"
            )

            answer = llm_generate(prompt)
            st.success("Réponse")
            st.write(answer)

            df = pd.DataFrame(sources)
            df["text"] = df["text"].str.replace("\n", " ").str.slice(0, 220) + "..."
            st.caption("Passages retrouvés (retrieval)")
            st.dataframe(df, use_container_width=True)

            log_event(
                {
                    "type": "rag_llm",
                    "question": question,
                    "best_score": best,
                    "answer": answer,
                    "sources": sources,
                }
            )

# 3) Outil devis
with tab3:
    st.subheader("Outils : Devis automatique")

    products = ["CRM Starter", "CRM Pro", "Sales Intelligence"]
    options = {
        "CRM Starter": ["", "Support Standard", "Support Premium"],
        "CRM Pro": ["", "Support Premium", "Onboarding (formation 1 jour)"],
        "Sales Intelligence": ["", "Pack IA", "Pack IA + Formation"],
    }

    c1, c2, c3 = st.columns(3)
    with c1:
        product = st.selectbox("Produit", products)
    with c2:
        option = st.selectbox("Option", options[product])
    with c3:
        qty = st.number_input("Quantité", 1, 1000, 10)

    if st.button("Calculer le devis"):
        quote = compute_quote(product, int(qty), option if option else None)
        if not quote["ok"]:
            st.error(quote["error"])
        else:
            st.session_state["quote"] = quote
            st.success(f"Total: {quote['total']:,.2f}")
            st.json(quote)
            log_event({"type": "quote", "quote": quote})

# 4) Outil créer lead (préparation)
with tab4:
    st.subheader("Outil : Créer Lead")
    quote = st.session_state.get("quote")

    if not quote:
        st.info("Calculez d'abord un devis dans l'onglet 'Devis automatique'.")
    else:
        st.write("Devis actuel :")
        st.json(quote)

        name = st.text_input("Nom client", "Société ABC")
        email = st.text_input("Email client", "contact@abc.tn")
        need = st.text_area("Besoin (résumé)", user_need)

        st.session_state["lead_form"] = {"name": name, "email": email, "need": need}

# 5) Validation + logs
with tab5:
    st.subheader("Validation humaine")
    quote = st.session_state.get("quote")
    lead = st.session_state.get("lead_form")

    if not quote or not lead:
        st.info("Calculez un devis et remplissez le lead dans l'onglet 'Créer Lead'.")
    else:
        needs_review = float(quote["total"]) > float(review_amount)

        if needs_review:
            st.warning("Validation requise avant envoi (montant élevé).")
            approved = st.checkbox("Approuver avant envoi")
        else:
            st.success("Pas de validation requise (montant sous seuil).")
            approved = True

        if st.button("Finaliser (enregistrer le lead)"):
            if not approved:
                st.error("Action bloquée : validation requise.")
                log_event({"type": "lead_blocked", "quote": quote, "lead": lead})
            else:
                res = create_lead(lead["name"], lead["email"], lead["need"], float(quote["total"]))
                st.success(f"Lead enregistré ({res['timestamp']})")
                log_event({"type": "lead_created", "quote": quote, "lead": lead})

    st.divider()
    st.subheader("Logs / audit trail")

    if os.path.exists("leads.csv"):
        st.caption("leads.csv")
        st.dataframe(pd.read_csv("leads.csv"), use_container_width=True)
    else:
        st.info("Aucun lead pour l’instant.")

    if os.path.exists(LOG_PATH):
        rows = []
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        st.caption("logs.jsonl (dernières lignes)")
        st.dataframe(pd.DataFrame(rows[-50:]), use_container_width=True)
    else:
        st.info("Aucun log pour l’instant.")