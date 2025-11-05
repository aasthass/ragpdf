"""
app.py — Streamlit RAG agent 

Preconditions:
- .env contains AZURE_COGNITIVE_ENDPOINT, AZURE_API_KEY, OPENAI_API_KEY
- venv has required packages: streamlit, python-dotenv, azure-ai-formrecognizer,
  chromadb, openai (or new OpenAI SDK used below), tabulate
- Add PDFs to UI via file uploader; app will extract text/tables with Azure Form Recognizer,
  store chunks in a Chroma collection, then answer queries using retrieved context.

How to run:
$ streamlit run app.py
"""

"""
app.py — Streamlit RAG agent with 4 explicit nodes:
plan -> retrieve -> answer -> reflect
"""

import os
import re
import json
import glob
import streamlit as st
from dotenv import load_dotenv
from tabulate import tabulate
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from typing import Dict

# -------------------------
# Load environment
# -------------------------
load_dotenv()

AZURE_COGNITIVE_ENDPOINT = os.getenv("AZURE_COGNITIVE_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client_openai = OpenAI(api_key=OPENAI_API_KEY)
chroma_client = chromadb.Client()
COLLECTION_NAME = "pdf_rag_collection"

# -------------------------
# Agent Class
# -------------------------
class LangGraphAgent:
    def __init__(self, chroma_client, openai_client):
        self.chroma_client = chroma_client
        self.openai = openai_client

        # attach correct embedding model (1536-dim)
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name="text-embedding-ada-002"
        )
        try:
            self.collection = self.chroma_client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=openai_ef
            )
        except:
            self.collection = None

    # PLAN
    def plan(self, user_question: str) -> Dict:
        st.sidebar.markdown("### PLAN")
        q = user_question.strip()
        intent = "question" if q.endswith("?") else "statement"
        retrieve_needed = True
        st.sidebar.write(f"**Question:** {q}")
        st.sidebar.write(f"**Intent:** {intent}")
        st.sidebar.write(f"**Retrieve:** {retrieve_needed}")
        return {"question": q, "intent": intent, "retrieve_needed": retrieve_needed}

    # RETRIEVE
    def retrieve(self, plan_output: Dict, top_k=5) -> Dict:
        st.sidebar.markdown("### RETRIEVE")
        if self.collection is None:
            st.sidebar.warning("⚠️ No data loaded yet. Upload or load PDFs first.")
            return {"retrieved": [], "query": plan_output["question"]}

        q = plan_output["question"]
        result = self.collection.query(
            query_texts=[q], n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        docs = result["documents"][0]
        metas = result["metadatas"][0]
        scores = result["distances"][0]

        for i, (doc, meta, score) in enumerate(zip(docs, metas, scores)):
            st.sidebar.write(
                f"- **Doc {i+1}** | Page **{meta.get('page_number')}** | "
                f"Type `{meta.get('type')}` | Score `{score:.3f}`"
            )

        retrieved = [{"text": d, "meta": m, "score": s} for d, m, s in zip(docs, metas, scores)]
        return {"retrieved": retrieved, "query": q}

    # ANSWER
    def answer(self, retrieve_output: Dict, temperature=0.0):
        st.sidebar.markdown("### ANSWER")
        q = retrieve_output["query"]
        retrieved = retrieve_output["retrieved"]

        context = "\n\n---\n\n".join([
            f"[page {r['meta'].get('page_number', '?')}]\n{r['text'][:1500]}" for r in retrieved
        ]) if retrieved else "No context available."

        system_prompt = (
            "You are a helpful assistant. Use ONLY the provided context to answer. "
            "If not answerable, say you cannot find the answer."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer:"

        try:
            resp = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_prompt}],
                temperature=temperature
            )
            answer_text = resp.choices[0].message.content.strip()
        except:
            answer_text = "⚠️ Answer generation failed."

        st.sidebar.write("Generated answer displayed below in chat.")
        return {"question": q, "answer": answer_text}

    # REFLECT
    def reflect(self, answer_output: Dict):
        st.sidebar.markdown("### REFLECT")

        judge_prompt = (
            "Rate how well the answer addresses the question (1-10) and give a one-sentence justification.\n"
            f"Question: {answer_output['question']}\n"
            f"Answer: {answer_output['answer']}\n"
            "Return JSON: {\"rating\": X, \"justification\": \"...\"}"
        )

        try:
            resp = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.0
            )
            clean_json = re.sub(r"```(json)?\n?(.*?)```", r"\2", resp.choices[0].message.content, flags=re.DOTALL).strip()
            parsed = json.loads(clean_json)
            rating = parsed.get("rating", 5)
            justification = parsed.get("justification", "")
        except:
            rating = 5
            justification = "Judge evaluation unavailable."

        st.sidebar.write(f"**Rating:** {rating}/10")
        st.sidebar.write(f"**Reason:** {justification}")

        return {"rating": rating, "justification": justification}

    # INGEST PDF
    def ingest_pdf_with_form_recognizer(self, pdf_bytes):
        credential = AzureKeyCredential(AZURE_API_KEY)
        client = DocumentAnalysisClient(AZURE_COGNITIVE_ENDPOINT, credential)

        with open("temp.pdf", "wb") as f:
            f.write(pdf_bytes)

        with open("temp.pdf", "rb") as f:
            res = client.begin_analyze_document("prebuilt-document", f).result().to_dict()

        try:
            self.chroma_client.delete_collection(name=COLLECTION_NAME)
        except:
            pass

        ef = embedding_functions.OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name="text-embedding-ada-002")
        self.collection = self.chroma_client.create_collection(name=COLLECTION_NAME, embedding_function=ef)

        idx = 1
        for p in res.get("pages", []):
            text = " ".join([line["content"] for line in p.get("lines", [])])
            if text.strip():
                self.collection.add(documents=[text], metadatas=[{"page_number": p["page_number"], "type": "raw"}], ids=[str(idx)])
                idx += 1

        st.success(f"✅ Ingested {idx-1} text chunks.")
        os.remove("temp.pdf")

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="LangGraph-like RAG Chat", layout="wide")
st.title("📄 LangGraph-style RAG Chat (Streamlit)")

# left: main chat; right: logs
col_main, col_logs = st.columns([3,1])

with col_logs:
    st.header("Agent Flow Log")
    st.info("Plan → Retrieve → Answer → Reflect will be logged here.")
    # we will append logs to session state below

if "agent_logs" not in st.session_state:
    st.session_state.agent_logs = []

if "messages" not in st.session_state:
    st.session_state.messages = []
# --- LOAD LOCAL PDFs FROM ./data/*.pdf ---
with col_main:
    st.subheader("Load Local PDF Knowledge Base")
    if st.button("Load PDFs into Chroma"):
        import glob

        agent = LangGraphAgent(chroma_client, client_openai)

        try:
            chroma_client.delete_collection(name=COLLECTION_NAME)
        except:
            pass

        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name="text-embedding-ada-002"
        )
        agent.collection = chroma_client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=openai_ef
        )

        pdf_files = glob.glob("./data/*.pdf")
        if not pdf_files:
            st.warning("⚠️ No PDFs found inside ./data/")
        else:
            for pdf_path in pdf_files:
                st.write(f"Processing: {os.path.basename(pdf_path)}")
                with open(pdf_path, "rb") as f:
                    agent.ingest_pdf_with_form_recognizer(f.read())
            st.success(f"✅ Loaded {len(pdf_files)} PDFs into Chroma.")
            st.rerun()


# File uploader & ingest control
with col_main:
    st.subheader("Upload PDF (Azure Form Recognizer will extract text & tables)")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        if st.button("Ingest uploaded PDF into Chroma"):
            pdf_bytes = uploaded_file.read()
            # create agent if needed
            agent = LangGraphAgent(chroma_client, client_openai)
            agent.ingest_pdf_with_form_recognizer(pdf_bytes)
            st.session_state.agent_logs.append("PDF ingested into Chroma collection.")
            st.rerun()

    st.markdown("---")
    st.subheader("Chat with the ingested PDF")
    user_prompt = st.chat_input("💬 Ask a question about the uploaded PDF")
    if user_prompt:
        # instantiate agent
        agent = LangGraphAgent(chroma_client, client_openai)

        # PLAN
        st.session_state.agent_logs.append("→ PLAN: interpreting question")
        plan_out = agent.plan(user_prompt)

        # RETRIEVE
        st.session_state.agent_logs.append("→ RETRIEVE: querying vector DB")
        retrieve_out = agent.retrieve(plan_out, top_k=5)

        # ANSWER
        st.session_state.agent_logs.append("→ ANSWER: calling LLM with context")
        answer_out = agent.answer(retrieve_out, temperature=0.0)

        # REFLECT
        st.session_state.agent_logs.append("→ REFLECT: self-evaluation of the answer")
        reflect_out = agent.reflect(answer_out)

        # show results in the chat UI
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        # --- CLEANED UP VERSION ---
        rating = reflect_out.get("rating", 5)
        justification = reflect_out.get("justification", "")
        st.session_state.messages.append({
        "role": "assistant",
        "content": answer_out["answer"]
       })

        # display full chat messages
        for message in st.session_state.messages[-6:]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # append logs to sidebar area
        st.session_state.agent_logs.append(f"Answer rating: {reflect_out['rating']}/10")

# display logs in right column
with col_logs:
    for log in st.session_state.agent_logs[-20:]:
        st.write(log)
