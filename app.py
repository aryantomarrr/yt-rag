import os, json, hmac
import streamlit as st
from dotenv import load_dotenv
from rag_core import Config, RAGPipeline, load_youtube, load_upload, BASELINE, ADVANCED
from evaluate import run_eval, METRICS

load_dotenv()
st.set_page_config(page_title="Advanced RAG Chat", page_icon="🎥", layout="wide")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")

st.title("🎥 Advanced RAG: chat with YouTube videos & documents")
st.caption("Hybrid retrieval · cross-encoder rerank · grounded answers with citations")

# ---------- Admin login (sirf evaluation ke liye) ----------
st.session_state.setdefault("is_admin", False)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("1 · Source")
    src = st.radio("Type", ["YouTube", "Upload file"], horizontal=True)
    url = st.text_input("YouTube URL") if src == "YouTube" else None
    up = st.file_uploader("PDF / TXT", type=["pdf", "txt"]) if src != "YouTube" else None
    chunk_chars = st.slider("Chunk size (chars)", 400, 2000, 1000, 100)
    overlap = st.slider("Chunk overlap", 0, 400, 150, 50)
    if st.button("Build index", type="primary"):
        try:
            with st.spinner("Ingesting, chunking, embedding..."):
                docs = (load_youtube(url, chunk_chars, overlap) if src == "YouTube"
                        else load_upload(up, chunk_chars, overlap))
                st.session_state.pipe = RAGPipeline(docs)
                st.session_state.messages = []
            st.success(f"Indexed {len(docs)} chunks")
        except Exception as e:
            st.error(f"Failed: {e}")

    st.header("2 · Pipeline")
    mode = st.radio("Mode", ["Advanced", "Baseline"], horizontal=True)
    k = st.slider("Top-k chunks", 1, 8, 4)
    temp = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    if mode == "Advanced":
        cfg = Config(k=k, temperature=temp,
                     rewrite=st.checkbox("Query rewriting", True),
                     multi_query=st.checkbox("Multi-query", True),
                     hybrid=st.checkbox("Hybrid (BM25 + dense)", True),
                     mmr=st.checkbox("MMR", True),
                     rerank=st.checkbox("Cross-encoder rerank", True),
                     compress=st.checkbox("Contextual compression", True),
                     min_score=st.slider("Min relevance (reranker)", 0.0, 0.5, 0.0, 0.01))
    else:
        cfg = Config(**{**BASELINE.__dict__, "k": k, "temperature": temp})

    st.divider()
    if st.session_state.is_admin:
        st.success("Admin mode")
        if st.button("Logout"):
            st.session_state.is_admin = False
            st.rerun()
    else:
        with st.expander("Admin login"):
            pw = st.text_input("Password", type="password")
            if st.button("Login"):
                if ADMIN_PASSWORD and hmac.compare_digest(pw, ADMIN_PASSWORD):
                    st.session_state.is_admin = True
                    st.rerun()
                else:
                    st.error("Wrong password")

def show_res(res):
    with st.expander(f"Sources ({len(res['sources'])}) · {res['latency']:.1f}s"):
        for i, s in enumerate(res["sources"], 1):
            head = f"**[{i}]** [{s['loc']}]({s['link']})" if s.get("link") else f"**[{i}]** {s['loc']}"
            if s["score"] is not None:
                head += f" · relevance {s['score']:.2f}"
            st.markdown(head)
            st.caption(s["text"])
    with st.expander("Pipeline trace"):
        st.json(res["trace"])

def chat_ui():
    st.session_state.setdefault("messages", [])
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("res"):
                show_res(m["res"])
    if q := st.chat_input("Ask something about the source..."):
        pipe = st.session_state.get("pipe")
        if not pipe:
            st.warning("Pehle sidebar mein 'Build index' karo.")
            return
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                res = pipe.answer(q, cfg, history)
            st.markdown(res["answer"])
            show_res(res)
        st.session_state.messages += [{"role": "user", "content": q},
                                      {"role": "assistant", "content": res["answer"], "res": res}]

def eval_ui():
    default = [{"question": "What is tool calling?", "ground_truth": "Write the true answer here."}]
    raw = st.text_area("Eval set (JSON)", json.dumps(default, indent=2), height=250)
    try:
        st.caption(f"≈ {len(json.loads(raw)) * 6} LLM calls. Cache hone par rerun free.")
    except Exception:
        st.caption("JSON invalid")
    if st.button("Run evaluation"):
        if "pipe" not in st.session_state:
            st.warning("Pehle index build karo.")
        else:
            try:
                with st.spinner("Running baseline + advanced + judge..."):
                    st.session_state.eval_df = run_eval(
                        st.session_state.pipe, json.loads(raw),
                        {"baseline": BASELINE, "advanced": ADVANCED})
            except Exception as e:
                st.error(f"Eval failed: {e}")
    if "eval_df" in st.session_state:
        df = st.session_state.eval_df
        summary = df.groupby("mode")[METRICS + ["latency_s"]].mean().round(3)
        st.dataframe(summary)
        st.bar_chart(summary[METRICS].T)
        with st.expander("Per-question results"):
            st.dataframe(df)

# ---------- Layout ----------
if st.session_state.is_admin:
    tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 Evaluation (admin)"])
    with tab_chat:
        chat_ui()
    with tab_eval:
        eval_ui()
else:
    chat_ui()