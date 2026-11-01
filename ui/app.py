"""Streamlit chat UI for the Answer Trail API. Run the API first (see README)."""

import os

import requests
import streamlit as st

DEFAULT_API = os.environ.get("RAG_API_URL", "http://127.0.0.1:8000/api")
FALLBACK_MARKER = "[extractive fallback"  # server prefix when no LLM is configured

st.set_page_config(page_title="Doc Q&A", page_icon=":books:", layout="wide")


# ---------- API helpers ----------
def api(method: str, path: str, **kwargs):
    headers = {"Authorization": f"Token {st.session_state.token}"} if st.session_state.get("token") else {}
    return requests.request(method, f"{st.session_state.api_url}{path}", headers=headers, timeout=150, **kwargs)


def api_error(resp) -> str:
    try:
        data = resp.json()
    except ValueError:
        return f"HTTP {resp.status_code}"
    if isinstance(data, dict):
        return "; ".join(f"{k}: {v}" for k, v in data.items())
    return str(data)


def authenticate(mode: str, username: str, password: str) -> None:
    path = "/auth/register/" if mode == "Register" else "/auth/token/"
    try:
        resp = api("POST", path, json={"username": username, "password": password})
    except requests.RequestException as exc:
        st.error(f"Cannot reach the API at {st.session_state.api_url}: {exc.__class__.__name__}")
        return
    if resp.ok:
        st.session_state.token = resp.json()["token"]
        st.session_state.username = username
        st.session_state.messages = []
        st.rerun()
    else:
        st.error(api_error(resp))


# ---------- state ----------
st.session_state.setdefault("api_url", DEFAULT_API)
st.session_state.setdefault("token", None)
st.session_state.setdefault("messages", [])

# ---------- sign-in (main area) ----------
if not st.session_state.token:
    st.title("Doc Q&A")
    st.write("Upload your documents and ask questions. Answers cite the passages they came from.")
    _, centre, _ = st.columns([1, 2, 1])
    with centre:
        mode = st.radio("Action", ["Sign in", "Register"], horizontal=True, label_visibility="collapsed")
        with st.form("auth"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button(mode, use_container_width=True, type="primary"):
                authenticate(mode, username, password)
        with st.expander("API settings"):
            st.session_state.api_url = st.text_input("API URL", st.session_state.api_url)
    st.stop()

# ---------- sidebar ----------
with st.sidebar:
    st.title("Doc Q&A")
    st.caption(f"Signed in as **{st.session_state.username}**")
    if st.button("Sign out", use_container_width=True):
        st.session_state.token = None
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("Add documents")
    files = st.file_uploader("PDF, TXT or MD", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if files and st.button("Upload", type="primary", use_container_width=True):
        for f in files:
            with st.spinner(f"Indexing {f.name}..."):
                resp = api("POST", "/documents/", files={"file": (f.name, f.getvalue())})
            if resp.ok:
                st.success(f"{f.name}: {resp.json()['chunk_count']} chunks")
            else:
                st.error(f"{f.name}: {api_error(resp)}")

    st.divider()
    st.subheader("Your documents")
    docs = api("GET", "/documents/").json() if st.session_state.token else []
    doc_ids = {None: "All documents"}
    for d in docs:
        doc_ids[d["id"]] = d["title"]
        cols = st.columns([5, 1])
        cols[0].write(f"{d['title']}  \n:gray[{d['chunk_count']} chunks]")
        if cols[1].button("🗑", key=f"del{d['id']}", help="Delete"):
            api("DELETE", f"/documents/{d['id']}/")
            st.rerun()
    if not docs:
        st.caption("Nothing uploaded yet.")

    st.divider()
    scope = st.selectbox("Search in", list(doc_ids), format_func=doc_ids.get)
    top_k = st.slider("Passages to retrieve", 1, 10, 4)
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------- chat ----------
def escape_md(text: str) -> str:
    """Show retrieved document text literally (e.g. __init__ must not render as bold)."""
    for ch in "\\`*_[]#$~<>":
        text = text.replace(ch, "\\" + ch)
    return text


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**[{s['n']}] {s['document_title']}** · chunk {s['chunk_index']} · similarity {s['score']:.2f}")
            st.caption(escape_md(s["text"]))


st.header("Ask your documents")
if not docs:
    st.info("Upload a document in the sidebar, then ask a question.")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        render_sources(m.get("sources", []))

if question := st.chat_input("Ask a question about your documents", disabled=not docs):
    st.session_state.messages.append({"role": "user", "content": escape_md(question)})
    with st.chat_message("user"):
        st.markdown(escape_md(question))
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            body = {"question": question, "top_k": top_k}
            if scope is not None:
                body["document_id"] = scope
            try:
                resp = api("POST", "/ask/", json=body)
            except requests.RequestException as exc:
                resp = None
                msg = f"Could not get an answer ({exc.__class__.__name__}). Is the API running and responding?"
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
        if resp is None:
            pass
        elif resp.ok:
            data = resp.json()
            answer = data["answer"]
            if answer.startswith(FALLBACK_MARKER):
                st.caption("No language model configured on the server: showing the best matching passage instead of a written answer.")
                answer = answer.split("] ", 1)[1]
                shown = escape_md(answer)  # raw document text, not model-written markdown
            else:
                shown = answer
            st.markdown(shown)
            render_sources(data["sources"])
            st.session_state.messages.append({"role": "assistant", "content": shown, "sources": data["sources"]})
        else:
            msg = f"Request failed: {api_error(resp)}"
            st.error(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
