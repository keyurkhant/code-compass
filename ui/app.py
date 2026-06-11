import streamlit as st
import httpx

API_URL = "http://localhost:8000"

st.set_page_config(page_title="code-compass", page_icon="🧭", layout="wide")
st.title("🧭 code-compass")
st.caption("Ask questions about your codebase — answers grounded in real code with citations.")

# ---------------------------------------------------------------------------
# Sidebar configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")
    api_url = st.text_input("API URL", value=API_URL, help="Base URL of the code-compass API server.")
    st.divider()

    st.subheader("Filters")
    filter_lang = st.text_input(
        "Language filter",
        value="",
        placeholder="e.g. python",
        help="Only retrieve chunks written in this language.",
    )
    filter_repo = st.text_input(
        "Repo filter",
        value="",
        placeholder="e.g. requests",
        help="Restrict results to chunks from this repo name.",
    )
    st.divider()

    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages: list[dict] = []

# ---------------------------------------------------------------------------
# Render chat history
# ---------------------------------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("citations"):
            with st.expander(f"Citations ({len(message['citations'])})"):
                for citation in message["citations"]:
                    st.markdown(
                        f"**`{citation['path']}`** — lines {citation['start_line']}–{citation['end_line']}"
                    )

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask a question about the codebase..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Build filters payload
    filters: dict | None = None
    where_clauses = []
    if filter_lang.strip():
        where_clauses.append({"language": {"$eq": filter_lang.strip()}})
    if filter_repo.strip():
        where_clauses.append({"repo": {"$eq": filter_repo.strip()}})
    if len(where_clauses) == 1:
        filters = where_clauses[0]
    elif len(where_clauses) > 1:
        filters = {"$and": where_clauses}

    # Call the API
    with st.chat_message("assistant"):
        with st.spinner("Searching the codebase..."):
            try:
                response = httpx.post(
                    f"{api_url.rstrip('/')}/ask",
                    json={"question": prompt, "filters": filters},
                    timeout=60.0,
                )
                response.raise_for_status()
                data = response.json()

                answer_text = data.get("answer", "No answer returned.")
                citations = data.get("citations", [])
                retrieved_count = data.get("retrieved_chunk_count", 0)

                st.markdown(answer_text)

                if citations:
                    with st.expander(f"Citations ({len(citations)}) — retrieved {retrieved_count} chunks"):
                        for citation in citations:
                            path = citation.get("path", "unknown")
                            start = citation.get("start_line", 0)
                            end = citation.get("end_line", 0)
                            chunk_id = citation.get("chunk_id", "")
                            st.markdown(
                                f"**`{path}`** — lines {start}–{end}"
                                + (f" (chunk `{chunk_id}`)" if chunk_id else "")
                            )
                else:
                    st.caption("No citations were extracted from this answer.")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer_text,
                        "citations": citations,
                    }
                )

            except httpx.ConnectError:
                error_msg = (
                    f"Cannot connect to the API at **{api_url}**. "
                    "Make sure the server is running (`codecompass serve`)."
                )
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg, "citations": []}
                )

            except httpx.HTTPStatusError as exc:
                error_msg = f"API returned an error: `{exc.response.status_code}` — {exc.response.text}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg, "citations": []}
                )

            except Exception as exc:
                error_msg = f"Unexpected error: {exc}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg, "citations": []}
                )
