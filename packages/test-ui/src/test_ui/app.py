"""Chat interface for the retrieval-system /ws/search endpoint.

Run: uv run streamlit run packages/test-ui/src/test_ui/app.py
"""
import asyncio
import json

import streamlit as st
import websockets

st.set_page_config(page_title="Retrieval System Chat", layout="wide")
st.title("Retrieval System")

with st.sidebar:
    st.subheader("Connection")
    ws_url = st.text_input("WebSocket URL", value="ws://localhost:8010/ws/search")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []


async def run_query(url: str, q: str) -> dict:
    result = {"instant": None, "ai_mode": None}
    async with websockets.connect(url, open_timeout=10) as ws:
        await ws.send(json.dumps({"query": q}))
        try:
            while True:
                msg = json.loads(await ws.recv())
                if msg["type"] == "instant_result":
                    result["instant"] = msg
                elif msg["type"] in ("ai_mode_done", "ai_mode_error"):
                    result["ai_mode"] = msg
        except websockets.exceptions.ConnectionClosedOK:
            pass
    return result


def render_instant(instant: dict):
    if instant is None:
        return
    with st.expander("Instant results (ES + Milvus)"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Elasticsearch**")
            if instant.get("es_error"):
                st.error(instant["es_error"])
            else:
                st.json(instant.get("es_result"), expanded=False)
        with col2:
            st.markdown("**Milvus**")
            if instant.get("milvus_error"):
                st.error(instant["milvus_error"])
            else:
                st.json(instant.get("milvus_result"), expanded=False)


def render_assistant_turn(turn: dict):
    render_instant(turn.get("instant"))

    ai_mode = turn.get("ai_mode")
    if ai_mode is None:
        st.warning("No AI Mode response received.")
        return

    if ai_mode["type"] == "ai_mode_error":
        st.error(ai_mode["error"])
        return

    st.markdown(ai_mode["answer"])
    citations = ai_mode.get("citations", [])
    if citations:
        with st.expander(f"Citations ({len(citations)})"):
            st.json(citations)


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            render_assistant_turn(message["content"])

query = st.chat_input("Ask a legal/tax question...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            try:
                turn = asyncio.run(run_query(ws_url, query))
            except Exception as e:
                turn = {"instant": None, "ai_mode": {"type": "ai_mode_error", "error": f"{type(e).__name__}: {e}"}}
        render_assistant_turn(turn)

    st.session_state.messages.append({"role": "assistant", "content": turn})
