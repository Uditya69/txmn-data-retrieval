"""Manual test/validation UI for the retrieval-system /ws/search endpoint.

Run: uv run streamlit run packages/test-ui/src/test_ui/app.py
"""
import asyncio
import json
import time

import streamlit as st
import websockets

st.set_page_config(page_title="Retrieval System Tester", layout="wide")
st.title("Retrieval System — Manual Test UI")

with st.sidebar:
    ws_url = st.text_input("WebSocket URL", value="ws://localhost:8000/ws/search")
    query = st.text_area("Query", value="", height=100)
    send = st.button("Send", type="primary", use_container_width=True)


async def run_query(url: str, q: str, events: list):
    async with websockets.connect(url, open_timeout=10) as ws:
        await ws.send(json.dumps({"query": q}))
        t0 = time.monotonic()
        try:
            while True:
                raw = await ws.recv()
                events.append((time.monotonic() - t0, json.loads(raw)))
        except websockets.exceptions.ConnectionClosedOK:
            pass


def render_instant(msg: dict):
    st.subheader("Instant result", divider=True)
    es = msg.get("es_result")
    es_err = msg.get("es_error")
    milvus = msg.get("milvus_result")
    milvus_err = msg.get("milvus_error")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Elasticsearch**")
        if es_err:
            st.error(es_err)
        else:
            st.json(es, expanded=False)
    with col2:
        st.markdown("**Milvus**")
        if milvus_err:
            st.error(milvus_err)
        else:
            st.json(milvus, expanded=False)


def render_ai_mode(msg: dict):
    st.subheader("AI Mode result", divider=True)
    if msg["type"] == "ai_mode_error":
        st.error(msg["error"])
        return
    st.markdown(msg["answer"])
    with st.expander(f"Citations ({len(msg.get('citations', []))})"):
        st.json(msg["citations"])


if send:
    if not query.strip():
        st.warning("Enter a query first.")
    else:
        events = []
        status = st.status("Querying...", expanded=True)
        try:
            asyncio.run(run_query(ws_url, query, events))
            status.update(label="Done", state="complete")
        except Exception as e:
            status.update(label="Failed", state="error")
            st.error(f"{type(e).__name__}: {e}")

        for elapsed, msg in events:
            st.caption(f"t+{elapsed:.2f}s — {msg['type']}")
            if msg["type"] == "instant_result":
                render_instant(msg)
            elif msg["type"] in ("ai_mode_done", "ai_mode_error"):
                render_ai_mode(msg)
            with st.expander("raw"):
                st.json(msg)
else:
    st.info("Enter a query in the sidebar and click Send.")
