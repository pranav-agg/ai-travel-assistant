"""Streamlit UI for the AI Travel Planning Assistant


    streamlit run app.py
"""


from __future__ import annotations


import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage


from src.agent import ainvoke, build_agent
from src.config import (
    CHAT_MODEL,
    DESTINATION,
    EMBEDDING_MODEL,
)
from src.dates import build_date_anchor
from src.mcp_client import load_mcp_tools, probe_servers
from src.prompts import SAMPLE_QUESTIONS
from src.retriever import collection_stats
from src.runtime import run_async


st.set_page_config(
    page_title=f"{DESTINATION} Travel Assistant",
    page_icon="🧭",
    layout="centered",
)




# --- Resources ----------------------------------------------------------
# NOTE: the agent is cached, the *prompt* is not. The prompt is a callable
# re-rendered on every invocation so its date anchor cannot go stale




def get_agent():
    """Build the agent once per session, keeping MCP subprocesses alive."""
    if "agent" not in st.session_state:
        tools = run_async(load_mcp_tools())
        st.session_state.mcp_tool_names = [t.name for t in tools]
        st.session_state.agent = build_agent(tools)
    return st.session_state.agent




@st.cache_data(ttl=60)
def cached_kb_stats() -> dict:
    return collection_stats()




def probe() -> dict:
    if "mcp_status" not in st.session_state:
        st.session_state.mcp_status = run_async(probe_servers())
    return st.session_state.mcp_status




# --- Sidebar ------------------------------------------------------------


with st.sidebar:
    st.header("Status")

    stats = cached_kb_stats()
    if stats.get("available"):
        st.success(f"Knowledge base: {stats['chunks']} chunks")
        with st.expander("Sources indexed"):
            for title, count in sorted(stats["sources"].items()):
                st.caption(f"{title} — {count} chunks")
            st.caption("---")
            activities = stats.get("activities", {})
            st.caption(
                "Activity tags: "
                + ", ".join(f"{k} {v}" for k, v in sorted(activities.items()))
            )
    else:
        st.error("Knowledge base not built")
        st.caption(stats.get("error", ""))
        st.code("python -m src.ingest", language="bash")


    st.divider()
    st.subheader("MCP servers")
    try:
        for name, info in probe().items():
            if info["ok"]:
                st.success(f"{name} — {len(info['tools'])} tool(s)")
                st.caption(", ".join(info["tools"]))
            else:
                st.error(f"{name} — unavailable")
                st.caption(info["error"][:200])
    except Exception as exc:  # noqa: BLE001
        st.error("Could not probe MCP servers")
        st.caption(str(exc)[:200])


    st.divider()
    with st.expander("Date anchor sent to the model"):
        st.code(build_date_anchor(), language="text")
        st.caption(
            "Regenerated every turn. The model has no clock of its own."
        )


    st.divider()
    st.caption(f"Chat model: `{CHAT_MODEL}`")
    st.caption(f"Embeddings: `{EMBEDDING_MODEL}`")


    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.turns = []
        st.rerun()


    if st.button("Refresh app status", use_container_width=True):
        cached_kb_stats.clear()
        st.session_state.pop("mcp_status", None)
        st.rerun()




# --- State --------------------------------------------------------------


st.session_state.setdefault("messages", [])  # LangChain messages for the agent
st.session_state.setdefault("turns", [])     # display records for the UI
st.session_state.setdefault("pending", None)




# --- Header and sample questions ---------------------------------------


st.title("🧭 Singapore Travel Assistant")
st.caption(
    "Destination knowledge from travel guides (RAG) · live weather and "
    "exchange rates via MCP tools"
)


if not st.session_state.turns:
    st.write("**Try one of these:**")
    columns = st.columns(2)
    for index, (label, question) in enumerate(SAMPLE_QUESTIONS):
        with columns[index % 2]:
            if st.button(label, key=f"sample_{index}", use_container_width=True):
                st.session_state.pending = question
                st.rerun()
    st.caption(
        "The **Combined** examples are the brief's core scenario: guide "
        "content plus a live forecast, merged into one weather-aware plan."
    )




# --- Replay the conversation -------------------------------------------




def render_provenance(turn: dict) -> None:
    """Show which tools ran and which sources the answer cited."""
    tool_calls = turn.get("tool_calls") or []
    citations = turn.get("citations") or []
    if not tool_calls and not citations:
        return


    label = f"🔍 Sources & tools ({len(tool_calls)} call(s), " \
            f"{len(citations)} citation(s))"
    with st.expander(label):
        if tool_calls:
            st.markdown("**Tools invoked**")
            for call in tool_calls:
                icon = "✅" if call["ok"] else "⚠️"
                args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
                st.markdown(
                    f"{icon} `{call['name']}({args})` — {call['summary']}"
                )
        if citations:
            st.markdown("**Knowledge-base citations**")
            for citation in citations:
                title = f"[{citation['index']}] {citation['title']} — " \
                        f"{citation['section']}"
                if citation["url"]:
                    st.markdown(f"- [{title}]({citation['url']})")
                else:
                    st.markdown(f"- {title}")




for turn in st.session_state.turns:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant":
            render_provenance(turn)




# --- Handle input -------------------------------------------------------


typed = st.chat_input("Ask about Singapore, the weather, or your budget...")
question = typed or st.session_state.pending
st.session_state.pending = None


if question:
    st.session_state.turns.append({"role": "user", "content": question})
    st.session_state.messages.append(HumanMessage(content=question))


    with st.chat_message("user"):
        st.markdown(question)


    with st.chat_message("assistant"):
        with st.status("Thinking...", expanded=True) as status:
            try:
                agent = get_agent()
                status.write("Selecting tools...")
                result = run_async(ainvoke(agent, st.session_state.messages))


                for call in result.tool_calls:
                    icon = "✅" if call.ok else "⚠️"
                    status.write(f"{icon} `{call.name}` — {call.summary}")


                status.update(label="Done", state="complete", expanded=False)
            except Exception as exc:  # noqa: BLE001
                status.update(label="Failed", state="error")
                st.error(
                    "Could not complete that request. Check the sidebar for "
                    f"knowledge-base and MCP status.\n\n`{exc}`"
                )
                st.stop()


        st.markdown(result.answer)


        record = {
            "role": "assistant",
            "content": result.answer,
            "tool_calls": [
                {
                    "name": c.name,
                    "args": c.args,
                    "ok": c.ok,
                    "summary": c.summary,
                }
                for c in result.tool_calls
            ],
            "citations": result.citations,
        }
        render_provenance(record)


    st.session_state.turns.append(record)
    st.session_state.messages.append(AIMessage(content=result.answer))
