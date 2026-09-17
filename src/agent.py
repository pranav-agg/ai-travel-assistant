"""The LangGraph ReAct agent


Tool set:
    search_travel_knowledge   local RAG over the travel guides
    get_weather_forecast      MCP -> Open-Meteo
    convert_currency          MCP -> Frankfurter
    list_supported_currencies MCP -> Frankfurter


Retrieval is a tool alongside the MCP tools rather than a fixed pre-retrieval
step, which is what makes "appropriate tool selection based on user intent"
observable in the trace.

"""


from __future__ import annotations


from dataclasses import dataclass, field
from typing import Any


from src.config import CHAT_MODEL, OPENAI_API_KEY
from src.prompts import build_system_prompt
from src.retriever import search_travel_knowledge


# Keep the last N messages of history. Enough for the multi-turn scenarios in
# the brief without letting a long session bloat every request.
MAX_HISTORY_MESSAGES = 20




@dataclass
class ToolCallRecord:
    """One tool invocation, for the UI's provenance panel."""


    name: str
    args: dict
    ok: bool
    summary: str = ""




@dataclass
class AgentResult:
    answer: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    error: str | None = None




def _prompt_with_fresh_date(state) -> list:
    """Render the system prompt on every invocation.
    
    `create_react_agent` freezes a
    string prompt at construction time, and the date anchor inside it would
    then be stuck at whenever the app started.
    """
    from langchain_core.messages import SystemMessage


    messages = state["messages"]
    return [SystemMessage(content=build_system_prompt())] + list(messages)




def build_agent(mcp_tools: list):
    """Construct the agent over RAG + MCP tools."""
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent


    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env."
        )


    model = ChatOpenAI(
        model=CHAT_MODEL,
        temperature=0.2,
        api_key=OPENAI_API_KEY,
        max_tokens=4096,
        timeout=90.0,
    )
    tools = [search_travel_knowledge, *mcp_tools]
    return create_react_agent(model, tools, prompt=_prompt_with_fresh_date)




def trim_history(messages: list) -> list:
    """Keep the conversation bounded without losing the thread.


    The system message is injected fresh per turn by `_prompt_with_fresh_date`,
    so only the dialogue itself is trimmed here.
    """
    if len(messages) <= MAX_HISTORY_MESSAGES:
        return messages
    return messages[-MAX_HISTORY_MESSAGES:]




def extract_text(message: Any) -> str:
    """Get the plain text out of a message, whatever shape the provider used.


    Providers can return either strings or structured content blocks. Rendering
    those blocks directly would put a Python repr on screen, so normalise it
    here rather than at each call site.
    """
    text = getattr(message, "text", None)
    if callable(text):
        try:
            return (text() or "").strip()
        except Exception:
            pass


    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(p for p in parts if p).strip()
    return str(content).strip()




def _summarise_tool_result(name: str, content: Any) -> tuple[bool, str]:
    """Turn a tool result into a one-line status for the provenance panel."""
    text = content if isinstance(content, str) else str(content)


    if "Knowledge base unavailable:" in text:
        return False, "knowledge base unavailable"
    if "Retrieval failed:" in text:
        return False, "retrieval failed"
    if "NO_RELEVANT_KNOWLEDGE_FOUND" in text:
        return False, "no relevant passage found"
    if '"error"' in text or text.strip().startswith("Error"):
        return False, "tool reported an error"


    if name == "get_weather_forecast":
        return True, "forecast retrieved"
    if name == "convert_currency":
        return True, "rate retrieved"
    if name == "search_travel_knowledge":
        hits = text.count("\nURL: ")
        return True, f"{hits} passage(s) retrieved"
    return True, "ok"


def _knowledge_failure_answer(tool_output: Any) -> str | None:
    """Return a deterministic answer for infrastructure failures in RAG."""
    text = tool_output if isinstance(tool_output, str) else str(tool_output)
    if "Knowledge base unavailable:" in text:
        reason = text.split("Knowledge base unavailable:", 1)[1].strip()
    elif "Retrieval failed:" in text:
        reason = text.split("Retrieval failed:", 1)[1].strip()
    else:
        return None

    return (
        "The local travel-guide index is built, but guide search failed while "
        "embedding your query with OpenAI.\n\n"
        f"Reason: `{reason}`\n\n"
        "This is not an attractions/opening-hours problem and the guides do "
        "not need to \"come back online\". Check that this machine can reach "
        "`https://api.openai.com`, that `OPENAI_API_KEY` is valid, and that "
        "the OpenAI account is not currently rate-limited. After fixing that, "
        "clear the conversation and ask again."
    )




async def ainvoke(agent, messages: list) -> AgentResult:
    """Run one turn and unpack the trace for the UI."""
    from langchain_core.messages import AIMessage, ToolMessage


    from src import retriever


    retriever.LAST_CITATIONS = []


    try:
        response = await agent.ainvoke({"messages": trim_history(messages)})
    except Exception as exc:  # noqa: BLE001 - surfaced in the chat, not raised
        return AgentResult(
            answer=(
                "Something went wrong while answering. The underlying error "
                f"was: {exc}"
            ),
            error=str(exc),
        )


    produced = response["messages"]


    # Pair each tool call with its result message.
    call_args: dict[str, dict] = {}
    order: list[tuple[str, str]] = []  # (call_id, tool_name)
    for message in produced:
        if isinstance(message, AIMessage):
            for call in message.tool_calls or []:
                call_args[call["id"]] = call.get("args", {})
                order.append((call["id"], call["name"]))


    results: dict[str, Any] = {}
    for message in produced:
        if isinstance(message, ToolMessage):
            results[message.tool_call_id] = message.content


    records: list[ToolCallRecord] = []
    knowledge_failure: str | None = None
    for call_id, name in order:
        output = results.get(call_id, "")
        ok, summary = _summarise_tool_result(name, output)
        records.append(
            ToolCallRecord(
                name=name, args=call_args.get(call_id, {}), ok=ok, summary=summary
            )
        )
        if name == "search_travel_knowledge" and not ok:
            knowledge_failure = knowledge_failure or _knowledge_failure_answer(output)


    if knowledge_failure:
        return AgentResult(
            answer=knowledge_failure,
            tool_calls=records,
            citations=[c.as_dict() for c in retriever.LAST_CITATIONS],
            error=knowledge_failure,
        )


    answer = ""
    for message in reversed(produced):
        if isinstance(message, AIMessage):
            candidate = extract_text(message)
            if candidate:
                answer = candidate
                break


    return AgentResult(
        answer=answer or "No answer was produced.",
        tool_calls=records,
        citations=[c.as_dict() for c in retriever.LAST_CITATIONS],
    )
