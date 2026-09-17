"""RAG retrieval, exposed to the agent as a LangChain tool.
Retrieval is a *tool*, not a fixed pre-retrieval step.


Two behaviours the prompt depends on:
  * Results come back numbered, so the model can cite [1], [2] and the UI can
    resolve those back to clickable sources.
  * When nothing clears the relevance threshold the tool returns the literal
    sentinel NO_RELEVANT_KNOWLEDGE_FOUND, which the system prompt binds to
    "say the knowledge base doesn't cover this". 
"""


from __future__ import annotations


from dataclasses import dataclass
from functools import lru_cache


from langchain_core.tools import tool


from src.config import (
    COLLECTION_NAME,
    RELEVANCE_THRESHOLD,
    RETRIEVAL_FETCH_K,
    RETRIEVAL_K,
    RETRIEVAL_LAMBDA,
    VECTORSTORE_DIR,
)
from src.embeddings import EmbeddingConfigError, check_fingerprint, get_embeddings


NO_KNOWLEDGE_SENTINEL = "NO_RELEVANT_KNOWLEDGE_FOUND"


# Passed as a Chroma `where` clause. Chroma evaluates this inside the index,
# so asking for 5 indoor chunks returns 5 indoor chunks -- unlike post-hoc
# filtering, which quietly returns fewer when matching content is sparse.
# "mixed" is included with "indoor" because a mixed venue still works in rain.
_ACTIVITY_FILTERS = {
    "indoor": {"activity_type": {"$in": ["indoor", "mixed"]}},
    "outdoor": {"activity_type": {"$in": ["outdoor", "mixed"]}},
}




@dataclass
class Citation:
    """One retrieved chunk, as the UI's source panel renders it."""

    index: int
    title: str
    url: str
    section: str


    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "title": self.title,
            "url": self.url,
            "section": self.section,
        }




# Citations from the most recent retrieval, for the UI's source expander.
# Reset by the app at the start of each turn.
LAST_CITATIONS: list[Citation] = []


class KnowledgeBaseUnavailable(RuntimeError):
    """Raised when the vector store has not been built yet."""




@lru_cache(maxsize=1)
def get_vectorstore():
    """Open the persistent Chroma collection once per process."""
    from langchain_chroma import Chroma


    if not VECTORSTORE_DIR.exists():
        raise KnowledgeBaseUnavailable(
            f"No vector store at {VECTORSTORE_DIR}. "
            "Run `python -m src.ingest` first."
        )


    try:
        # Refuse to query a store written by a different embedding model,
        # rather than returning meaningless neighbours.
        check_fingerprint()
        embeddings = get_embeddings()
    except EmbeddingConfigError as exc:
        raise KnowledgeBaseUnavailable(str(exc)) from exc


    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )




def collection_stats() -> dict:
    """Chunk count and source breakdown, for the sidebar."""
    try:
        store = get_vectorstore()
        raw = store.get(include=["metadatas"])
        metadatas = raw.get("metadatas") or []
        sources: dict[str, int] = {}
        activities: dict[str, int] = {}
        for meta in metadatas:
            sources[meta.get("source_title", "?")] = (
                sources.get(meta.get("source_title", "?"), 0) + 1
            )
            activities[meta.get("activity_type", "na")] = (
                activities.get(meta.get("activity_type", "na"), 0) + 1
            )
        return {
            "available": True,
            "chunks": len(metadatas),
            "sources": sources,
            "activities": activities,
        }
    except Exception as exc:  # noqa: BLE001 - surfaced in the sidebar, not fatal
        return {"available": False, "error": str(exc)}




def format_results(docs_with_scores: list[tuple]) -> tuple[str, list[Citation]]:
    """Render retrieved chunks as a numbered block the model can cite from."""
    blocks: list[str] = []
    citations: list[Citation] = []


    for position, (doc, score) in enumerate(docs_with_scores, start=1):
        meta = doc.metadata
        citation = Citation(
            index=position,
            title=meta.get("source_title", "Unknown source"),
            url=meta.get("source_url", ""),
            section=meta.get("section", "Overview"),
        )
        citations.append(citation)
        blocks.append(
            f"[{position}] {citation.title} — {citation.section}\n"
            f"URL: {citation.url}\n"
            f"Activity type: {meta.get('activity_type', 'na')}\n"
            f"{doc.page_content}"
        )


    return "\n\n---\n\n".join(blocks), citations




@tool
def search_travel_knowledge(query: str, activity_type: str = "any") -> str:
    """Search the Singapore travel guides for destination knowledge.


    Use this for anything about the destination itself: attractions,
    neighbourhoods, getting around, food, culture, practical tips and
    itinerary ideas. Do NOT use it for weather or exchange rates.


    Args:
        query: What to look for, in natural language.
        activity_type: "indoor" to restrict to rain-friendly options,
            "outdoor" for open-air ones, or "any" (default). Use "indoor"
            when the forecast for a day is poor.


    Returns:
        Numbered excerpts with source titles and URLs — cite them as [1],
        [2] in your answer. Returns NO_RELEVANT_KNOWLEDGE_FOUND when the
        guides do not cover the question, in which case say so plainly
        rather than answering from your own knowledge.
    """
    global LAST_CITATIONS


    try:
        store = get_vectorstore()
    except KnowledgeBaseUnavailable as exc:
        LAST_CITATIONS = []
        return f"{NO_KNOWLEDGE_SENTINEL}\nKnowledge base unavailable: {exc}"


    where = _ACTIVITY_FILTERS.get(activity_type.strip().lower())


    try:
        # MMR so a three-day itinerary draws on several sections rather than
        # five near-duplicate chunks.
        docs = store.max_marginal_relevance_search(
            query,
            k=RETRIEVAL_K,
            fetch_k=RETRIEVAL_FETCH_K,
            lambda_mult=RETRIEVAL_LAMBDA,
            filter=where,
        )
        # MMR does not return scores, so score the same query separately to
        # apply the relevance threshold.
        scored = store.similarity_search_with_relevance_scores(
            query, k=RETRIEVAL_K, filter=where
        )
    except Exception as exc:  # noqa: BLE001
        LAST_CITATIONS = []
        return f"{NO_KNOWLEDGE_SENTINEL}\nRetrieval failed: {exc}"


    best = max((score for _, score in scored), default=0.0)
    if not docs or best < RELEVANCE_THRESHOLD:
        LAST_CITATIONS = []
        return (
            f"{NO_KNOWLEDGE_SENTINEL}\n"
            f"No passage in the travel guides was relevant enough to answer "
            f"'{query}'"
            + (f" with activity_type={activity_type}." if where else ".")
        )


    rendered, citations = format_results([(doc, None) for doc in docs])
    LAST_CITATIONS = citations
    return rendered