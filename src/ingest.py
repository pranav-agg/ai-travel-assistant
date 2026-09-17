"""Build the knowledge base: fetch -> chunk -> tag -> embed -> Chroma.

    python -m src.ingest              # full build
    python -m src.ingest --no-embed   # fetch + chunk only, no API cost
    python -m src.ingest --reset      # discard the existing collection first

"""


from __future__ import annotations


import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone


import httpx


from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    EMBEDDING_REQUEST_INTERVAL_SECONDS,
    HTTP_TIMEOUT_SECONDS,
    RAW_DIR,
    SOURCES_FILE,
    VECTORSTORE_DIR,
)
from src.embeddings import EmbeddingFingerprint, get_embeddings


WIKIVOYAGE_API = "https://en.wikivoyage.org/w/api.php"
USER_AGENT = (
    "AI-Travel-Planning-Assistant (educational assignment)"
)


# --- Indoor / outdoor tagging ------------------------------

INDOOR_TERMS = (
    "museum", "gallery", "aquarium", "mall", "shopping centre", "shopping center",
    "indoor", "cinema", "theatre", "theater", "spa", "hawker centre",
    "hawker center", "food court", "temple", "mosque", "church", "cathedral",
    "science centre", "science center", "exhibition", "arcade", "library",
    "conservatory", "cloud forest", "flower dome", "casino", "planetarium",
)
OUTDOOR_TERMS = (
    "garden", "park", "beach", "walk", "trail", "hike", "cycling", "boardwalk",
    "river cruise", "zoo", "safari", "reservoir", "island", "outdoor",
    "rooftop", "skyline", "waterfront", "nature reserve", "treetop", "boat",
)




def classify_activity(text: str) -> str:
    """Tag a chunk indoor / outdoor / mixed / n·a for weather-aware filtering."""
    lowered = text.lower()
    indoor = sum(term in lowered for term in INDOOR_TERMS)
    outdoor = sum(term in lowered for term in OUTDOOR_TERMS)
    if indoor == 0 and outdoor == 0:
        return "na"
    if indoor and outdoor:
        return "mixed"
    return "indoor" if indoor else "outdoor"




# --- Fetching -----------------------------------------------------------




@dataclass
class Source:
    id: str
    title: str
    url: str
    fetcher: str
    licence: str
    redistributable: bool
    enabled: bool
    page: str | None = None




def load_sources(enabled_only: bool = True) -> list[Source]:
    data = json.loads(SOURCES_FILE.read_text(encoding="utf-8"))
    sources = [
        Source(
            id=s["id"],
            title=s["title"],
            url=s["url"],
            fetcher=s["fetcher"],
            licence=s["licence"],
            redistributable=s["redistributable"],
            enabled=s["enabled"],
            page=s.get("page"),
        )
        for s in data["sources"]
    ]
    return [s for s in sources if s.enabled] if enabled_only else sources




def _wikitext_headings_to_markdown(text: str) -> str:
    "Convert MediaWiki `== Heading ==` markers to markdown `## Heading`."
    def repl(match: re.Match) -> str:
        level = len(match.group(1))
        return f"\n{'#' * min(level, 6)} {match.group(2).strip()}\n"


    return re.sub(r"^(={2,6})\s*(.+?)\s*\1\s*$", repl, text, flags=re.MULTILINE)




def fetch_wikivoyage(source: Source) -> str:
    """Fetch a Wikivoyage page as plain text with headings preserved."""
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": "1",
        "redirects": "1",
        "titles": source.page or source.title,
    }
    with httpx.Client(timeout=HTTP_TIMEOUT_SECONDS,
                      headers={"User-Agent": USER_AGENT}) as client:
        response = client.get(WIKIVOYAGE_API, params=params)
        response.raise_for_status()
        payload = response.json()


    pages = payload.get("query", {}).get("pages", {})
    for page in pages.values():
        extract = page.get("extract")
        if extract:
            return _wikitext_headings_to_markdown(extract)
    raise ValueError(f"No extract returned for '{source.page}'")




def fetch_html(source: Source) -> str:
    """Generic HTML fetcher for non-wiki sources."""
    from bs4 import BeautifulSoup


    with httpx.Client(timeout=HTTP_TIMEOUT_SECONDS, follow_redirects=True,
                      headers={"User-Agent": USER_AGENT}) as client:
        response = client.get(source.url)
        response.raise_for_status()
        html = response.text


    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()


    parts: list[str] = []
    for element in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = element.get_text(" ", strip=True)
        if not text:
            continue
        if element.name.startswith("h"):
            parts.append(f"\n{'#' * int(element.name[1])} {text}\n")
        else:
            parts.append(text)
    return "\n".join(parts)


FETCHERS = {"wikivoyage": fetch_wikivoyage, "html": fetch_html}


def fetch_source(source: Source) -> str:
    fetcher = FETCHERS.get(source.fetcher)
    if fetcher is None:
        raise ValueError(f"Unknown fetcher '{source.fetcher}' for {source.id}")
    return fetcher(source)




def write_raw(source: Source, body: str) -> None:
    """Persist fetched text with YAML front matter carrying its provenance."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    front_matter = (
        "---\n"
        f"title: {source.title}\n"
        f"url: {source.url}\n"
        f"licence: {source.licence}\n"
        f"redistributable: {str(source.redistributable).lower()}\n"
        f"retrieved_at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
        "---\n\n"
    )
    (RAW_DIR / f"{source.id}.md").write_text(front_matter + body, encoding="utf-8")




# --- Chunking -----------------------------------------------------------




def chunk_document(source: Source, body: str) -> list:
    """Header-aware split, then size-bounded split, then activity tagging."""
    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )


    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


    sections = header_splitter.split_text(body)
    chunks = size_splitter.split_documents(sections)


    enriched = []
    for index, chunk in enumerate(chunks):
        text = chunk.page_content.strip()
        if len(text) < 80:  # drop navigational scraps
            continue
        section = (
            chunk.metadata.get("h3")
            or chunk.metadata.get("h2")
            or chunk.metadata.get("h1")
            or "Overview"
        )
        chunk.page_content = text
        # Deterministic ID -> reruns upsert instead of duplicating.
        chunk.metadata = {
            "source_id": source.id,
            "source_title": source.title,
            "source_url": source.url,
            "licence": source.licence,
            "section": section,
            "activity_type": classify_activity(text),
            "chunk_index": index,
        }
        chunk.metadata["chunk_id"] = hashlib.sha1(
            f"{source.url}|{section}|{index}".encode("utf-8")
        ).hexdigest()
        enriched.append(chunk)
    return enriched




# --- Embedding ----------------------------------------------------------


def _batched(items: list, size: int):
    for start in range(0, len(items), size):
        yield start, items[start:start + size]


def _is_rate_limit_error(exc: Exception) -> bool:
    return "RateLimit" in exc.__class__.__name__




def build_vectorstore(chunks: list, reset: bool = False):
    from langchain_chroma import Chroma


    from src.embeddings import EmbeddingConfigError


    try:
        embeddings = get_embeddings()
    except EmbeddingConfigError as exc:
        raise SystemExit(
            f"{exc}\n\nOr run with --no-embed to verify fetching and chunking "
            "first, which costs nothing."
        ) from exc


    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )


    if reset:
        # Chroma keeps a collection's embedding dimension even after deleting
        # its rows. When switching providers/models, delete and recreate the
        # collection so the first OpenAI embedding can set the new dimension.
        store.delete_collection()
        store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(VECTORSTORE_DIR),
        )


    batch_size = max(1, EMBEDDING_BATCH_SIZE)
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    for batch_number, (start, batch) in enumerate(
        _batched(chunks, batch_size), start=1
    ):
        try:
            store.add_documents(batch, ids=[c.metadata["chunk_id"] for c in batch])
        except Exception as exc:
            if _is_rate_limit_error(exc):
                raise SystemExit(
                    "OpenAI rate-limited the ingest. This project batches "
                    "embedding requests, but your current account/key may "
                    "still need a slower setting.\n\n"
                    "Try this and rerun the ingest:\n\n"
                    "    EMBEDDING_BATCH_SIZE=16 "
                    "EMBEDDING_REQUEST_INTERVAL_SECONDS=5 "
                    "python -m src.ingest --reset\n\n"
                    "You can also check your OpenAI project limits and billing."
                ) from exc
            raise

        end = start + len(batch)
        print(f"  embedded batch {batch_number}/{total_batches} "
              f"({end}/{len(chunks)} chunks)")

        if batch_number < total_batches and EMBEDDING_REQUEST_INTERVAL_SECONDS > 0:
            time.sleep(EMBEDDING_REQUEST_INTERVAL_SECONDS)

    # Record which model wrote this store, so the app refuses to query it
    # with a different one later.
    EmbeddingFingerprint.write(EMBEDDING_MODEL)
    return store




# --- Entry point --------------------------------------------------------




def main() -> int:
    parser = argparse.ArgumentParser(description="Build the travel knowledge base.")
    parser.add_argument("--no-embed", action="store_true",
                        help="Fetch and chunk only; skip embedding (no API cost).")
    parser.add_argument("--reset", action="store_true",
                        help="Discard the existing collection before writing.")
    args = parser.parse_args()


    sources = load_sources()
    if not sources:
        print("No enabled sources in knowledge_base/sources.json.", file=sys.stderr)
        return 1


    all_chunks: list = []
    failures: list[tuple[str, str]] = []


    for source in sources:
        try:
            body = fetch_source(source)
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures.append((source.id, str(exc)))
            print(f"  !! {source.id}: {exc}", file=sys.stderr)
            continue


        write_raw(source, body)
        chunks = chunk_document(source, body)
        all_chunks.extend(chunks)


        tags: dict[str, int] = {}
        for chunk in chunks:
            tag = chunk.metadata["activity_type"]
            tags[tag] = tags.get(tag, 0) + 1
        tag_summary = ", ".join(f"{k}:{v}" for k, v in sorted(tags.items()))
        print(f"  ok {source.id:<34} {len(body):>7,} chars -> "
              f"{len(chunks):>3} chunks  ({tag_summary})")


    print(f"\n{len(sources) - len(failures)}/{len(sources)} sources fetched, "
          f"{len(all_chunks)} chunks total.")


    if failures:
        print(f"{len(failures)} source(s) failed:", file=sys.stderr)
        for source_id, reason in failures:
            print(f"  - {source_id}: {reason}", file=sys.stderr)


    if not all_chunks:
        print("Nothing to embed.", file=sys.stderr)
        return 1


    if args.no_embed:
        print("\n--no-embed set; stopping before embedding.")
        return 0


    print(f"\nEmbedding with {EMBEDDING_MODEL} into {VECTORSTORE_DIR} ...")
    build_vectorstore(all_chunks, reset=args.reset)
    print(f"Done. Collection '{COLLECTION_NAME}' holds {len(all_chunks)} chunks.")
    return 0




if __name__ == "__main__":
    raise SystemExit(main())
