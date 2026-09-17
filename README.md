# AI Travel Planning Assistant — Singapore


A context-aware travel assistant that combines a document-based knowledge base
(RAG) with current information retrieved through MCP tools.


Ask it *"Create a three-day Singapore itinerary for next week and adjust it
according to the weather forecast"* and it retrieves attractions and itinerary
ideas from travel guides, calls an MCP weather tool for the forecast, and
produces a day-wise plan with indoor alternatives for the wet days — labelling
which parts came from the guides, which from a live service, and which are its
own suggestions.


---


## Quick start


```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

 # then add OPENAI_API_KEY in .env file


python -m src.ingest --no-embed   # verify fetching + chunking, no API cost
python -m src.ingest              # build the vector store


streamlit run app.py
```


### One OpenAI key

The app uses OpenAI for both chat generation and embeddings:

| Purpose | Provider | Model | Key |
|---|---|---|---|
| Generation | OpenAI | `gpt-4.1-mini` | `OPENAI_API_KEY` |
| Embeddings | OpenAI | `text-embedding-3-small` | `OPENAI_API_KEY` |

After adding any new source, you can rebuild the vector store with
`python -m src.ingest --reset`.

---


## Architecture


```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit UI (app.py)                     │
│   chat · provenance panel · MCP status · date-anchor view    │
└──────────────────────────┬──────────────────────────────────┘
                           │  st.session_state (conversation memory)
┌──────────────────────────▼──────────────────────────────────┐
│           LangGraph ReAct agent  (src/agent.py)              │
│   system prompt rendered PER TURN · tool selection by intent │
└───────┬───────────────────────────────────┬─────────────────┘
        │                                   │
  ┌─────▼──────────────┐          ┌─────────▼──────────────────┐
  │ search_travel_     │          │  MCP tools                 │
  │ knowledge (tool)   │          │  MultiServerMCPClient      │
  │ src/retriever.py   │          │  ├─ stdio → weather server │
  └─────┬──────────────┘          │  └─ stdio → currency server│
        │                          └────────┬──────────────────┘
  ┌─────▼──────────────┐                    │
  │  Chroma collection │           ┌────────▼───────────────────┐
  │  + metadata        │           │ Open-Meteo · Frankfurter   │
  │  (title, url, tag) │           └────────────────────────────┘
  └────────────────────┘
```


**Retrieval is a tool, not a fixed pre-retrieval step.** The model chooses
between searching the guides and calling an MCP tool, which is what makes
"select the appropriate tool based on the user request" visible in the trace.


| File | Responsibility |
|---|---|
| `app.py` | Streamlit chat UI, provenance panel, status sidebar |
| `src/agent.py` | ReAct agent, history trimming, trace unpacking |
| `src/prompts.py` | System prompt + date anchor injection |
| `src/dates.py` | Date anchoring and FX staleness — the calendar authority |
| `src/retriever.py` | Chroma retrieval exposed as a LangChain tool |
| `src/embeddings.py` | OpenAI embedding provider + the model-mismatch guard |
| `src/ingest.py` | Fetch → chunk → tag → embed pipeline |
| `src/mcp_client.py` | Spawns and connects to both MCP servers |
| `src/runtime.py` | One persistent event loop for the process |
| `mcp_servers/weather_server.py` | MCP server → Open-Meteo |
| `mcp_servers/currency_server.py` | MCP server → Frankfurter |

---


## Knowledge base


Six Wikivoyage pages, declared in `knowledge_base/sources.json`:


| Source | Covers |
|---|---|
| Wikivoyage — Singapore | Overview, transport, culture, food, itineraries |
| Wikivoyage — Riverside | Attractions, museums |
| Wikivoyage — Bugis & Kampong Glam | Neighbourhoods, culture, food |
| Wikivoyage — Chinatown | Neighbourhoods, culture, attractions |
| Wikivoyage — Orchard | Shopping, indoor options |
| Wikivoyage — Sentosa | Attractions, outdoor, family |


**Licensing.** Wikivoyage is CC BY-SA 4.0, so this content can be redistributed
with attribution. Two Visit Singapore pages are listed in `sources.json` but enabling
them did not return any relevant info. So they can be set **disabled** (`enabled: false`) 
because Singapore Tourism Board content is not freely redistributable. 

`python -m src.ingest` to rebuild it from source.


### RAG workflow


1. **Fetch** — Wikivoyage plain-text extracts via the MediaWiki API, with
   `== Heading ==` markers converted to markdown so headings survive.
2. **Chunk** — `MarkdownHeaderTextSplitter` first, then
   `RecursiveCharacterTextSplitter` (1000 chars, 150 overlap). Header-aware
   splitting keeps "See → Gardens by the Bay" intact instead of cutting
   mid-section; it is the single biggest retrieval-quality lever here.
3. **Tag** — every chunk gets `activity_type ∈ {indoor, outdoor, mixed, na}`
   from a keyword heuristic. This is what powers the rainy-day swap.
4. **Embed** — `text-embedding-3-small` into a persistent Chroma collection.
   Chunk IDs are `sha1(source_url|section|chunk_index)`, so re-running ingest
   upserts in place: fixing one source costs one source's embeddings.
5. **Retrieve** — MMR (`k=5`, `fetch_k=20`, `λ=0.6`) so a three-day itinerary
   draws on several sections rather than five near-duplicate chunks. The
   indoor/outdoor filter is a Chroma `where` clause evaluated **inside** the
   index, so asking for 5 indoor chunks returns 5 indoor chunks.
6. **Ground** — results come back numbered with source title and URL; the
   prompt requires `[n]` citations and the UI resolves them to clickable links.


**When the guides don't cover something**, the retriever returns the literal
sentinel `NO_RELEVANT_KNOWLEDGE_FOUND`, which the system prompt binds to "say
so plainly".

---


## MCP tools

Two stdio servers built with `FastMCP`, spawned as subprocesses using the same
interpreter that runs the app.


### `weather` — Open-Meteo (keyless)


`get_weather_forecast(city, start_date, num_days)` → per-day conditions with a
derived `outdoor_suitability` of `good` / `mixed` / `poor`, so the model need
not re-derive meteorology. Thunderstorms force `poor` regardless of stated
probability — Singapore's afternoon showers are short, but storms genuinely
stop outdoor plans.


### `currency` — Frankfurter / ECB (keyless)


`convert_currency(amount, from_currency, to_currency)` and
`list_supported_currencies()`. INR, USD and SGD are all verified supported, and
their presence is asserted at server startup.


Test both servers standalone before wiring them up:


```bash
python mcp_servers/weather_server.py     # should sit waiting on stdio
python mcp_servers/currency_server.py    # prints the currency-validation result
```


---


## Prompt strategy


The system prompt (`src/prompts.py`) has one job beyond being helpful: keep
three registers of information separate.


| Register | Marked as | Rule |
|---|---|---|
| Knowledge-base facts | 📚 with `[n]` citations | Must come from `search_travel_knowledge` |
| Live service data | 🌤️ / 💱 with tool name + date | Must come from an MCP tool |
| Model's own reasoning | 💡 explicitly flagged | Never presented as sourced fact |


The rest of the prompt exists to stop one register leaking into another:


- **Channel separation both ways.** Destination facts only from the guides;
  weather and rates only from MCP. And never an MCP call for a question the
  guides already answer.
- **Sentinel binding.** `NO_RELEVANT_KNOWLEDGE_FOUND` maps to a plain
  admission, not a fallback to model memory.
- **Dates are copied, not computed**.
- **Rates are never called live** unless they are.
- **Tool errors are disclosed**.
- **Preferences persist** — children, budget, diet, mobility carry forward
  without re-asking, which is what makes the multi-turn scenarios work.


---