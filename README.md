# ✈️ SmartTrip AI — Agentic AI-Based Smart Trip Planner

An intelligent multi-agent travel planner powered by **LangGraph**, **MCP**, and **RAG** that creates complete, budget-checked day-by-day itineraries with weather awareness, visa rules, and local event information.

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────┐
│                    User Query                        │
│  "Plan a 4-day trip to Singapore under Rs 80,000"    │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Planner Agent  │ ← Parses query, fetches RAG knowledge
              │  (GPT-4o-mini)  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Weather Agent   │ ← Open-Meteo API forecasts
              │  (GPT-4o-mini)  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Itinerary Agent │ ← RAG guides + Tavily events
              │  (GPT-4o-mini)  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Budget Agent   │ ← MCP flights/hotels + cost check
              │  (GPT-4o-mini)  │
              └────────┬────────┘
                       │
                ┌──────┴──────┐
                │ Budget OK?  │
                └──────┬──────┘
               Yes     │     No (max 2 retries)
                │      │      │
                │      │      └──→ Back to Planner Agent
                ▼      ▼
         ┌──────────────────┐
         │ Booking Summary  │ ← Final itinerary + cost breakdown
         │     Agent        │
         └──────────────────┘
```

## 🧩 Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Orchestration** | LangGraph | Planner-executor pattern with conditional re-planning |
| **MCP Server** | FastMCP | `search_flights`, `search_hotels`, `convert_currency` tools |
| **RAG Knowledge Base** | JSON-based | Visa policies, destination guides, travel insurance |
| **Web Search** | Tavily API | Local events, festivals, closures, travel advisories |
| **Weather** | Open-Meteo API | Weather forecasts (free, no API key needed) |
| **Currency** | Frankfurter API | Live exchange rates with fallback |
| **LLM** | OpenAI GPT-4o-mini | Agent reasoning via LangChain |
| **UI** | Streamlit | Structured trip form, results, quick tools, and chat |

## 📁 Project Structure

```
SmartTrip-Ai/
├── app.py                    # Streamlit UI (main entry point)
├── agents/
│   └── agent_nodes.py        # 5 agent definitions (Planner, Itinerary, Budget, Weather, Summary)
├── graph/
│   ├── state.py              # Shared TripState TypedDict
│   ├── workflow.py           # LangGraph workflow with conditional edges
│   └── structured_workflow.py # Form planner with budget check and MCP calls
├── tools/
│   └── tool_wrappers.py      # LangChain tool wrappers for all APIs
├── rag/
│   ├── knowledge_base.py     # RAG search over visa, guides, insurance
│   ├── retrieval.py           # Source-linked travel guidance for the form planner
│   └── travel_knowledge.json  # Curated destination and visa notes
├── mcp_server/
│   └── server.py             # FastMCP server with 3 tools
├── database/
│   ├── flights.json          # Mock flight data
│   ├── hotels.json           # Mock hotel data
│   ├── visa_policies.json    # Visa policies (10 countries)
│   ├── destination_guides.json # City guides (5 cities)
│   └── travel_insurance.json # Insurance plans
├── utils/
│   └── helpers.py            # Utility functions
├── test_client.py            # MCP server test client
├── requirements.txt
├── .env                      # API keys
└── README.md
```

## 🚀 Setup & Run

### 1. Create the project environment and install dependencies (PowerShell)

```bash
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Use `.venv`; an older `venv` directory may refer to a Python installation on another machine.

### 2. Set API Keys

Optional OpenAI and Tavily integrations are configured in the project `.env` file:

```
OPENAI_API_KEY=<your-openai-api-key>
TAVILY_API_KEY=<your-tavily-api-key>
```

### 3. Run the Streamlit App

```bash
python -m streamlit run app.py
```

The structured form works without API keys. OpenAI adds itinerary synthesis and Tavily enables event search. Keys stay server-side and are not shown in the page.

### 4. (Optional) Test MCP Server Standalone

```bash
python test_client.py
```

### 5. (Optional) Run Graph Directly

```bash
python graph/workflow.py
```

## 💬 Sample Queries

| Query | What it does |
|-------|-------------|
| "Plan a 4-day trip to Singapore in December under Rs 80,000" | Full itinerary with flights, hotels, activities, budget |
| "Do I need a visa to visit Thailand on an Indian passport?" | Visa lookup from RAG knowledge base |
| "What's the weather like in Munnar next week?" | Weather forecast from Open-Meteo |
| "Suggest cheaper hotel options for my itinerary" | Budget-optimized hotel search |
| "Plan a budget trip to Bangkok for 5 days under 50000 INR" | Budget-constrained trip planning |

## 📤 Expected Output

- ✅ Day-by-day itinerary with activities, food, and transport
- ✅ Cost breakdown in user's currency (INR)
- ✅ Visa notes with official sources
- ✅ Weather-adjusted activity suggestions
- ✅ Local events and travel advisories
- ✅ Flight and hotel recommendations
- ✅ Travel insurance suggestions

## ⚠️ Guardrails & Disclaimers

- All prices are **indicative** and may vary from actual booking prices
- Visa notes are guidance, not an eligibility decision — **always verify with the official embassy or immigration site**
- Flight and hotel data uses **mock data** for demonstration purposes
- Weather forecasts are from Open-Meteo and subject to change
- Forecasts beyond Open-Meteo's available date range and events without Tavily access are called out as unavailable
- The system asks for missing essentials (dates, budget) before planning

## 🔧 Data Sources

| Data | Source | Type |
|------|--------|------|
| Flights | `database/flights.json` | Mock |
| Hotels | `database/hotels.json` | Mock |
| Currency | Frankfurter API | Live |
| Weather | Open-Meteo API | Live |
| Events | Tavily Web Search | Live |
| Visa | `database/visa_policies.json` | Static RAG |
| Guides | `database/destination_guides.json` | Static RAG |
| Insurance | `database/travel_insurance.json` | Static RAG |
