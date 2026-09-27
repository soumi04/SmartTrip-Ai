"""SmartTrip AI — Intelligent Multi-Agent Travel Planner."""

import asyncio
import sys
from datetime import date
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
load_dotenv()

st.set_page_config(
    page_title="SmartTrip AI | Travel Desk & Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

    /* Global Dark Theme High-Contrast Enforcement */
    :root {
        --bg-main: #090d16;
        --card-bg: #111827;
        --card-border: #1e293b;
        --text-pure: #ffffff;
        --text-body: #e2e8f0;
        --text-muted: #94a3b8;
        --accent-cyan: #38bdf8;
        --accent-indigo: #818cf8;
        --accent-emerald: #10b981;
    }

    html, body, [class*="css"], [data-testid="stMarkdownContainer"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: #e2e8f0 !important;
    }

    /* App Background */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #131c31 0%, #090d16 60%, #04060a 100%) !important;
        color: #e2e8f0 !important;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1160px;
        padding-top: 1.5rem;
        padding-bottom: 3.5rem;
    }

    /* Hero Header */
    .hero-badge {
        display: inline-block;
        background: rgba(56, 189, 248, 0.12);
        color: #38bdf8 !important;
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 2.85rem !important;
        line-height: 1.15 !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        margin: 0 0 0.5rem 0 !important;
        letter-spacing: -0.02em !important;
    }

    .hero-desc {
        color: #94a3b8 !important;
        font-size: 1.05rem !important;
        margin-bottom: 1.8rem !important;
        line-height: 1.5 !important;
    }

    /* Section Headings */
    .section-heading {
        border-top: 1px solid #1e293b;
        padding-top: 1.3rem;
        margin-top: 1.8rem;
        margin-bottom: 1rem;
    }

    .section-heading h2 {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        margin: 0 !important;
    }

    h1, h2, h3, h4 {
        color: #ffffff !important;
    }

    p, span, label, div {
        color: #e2e8f0;
    }

    /* Widget Labels */
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
    }

    /* High-Contrast Tabs */
    button[data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 0.6rem 1.25rem !important;
        margin-right: 0.4rem !important;
        border: none !important;
    }

    button[data-baseweb="tab"] div,
    button[data-baseweb="tab"] p,
    button[data-baseweb="tab"] span {
        color: #94a3b8 !important;
        font-size: 0.96rem !important;
        font-weight: 600 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background: rgba(56, 189, 248, 0.12) !important;
        border-bottom: 3px solid #38bdf8 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] div,
    button[data-baseweb="tab"][aria-selected="true"] p,
    button[data-baseweb="tab"][aria-selected="true"] span {
        color: #38bdf8 !important;
        font-weight: 700 !important;
    }

    /* Input Fields (Text, Number, Date, Select) */
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stDateInput"] input,
    input[type="text"],
    input[type="number"] {
        background-color: #111827 !important;
        color: #ffffff !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        font-size: 0.95rem !important;
        padding: 0.5rem 0.75rem !important;
    }

    [data-testid="stTextInput"] input:focus,
    [data-testid="stNumberInput"] input:focus,
    [data-testid="stDateInput"] input:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 1px #38bdf8 !important;
    }

    /* Placeholders */
    ::placeholder {
        color: #64748b !important;
        opacity: 1 !important;
    }

    /* Selectboxes and Multiselect */
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stMultiSelect"] [data-baseweb="select"] > div {
        background-color: #111827 !important;
        color: #ffffff !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }

    [data-testid="stSelectbox"] svg,
    [data-testid="stMultiSelect"] svg {
        fill: #94a3b8 !important;
    }

    /* Form Container */
    [data-testid="stForm"] {
        background: rgba(17, 24, 39, 0.75) !important;
        border: 1px solid #1e293b !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5) !important;
    }

    /* Submit Button */
    [data-testid="stFormSubmitButton"] button,
    button[kind="primary"] {
        background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
        border: none !important;
        border-radius: 8px !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.02rem !important;
        min-height: 2.9rem !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4) !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stFormSubmitButton"] button:hover,
    button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.6) !important;
    }

    /* Metric Cards */
    [data-testid="stMetric"] {
        background: #111827 !important;
        border: 1px solid #1e293b !important;
        border-left: 4px solid #38bdf8 !important;
        padding: 0.9rem 1.1rem !important;
        border-radius: 8px !important;
    }

    [data-testid="stMetricLabel"] p {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricValue"] div {
        color: #38bdf8 !important;
        font-weight: 700 !important;
        font-size: 1.35rem !important;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background: rgba(17, 24, 39, 0.75) !important;
        border: 1px solid #1e293b !important;
        border-radius: 8px !important;
        margin-bottom: 0.6rem !important;
    }

    [data-testid="stExpander"] details summary span,
    [data-testid="stExpander"] details summary p {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
    }

    /* Chat Messages */
    [data-testid="stChatMessage"] {
        background: rgba(17, 24, 39, 0.75) !important;
        border: 1px solid #1e293b !important;
        border-radius: 10px !important;
        margin-bottom: 0.8rem !important;
    }

    /* Sidebar info box */
    .mode-card {
        background: rgba(17, 24, 39, 0.85) !important;
        border: 1px solid #1e293b !important;
        border-left: 3px solid #10b981 !important;
        border-radius: 8px !important;
        padding: 0.85rem 1rem !important;
        margin: 0.75rem 0 !important;
        color: #cbd5e1 !important;
        font-size: 0.88rem !important;
    }

    a {
        color: #38bdf8 !important;
        text-decoration: underline !important;
    }

    hr {
        border-color: #1e293b !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _value_text(value: Any) -> str:
    if value is None or value == "":
        return "Not provided"
    if isinstance(value, (dict, list)):
        return str(value)
    return str(value)


def _money(value: Any, currency: str) -> str:
    try:
        return f"{currency} {float(value):,.0f}"
    except (TypeError, ValueError):
        return "Not available"


def _render_record(record: Any) -> None:
    if isinstance(record, str):
        st.write(record)
        return
    if not isinstance(record, dict):
        st.write(_value_text(record))
        return

    for key, value in record.items():
        if key.lower() in {"url", "source", "official_source"} and isinstance(value, str):
            if value.startswith(("https://", "http://")):
                st.markdown(f"[{key.replace('_', ' ').title()}]({value})")
            else:
                st.write(f"{key.replace('_', ' ').title()}: {_value_text(value)}")
        elif value is not None and value != "":
            st.write(f"**{key.replace('_', ' ').title()}:** {_value_text(value)}")


def _render_items(items: Any, empty_message: str) -> bool:
    if not isinstance(items, list) or not items:
        st.caption(empty_message)
        return False
    for item in items:
        _render_record(item)
        st.divider()
    return True


def _render_options(title: str, selected: Any, options: Any) -> bool:
    st.markdown(f"#### {title}")
    has_data = selected is not None or bool(options)
    if selected is not None:
        _render_record(selected)
    else:
        st.caption("No selected option was returned.")

    if isinstance(options, list) and options:
        for index, option in enumerate(options, start=1):
            with st.expander(f"Option {index}: {option.get('airline', option.get('name', 'Alternative'))}"):
                _render_record(option)
    elif not selected:
        st.caption("No alternatives were returned.")
    return has_data


def _render_results(result: dict[str, Any]) -> None:
    costs = result.get("costs")
    costs = costs if isinstance(costs, dict) else {}
    currency = str(costs.get("currency") or st.session_state.get("trip_currency", "INR"))
    total = costs.get("total")
    budget = costs.get("budget")
    within_budget = costs.get("within_budget")
    if not isinstance(within_budget, bool):
        try:
            within_budget = float(total) <= float(budget)
        except (TypeError, ValueError):
            within_budget = None

    st.markdown('<div class="section-heading"><h2>Budget at a Glance</h2></div>', unsafe_allow_html=True)
    if within_budget is True:
        st.success("✅ The estimated cost is comfortably within your allocated budget.")
    elif within_budget is False:
        st.warning("⚠️ The estimated cost exceeds the stated budget. Cheaper alternatives are listed below.")
    else:
        st.info("Budget status calculated based on average local tariffs.")

    metric_columns = st.columns(4)
    for column, label, key in zip(
        metric_columns,
        ("Flights (Round-Trip)", "Hotels", "Local Expenses", "Estimated Total"),
        ("flights", "hotel", "local_estimate", "total"),
    ):
        column.metric(label, _money(costs.get(key), currency))
    st.caption("All prices are indicative based on real-time averages. Final tariffs may vary.")
    if costs.get("budget") is not None:
        st.write(f"**Target Budget:** {_money(costs.get('budget'), currency)}")

    itinerary = result.get("itinerary")
    st.markdown('<div class="section-heading"><h2>Day-by-Day Itinerary</h2></div>', unsafe_allow_html=True)
    if isinstance(itinerary, list) and itinerary:
        for index, day in enumerate(itinerary, start=1):
            if not isinstance(day, dict):
                with st.expander(f"Day {index}"):
                    _render_record(day)
                continue
            day_date = day.get("date") or f"Day {index}"
            theme = day.get("theme")
            title = f"Day {index} · {day_date} — {theme}" if theme else f"Day {index} · {day_date}"
            with st.expander(title, expanded=index == 1):
                for period in ("morning", "afternoon", "evening"):
                    if day.get(period):
                        st.write(f"**{period.title()}**  \n{_value_text(day[period])}")
                if day.get("weather_note"):
                    st.caption(f"🌤️ **Weather Alert:** {_value_text(day['weather_note'])}")
    else:
        st.info("The planner did not return an itinerary.")

    st.markdown('<div class="section-heading"><h2>Flights & Accommodations</h2></div>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        _render_options(
            "✈️ Selected Flight", result.get("selected_flight"), result.get("flight_options")
        )
    with right:
        _render_options(
            "🏨 Selected Hotel", result.get("selected_hotel"), result.get("hotel_options")
        )

    st.markdown('<div class="section-heading"><h2>Visa Guidance (RAG)</h2></div>', unsafe_allow_html=True)
    visa = result.get("visa_notes")
    if isinstance(visa, dict):
        summary = visa.get("summary")
        if summary:
            st.write(summary)
        else:
            st.caption("No visa summary was returned. Check current official guidance before travel.")
        sources = visa.get("sources")
        if isinstance(sources, list) and sources:
            st.markdown("**Official Consular Sources**")
            for source in sources:
                _render_record(source)
    elif visa:
        _render_record(visa)

    events = result.get("events")
    events = events.get("items", []) if isinstance(events, dict) else events
    weather = result.get("weather")
    weather = weather.get("items", []) if isinstance(weather, dict) else weather
    st.markdown('<div class="section-heading"><h2>On the Ground: Sights & Forecast</h2></div>', unsafe_allow_html=True)
    events_column, weather_column = st.columns(2, gap="large")
    with events_column:
        st.markdown("#### 🎟️ Top Sights")
        _render_items(events, "Top sights available in knowledge base.")
    with weather_column:
        st.markdown("#### 🌤️ Live Forecast (Open-Meteo)")
        _render_items(weather, "Weather forecasts available via quick tools.")

    if result.get("final_itinerary"):
        with st.expander("📄 View Full Markdown Itinerary Report"):
            st.markdown(result["final_itinerary"])


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## ⚙️ Settings & Engine")
    st.caption("Optional OpenAI and Tavily integrations use server-side .env settings. API keys are not shown in the page.")

    st.markdown("""
    <div class="mode-card">
        <b>🛡️ Resilient Multi-Agent Engine:</b><br>
        • Live Weather (Open-Meteo)<br>
        • Live Currency Rates (Frankfurter)<br>
        • Curated RAG Visa & City Guides<br>
        • Mock Flight & Hotel MCP Tools<br>
        <i>Runs seamlessly with or without OpenAI quota!</i>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 📋 Quick Sample Queries")
    samples = [
        "Plan a 4-day trip to Singapore in December under Rs 80,000 from Chennai",
        "Do I need a visa to visit Thailand on an Indian passport?",
        "What's the weather like in Munnar next week?",
        "Suggest a 3-day Dubai itinerary under 1 lakh from Delhi",
        "Plan a budget trip to Bangkok for 5 days under 50000 INR",
    ]
    for s in samples:
        if st.button(s, key=f"s_{hash(s)}", use_container_width=True):
            st.session_state["prefill_chat"] = s

    st.markdown("---")
    st.markdown("""
    **Architecture (LangGraph):**
    ```
    Planner → Weather → Itinerary → Budget
      ↑                                 ↓
      └── (Re-plan if over budget) ←────┘
                                        ↓
                                 Booking Summary
    ```
    """)


# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="hero-badge">SmartTrip AI · Intelligent Travel Planner</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">Plan the journey.<br>Keep every detail in view.</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-desc">Agentic multi-agent travel planner powered by LangGraph, MCP databases, and RAG knowledge bases.</p>',
    unsafe_allow_html=True,
)


# ============================================================
# TABS
# ============================================================

tab_form, tab_chat, tab_tools, tab_kb = st.tabs([
    "📝 Trip Planner Desk",
    "💬 AI Travel Assistant (Chat)",
    "🔍 Quick Lookup Tools",
    "📚 Knowledge Base Explorer",
])


# ============================================================
# TAB 1: FORM PLANNER
# ============================================================

with tab_form:
    with st.form("trip_planner", clear_on_submit=False):
        route_left, route_right = st.columns(2, gap="large")
        with route_left:
            origin = st.text_input("Departure City (Origin)", value="Chennai", placeholder="e.g. Chennai, Delhi, Mumbai")
        with route_right:
            destination = st.text_input("Destination City / Country", value="Singapore", placeholder="e.g. Singapore, Bangkok, Dubai")

        date_left, date_right = st.columns(2, gap="large")
        with date_left:
            start_date = st.date_input("Departure Date", value=date(2026, 12, 10))
        with date_right:
            end_date = st.date_input("Return Date", value=date(2026, 12, 13))

        budget_left, currency_right = st.columns([2, 1], gap="large")
        with budget_left:
            budget = st.number_input("Total Budget", min_value=1000.0, value=80000.0, step=5000.0)
        with currency_right:
            currency = st.selectbox("Preferred Currency", ["INR", "USD", "EUR", "SGD", "THB", "AED"])

        details_left, details_right = st.columns([2, 1], gap="large")
        with details_left:
            nationality = st.text_input("Passport Nationality", value="India", placeholder="For visa rules")
        with details_right:
            travelers = st.number_input("Number of Travellers", min_value=1, value=1, step=1)

        interests = st.multiselect(
            "Travel Interests",
            ["Culture", "Food & Dining", "Nature & Wildlife", "History", "Architecture", "Shopping", "Adventure", "Wellness", "Family-friendly"],
            default=["Culture", "Food & Dining"],
        )
        submitted = st.form_submit_button("Plan My Trip", use_container_width=True)

    if submitted:
        st.session_state.pop("trip_result", None)
        errors = []
        if not origin.strip():
            errors.append("Please specify your origin city.")
        if not destination.strip():
            errors.append("Please specify your destination.")
        if end_date <= start_date:
            errors.append("Return date must be after departure date.")
        if budget <= 0:
            errors.append("Budget must be greater than zero.")

        if errors:
            for error in errors:
                st.error(error)
        else:
            trip = {
                "origin": origin.strip(),
                "destination": destination.strip(),
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "budget": float(budget),
                "currency": currency,
                "nationality": nationality.strip(),
                "travelers": int(travelers),
                "interests": interests,
            }
            st.session_state["trip_currency"] = currency
            with st.spinner("🤖 Coordinating Planner, Weather, Itinerary, Budget, and Booking Agents..."):
                try:
                    from graph.structured_workflow import plan_trip
                    result = asyncio.run(plan_trip(trip))
                    if not isinstance(result, dict):
                        raise TypeError("Planner returned an unexpected format.")
                    st.session_state["trip_result"] = result
                except Exception as e:
                    import traceback
                    st.error(f"Trip planning encountered an issue: {str(e)}")
                    with st.expander("Show details"):
                        st.code(traceback.format_exc())

    result = st.session_state.get("trip_result")
    if isinstance(result, dict):
        _render_results(result)


# ============================================================
# TAB 2: AI CHAT ASSISTANT
# ============================================================

with tab_chat:
    st.markdown("### 💬 Ask Anything About Your Trip")
    st.caption("Ask natural language queries like visa rules, weather forecasts, budget trip plans, or hotel recommendations.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prefill = st.session_state.pop("prefill_chat", "")
    user_input = st.chat_input("Ask a travel question or plan a trip...")

    query_to_run = user_input or prefill

    if query_to_run:
        st.session_state.chat_history.append({"role": "user", "content": query_to_run})
        with st.chat_message("user"):
            st.markdown(query_to_run)

        with st.chat_message("assistant"):
            with st.spinner("Consulting multi-agent travel graph..."):
                try:
                    from graph.workflow import run_trip_planner
                    planner_res = run_trip_planner(query_to_run)
                    response_md = planner_res.get("final_itinerary", "Could not produce an itinerary.")
                    st.markdown(response_md)
                    st.session_state.chat_history.append({"role": "assistant", "content": response_md})
                except Exception as e:
                    err_msg = f"❌ An error occurred: {str(e)}"
                    st.error(err_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": err_msg})


# ============================================================
# TAB 3: QUICK TOOLS
# ============================================================

with tab_tools:
    st.markdown("### 🔍 Standalone Travel Tools")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### 🌤️ Live Weather Forecast (Open-Meteo)")
        w_city = st.text_input("City", value="Singapore", key="w_city_tab")
        w_days = st.slider("Days", 1, 14, 5, key="w_days_tab")
        if st.button("Get Weather", key="btn_w"):
            from tools.tool_wrappers import get_weather_forecast
            w_res = get_weather_forecast.invoke({"city": w_city, "num_days": w_days})
            if "error" in w_res:
                st.error(w_res["error"])
            else:
                st.success(f"Weather Forecast for {w_res.get('city', w_city)}")
                for f in w_res.get("forecast", []):
                    st.write(f"📅 **{f['date']}**: {f['condition']} ({f['min_temp_c']}°C – {f['max_temp_c']}°C, Rain: {f['precipitation_mm']}mm)")

        st.markdown("---")
        st.markdown("#### 💱 Live Currency Converter (Frankfurter)")
        c_amt = st.number_input("Amount", value=80000.0, step=1000.0, key="c_amt_tab")
        c_from = st.selectbox("From Currency", ["INR", "USD", "EUR", "SGD", "THB", "AED", "JPY"], key="c_from_tab")
        c_to = st.selectbox("To Currency", ["SGD", "THB", "AED", "USD", "EUR", "JPY", "INR"], key="c_to_tab")
        if st.button("Convert Currency", key="btn_c"):
            from tools.tool_wrappers import convert_currency
            c_res = convert_currency.invoke({"amount": c_amt, "from_currency": c_from, "to_currency": c_to})
            if "error" in c_res:
                st.error(c_res["error"])
            else:
                st.success(f"💱 {c_from} {c_amt:,.2f} = **{c_to} {c_res['converted_amount']:,.2f}**")
                st.caption(f"Rate: {c_res['exchange_rate']} | Source: {c_res['rate_source']}")

    with col2:
        st.markdown("#### 🛂 Consular Visa Policy (RAG)")
        v_country = st.text_input("Destination Country", value="Thailand", key="v_country_tab")
        if st.button("Check Visa Rules", key="btn_v"):
            from tools.tool_wrappers import lookup_visa_policy
            v_res = lookup_visa_policy.invoke({"country": v_country})
            if "message" in v_res and "not found" in v_res.get("message", ""):
                st.warning(v_res["message"])
            else:
                v_req = "✅ Required" if v_res.get("visa_required") else "❌ Not Required (Visa-Free / VoA)"
                st.markdown(f"""
                - **Visa Status:** {v_req}
                - **Visa Type:** {v_res.get('visa_type', 'N/A')}
                - **Permitted Stay:** {v_res.get('duration_allowed', 'N/A')}
                - **Fee:** ₹{v_res.get('fee_inr', 0):,}
                - **Source:** [{v_res.get('official_source', '#')}]({v_res.get('official_source', '#')})
                """)
                st.info(v_res.get("notes", ""))

        st.markdown("---")
        st.markdown("#### ✈️ Flight Search (MCP Database)")
        f_orig = st.text_input("Origin", value="Chennai", key="f_orig_tab")
        f_dest = st.text_input("Destination", value="Singapore", key="f_dest_tab")
        f_date = st.text_input("Date (YYYY-MM-DD)", value="2026-12-10", key="f_date_tab")
        if st.button("Find Flights", key="btn_f"):
            from tools.tool_wrappers import search_flights
            f_res = search_flights.invoke({"origin": f_orig, "destination": f_dest, "date": f_date})
            flights = f_res.get("flights", [])
            if flights:
                for fl in flights:
                    st.write(f"✈️ **{fl['airline']}** (`{fl['flight_id']}`) — {fl['departure']} → {fl['arrival']} — **₹{fl['price_inr']:,}**")
            else:
                st.info("No matching flights found for this specific route/date.")


# ============================================================
# TAB 4: KNOWLEDGE BASE
# ============================================================

with tab_kb:
    st.markdown("### 📚 Curated Knowledge Base Explorer")
    kb1, kb2, kb3 = st.tabs(["🛂 Visa Policies (10 Countries)", "🗺️ Destination City Guides", "🛡️ Travel Insurance Plans"])

    with kb1:
        from rag.knowledge_base import load_visa_policies
        for p in load_visa_policies():
            status_txt = "Visa Required" if p["visa_required"] else "Visa-Free / VoA"
            with st.expander(f"🛂 {p['country']} — {status_txt}"):
                st.write(f"**Type:** {p['visa_type']} | **Duration:** {p['duration_allowed']} | **Fee:** ₹{p['fee_inr']}")
                st.write(f"**Notes:** {p['notes']}")
                st.markdown(f"**Official Source:** [{p['official_source']}]({p['official_source']})")

    with kb2:
        from rag.knowledge_base import load_destination_guides
        for g in load_destination_guides():
            with st.expander(f"🗺️ {g['city']}, {g['country']} — Best time: {g['best_time_to_visit']}"):
                st.write(f"**Currency:** {g['currency']} | **Language:** {g['language']}")
                st.markdown("**Top Attractions:**")
                for a in g.get("top_attractions", []):
                    st.write(f"- {a['name']} ({a['type']}) — Entry: ₹{a.get('entry_fee_inr', 0)} ({a.get('time_needed', '')})")
                st.markdown("**Tips:**")
                for t in g.get("tips", []):
                    st.write(f"- {t}")

    with kb3:
        from rag.knowledge_base import load_travel_insurance
        for ins in load_travel_insurance():
            with st.expander(f"🛡️ {ins['plan_name']} — {ins['coverage_type']}"):
                st.write(f"**Premium:** ₹{ins['premium_inr_per_day']}/day | **Max Coverage:** ₹{ins['max_coverage_inr']:,}")
                st.write("**Covers:** " + ", ".join(ins.get("covers", [])))
                st.write("**Excludes:** " + ", ".join(ins.get("excludes", [])))


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption("Prices and currency rates are indicative. Verify visa rules and travel conditions with current official sources.")