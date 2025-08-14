import streamlit as st
import json
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_community.utilities import GoogleSerperAPIWrapper
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
import tempfile
from fpdf import FPDF

# ------------------- Page & Env -------------------
st.set_page_config(page_title="AI Travel Planner", layout="wide")
load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ------------------- Initialize Services -------------------
# NOTE: Streamlit re-runs the script on every interaction. We keep these in the global scope
# so they only actually initialize once per process.
try:
    llm = ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model="gemini-2.0-flash")
except Exception as e:
    st.error(f"LLM initialization failed: {str(e)}")
    st.stop()

try:
    search = GoogleSerperAPIWrapper()  # uses SERPER_API_KEY from env
except Exception as e:
    st.error(f"Serper API initialization failed: {str(e)}")
    st.stop()

# ------------------- Utilities -------------------
def export_to_pdf(itinerary_text: str) -> str:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in itinerary_text.split("\n"):
        # FPDF is latin-1; replace unknowns to avoid crash
        safe = line.encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 10, safe)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp.name)
    return tmp.name

# ------------------- Base LLM/Tools (uncached) -------------------
def _llm_generate_itinerary(preferences: dict) -> str:
    prompt = f"""
    Using the following preferences, create a detailed itinerary:
    {json.dumps(preferences, indent=2)}
    Include sections for each day, dining options, and downtime.
    """
    return llm.invoke(prompt).content.strip()

def _llm_recommend_activities(preferences: dict, itinerary: str) -> str:
    prompt = f"""
    Based on these preferences and the itinerary, suggest unique local activities.
    Preferences: {json.dumps(preferences, indent=2)}
    Itinerary: {itinerary}
    Provide bullet-point suggestions for each day if possible.
    """
    return llm.invoke(prompt).content.strip()

def _llm_food_culture(preferences: dict) -> str:
    prompt = f"""
    For a trip to {preferences.get('destination','')} with a {preferences.get('budget_type','mid-range')} budget:
    1) Suggest popular local dishes and recommended dining options.
    2) Provide important cultural norms, etiquette tips, and things travelers should be aware of.
    Format sections as 'Food & Dining' and 'Culture & Etiquette'.
    """
    return llm.invoke(prompt).content.strip()

def _llm_packing_list(preferences: dict) -> str:
    prompt = f"""
    Generate a comprehensive packing list for a {preferences.get('holiday_type','general')} holiday in {preferences.get('destination','')}
    during {preferences.get('month','')} for {preferences.get('duration',0)} days.
    Include essentials based on expected weather and trip type.
    """
    return llm.invoke(prompt).content.strip()

def _llm_weather(preferences: dict) -> str:
    prompt = f"""
    Based on the destination and month, provide a concise weather overview:
    - Typical temperatures
    - Precipitation likelihood
    - Packing/comfort advice
    Destination: {preferences.get('destination','')}
    Month: {preferences.get('month','')}
    """
    return llm.invoke(prompt).content.strip()

def _serper_links(preferences: dict):
    destination = preferences.get("destination", "")
    month = preferences.get("month", "")
    query = f"Travel tips and guides for {destination} in {month}"
    results = search.results(query)
    organic = results.get("organic", [])
    return [
        {"title": r.get("title", "No title"), "link": r.get("link", "")}
        for r in organic[:5]
    ]

def _llm_chat(state: dict) -> dict:
    prompt = f"""
    Context:
    Preferences: {json.dumps(state.get('preferences', {}), indent=2)}
    Itinerary: {state.get('itinerary', '')}

    User Question:
    {state.get('user_question', '')}

    Respond conversationally with insights or suggestions. Keep it brief.
    Return JSON: {{ "chat_response": "..." }}
    """
    content = llm.invoke(prompt).content
    try:
        parsed = json.loads(content.strip())
        response = parsed.get("chat_response", content.strip())
    except json.JSONDecodeError:
        response = content.strip()
    return {"chat_response": response}

# ------------------- Cache Layers -------------------
# We cache by a deterministic key (frozen preferences) so re-clicks are instant and don't re-generate.
@st.cache_data(show_spinner=False)
def cached_generate_itinerary(preferences_json: str) -> str:
    return _llm_generate_itinerary(json.loads(preferences_json))

@st.cache_data(show_spinner=False)
def cached_recommend_activities(preferences_json: str, itinerary: str) -> str:
    return _llm_recommend_activities(json.loads(preferences_json), itinerary)

@st.cache_data(show_spinner=False)
def cached_food_culture(preferences_json: str) -> str:
    return _llm_food_culture(json.loads(preferences_json))

@st.cache_data(show_spinner=False)
def cached_packing_list(preferences_json: str) -> str:
    return _llm_packing_list(json.loads(preferences_json))

@st.cache_data(show_spinner=False)
def cached_weather(preferences_json: str) -> str:
    return _llm_weather(json.loads(preferences_json))

@st.cache_data(show_spinner=False)
def cached_links(preferences_json: str):
    return _serper_links(json.loads(preferences_json))

# ------------------- (Optional) LangGraph Shell -------------------
class GraphState(TypedDict):
    preferences_text: str
    preferences: dict
    itinerary: str
    activity_suggestions: str
    useful_links: list[dict]
    weather_forecast: str
    packing_list: str
    food_culture_info: str
    chat_history: Annotated[list[dict], "List of question-response pairs"]
    user_question: str
    chat_response: str

def generate_itinerary_node(state: GraphState):
    # Use cached generator
    prefs_json = json.dumps(state["preferences"], sort_keys=True)
    itinerary = cached_generate_itinerary(prefs_json)
    return {"itinerary": itinerary}

workflow = StateGraph(GraphState)
workflow.add_node("generate_itinerary", generate_itinerary_node)
workflow.set_entry_point("generate_itinerary")
workflow.add_edge("generate_itinerary", END)
graph = workflow.compile()

# ------------------- Session State -------------------
if "state" not in st.session_state:
    st.session_state.state = {
        "preferences_text": "",
        "preferences": {},
        "itinerary": "",
        "activity_suggestions": "",
        "useful_links": [],
        "weather_forecast": "",
        "packing_list": "",
        "food_culture_info": "",
        "chat_history": [],
        "user_question": "",
        "chat_response": ""
    }

# ------------------- Header -------------------
st.markdown("<h1 style='color:#0077b6;'>🌍 AI-Powered Travel Itinerary Planner</h1>", unsafe_allow_html=True)
st.caption("Plan smarter, travel better – all with AI ✈️")

# ------------------- Form -------------------
with st.form("travel_form"):
    st.subheader("✏️ Tell us about your trip")
    c1, c2 = st.columns(2)
    with c1:
        destination = st.text_input("Destination")
        month = st.selectbox(
            "Month of Travel",
            ["January","February","March","April","May","June","July","August","September","October","November","December"]
        )
        duration = st.slider("Number of Days", 1, 30, 7)
        num_people = st.selectbox("Number of People", ["1", "2", "3", "4-6", "7-10", "10+"])
        accommodation_type = st.selectbox(
            "Accommodation Preferences",
            ["Hotel", "Resort", "Airbnb / Apartment", "No Preference"]
        )
    with c2:
        holiday_type = st.selectbox(
            "Holiday Type",
            ["Any", "Party", "Skiing", "Backpacking", "Family", "Beach", "Festival", "Adventure", "City Break", "Romantic", "Cruise"]
        )
        budget_type = st.selectbox("Budget Type", ["Budget", "Mid-Range", "Luxury", "Backpacker", "Family"])
        visa_status_options = st.selectbox(
            "Visa Status",
            ["No visa required", "Visa required", "Visa on arrival", "E-visa available", "I don't know / Not sure"]
        )
        comments = st.text_area("Additional Comments")

    submit_btn = st.form_submit_button("✨ Generate Itinerary")

# ------------------- Submission -------------------
if submit_btn:
    preferences = {
        "destination": destination,
        "month": month,
        "duration": duration,
        "num_people": num_people,
        "holiday_type": holiday_type,
        "budget_type": budget_type,
        "accommodation": accommodation_type,
        "visa_status_options": visa_status_options,
        "comments": comments
    }
    preferences_text = (
        f"Destination: {destination}\nMonth: {month}\nDuration: {duration} days\nPeople: {num_people}\n"
        f"Type: {holiday_type}\nBudget: {budget_type}\nAccommodation: {accommodation_type}\n"
        f"Visa: {visa_status_options}\nComments: {comments}"
    )

    # Reset extras/chat when generating a new itinerary
    st.session_state.state.update({
        "preferences_text": preferences_text,
        "preferences": preferences,
        "activity_suggestions": "",
        "useful_links": [],
        "weather_forecast": "",
        "packing_list": "",
        "food_culture_info": "",
        "user_question": "",
        "chat_response": ""
    })

    with st.spinner("Generating itinerary..."):
        result = graph.invoke(st.session_state.state)
        st.session_state.state.update(result)

    if st.session_state.state.get("itinerary"):
        st.success("✅ Itinerary Created")
    else:
        st.error("❌ Failed to generate itinerary.")

# ------------------- Main UI: Tabs -------------------
if st.session_state.state.get("itinerary"):
    tab_itin, tab_extras, tab_chat, tab_export = st.tabs(["🧭 Itinerary", "📌 Extras", "💬 Chat", "📄 Export"])

    # --- Itinerary Tab (rendered once, never elsewhere) ---
    with tab_itin:
        st.markdown("### Your Travel Itinerary")
        st.markdown(st.session_state.state["itinerary"])

    # --- Extras Tab ---
    with tab_extras:
        st.markdown("#### Explore more")
        b1, b2, b3, b4, b5 = st.columns(5)

        prefs_json_key = json.dumps(st.session_state.state["preferences"], sort_keys=True)

        with b1:
            if st.button("Activities", key="x_activities"):
                st.session_state.state["activity_suggestions"] = cached_recommend_activities(
                    prefs_json_key,
                    st.session_state.state["itinerary"]
                )
        with b2:
            if st.button("Useful Links", key="x_links"):
                st.session_state.state["useful_links"] = cached_links(prefs_json_key)
        with b3:
            if st.button("Weather", key="x_weather"):
                st.session_state.state["weather_forecast"] = cached_weather(prefs_json_key)
        with b4:
            if st.button("Packing List", key="x_pack"):
                st.session_state.state["packing_list"] = cached_packing_list(prefs_json_key)
        with b5:
            if st.button("Food & Culture", key="x_food"):
                st.session_state.state["food_culture_info"] = cached_food_culture(prefs_json_key)

        # Only extras are shown here (no itinerary duplication)
        if st.session_state.state.get("activity_suggestions"):
            with st.expander("🎯 Activity Suggestions", expanded=True):
                st.markdown(st.session_state.state["activity_suggestions"])

        if st.session_state.state.get("useful_links"):
            with st.expander("🔗 Useful Links", expanded=True):
                for link in st.session_state.state["useful_links"]:
                    st.markdown(f"- [{link['title']}]({link['link']})")

        if st.session_state.state.get("weather_forecast"):
            with st.expander("🌤️ Weather Forecast", expanded=True):
                st.markdown(st.session_state.state["weather_forecast"])

        if st.session_state.state.get("packing_list"):
            with st.expander("🎒 Packing List", expanded=True):
                st.markdown(st.session_state.state["packing_list"])

        if st.session_state.state.get("food_culture_info"):
            with st.expander("🍽️ Food & Culture Info", expanded=True):
                st.markdown(st.session_state.state["food_culture_info"])

    # --- Chat Tab ---
    with tab_chat:
        st.markdown("### Chat About Your Itinerary")
        for chat in st.session_state.state.get("chat_history", []):
            with st.chat_message("user"):
                st.markdown(chat["question"])
            with st.chat_message("assistant"):
                st.markdown(chat["response"])

        user_q = st.chat_input("Ask something about your trip")
        if user_q:
            st.session_state.state["user_question"] = user_q
            with st.spinner("Thinking..."):
                resp = _llm_chat(st.session_state.state)
            st.session_state.state["chat_response"] = resp["chat_response"]
            st.session_state.state["chat_history"] = st.session_state.state.get("chat_history", []) + [
                {"question": user_q, "response": resp["chat_response"]}
            ]
            # No experimental_rerun here; Streamlit will re-run automatically after input

    # --- Export Tab ---
    with tab_export:
        st.markdown("### Export Itinerary")
        if st.button("Generate PDF", key="btn_pdf"):
            path = export_to_pdf(st.session_state.state["itinerary"])
            with open(path, "rb") as f:
                st.download_button("Download Itinerary PDF", f, file_name="itinerary.pdf")

else:
    st.info("📌 Fill the form and generate an itinerary to begin.")
