
import streamlit as st
import json
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
import json
from langchain_community.utilities import GoogleSerperAPIWrapper
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
import tempfile
from fpdf import FPDF

#API Keys
SERPER_API_KEY= os.getenv("SERPER_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize LLM
st.set_page_config(page_title="AI Travel Planner", layout="wide")
try:
    llm = ChatGoogleGenerativeAI(api_key=google_api_key, model='gemini-2.0-flash')
except Exception as e:
    st.error(f"LLM initialization failed: {str(e)}")
    st.stop()

# Initialize GoogleSerperAPIWrapper
try:
    search = GoogleSerperAPIWrapper()
except Exception as e:
    st.error(f"Serper API initialization failed: {str(e)}")
    st.stop()


# Export to PDF
def export_to_pdf(itinerary_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    try:
        for line in itinerary_text.split("\n"):
            line = line.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 10, line)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf.output(temp_file.name)
        return temp_file.name
    except Exception as e:
        raise Exception(f"PDF generation failed: {str(e)}")


def chat_node(state):
    prompt = f"""
    Context:
    Preferences: {json.dumps(state['preferences'], indent=2)}
    Itinerary: {state['itinerary']}

    User Question:
    {state['user_question']}

    Respond conversationally with insights or suggestions : keep your response brief
    {{ "chat_response": "Your response here" }}
    """
    try:
        result = llm.invoke(prompt).content
        try:
            parsed = json.loads(result.strip())
            response = parsed.get("chat_response", result.strip())
        except json.JSONDecodeError:
            response = result.strip()
        chat_entry = {"question": state['user_question'], "response": response}
        chat_history = state.get('chat_history', []) + [chat_entry]
        return {"chat_response": response, "chat_history": chat_history}
    except Exception as e:
        return {"chat_response": "", "warning": str(e)}

def fetch_useful_links(state):
    search = GoogleSerperAPIWrapper()
    destination = state['preferences'].get('destination', '')
    month = state['preferences'].get('month', '')
    query = f"Travel tips and guides for {destination} in {month}"
    try:
        search_results = search.results(query)
        organic_results = search_results.get("organic", [])
        links = [
            {"title": result.get("title", "No title"), "link": result.get("link", "")}
            for result in organic_results[:5]
        ]
        return {"useful_links": links}
    except Exception as e:
        return {"useful_links": [], "warning": f"Failed to fetch links: {str(e)}"}


def food_culture_recommender(state):
    prompt = f"""
    For a trip to {state['preferences'].get('destination', '')} with a {state['preferences'].get('budget_type', 'mid-range')} budget:
    1. Suggest popular local dishes and recommended dining options.
    2. Provide important cultural norms, etiquette tips, and things travelers should be aware of.
    Format the response with clear sections for 'Food & Dining' and 'Culture & Etiquette'.
    """
    try:
        result = llm.invoke(prompt).content
        return {"food_culture_info": result.strip()}
    except Exception as e:
        return {"food_culture_info": "", "warning": str(e)}


def generate_itinerary(state):

    prompt = f"""
    Using the following preferences, create a detailed itinerary:
    {json.dumps(state['preferences'], indent=2)}

    Include sections for each day, dining options, and downtime.
    """
    try:
        result = llm.invoke(prompt).content
        return {"itinerary": result.strip()}
    except Exception as e:
        return {"itinerary": "", "warning": str(e)}

def packing_list_generator(state):
    prompt = f"""
    Generate a comprehensive packing list for a {state['preferences'].get('holiday_type', 'general')} holiday in {state['preferences'].get('destination', '')} during {state['preferences'].get('month', '')} for {state['preferences'].get('duration', 0)} days.
    Include essentials based on expected weather and trip type.
    """
    try:
        result = llm.invoke(prompt).content
        return {"packing_list": result.strip()}
    except Exception as e:
        return {"packing_list": "", "warning": str(e)}

def recommend_activities(state):
    prompt = f"""
    Based on the following preferences and itinerary, suggest unique local activities:
    Preferences: {json.dumps(state['preferences'], indent=2)}
    Itinerary: {state['itinerary']}

    Provide suggestions in bullet points for each day if possible.
    """
    try:
        result = llm.invoke(prompt).content
        return {"activity_suggestions": result.strip()}
    except Exception as e:
        return {"activity_suggestions": "", "warning": str(e)}

def weather_forecaster(state):

    prompt = f"""
    Based on the destination and month, provide a detailed weather forecast including temperature, precipitation, and advice for travelers:
    Destination: {state['preferences'].get('destination', '')}
    Month: {state['preferences'].get('month', '')}
    """
    try:
        result = llm.invoke(prompt).content
        return {"weather_forecast": result.strip()}
    except Exception as e:
        return {"weather_forecast": "", "warning": str(e)}


# Define state
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

# ------------------- LangGraph -------------------

workflow = StateGraph(GraphState)
workflow.add_node("generate_itinerary", generate_itinerary)
# workflow.add_node("fetch_useful_links", fetch_useful_links)
# workflow.add_node("weather_forecaster", weather_forecaster)
# workflow.add_node("recommend_activities", recommend_activities)
# workflow.add_node("packing_list_generator", packing_list_generator)
# workflow.add_node("food_culture_recommender", food_culture_recommender)
# workflow.add_node("chat", chat_node)
workflow.set_entry_point("generate_itinerary")
workflow.add_edge("generate_itinerary", END)
# workflow.add_edge("recommend_activities", END)
# workflow.add_edge("fetch_useful_links", END)
# workflow.add_edge("weather_forecaster", END)
# workflow.add_edge("packing_list_generator", END)
# workflow.add_edge("food_culture_recommender", END)
# workflow.add_edge("chat", END)
graph = workflow.compile()




# ------------------- Initial Setup -------------------
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


# ------------------- Custom Header -------------------
st.markdown("<h1 style='color:#0077b6;'>🌍 AI-Powered Travel Itinerary Planner</h1>", unsafe_allow_html=True)
st.markdown("Plan smarter, travel better – all with AI ✈️")

# ------------------- Initial State -------------------
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

#if "itinerary_displayed" not in st.session_state:
    #st.session_state.itinerary_displayed = False

# ------------------- Form UI -------------------
with st.form("travel_form"):
    st.markdown("### ✏️ Tell us about your trip")
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            destination = st.text_input("Destination")
            month = st.selectbox("Month of Travel", ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])
            duration = st.slider("Number of Days", 1, 30, 7)
            num_people = st.selectbox("Number of People", ["1", "2", "3", "4-6", "7-10", "10+"])
            accommodation_type = st.selectbox("Accommodation Preferences", ["Hotel", "Resort", "Airbnb / Apartment", "No Preference"])
        with col2:
            holiday_type = st.selectbox("Holiday Type", ["Any", "Party", "Skiing", "Backpacking", "Family", "Beach", "Festival", "Adventure", "City Break", "Romantic", "Cruise"])
            budget_type = st.selectbox("Budget Type", ["Budget", "Mid-Range", "Luxury", "Backpacker", "Family"])
            visa_status_options = st.selectbox("Visa Status", [
                "No visa required",
                "Visa required",
                "Visa on arrival",
                "E-visa available",
                "I don't know / Not sure"
            ])
            comments = st.text_area("Additional Comments")

    submit_btn = st.form_submit_button("✨ Generate Itinerary")

# ------------------- Submission Handling -------------------
if submit_btn:
    preferences_text = f"Destination: {destination}\nMonth: {month}\nDuration: {duration} days\nPeople: {num_people}\nType: {holiday_type}\nBudget: {budget_type}\nAccommodation: {accommodation_type}\nVisa: {visa_status_options}\nComments: {comments}"
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
    st.session_state.state.update({
        "preferences_text": preferences_text,
        "preferences": preferences,
        "chat_history": [],
        "user_question": "",
        "chat_response": "",
        "activity_suggestions": "",
        "useful_links": [],
        "weather_forecast": "",
        "packing_list": "",
        "food_culture_info": ""
    })

    with st.spinner("Generating itinerary..."):
        result = graph.invoke(st.session_state.state)
        st.session_state.state.update(result)
        if result.get("itinerary"):
            st.success("✅ Itinerary Created")
            st.session_state.itinerary_displayed = False  # allow initial display
        else:
            st.error("❌ Failed to generate itinerary.")

# ------------------- Layout After Itinerary -------------------
if st.session_state.state.get("itinerary"):
    # Initialize flags for itinerary display and expander states
    if "itinerary_displayed" not in st.session_state:
        st.session_state.itinerary_displayed = False
    if "expanded_section" not in st.session_state:
        st.session_state.expanded_section = None

    # Single container to enforce strict top-to-bottom layout
    with st.container():
        # Itinerary display with CSS to pin it at the top
        if not st.session_state.itinerary_displayed:
            st.markdown(
                """
                <div style='position: sticky; top: 0; z-index: 100; background-color: white; padding: 10px;'>
                    <h3>🧭 Travel Itinerary</h3>
                    {itinerary}
                </div>
                """.format(itinerary=st.session_state.state["itinerary"]),
                unsafe_allow_html=True
            )
            st.session_state.itinerary_displayed = True

        # Horizontal divider for visual separation
        st.markdown("---")

        # Columns for Extras and Chat below itinerary
        col_itin, col_chat = st.columns([3, 2])

        with col_itin:
            # Extras Buttons
            st.markdown("#### 📌 Extras")
            col_btn1, col_btn2, col_btn3, col_btn4, col_btn5 = st.columns(5)
            with col_btn1:
                if st.button("Activities"):
                    with st.spinner("Fetching activity suggestions..."):
                        result = recommend_activities(st.session_state.state)
                        st.session_state.state.update(result)
                        st.session_state.expanded_section = "activities"
            with col_btn2:
                if st.button("Links"):
                    with st.spinner("Fetching useful links..."):
                        result = fetch_useful_links(st.session_state.state)
                        st.session_state.state.update(result)
                        st.session_state.expanded_section = "links"
            with col_btn3:
                if st.button("Weather"):
                    with st.spinner("Fetching weather forecast..."):
                        result = weather_forecaster(st.session_state.state)
                        st.session_state.state.update(result)
                        st.session_state.expanded_section = "weather"
            with col_btn4:
                if st.button("Packing"):
                    with st.spinner("Generating packing list..."):
                        result = packing_list_generator(st.session_state.state)
                        st.session_state.state.update(result)
                        st.session_state.expanded_section = "packing"
            with col_btn5:
                if st.button("Food & Culture"):
                    with st.spinner("Fetching info..."):
                        result = food_culture_recommender(st.session_state.state)
                        st.session_state.state.update(result)
                        st.session_state.expanded_section = "food_culture"

            # Expanders for each section, displayed below buttons
            if st.session_state.state.get("activity_suggestions"):
                with st.expander("🎯 Activity Suggestions", expanded=st.session_state.expanded_section == "activities"):
                    st.markdown(st.session_state.state["activity_suggestions"])

            if st.session_state.state.get("useful_links"):
                with st.expander("🔗 Useful Links", expanded=st.session_state.expanded_section == "links"):
                    for link in st.session_state.state["useful_links"]:
                        st.markdown(f"- [{link['title']}]({link['link']})")

            if st.session_state.state.get("weather_forecast"):
                with st.expander("🌤️ Weather Forecast", expanded=st.session_state.expanded_section == "weather"):
                    st.markdown(st.session_state.state["weather_forecast"])

            if st.session_state.state.get("packing_list"):
                with st.expander("🎒 Packing List", expanded=st.session_state.expanded_section == "packing"):
                    st.markdown(st.session_state.state["packing_list"])

            if st.session_state.state.get("food_culture_info"):
                with st.expander("🍽️ Food & Culture Info", expanded=st.session_state.expanded_section == "food_culture"):
                    st.markdown(st.session_state.state["food_culture_info"])

            # PDF export below expanders
            if st.button("📄 Export as PDF"):
                pdf_path = export_to_pdf(st.session_state.state["itinerary"])
                if pdf_path:
                    with open(pdf_path, "rb") as f:
                        st.download_button("Download PDF", f, file_name="itinerary.pdf")

        with col_chat:
            st.markdown("### 💬 Chat About Your Itinerary")
            for chat in st.session_state.state["chat_history"]:
                with st.chat_message("user"):
                    st.markdown(chat["question"])
                with st.chat_message("assistant"):
                    st.markdown(chat["response"])

            if user_input := st.chat_input("Ask something about your trip"):
                st.session_state.state["user_question"] = user_input
                with st.spinner("Generating response..."):
                    result = chat_node(st.session_state.state)
                    st.session_state.state.update(result)
                    st.rerun()
else:
    st.info("📌 Fill the form and generate an itinerary to begin.")
