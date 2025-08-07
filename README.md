# 🌍 AI Travel Planner – Powered by Agentic AI

An intelligent, agent-driven travel planner built using **Streamlit**, **LangGraph**, and **LangChain**, and deployed on **Hugging Face Spaces**.

This project uses **Agentic AI** from the moment a user engages with the app to the final generation of a custom travel plan. The system reasons through the request in multiple steps, makes decisions, and returns thoughtful, personalized travel recommendations.

> 🚀 **Live App**: [Try it on Hugging Face Spaces](https://huggingface.co/spaces/olayimika01/AI-Travel-Planner)

## This app serves as a smart travel assistant that collects the following input from the user:

🌍 Destination

📅 Month of Travel (e.g., January)

📆 Number of Days

👥 Number of People

🏖️ Holiday Type (e.g., Adventure, Relaxation, Family)

💸 Budget Type (e.g., Economy, Premium)

💰 Budget

✍️ Additional Comments

Once the data is submitted, the Agentic AI system takes over to understand the user's travel goals
Decide what information needs to be gathered or reasoned through
Generate a rich travel itinerary with intelligent recommendations


---
## Add your own API keys
Create a .env file in the root directory:
GOOGLE_API_KEY=your_google_api_key
GEMINI_API_KEY=your_gemini_api_key

## 📂 Repository Contents

```bash
.
├── Agentic_AI.ipynb       # Jupyter Notebook with full agentic AI logic
├── app.py                 # Streamlit app for public deployment
└── README.md              # This file

