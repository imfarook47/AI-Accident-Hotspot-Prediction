import streamlit as st
import os

from dotenv import load_dotenv
from groq import Groq

# ----------------------------------
# Load Environment Variables
# ----------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ----------------------------------
# Groq Client
# ----------------------------------
client = Groq(
    api_key=GROQ_API_KEY
)

# ----------------------------------
# Page Configuration
# ----------------------------------
st.set_page_config(
    page_title="AI Road Safety Chatbot",
    layout="wide"
)

# ----------------------------------
# Custom UI
# ----------------------------------
st.markdown("""
<style>

/* Main Background */
.main {

    background-color: #0f172a;
    color: white;
}

/* Chat Input */
.stChatInput input {

    background-color: white !important;
    color: black !important;

    border-radius: 12px !important;
}

/* Chat Messages */
[data-testid="stChatMessage"] {

    border-radius: 15px;
    padding: 12px;
    margin-bottom: 12px;
}

/* Sidebar */
[data-testid="stSidebar"] {

    background-color: #111827;
}

/* Sidebar Text */
[data-testid="stSidebar"] * {

    color: white !important;
}

/* Buttons */
.stButton button {

    background-color: #2563eb;
    color: white !important;

    border-radius: 10px;

    border: none;

    padding: 10px;
}

/* Title */
h1 {

    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------
# Title
# ----------------------------------
st.title("🤖 AI Road Safety Assistant")

st.markdown("""
### Ask anything about:

- 🚗 Route safety
- 🌧 Weather impact
- 🚦 Traffic congestion
- 🛣 Highway driving
- 🌙 Night travel safety
- ⚠ Accident prevention
- ⏰ Best travel timing
""")

# ----------------------------------
# Session State
# ----------------------------------
if "messages" not in st.session_state:

    st.session_state.messages = []

# ----------------------------------
# Sidebar Controls
# ----------------------------------
st.sidebar.title("⚙ Chat Controls")

# Clear Chat Button
if st.sidebar.button("🗑 Clear Chat"):

    st.session_state.messages = []

    st.rerun()

# Sidebar History
st.sidebar.markdown("## 💬 Previous Questions")

for msg in st.session_state.messages:

    if msg["role"] == "user":

        st.sidebar.write(
            f"🧑 {msg['content'][:40]}"
        )

# ----------------------------------
# Display Chat History
# ----------------------------------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# ----------------------------------
# Chat Input
# ----------------------------------
prompt = st.chat_input(
    "Ask AI about road safety..."
)

# ----------------------------------
# AI Response
# ----------------------------------
if prompt:

    # Store User Message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    # Display User Message
    with st.chat_message("user"):

        st.markdown(prompt)

    # Assistant Response
    with st.chat_message("assistant"):

        with st.spinner(
            "🤖 AI is analyzing road safety..."
        ):

            try:

                response = client.chat.completions.create(

                    model="llama-3.3-70b-versatile",

                    messages=[

                        {
                            "role": "system",

                            "content":
                            """
                            You are an advanced AI Road Safety Assistant.

                            Remember previous conversation context
                            and answer conversationally.

                            Help users with:

                            - Accident prevention
                            - Safe driving practices
                            - Traffic awareness
                            - Weather-related risks
                            - Highway safety
                            - Fog and rain driving
                            - Night travel precautions
                            - Emergency driving guidance
                            - Defensive driving
                            - Best travel timings

                            Give practical, intelligent,
                            human-like responses.
                            """
                        },

                        *st.session_state.messages
                    ]
                )

                ai_reply = (
                    response.choices[0]
                    .message.content
                )

            except Exception as e:

                ai_reply = f"AI Error: {e}"

            st.markdown(ai_reply)

    # Store Assistant Message
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": ai_reply
        }
    )