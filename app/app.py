import streamlit as st
import requests
from streamlit_chat import message

st.title("💬 Protein Chat")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

user_input = st.text_input("You:", key="input")

if user_input:
    st.session_state["messages"].append({"text": user_input, "is_user": True})

    url = "http://127.0.0.1:8000/predict"
    data = {"feature": user_input}

    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        prediction = dict(response.json())
        print(prediction)
    except Exception as e:
        prediction_str = f"Error: {e}"

    bot_reply = f"AMR Resistance predicted: {prediction['prediction']}"
    st.session_state["messages"].append({"text": bot_reply, "is_user": False})

# Display messages with unique keys
for i, msg in enumerate(st.session_state["messages"]):
    message(
        msg["text"],
        is_user=msg["is_user"],
        key=f"msg_{i}"  # unique key
    )
