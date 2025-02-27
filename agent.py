import os
import streamlit as st
from scraper import scrape_and_store
from chatbot import user_query
import time

st.set_page_config(page_title="Chatbot", layout="wide")

# Sidebar for URL input and scraping
with st.sidebar:
    st.header("📂 Documentation Scraper")
    base_url = st.text_input("Base URL", "https://docs.mparticle.com/")
    start_url = st.text_input("Start URL", "https://docs.mparticle.com/developers/developersjourney/")
    scrape_button = st.button("Scrape & Store Data")

    if scrape_button:
        st.success("Scraping started...")
        scrape_and_store(start_url, base_url)
        st.success("Scraping complete!")

# Chatbot UI
st.title("💬 AI Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat messages
for message in st.session_state["messages"]:
    with st.chat_message("user" if message["role"] == "user" else "assistant"):
        st.markdown(message["content"])

# Chat input at the bottom
query = st.chat_input("Ask anything...")
if query:
    # Store user message
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Get bot response with typewriter effect
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_text = ""
        print("Query received:", query)

        response = user_query(query)  # Empty context initially

        bot_response = response

        for char in bot_response:
            response_text += char
            response_placeholder.markdown(response_text)
            time.sleep(0.03)  # Slow typewriter effect
    
    # Store bot response
    st.session_state["messages"].append({"role": "assistant", "content": bot_response})
