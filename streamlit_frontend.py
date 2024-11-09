import streamlit as st
import requests
import time

url = "http://127.0.0.1:8000/"

st.title("Query Reformulation App")


def api_status():
    try:
        response = requests.get(url + "health")
        if response.status_code == 200 and response.json().get("status") == "ready":
            return True
    except requests.exceptions.RequestException:
        return False
    return False


with st.spinner("Initializing FastAPI Server... Please wait."):
    while not api_status():
        time.sleep(1)

input_text = st.text_area("Enter your query:", "")

if st.button("Reformulate Query"):
    if input_text:
        payload = {"query": input_text}

        response = requests.post(url + "reformulate_query", json=payload)

        if response.status_code == 200 and response.json():
            reformulated_queries = response.json()
            for idx, query in enumerate(reformulated_queries):
                st.write(f"{idx + 1}: {query}")
        else:
            st.error("Error: Unable to reformulate the query.")
    else:
        st.warning("Please enter a query.")
