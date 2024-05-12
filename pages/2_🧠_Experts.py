import streamlit as st
from utils import process_vector_space, initialize_session_state
import os

# Initialize session state with needed variables
initialize_session_state()

if "text_area_key" not in st.session_state:
    st.session_state.text_area_key = 0

# Set page configuration including title and icon
st.set_page_config(page_title="Experts",
                   page_icon="🧠")

# Add textbox for expert to input their content to the bot
st.title("🧠 Experts")

# Display a markdown section with instructions on what experts can do
st.markdown("You can enrich me with content by entering your response in the text box below.")

# Add a text area for expert to input their response
raw_text = st.text_area("Enter your response here:", key=f"text_area_{st.session_state.text_area_key}")

# Add a button to save the expert input
def save_response():
    length = len(os.listdir("text_files"))
    with open(f"text_files/expert_responses{length}.txt", "a") as f:
        f.write(raw_text + "\n")
    process_vector_space()
    st.session_state.text_area_key += 1
    
if st.button("Add Content", on_click=save_response):
    st.success("Content added successfully! 🚀🧠")
