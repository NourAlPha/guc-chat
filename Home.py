import streamlit as st
from utils import initialize_session_state

# Initialize session state with needed variables
initialize_session_state()

# Set page configuration including title and icon
st.set_page_config(page_title="GUC Chat",
                   page_icon="graphics/guc_logo_nb_Lsx_icon.ico")

st.image("graphics/guc-logo-nb.png", width=200)

# Displaying a markdown header with a welcoming message and description
st.markdown("""
            # Hello! I am GUC Chat 🤖👋
            A conversational AI bot that can help you with any questions you may have.
            """)
# Displaying a markdown section with instructions on what users can do
st.markdown("""
            ### 🛠 What You Can Do:
            - Go To `Chat` page to chat with bot
            - Go To `Experts` page to add content to bot manually
            - Go To `Upload PDF` page to upload a PDF file to bot database
            """)