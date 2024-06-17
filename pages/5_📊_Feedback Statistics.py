import streamlit as st
import plotly.graph_objects as go
import json
from auth import authenticate_admin
from utils import initialize_session_state

def main():
    
    # Set page configuration including title and icon
    st.set_page_config(page_title="Feedback Statistics",
                        page_icon="📊")
    
    # Authenticate admin
    if "authentication_status" not in st.session_state or not st.session_state.authentication_status:
        authenticate_admin()
    else:
        st.session_state.authenticator.logout('Logout', 'sidebar')
    if not st.session_state.authentication_status:
        return
    
    # Initialize session state with needed variables
    initialize_session_state()
    
    # Add a title to the page
    st.title("📊 Feedback Statistics")
    
    # Add a description to the page
    st.markdown("Below is the user feedback statistics of the chatbot responses, with Great being the highest and Very Poor being the lowest.")
    
    feedbacks = json.load(open("feedback.json"))
    labels = list(feedbacks.keys())
    values = list(feedbacks.values())

    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    fig.update_layout(height=600)
    st.plotly_chart(fig, theme=None, height=800)

    if st.button("Reset Statistics"):
        with open("feedback.json", "w") as f:
            json.dump({
                "Great": 0,
                "Good": 0,
                "Average": 0,
                "Poor": 0,
                "Very Poor": 0
            }, f)
        st.rerun()


if __name__ == "__main__":
    main()