import streamlit as st
from utils import process_vector_space, initialize_session_state
import os
from auth import authenticate_admin

def main():
    # Set page configuration including title and icon
    st.set_page_config(page_title="Experts",
                    page_icon="🧠")
    
    # Authenticate admin
    if "authentication_status" not in st.session_state or not st.session_state.authentication_status:
        authenticate_admin()
    else:
        st.session_state.authenticator.logout('Logout', 'sidebar')
    if not st.session_state.authentication_status:
        return
    
    # Initialize session state with needed variables
    initialize_session_state()

    if "text_area_key" not in st.session_state:
        st.session_state.text_area_key = 0
    if "text_box_key" not in st.session_state:
        st.session_state.text_box_key = 0

    # Add textbox for expert to input their content to the bot
    st.title("🧠 Experts")

    # Display a markdown section with instructions on what experts can do
    st.markdown("You can enrich me with content by entering your response in the text box below.")
    
    # Add a text box for the name of the file of the expert response
    file_name = st.text_input("Enter the name of the file to be saved in the database:", key=f"file_name_{st.session_state.text_box_key}")

    # Add a text area for expert to input their response
    raw_text = st.text_area("Enter your response here:", key=f"text_area_{st.session_state.text_area_key}")

    # Add a button to save the expert input
    def save_response():
        if file_name == "":
            st.warning("Please enter a file name before saving.")
            return
        if raw_text == "":
            st.warning("Please enter a response before saving.")
            return
        with open(f"text_files/{file_name}.txt", "a") as f:
            f.write("\n\n\n" + raw_text + "\n\n\n")
        process_vector_space()
        st.session_state.text_area_key += 1
        st.session_state.text_box_key += 1
        st.success("Content added successfully! 🚀🧠")
    
    st.button("Add Content", on_click=save_response)
        
if __name__ == "__main__":
    main()
