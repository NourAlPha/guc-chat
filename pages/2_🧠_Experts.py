import streamlit as st
from utils import process_vector_space_level1, process_vector_space_level2_rules, initialize_session_state, summarizeDocAndSave
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
    
    if "add_rules" not in st.session_state:
        st.session_state.add_rules = False
    
    
    # Initialize session state with needed variables
    initialize_session_state()

    if "text_area_key" not in st.session_state:
        st.session_state.text_area_key = 0
    if "text_box_key" not in st.session_state:
        st.session_state.text_box_key = 0

    def on_click_rule():
        st.session_state.add_rules = True

    def on_click_context():
        st.session_state.add_rules = False
    # Add textbox for expert to input their content to the bot
    st.title("🧠 Experts" + (" Rules" if st.session_state.add_rules else " Content"))

    # Display a markdown section with instructions on what experts can do
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.session_state.add_rules:
            st.markdown("Enrich me with rules in the text box below.")
        else:
            st.markdown("Enrich me with content in the text box below.")
    with col2:
        if st.session_state.add_rules:
            st.button("Switch to Content", on_click=on_click_context)
        else:
            st.button("Switch to Rules", on_click=on_click_rule)
    
    # Add a text box for the name of the file of the expert response
    if st.session_state.add_rules:
        file_name = st.text_input("Enter the file name (without \"rule\" as suffix) to be saved in the database:", key=f"file_name_{st.session_state.text_box_key}")
        file_name += " rule"
    else:
        file_name = st.text_input("Enter the name of the file to be saved in the database:", key=f"file_name_{st.session_state.text_box_key}")
        
    # Add a text area for expert to input their response
    if st.session_state.add_rules:
        raw_text = st.text_area("Enter your response here:", key=f"text_area_{st.session_state.text_area_key}", max_chars=2000)
    else:
        raw_text = st.text_area("Enter your response here:", key=f"text_area_{st.session_state.text_area_key}")
    # Add a button to save the expert input
    def save_response_context():
        if file_name == "":
            st.warning("Please enter a file name before saving.")
            return
        if raw_text == "":
            st.warning("Please enter a response before saving.")
            return
        with open(f"text_files/{file_name}.txt", "a") as f:
            f.write("\n\n\n" + raw_text + "\n\n\n")
        summarizeDocAndSave(file_name + ".txt")
        process_vector_space_level1()
        st.session_state.text_area_key += 1
        st.session_state.text_box_key += 1
        st.success("Content added successfully! 🚀🧠")
    
    def save_response_rule():
        if file_name == "":
            st.warning("Please enter a file name before saving.")
            return
        if raw_text == "":
            st.warning("Please enter a response before saving.")
            return
        with open(f"rules/{file_name}.txt", "a") as f:
            f.write(raw_text)
        process_vector_space_level2_rules()
        st.session_state.text_area_key += 1
        st.session_state.text_box_key += 1
        st.success("Rule added successfully! 🚀🧠")
    

    st.button("Add Content" if not st.session_state.add_rules else "Add Rule", on_click=(save_response_context if not st.session_state.add_rules else save_response_rule))
        
if __name__ == "__main__":
    main()
