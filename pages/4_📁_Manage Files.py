import streamlit as st
from utils import initialize_session_state, process_vector_space
import os
from streamlit_pdf_viewer import pdf_viewer
from streamlit_extras.stylable_container import stylable_container
from auth import authenticate_admin

def main():
    # Set page configuration including title and icon
    st.set_page_config(page_title="Manage Files",
                    page_icon="📁")
    
    # Authenticate admin
    authenticate_admin()
    if not st.session_state.authentication_status:
        return

    # Initialize session state with needed variables
    initialize_session_state()

    # Add a title to the page
    st.title("📁 Manage Files")

    # Add a description to the page
    st.markdown("Manage the files in my knowledge base.")

    # Add a select box to the page to select from the pdf_files directory
    all_file_list = os.listdir("pdf_files") + os.listdir("text_files")
    file_list = st.selectbox("Select a file from the knowledge base:", all_file_list, placeholder="Select a file", index=None)

    def delete_file():
        if file_list in os.listdir("pdf_files"):
            os.remove(f"pdf_files/{file_list}")
        else:
            os.remove(f"text_files/{file_list}")
        process_vector_space()
        st.success("File deleted successfully! 🚫🚀")

    def exclude_file():
        with open("excluded_files.txt", "a") as f:
            f.write(file_list + "\n")
        process_vector_space()
        st.success("File excluded successfully! ❌🚀")
        
    def include_file():
        with open("excluded_files.txt", "r") as f:
            excluded_files = f.read().splitlines()
        excluded_files.remove(file_list)
        with open("excluded_files.txt", "w") as f:
            for file in excluded_files:
                f.write(file + "\n")
        process_vector_space()
        st.success("File included successfully! ✅🚀")

    # Display the content of the selected file
    if file_list:
        col0, col1, col2 = st.columns([3, 4, 5])
        with col1:
            with stylable_container(
                "red",
                css_styles="""
                button {
                    background-color: #FF8F95;
                    color: black;
                }""",
            ):
                st.button("Delete File", on_click=delete_file)
        with col2:
            with stylable_container(
                "blue",
                css_styles="""
                button {
                    background-color: #BFDBF7;
                    color: black;
                }""",
            ):
                with open("excluded_files.txt", "r") as f:
                    excluded_files = f.read().splitlines()
                is_excluded = True if file_list in excluded_files else False
                if is_excluded:
                    st.button("Include File", on_click=include_file)
                else:
                    st.button("Exclude File", on_click=exclude_file)
        if file_list in os.listdir("pdf_files"):
            st.markdown(f"### {file_list}")
            st.markdown(f"#### Content:")
            pdf_viewer(f"pdf_files/{file_list}")
        else:
            with open(f"text_files/{file_list}", "r") as f:
                text = f.read()
            st.markdown(f"### {file_list}")
            st.markdown(f"#### Content:")
            st.write(text)
            
if __name__ == "__main__":
    main()
