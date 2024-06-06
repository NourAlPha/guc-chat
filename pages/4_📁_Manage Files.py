import streamlit as st
from utils import initialize_session_state, process_vector_space_level1, summarizeDocAndSave, process_vector_space_level2_rules
import os
from streamlit_pdf_viewer import pdf_viewer
from streamlit_extras.stylable_container import stylable_container
from auth import authenticate_admin

def main():
    # Set page configuration including title and icon
    st.set_page_config(page_title="Manage Files",
                    page_icon="📁")
    
    # Authenticate admin
    if "authentication_status" not in st.session_state or not st.session_state.authentication_status:
        authenticate_admin()
    else:
        st.session_state.authenticator.logout('Logout', 'sidebar')
    if not st.session_state.authentication_status:
        return

    # Initialize session state with needed variables
    initialize_session_state()
    
    if "rules" not in st.session_state:
        st.session_state.rules = False

    # Add a title to the page
    st.title("📁 Manage " + ("Content" if not st.session_state.rules else "Rules"))

    # Add a description to the page
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("Manage the files in my knowledge base.")
    with col2:
        if st.session_state.rules:
            st.button("Switch to Content", on_click=lambda: st.session_state.__setitem__("rules", False))
        else:
            st.button("Switch to Rules", on_click=lambda: st.session_state.__setitem__("rules", True))

    # Add a select box to the page to select from the pdf_files directory
    all_file_list = os.listdir("pdf_files") + os.listdir("text_files") if not st.session_state.rules else os.listdir("rules")
    file_list = st.selectbox("Select a file from the knowledge base:", all_file_list, placeholder="Select a file", index=None)

    def delete_file():
        if st.session_state.rules:
            if os.path.exists(f"rules/{file_list}"):
                os.remove(f"rules/{file_list}")
            process_vector_space_level2_rules()
            st.success("Rule deleted successfully! 🚫🚀")
            return
        if file_list in os.listdir("pdf_files"):
            if os.path.exists(f"pdf_files/{file_list}"):
                os.remove(f"pdf_files/{file_list}")
            if os.path.exists(f"summarized_files/{file_list[:-4] + '_pdf'}.txt"):
                os.remove(f"summarized_files/{file_list[:-4] + '_pdf'}.txt")
            if os.path.exists(f"faiss_index/{file_list[:-4] + '_pdf'}"):
                os.remove(f"faiss_index/{file_list[:-4] + '_pdf'}/index.faiss")
                os.remove(f"faiss_index/{file_list[:-4] + '_pdf'}/index.pkl")
                os.rmdir(f"faiss_index/{file_list[:-4] + '_pdf'}")
        else:
            if os.path.exists(f"text_files/{file_list}"):
                os.remove(f"text_files/{file_list}")
            if os.path.exists(f"summarized_files/{file_list[:-4] + '_txt'}.txt"):
                os.remove(f"summarized_files/{file_list[:-4] + '_txt'}.txt")
            if os.path.exists(f"faiss_index/{file_list[:-4] + '_txt'}"):
                os.remove(f"faiss_index/{file_list[:-4] + '_txt'}/index.faiss")
                os.remove(f"faiss_index/{file_list[:-4] + '_txt'}/index.pkl")
                os.rmdir(f"faiss_index/{file_list[:-4] + '_txt'}")
        process_vector_space_level1()
        st.success("File deleted successfully! 🚫🚀")

    def exclude_file():
        if st.session_state.rules:
            with open("excluded_files.txt", "a") as f:
                f.write(file_list[:-4] + ("_pdf.txt" if file_list[-4:] == ".pdf" else "_txt.txt") + "\n")
            process_vector_space_level2_rules()
            st.success("Rule excluded successfully! ❌🚀")
            return
        with open("excluded_files.txt", "a") as f:
            f.write(file_list[:-4] + ("_pdf.txt" if file_list[-4:] == ".pdf" else "_txt.txt") + "\n")
        process_vector_space_level1()
        st.success("File excluded successfully! ❌🚀")
        
    def include_file():
        if st.session_state.rules:
            with open("excluded_files.txt", "r") as f:
                excluded_files = f.read().splitlines()
            excluded_files.remove(file_list[:-4] + ("_pdf.txt" if file_list[-4:] == ".pdf" else "_txt.txt"))
            with open("excluded_files.txt", "w") as f:
                for file in excluded_files:
                    f.write(file + "\n")
            process_vector_space_level2_rules()
            st.success("Rule included successfully! ✅🚀")
            return
        with open("excluded_files.txt", "r") as f:
            excluded_files = f.read().splitlines()
        excluded_files.remove(file_list[:-4] + ("_pdf.txt" if file_list[-4:] == ".pdf" else "_txt.txt"))
        with open("excluded_files.txt", "w") as f:
            for file in excluded_files:
                f.write(file + "\n")
        process_vector_space_level1()
        st.success("File included successfully! ✅🚀")
    def save_changes(update_text):
        if st.session_state.rules:
            with open(f"rules/{file_list}", "w") as f:
                f.write(update_text)
            process_vector_space_level2_rules()
            st.success("Changes saved successfully! 📝🚀")
            return
        with open(f"text_files/{file_list}", "w") as f:
            f.write(update_text)
        summarizeDocAndSave(file_list)
        process_vector_space_level1()
        st.success("Changes saved successfully! 📝🚀")

    # Display the content of the selected file
    if file_list:
        if file_list in os.listdir("pdf_files"):
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
                    is_excluded = True if (file_list[:-4] + "_pdf.txt") in excluded_files else False
                    if is_excluded:
                        st.button("Include File", on_click=include_file)
                    else:
                        st.button("Exclude File", on_click=exclude_file)
            pdf_viewer(f"pdf_files/{file_list}")
        else:
            with open(f"text_files/{file_list}" if not st.session_state.rules else f"rules/{file_list}", "r") as f:
                text = f.read()
            update_text = st.text_area("File content: ", text, height=300)
            col0, col1, col2, col3 = st.columns([2, 5, 5, 6])
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
                    is_excluded = True if (file_list[:-4] + "_txt.txt") in excluded_files else False
                    if is_excluded:
                        st.button("Include File", on_click=include_file)
                    else:
                        st.button("Exclude File", on_click=exclude_file)
            with col3:
                with stylable_container(
                    "green",
                    css_styles="""
                    button {
                        background-color: #C3E88D;
                        color: black;
                    }""",
                ):
                    st.button("Save Changes", on_click=save_changes, args=(update_text,))
            
        
if __name__ == "__main__":
    main()
