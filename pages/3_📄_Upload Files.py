import streamlit as st
from utils import initialize_session_state, process_vector_space_level1, summarizeDocAndSave
from auth import authenticate_admin
import textract
import os

def main():
    # Set page configuration including title and icon
    st.set_page_config(page_title="Upload PDF",
                        page_icon="📄")
    
    # Authenticate admin
    if "authentication_status" not in st.session_state or not st.session_state.authentication_status:
        authenticate_admin()
    else:
        st.session_state.authenticator.logout('Logout', 'sidebar')
    if not st.session_state.authentication_status:
        return

    # Initialize session state with needed variables
    initialize_session_state()

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    # Add a title to the page
    st.title("📄 Upload Files")

    # Add a description to the page
    st.markdown("Upload file(s) to add content to my knowledge base.")

    # Add a file uploader to the page
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf", "doc", "docx", "txt", "xls", "xlsx"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )

    def process_files():
        if not uploaded_file:
            return
        for file in uploaded_file:
            if file.type == "application/pdf":
                process_pdf_files(file)
            elif file.type == "application/msword" or file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                process_textract_files(file)
            elif file.type == "text/plain":
                process_text_files(file)
            elif file.type == "application/vnd.ms-excel" or file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                process_textract_files(file)
            else:
                st.warning(f"Unsupported file type: {file.type}")
            summarizeDocAndSave(file.name.split(".")[0] + ("_pdf" if file.type == "application/pdf" else "_txt"))
        process_vector_space_level1()
        st.session_state.uploader_key += 1
    
    def process_pdf_files(file):
        if not uploaded_file:
            return
        with open(f"pdf_files/{file.name}", "wb") as f:
            f.write(file.getbuffer())
    
    def process_textract_files(file):
        if not uploaded_file:
            return
        if not os.path.exists("temp_files"):
            os.makedirs("temp_files")
        with open(f"temp_files/{file.name}", "wb") as f:
            f.write(file.getbuffer())
        text = textract.process(f"temp_files/{file.name}")
        file_name = file.name.split(".")[0]
        with open(f"text_files/{file_name}.txt", "w") as f:
            f.write(text.decode("utf-8"))
        os.remove(f"temp_files/{file.name}")      
    
    def process_text_files(file):
        if not uploaded_file:
            return
        with open(f"text_files/{file.name}", "w") as f:
            f.write(file.getvalue().decode("utf-8"))
    
    disable_button = uploaded_file == []
    if st.button("Save to Database", on_click=process_files, disabled=disable_button):
        st.success("PDF file(s) uploaded successfully! 📄🚀")

if __name__ == "__main__":
    main()