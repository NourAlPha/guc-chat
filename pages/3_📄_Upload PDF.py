import streamlit as st
from utils import initialize_session_state, process_vector_space

# Initialize session state with needed variables
initialize_session_state()

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# Set page configuration including title and icon
st.set_page_config(page_title="Upload PDF",
                     page_icon="📄")

# Add a title to the page
st.title("📄 Upload PDF")

# Add a description to the page
st.markdown("Upload a PDF file to add its content to my knowledge base.")

# Add a file uploader to the page
uploaded_file = st.file_uploader(
    "Choose a PDF file", type="pdf",
    accept_multiple_files=True,
    help="Only PDF files are allowed.",
    key=f"uploader_{st.session_state.uploader_key}",
)

def process_pdf_files():
    if not uploaded_file:
        return
    for file in uploaded_file:
        with open(f"pdf_files/{file.name}", "wb") as f:
            f.write(file.getbuffer())
    process_vector_space()
    st.session_state.uploader_key += 1

if st.button("Save to Database", on_click=process_pdf_files):
    st.success("PDF file(s) uploaded successfully! 📄🚀")