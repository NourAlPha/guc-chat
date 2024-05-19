import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai

def get_pdf_to_text(pdf_docs_path):
    # Get a list of all PDF documents in the specified folder
    pdf_docs = [os.path.join(pdf_docs_path, f) for f in os.listdir(pdf_docs_path) if f.endswith(".pdf")]
    
    # Get Excluded Files
    with open("excluded_files.txt", "r") as f:
        excluded_files = f.read().splitlines()
    
    text = " "
    # Iterate through each PDF document path in the list
    for pdf in pdf_docs:
        if(pdf.split("/")[1] in excluded_files):
            continue
        # Create a PdfReader object for the current PDF document
        pdf_reader = PdfReader(pdf)
        # Iterate through each page in the PDF document
        for page in pdf_reader.pages:
            # Extract text from the current page and append it to the 'text' string
            text += page.extract_text()

    # Return the concatenated text from all PDF documents
    return text

def get_pdf_file(pdf_file_path):
    # Create a PdfReader object for the specified PDF file
    pdf_reader = PdfReader(pdf_file_path)
    text = " "
    # Iterate through each page in the PDF document
    for page in pdf_reader.pages:
        # Extract text from the current page and append it to the 'text' string
        text += page.extract_text()
    return text

def get_text_files(text_files_path):
    # Get a list of all text files in the specified folder
    text_files = [os.path.join(text_files_path, f) for f in os.listdir(text_files_path) if f.endswith(".txt")]
    
    # Get Excluded Files
    with open("excluded_files.txt", "r") as f:
        excluded_files = f.read().splitlines()
    
    text = " "
    # Iterate through each text file path in the list
    for file in text_files:
        if(file.split("/")[1] in excluded_files):
            continue
        # Open the current text file and read its contents
        with open(file, "r") as f:
            text += f.read()

    # Return the concatenated text from all text files
    return text

def get_text_file(text_file_path):
    # Open the specified text file and read its contents
    with open(text_file_path, "r") as f:
        text = f.read()
    return text

# The RecursiveCharacterTextSplitter takes a large text and splits it based on a specified chunk size.
def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_summarized_list():
    files = os.listdir("summarized_files")
    docs = []
    for file in files:
        with open(f"summarized_files/{file}", "r") as f:
            docs.append(Document(page_content=f.read(), metadata={"source": file}))

    db = FAISS.from_documents(docs, st.session_state.embeddings)
    db.save_local("./faiss_index/summarized_files")
    return db

def summarizeDocAndSave(file_name):
    if file_name[-4:] == ".pdf":
        loader = PyPDFLoader(f"pdf_files/{file_name}")
    else:
        loader = TextLoader(f"text_files/{file_name}")
    docs = loader.load_and_split()
    
    prompt_template = """Write a concise summary in english of the following:
    "{text}"
    CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)
    
    llm_chain = LLMChain(llm=st.session_state.model, prompt=prompt)
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
    
    summary = stuff_chain.invoke(docs)
    print(summary)
    
    with open(f"summarized_files/{file_name.split('.')[0] + ('_pdf' if file_name[-4:] == '.pdf' else '_txt')}.txt", "w") as f:
        f.write(summary["output_text"])
    

def add_vector_store(text_chunks, filename):
    # Create a vector store using FAISS from the provided text chunks and embeddings
    vector_store = FAISS.from_texts(text_chunks, embedding=st.session_state.embeddings)

    # Save the vector store locally with the name "faiss_index"
    vector_store.save_local(f"./faiss_index/{filename.split('.')[0] + ('_pdf' if filename[-4:] == '.pdf' else '_txt')}")

def get_conversational_chain_docs():
    # Define a prompt template for asking questions based on a given context
    prompt_template = """    
    Given the chat history as an array of pair, the first element being the role and the second element being the content.
    You are in the middle of a chat with a human. Answer the question based on your powerful knowledge and the given context.\n\n
    
    Chat History:\n{chat_history}\n
    Context:\n{context}\n
    Question: \n{question}\n
    """

    # Create a prompt template with input variables "context" and "question"
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question", "chat_history"]
    )

    # Load a question-answering chain with the specified model and prompt
    chain = load_qa_chain(llm=st.session_state.model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    
    db = st.session_state.db
    result_doc = db.similarity_search(user_question, k=4)
    docs_to_search = [doc.metadata["source"] for doc in result_doc]
    all_result = []
    for file in docs_to_search:
        cur_db = FAISS.load_local(f"./faiss_index/{file.split('.')[0] + ('_pdf' if file[-4:] == '.pdf' else '_txt')}", st.session_state.embeddings, allow_dangerous_deserialization=True)
        all_result.extend(cur_db.similarity_search_with_score(user_question, k=4))
    all_result.sort(key=lambda x: x[1])
    all_result = all_result[:4]
    docs = [result[0] for result in all_result]    
    
    try:
        # Use the conversational chain to get a response based on the user question and retrieved documents
        response = st.session_state.chain(
            {
                "input_documents": docs,
                "question": user_question,
                "chat_history": st.session_state.messages
            },
            return_only_outputs=True)["output_text"]
    except Exception as e:
        try:
            response = st.session_state.model.invoke(user_question).content
        except Exception as e:
            response = "I'm sorry, I don't have an answer to that question."
            
    # return response
    return response
    
# Function to generate output based on a query
def make_output(query):
    # Query the QA chain and extract the result
    result = user_input(query)
    return result

# Function to modify the output by adding spaces between each word with a delay
def modify_output(input):
    # Iterate over each word in the input string
    for text in input.split(" "):
        # Yield the word with an added space
        yield text + " "
        # Introduce a small delay between each word
        time.sleep(0.05)
        
def initialize_session_state():
    # Initialize session state with needed variables
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.5)
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.chain = get_conversational_chain_docs()
        st.session_state.db = FAISS.load_local("./faiss_index/summarized_files", st.session_state.embeddings, allow_dangerous_deserialization=True)
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) # Loads API key

def process_vector_space_level1():
    st.session_state.db = get_summarized_list()
    
def process_vector_space_level2(filename):
    if filename[-4:] == ".pdf":
        text = get_pdf_file(f"pdf_files/{filename}")
    else:
        text = get_text_file(f"text_files/{filename}")
    chunks = get_chunks(text)
    add_vector_store(chunks, filename)