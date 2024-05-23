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
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory

def get_pdf_file(pdf_file_path):
    # Create a PdfReader object for the specified PDF file
    pdf_reader = PdfReader(pdf_file_path)
    text = " "
    # Iterate through each page in the PDF document
    for page in pdf_reader.pages:
        # Extract text from the current page and append it to the 'text' string
        text += page.extract_text()
    return text

def get_text_file(text_file_path):
    # Open the specified text file and read its contents
    with open(text_file_path, "r") as f:
        text = f.read()
    return text

# The RecursiveCharacterTextSplitter takes a large text and splits it based on a specified chunk size.
def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

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
    
    with open(f"summarized_files/{file_name.split('.')[0] + ('_pdf' if file_name[-4:] == '.pdf' else '_txt')}.txt", "w") as f:
        f.write(f"DOCUMENT NAME: {file_name.split('.')[0] + ('_pdf' if file_name[-4:] == '.pdf' else '_txt')}.txt\n\n"
                + summary["output_text"])
    process_vector_space_level2(file_name)
    

def add_vector_store(text_chunks, filename):
    # Create a vector store using FAISS from the provided text chunks and embeddings
    vector_store = FAISS.from_texts(text_chunks, embedding=st.session_state.embeddings)

    # Save the vector store locally with the name "faiss_index"
    vector_store.save_local(f"./faiss_index/{filename.split('.')[0] + ('_pdf' if filename[-4:] == '.pdf' else '_txt')}")

def process_conversational_chain_docs():
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
    st.session_state.chain = load_qa_chain(llm=st.session_state.model, chain_type="stuff", prompt=prompt)

def process_relevant_docs():
    prompt = PromptTemplate(
        template="""
            Given a set of textfiles and a query, you are asked to find the most relevant documents to the given query.
            Return only the names of the most relevant documents as a python list.\n\n
            
            textfiles: {context}\n\n
            query: {question}
            
            relevant documents list: 
        """,
        input_variables=["question", "textfiles"]
    )
    
    st.session_state.chain2 = load_qa_chain(llm=st.session_state.model2, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    
    docs_to_search_str = st.session_state.chain2({
        "input_documents": st.session_state.docs,
        "question": user_question
    })["output_text"]
    
    print(docs_to_search_str)
    
    try:
        docs_to_search = eval(docs_to_search_str)
    except Exception as e:
        docs_to_search = []
        for i in range(len(docs_to_search_str)):
            for j in range(i, len(docs_to_search_str)):
                if docs_to_search_str[i:j+1] in os.listdir("summarized_files"):
                    docs_to_search.append(docs_to_search_str[i:j+1])
    
    print(docs_to_search)
    docs = []
    for file in docs_to_search:
        cur_db = FAISS.load_local(f"./faiss_index/{file.split('.')[0]}", st.session_state.embeddings, allow_dangerous_deserialization=True)
        docs.extend(cur_db.similarity_search(user_question))
    
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
        if not os.path.exists("pdf_files"):
            os.makedirs("pdf_files")
        if not os.path.exists("text_files"):
            os.makedirs("text_files")
        if not os.path.exists("summarized_files"):
            os.makedirs("summarized_files")
        st.session_state.messages = []
        safety_settings = {
            HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }
        st.session_state.model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.5, safety_settings=safety_settings)
        st.session_state.model2 = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0, safety_settings=safety_settings)
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        process_conversational_chain_docs()
        process_relevant_docs()
        process_vector_space_level1()
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) # Loads API key

def process_vector_space_level1():
    st.session_state.docs = []
    excluded_files = []
    with open("excluded_files.txt", "r") as f:
        excluded_files = f.read().splitlines()
    for file in os.listdir("summarized_files"):
        if file in excluded_files:
            continue
        loader = TextLoader(f"summarized_files/{file}")
        docs = loader.load_and_split()
        st.session_state.docs.extend(docs)
    
def process_vector_space_level2(filename):
    if filename[-4:] == ".pdf":
        text = get_pdf_file(f"pdf_files/{filename}")
    else:
        text = get_text_file(f"text_files/{filename}")
    chunks = get_chunks(text)
    add_vector_store(chunks, filename)