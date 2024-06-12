import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import google.generativeai as genai
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory
from langchain_core.messages import HumanMessage, AIMessage

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
    "{context}"
    CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)
    
    stuff_chain = load_qa_chain(llm=st.session_state.model2, chain_type="stuff", prompt=prompt)
        
    summary = stuff_chain({
        "input_documents": docs,
    })
    
    with open(f"summarized_files/{file_name[:-4] + ('_pdf' if file_name[-4:] == '.pdf' else '_txt')}.txt", "w") as f:
        f.write(f"DOCUMENT NAME: {file_name[:-4] + ('_pdf' if file_name[-4:] == '.pdf' else '_txt')}.txt\n\n"
                + summary["output_text"])
    process_vector_space_level2(file_name)
    

def add_vector_store(text_chunks, filename):
    # Create a vector store using FAISS from the provided text chunks and embeddings
    vector_store = FAISS.from_texts(text_chunks, embedding=st.session_state.embeddings)

    # Save the vector store locally with the name "faiss_index"
    vector_store.save_local(f"./faiss_index/{filename[:-4] + ('_pdf' if filename[-4:] == '.pdf' else '_txt')}")

def process_conversational_chain_docs():
    # Define a prompt template for asking questions based on a given context
    prompt_template = """    
    You are a chat assistant bot for helping students in university named German University in Cairo (GUC). \
    Use the following pieces of retrieved context and rules only to formulate a single detailed answer for the list of questions given. \
    If you cannot formulate an answer from the given retrieved context and rules, tell the user to ask inside the GUC scope in a chatty way. \
    
    Context:\n{context}\n
    Rules:\n{rules}\n
    Questions:\n{questions}\n
    """

    # Create a prompt template with input variables "context" and "question"
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "rules", "questions"]
    )

    # Load a question-answering chain with the specified model and prompt
    st.session_state.chain = load_qa_chain(llm=st.session_state.model, chain_type="stuff", prompt=prompt)

def generate_query_based_on_chat_history(question):
    contextualize_q_system_prompt = """
    Given a chat history and the latest user question which might reference context in the chat history, \
    formulate a standalone question which can be understood without the chat history. \
    If the user question is greeting or thanking, return it as is. \
    Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    chat_history = []
    for message in st.session_state.messages:
        if message["role"] == "user":
            chat_history.append(HumanMessage(content=message["content"]))
        else:
            chat_history.append(AIMessage(content=message["content"]))
    chat_history = chat_history[:-1]
        
    query = contextualize_q_prompt.format(chat_history=chat_history, input=question)
    
    all = st.session_state.model2.invoke(query).content
            
    all = all.split("\n")
    for i in range(len(all)):
        all[i] = all[i].strip()
    all = list(set(all))
    all = [x for x in all if x != ""]
    return all

def generate_query_based_on_context(query, context):
    contextualize_q_system_prompt = """
    Given context and the latest user question which might reference the context given, formulate a five different \
    versions of the given user question to retrieve relevant documents from a vector database. \
    By generating multiple perspectives on the user question, your goal is to help \
    the user overcome some of the limitations of the distance-based similarity search. \
    If the user question is greeting or thanking, return it as is. \
    Provide these alternative questions separated by newlines. \
    Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
    """
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("context"),
            ("human", "{input}"),
        ]
    )
    
    query = contextualize_q_prompt.format(input=query, context=context)
    
    all = st.session_state.model2.invoke(query).content
        
    all = all.split("\n")
    for i in range(len(all)):
        all[i] = all[i].strip()
        if len(all[i]) > 0 and all[i][0] == '-':
            all[i] = all[i][1:]
        all[i] = all[i].strip()
    all = list(set(all))
    all = [x for x in all if x != ""]
    return all
    
def process_relevant_docs():
    prompt = PromptTemplate(
        template="""
            Given textfiles and user question which might reference context in the textfiles,\
            return a list of most relevant documents' names. Do NOT answer the question,\
            just return a list of most relevant documents' names otherwise say no relevant documents' names.\n\n
            textfiles:\n{context}\n\n
            question:\n{question}\n\n
            
            relevant documents list: 
        """,
        input_variables=["question", "textfiles"]
    )
    
    st.session_state.chain2 = load_qa_chain(llm=st.session_state.model2, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    
    try:
        questions = generate_query_based_on_chat_history(user_question)
        if len(questions) > 0:
            user_question = questions[0]    

        docs_to_search_str = st.session_state.chain2({
            "input_documents": st.session_state.docs,
            "question": user_question,
        })["output_text"]
        
        try:
            docs_to_search = eval(docs_to_search_str)
        except Exception as e:
            docs_to_search = []
            for i in range(len(docs_to_search_str)):
                for j in range(i, len(docs_to_search_str)):
                    if docs_to_search_str[i:j+1] in os.listdir("summarized_files"):
                        docs_to_search.append(docs_to_search_str[i:j+1])
        
        content_db = []
        rules_db = []
        all_context = []
        for file in docs_to_search:
            content_db.append(FAISS.load_local(f"./faiss_index/{file[:-4]}", st.session_state.embeddings, allow_dangerous_deserialization=True))
            for doc in list(content_db[-1].docstore._dict.values()):
                all_context.append(doc.page_content)
        if os.path.exists(f"./faiss_index/rules"):
            rules_db = FAISS.load_local(f"./faiss_index/rules", st.session_state.embeddings, allow_dangerous_deserialization=True)
            for doc in list(rules_db.docstore._dict.values()):
                all_context.append(doc.page_content)
        
        new_queries = generate_query_based_on_context(user_question, all_context)
        
        docs = []
        rules = []
        for query in new_queries:
            for db in content_db:
                cur_search = db.similarity_search(query)
                for doc in cur_search:
                    if doc not in docs:
                        docs.append(doc)
            if os.path.exists(f"./faiss_index/rules"):
                cur_search = rules_db.similarity_search(query)
                for doc in cur_search:
                    if doc not in rules:
                        rules.append(doc)
        
        # Use the conversational chain to get a response based on the user question and retrieved documents
        response = st.session_state.chain(
            {
                "input_documents": docs,
                "rules": rules,
                "questions": new_queries,
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
        if not os.path.exists("faiss_index"):
            os.makedirs("faiss_index")
        if not os.path.exists("rules"):
            os.makedirs("rules")
        st.session_state.messages = []
        safety_settings = {
            HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }
        st.session_state.model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, safety_settings=safety_settings)
        st.session_state.model2 = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, safety_settings=safety_settings)
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
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
    
def process_vector_space_level2_rules():
    if os.path.exists(f"./faiss_index/rules"):
        os.remove(f"./faiss_index/rules/index.faiss")
        os.remove(f"./faiss_index/rules/index.pkl")
        os.rmdir(f"./faiss_index/rules")
    excluded_files = []
    with open("excluded_files.txt", "r") as f:
        excluded_files = f.read().splitlines()
    dir = os.listdir("rules")
    chunks = []
    for file in dir:
        if (file[:-4] + "_txt.txt") in excluded_files:
            continue
        text = get_text_file(f"rules/{file}")
        chunks.append(text)
    if len(chunks) == 0:
        return
    vector_store = FAISS.from_texts(chunks, embedding=st.session_state.embeddings)
    vector_store.save_local(f"./faiss_index/rules")
        