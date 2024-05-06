import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import google.generativeai as genai
from langchain.prompts import PromptTemplate
import time

genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) # Loads API key

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.5)

def get_pdf_text(pdf_docs_path):
    # Get a list of all PDF documents in the specified folder
    pdf_docs = [os.path.join(pdf_docs_path, f) for f in os.listdir(pdf_docs_path) if f.endswith(".pdf")]
    
    text = " "
    # Iterate through each PDF document path in the list
    for pdf in pdf_docs:
        # Create a PdfReader object for the current PDF document
        pdf_reader = PdfReader(pdf)
        # Iterate through each page in the PDF document
        for page in pdf_reader.pages:
            # Extract text from the current page and append it to the 'text' string
            text += page.extract_text()

    # Return the concatenated text from all PDF documents
    return text

# The RecursiveCharacterTextSplitter takes a large text and splits it based on a specified chunk size.
def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):     
    # Create embeddings using a Google Generative AI model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Create a vector store using FAISS from the provided text chunks and embeddings
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    # Save the vector store locally with the name "faiss_index"
    vector_store.save_local("faiss_index")

def get_conversational_chain_docs():
    # Define a prompt template for asking questions based on a given context
    prompt_template = """    
    Given the chat history as an array of pair, the first element being the role and the second element being the content.
    You are in the middle of a chat with a human. Answer the question based on your powerful knowledge and the given context.\n\n
    
    Chat History:\n{chat_history}\n
    Context:\n{context}\n
    Question: \n{question}\n

    Answer:
    """

    # Create a prompt template with input variables "context" and "question"
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question", "chat_history"]
    )

    # Load a question-answering chain with the specified model and prompt
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    # Create embeddings for the user question using a Google Generative AI model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Load a FAISS vector database from a local file
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Perform similarity search in the vector database based on the user question
    docs = new_db.similarity_search(user_question)

    # Obtain a conversational question-answering chain
    chain = get_conversational_chain_docs()

    # Use the conversational chain to get a response based on the user question and retrieved documents
    response = chain(
        {
            "input_documents": docs,
            "question": user_question,
            "chat_history": st.session_state.messages
        },
        return_only_outputs=True)

    # Print the response to the console
    print(response["output_text"])
    
    # return response
    return response
    
# Function to generate output based on a query
def make_output(query):
    # Query the QA chain and extract the result
    answer = user_input(query)
    result = answer["output_text"]
    return result

# Function to modify the output by adding spaces between each word with a delay
def modify_output(input):
    # Iterate over each word in the input string
    for text in input.split(" "):
        # Yield the word with an added space
        yield text + " "
        # Introduce a small delay between each word
        time.sleep(0.05)

def main():
    raw_text = get_pdf_text("blogposts")
    text_chunks = get_chunks(raw_text)
    get_vector_store(text_chunks)

    # Set page configuration including title and icon
    st.set_page_config(page_title="ChatBot",
                    page_icon="🤔")
    # Display the title of the chat interface
    st.title("💭 Chat with GUC Bot")
    # Initialize session state to store chat messages if not already present
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Display previous chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    # Accept user input in the chat interface
    if prompt := st.chat_input("What is your question?"):
        # Display user input as a chat message
        with st.chat_message("user"):
            st.markdown(prompt)
        # Append user input to session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get response from the chatbot based on user input
        response = make_output(prompt)
        
        # Display response from the chatbot as a chat message
        with st.chat_message("assistant"):
            # Write response with modified output (if any)
            st.write_stream(modify_output(response))
        # Append chatbot response to session state
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()

