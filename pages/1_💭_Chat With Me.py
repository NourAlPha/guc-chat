import streamlit as st
from utils import make_output, modify_output, initialize_session_state
from streamlit_feedback import streamlit_feedback
import json
import functools

# Initialize session state with needed variables
initialize_session_state()

score_emoji_map = {
    "😀": "Great",
    "🙂": "Good",
    "😐": "Average",
    "🙁": "Poor",
    "😞": "Very Poor",
}

def _submit_feedback(user_feedback, feedback_key):
    feedbacks = json.load(open("feedback.json"))
    feedbacks[score_emoji_map[user_feedback["score"]]] += 1
    with open("feedback.json", "w") as f:
        json.dump(feedbacks, f)
    st.session_state["_" + feedback_key] = user_feedback
    
def insert_new_id():
    st.session_state.id_counter += 1
    st.session_state.id_list.append(st.session_state.id_counter)

def delete_from_session_state(key):
    if key in st.session_state:
        del st.session_state[key]

def disable_buttons():
    st.session_state.making_output = True

def main():
    # Set page configuration including title and icon
    st.set_page_config(page_title="Chat",
                    page_icon="🤔")
    # Display the title of the chat interface
    st.title("💭 What's in your mind?")
    
    col1, col2 = st.sidebar.columns([1, 1])
    
    if "regenerating" not in st.session_state:
        st.session_state.regenerating = False
        st.session_state.making_output = False
        st.session_state.messages = []
        st.session_state.id_list = []
        st.session_state.id_counter = 0
    
    # Add a refresh button to clear the chat interface
    if col1.button("🆕 New Chat", disabled=st.session_state.making_output):
        for id in st.session_state.id_list:
            feedback_key = f"feedback_{id}"
            delete_from_session_state(feedback_key)
            delete_from_session_state("_" + feedback_key)
            delete_from_session_state("feedback_submitted_" + feedback_key)
        st.session_state.id_list = []
        st.session_state.messages = []
        
    if col2.button("🔄 Regenerate", on_click=disable_buttons, disabled=st.session_state.making_output):
        if len(st.session_state.id_list) > 0:
            feedback_key = f"feedback_{st.session_state.id_list[-1]}"
            delete_from_session_state(feedback_key)
            delete_from_session_state("_" + feedback_key)
            delete_from_session_state("feedback_submitted_" + feedback_key)
            st.session_state.messages = st.session_state.messages[:-1]
            st.session_state.id_list = st.session_state.id_list[:-1]
            st.session_state.regenerating = True
    
    # Display previous chat messages
    for n, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                if int(n/2) >= len(st.session_state.id_list):
                    insert_new_id()
                feedback_key = f"feedback_{st.session_state.id_list[int(n/2)]}"
                if ("_" + feedback_key) in st.session_state:
                    st.session_state[feedback_key] = st.session_state["_" + feedback_key]
                if feedback_key not in st.session_state:
                    st.session_state[feedback_key] = None
                _submit_feedback_partial = functools.partial(_submit_feedback, feedback_key=feedback_key)
                streamlit_feedback(
                    feedback_type="faces",
                    key=feedback_key,
                    on_submit=_submit_feedback_partial,
                    align="center",
                    disable_with_score=st.session_state[feedback_key]["score"] if st.session_state[feedback_key] else None,
                )
    # Accept user input in the chat interface
    if prompt := st.chat_input("💭 What's in your mind?", on_submit=disable_buttons) or st.session_state.regenerating:
        # Display user input as a chat message
        if not st.session_state.regenerating:
            with st.chat_message("user"):
                st.markdown(prompt)
            # Append user input to session state
            st.session_state.messages.append({"role": "user", "content": prompt})
        
        if st.session_state.regenerating:
            prompt = st.session_state.messages[-1]["content"]
        
        # Get response from the chatbot based on user input
        response = make_output(prompt)
        
        # Display response from the chatbot as a chat message
        with st.chat_message("assistant"):
            # Append chatbot response to session state
            st.session_state.messages.append({"role": "assistant", "content": response})
            # Write response with modified output (if any)
            st.write_stream(modify_output(response))
        
        st.session_state.regenerating = False
        st.session_state.making_output = False
        st.rerun()


if __name__ == "__main__":
    main()

