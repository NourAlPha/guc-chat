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

def main():
    # Set page configuration including title and icon
    st.set_page_config(page_title="Chat",
                    page_icon="🤔")
    # Display the title of the chat interface
    st.title("💭 What's in your mind?")
    # Add a refresh button to clear the chat interface
    if st.sidebar.button("🆕 New Chat"):
        for n in range(int(len(st.session_state.messages)/2)):
            feedback_key = f"feedback_{n}"
            st.session_state["_" + feedback_key] = None
        st.session_state.messages = []
    # Display previous chat messages
    for n, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                feedback_key = f"feedback_{int(n/2)}"
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
    if prompt := st.chat_input("💭 What's in your mind?"):
        # Display user input as a chat message
        with st.chat_message("user"):
            st.markdown(prompt)
        # Append user input to session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get response from the chatbot based on user input
        response = make_output(prompt)
        
        # Display response from the chatbot as a chat message
        with st.chat_message("assistant"):
            # Append chatbot response to session state
            st.session_state.messages.append({"role": "assistant", "content": response})
            # Write response with modified output (if any)
            st.write_stream(modify_output(response))

        st.rerun()


if __name__ == "__main__":
    main()

