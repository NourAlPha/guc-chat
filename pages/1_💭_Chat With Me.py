import streamlit as st
from utils import make_output, modify_output, initialize_session_state
from streamlit_feedback import streamlit_feedback
import json

# Initialize session state with needed variables
initialize_session_state()

score_emoji_map = {
    "😀": "Great",
    "🙂": "Good",
    "😐": "Average",
    "🙁": "Poor",
    "😞": "Very Poor",
}

def _submit_feedback(user_feedback):
    feedbacks = json.load(open("feedback.json"))
    feedbacks[score_emoji_map[user_feedback["score"]]] += 1
    with open("feedback.json", "w") as f:
        json.dump(feedbacks, f)

def main():
    # Set page configuration including title and icon
    st.set_page_config(page_title="Chat",
                    page_icon="🤔")
    # Display the title of the chat interface
    st.title("💭 What's in your mind?")
    # Add a refresh button to clear the chat interface
    if st.sidebar.button("🔄 Refresh"):
        st.session_state.messages = []
    # Display previous chat messages
    for n, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                feedback_key = f"feedback_{int(n/2)}"
                if feedback_key not in st.session_state:
                    st.session_state[feedback_key] = None
                streamlit_feedback(
                    feedback_type="faces",
                    key=feedback_key,
                    on_submit=_submit_feedback,
                    align="center",
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
            # Write response with modified output (if any)
            st.write_stream(modify_output(response))
        # Append chatbot response to session state
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()


if __name__ == "__main__":
    main()

