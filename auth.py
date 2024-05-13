import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import streamlit as st

def authenticate_admin():
    
    if "authentication_status" not in st.session_state:
        st.session_state.authentication_status = False
    
    if st.session_state.authentication_status:
        st.session_state.authenticator.logout('Logout', 'sidebar')
        return
    
    if("config" not in st.session_state):
        with open("config.yaml", "r") as f:
            st.session_state.config = yaml.load(f, Loader=SafeLoader)

    if "authenticator" not in st.session_state:
        st.session_state.authenticator = stauth.Authenticate(
            st.session_state.config['credentials'],
            st.session_state.config['cookie']['name'],
            st.session_state.config['cookie']['key'],
            st.session_state.config['cookie']['expiry_days'],
        )
    
    name, authentication_status, username = st.session_state.authenticator.login()

    if authentication_status:
        st.session_state.authenticator.logout('Logout', 'sidebar')
        st.rerun()
    elif authentication_status == False:
        st.error('Username/password is incorrect')
    elif authentication_status == None:
        st.warning('Please enter your username and password')
    