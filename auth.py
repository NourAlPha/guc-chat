import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import streamlit as st

def authenticate_admin():
    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
    )

    name, authentication_status, username = authenticator.login()

    if authentication_status:
        authenticator.logout('Logout', 'main')
        return True
    elif authentication_status == False:
        st.error('Username/password is incorrect')
        return False
    elif authentication_status == None:
        st.warning('Please enter your username and password')
        return False
    
    