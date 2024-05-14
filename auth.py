import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import streamlit as st
from typing import Optional
from streamlit_authenticator.utilities.exceptions import DeprecationError
import time

def login(authenticator, location: str='main', max_concurrent_users: Optional[int]=None,
              max_login_attempts: Optional[int]=None, fields: dict=None,
              clear_on_submit: bool=False) -> tuple:
        """
        Creates a login widget.

        Parameters
        ----------
        location: str
            Location of the login widget i.e. main or sidebar.
        max_concurrent_users: int
            Maximum number of users allowed to login concurrently.
        max_login_attempts: int
            Maximum number of failed login attempts a user can make.
        fields: dict
            Rendered names of the fields/buttons.
        clear_on_submit: bool
            Clear on submit setting, True: clears inputs on submit, False: keeps inputs on submit.

        Returns
        -------
        str
            Name of the authenticated user.
        bool
            Status of authentication, None: no credentials entered, 
            False: incorrect credentials, True: correct credentials.
        str
            Username of the authenticated user.
        """
        if fields is None:
            fields = {'Form name':'Login', 'Username':'Username', 'Password':'Password',
                      'Login':'Login'}
        if location not in ['main', 'sidebar']:
            # Temporary deprecation error to be displayed until a future release
            raise DeprecationError("""Likely deprecation error, the 'form_name' parameter has been
                                   replaced with the 'fields' parameter. For further information please 
                                   refer to 
                                   https://github.com/mkhorasani/Streamlit-Authenticator/tree/main?tab=readme-ov-file#authenticatelogin""")
            # raise ValueError("Location must be one of 'main' or 'sidebar'")
        if not st.session_state['authentication_status']:
            token = authenticator.cookie_handler.get_cookie()
            if token:
                authenticator.authentication_handler.execute_login(token=token)
            time.sleep(0.7)
            if not st.session_state['authentication_status']:
                if location == 'main':
                    login_form = st.form('Login', clear_on_submit=clear_on_submit)
                elif location == 'sidebar':
                    login_form = st.sidebar.form('Login')
                login_form.subheader('Login' if 'Form name' not in fields else fields['Form name'])
                username = login_form.text_input('Username' if 'Username' not in fields
                                                 else fields['Username']).lower()
                password = login_form.text_input('Password' if 'Password' not in fields
                                                 else fields['Password'], type='password')
                if login_form.form_submit_button('Login' if 'Login' not in fields
                                                 else fields['Login']):
                    if authenticator.authentication_handler.check_credentials(username,
                                                                     password,
                                                                     max_concurrent_users,
                                                                     max_login_attempts):
                        authenticator.authentication_handler.execute_login(username=username)
                        authenticator.cookie_handler.set_cookie()
                        st.rerun()
        return (st.session_state['name'], st.session_state['authentication_status'],
                st.session_state['username'])

def authenticate_admin():
    
    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
    )
    
    name, authentication_status, username = login(authenticator=authenticator)
    
    if authentication_status:
        authenticator.logout('Logout', 'sidebar')
    elif authentication_status == False:
        st.error('Username/password is incorrect')
    elif authentication_status == None:
        st.warning('Please enter your username and password')
    