import streamlit as st
import pandas as pd
import sqlite3
from sqlite3 import Error

# Database connection and setup
def create_connection(db_file):
    """ Create a database connection to the SQLite database specified by db_file """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
    return conn

def setup_database(conn):
    """ Create table if not exists """
    try:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username text PRIMARY KEY,
                password text NOT NULL
            );
        ''')
        conn.commit()
    except Error as e:
        print(e)

# User authentication functions
def authenticate_user(conn, username, password):
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    rows = cur.fetchall()
    return len(rows) > 0

def sign_up_user(conn, username, password):
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return "User created successfully."
    except Error as e:
        return str(e)

def check_user_logged_in():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'username' not in st.session_state:
        st.session_state['username'] = ''

def login_user(username):
    st.session_state['logged_in'] = True
    st.session_state['username'] = username

def sign_out():
    st.session_state['logged_in'] = False
    st.session_state['username'] = ''
    st.session_state['current_page'] = None
    st.experimental_rerun()

def create_header():
    col1, col2 = st.columns([8, 2])
    with col1:
        st.write(f"Welcome, {st.session_state['username']}!")  # Customizable greeting
    with col2:
        if st.button("Sign Out"):
            sign_out()

# Test page function
def test_page():
    create_header()
    st.write("Welcome to the Test Page!")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("New"):
            st.session_state['current_page'] = 'new_experiment'
            st.experimental_rerun()
    with col2:
        if st.button("Download"):
            st.write("Download Started!")  # Placeholder action
    
    # Displaying the experiments table
    st.write("Experiments Table (Old Trials)")
    st.table(df)

# New Experiment page function
def new_experiment_page():
    create_header()
    st.write("New Experiment Page")
    
        
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Load Data"):
            st.write("Data loaded!")  # Placeholder for data loading
    with col2:
        if st.button("Load Model"):
            st.write("Model loaded!")  # Placeholder for model loading
    with col3:
        if st.button("Run"):
            st.write("Experiment running...")  # Placeholder for experiment run
    
    # Displaying the experiment results table
    st.write("Experiment Results Table")
    st.table(df)  # Placeholder for displaying experiment results
    if st.button("Back to Test Page"):
        st.session_state['current_page'] = 'test_page'
        st.experimental_rerun()

# Initialize database connection
conn = create_connection('auth.db')
setup_database(conn)

# Initialize data for the table
data = {
    'Experiment ID': [1, 2, 3],
    'Description': ['Trial 1', 'Trial 2', 'Trial 3'],
    'Result': ['Success', 'Failure', 'Success']
}
df = pd.DataFrame(data)

# Application routing
check_user_logged_in()
if st.session_state.get('logged_in', False):
    if st.session_state.get('current_page', '') == 'new_experiment':
        new_experiment_page()
    else:
        test_page()
else:
    st.title('Welcome to My App')
    tab1, tab2 = st.tabs(["Sign In", "Sign Up"])

    with tab1:
        st.header("Sign In")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button('Login'):
            if authenticate_user(conn, username, password):
                login_user(username)
                st.session_state['current_page'] = 'main'
                st.experimental_rerun()

    with tab2:
        st.header("Sign Up")
        new_username = st.text_input("Choose a username", key="signup_username")
        new_password = st.text_input("Choose a password", type="password", key="signup_password")
        if st.button('Sign Up'):
            result = sign_up_user(conn, new_username, new_password)
            st.success(result)
