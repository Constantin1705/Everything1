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
            CREATE TABLE IF NOT EXISTS experiments (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                password TEXT,
                experiment_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                eps NUMERIC,
                original_image TEXT,
                mask TEXT,
                original_image_prediction NUMERIC,
                original_image_mask_prediction NUMERIC,
                real_label NUMERIC
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
    st.rerun()

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
            st.rerun()
    with col2:
        if st.button("Download"):
            st.write("Download Started!")  # Placeholder action
    
    # Displaying the experiments table
    st.write("Experiments Table (Old Trials)")
    st.table(df)

def run_experiment(data, model):
    # Implement your experiment logic here
    st.write("Experiment running...")
    # Example: Complete processing
    st.write("Experiment complete!")

# New Experiment page function
def new_experiment_page():
    create_header()
    st.write("New Experiment Page")
    
        
    
    # Column for loading data
 
    
    data_file = st.file_uploader("Choose a data file", key="data_loader")
    if data_file is not None:
        st.success("Data loaded successfully!")

    # Column for loading model
   
        
    model_file = st.file_uploader("Choose a model file", key="model_loader")
    if model_file is not None:
        st.success("Model loaded successfully!")

    # Column for running the experiment
    #Model 2gb
    if st.button("Run"):
        if 'data_file' in st.session_state and 'model_file' in st.session_state:
            run_experiment(st.session_state['data_file'], st.session_state['model_file'])
        else:
            st.error("Please load both data and model files before running.")
    st.text_input("Choose your gradient")
    # Displaying the experiment results table
    st.write("Experiment Results Table")
    data = {
    'original_image_url': [1, 2, 3],
    'mask_image_url': [101, 102, 103],
    'original_class': [0.01, 0.02, 0.03],
    'original_predicted_class': ['/path/to/image1.jpg', '/path/to/image2.jpg', '/path/to/image3.jpg'],
    'masked_image_predicted class': ['/path/to/mask1.jpg', '/path/to/mask2.jpg', '/path/to/mask3.jpg'],
    }
    df1 = pd.DataFrame(data)
    st.table(df1)  # Placeholder for displaying experiment results
    if st.button("Back to Test Page"):
        st.session_state['current_page'] = 'test_page'
        st.rerun()

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
                st.rerun()

    with tab2:
        st.header("Sign Up")
        new_username = st.text_input("Choose a username", key="signup_username")
        new_password = st.text_input("Choose a password", type="password", key="signup_password")
        if st.button('Sign Up'):
            result = sign_up_user(conn, new_username, new_password)
            st.success(result)
