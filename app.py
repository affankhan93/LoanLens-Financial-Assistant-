import os 
import sys
import joblib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client
from streamlit_float import float_init 

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "Supa_database"))
from Supa_database.Authentication import login, signup 
from Orchestration import route_query 

load_dotenv() 

# Web-site color combination. 
COLOR_HEADER = "#8B9A6E"
COLOR_MAIN = "#F7F2EB"
COLOR_CHATBOT = "#EAE2D6"
COLOR_SIDEBAR = "#EEEEEE"
COLOR_SIGNUP = "#0D1C42"


st.set_page_config(
    page_title="LoanLens", 
    page_icon="🏦",
    layout='wide',
    initial_sidebar_state='expanded' 
)
float_init() 


# Authentication with supabase
supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_API_KEY'))

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def render_auth_page():
    st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {COLOR_SIGNUP}; }}

    .st-key-auth_card {{
        background-color: white;
        border-radius: 16px;
        padding: 2.5rem;
        max-width: 420px;
        margin: 4rem auto;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }}

    .auth-title {{
        color: {COLOR_SIGNUP};
        text-align: center;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }}

    .auth-subtitle {{
        text-align: center;
        color: #666;
        margin-bottom: 1.5rem;
    }}

    .st-key-auth_card [data-testid="stTab"] p {{
        color: #2C2C2A !important;
    }}
    .st-key-auth_card [data-testid="stTab"][aria-selected="true"] p {{
        color: {COLOR_SIGNUP} !important;
    }}

    /* Widget labels (Username / Password / Confirm password) */
    .st-key-auth_card label,
    .st-key-auth_card label * {{
        color: #2C2C2A !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

    card = st.container(key="auth_card")
    with card:
        st.markdown('<div class="auth-title">\U0001F4B0 LoanLens</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="auth-subtitle">Your loan eligibility &amp; financial Q&amp;A assistant</div>',
            unsafe_allow_html=True,
        )

        tab_login, tab_signup = st.tabs(['Log In', 'Sign Up'])

        with tab_login:
            with st.form("login_form"):
                username = st.text_input("Username", key='login_username')
                password = st.text_input("Password", type='password', key='login_pw')
                submitted = st.form_submit_button("Log In", use_container_width=True)
                if submitted:
                    success, message = login(username, password)
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.rerun()
                    else:
                        st.error(message)

        with tab_signup:
            with st.form("signup_form"):
                username = st.text_input("Username", key='signup_username')
                password = st.text_input("Password", type='password', key='signup_pw')
                confirm = st.text_input("Confirm password", type='password', key='signup_confirm')

                submitted = st.form_submit_button("Create Account", use_container_width=True)
                if submitted:
                    if password != confirm:
                        st.error("Passwords don't match.")
                    else:
                        success, message = signup(username, password)
                        if success:
                            st.success(f"{message} You can now log in.")
                        else:
                            st.error(message)


# ML Model 
@st.cache_resource
def load_artifact():
    return joblib.load(os.path.join(os.path.dirname(__file__), 'ML_Model', 'model_artifacts.pkl'))

ARTIFACT = load_artifact()
MODEL = ARTIFACT['model']
SCALER = ARTIFACT['scaler']
CATEGORIES = ARTIFACT['categories']

EXPECTED_COLUMNS = ARTIFACT["features_columns"]

CATEGORIES_COLUMNS = ['Marital_Status', 'Employment_Status', 'Loan_Purpose', 
                      'Education_Level', 'Gender', 'Employer_Category', 'Property_Area']



def build_features_row(inputs: dict)-> pd.DataFrame:
    row = {col: 0.0 for col in EXPECTED_COLUMNS}

    for col in ['Applicant_Income', 'Coapplicant_Income', 'Age', 'Dependents', 'Credit_Score',
                'Existing_Loans', 'DTI_Ratio', 'Savings', 'Collateral_Value', 'Loan_Amount', 'Loan_Term']:
        row[col] = inputs[col]

    row['Total_Income'] = inputs['Applicant_Income'] + inputs['Coapplicant_Income']
    row['Montly_Installments'] = inputs['Loan_Amount'] / inputs['Loan_Term']
    row['Collateral_Loan_Ratio'] = inputs['Collateral_Value'] / inputs['Loan_Amount']

    for col in CATEGORIES_COLUMNS:
        dummpy_col = f"{col}_{inputs[col]}"
        if dummpy_col in row:
            row[dummpy_col] = 1.0 # baseline category stays all-zero, matching drop = 'first

    return pd.DataFrame([row])[EXPECTED_COLUMNS]


# Loan form 
def render_loan_form():
    st.subheader("Check Your Loan Eligibility")
    with st.form("loan_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            applicant_income = st.number_input("Appicant Income", min_value = 0.0, value=15000.0)
            coapplicant_income = st.number_input("Co-applicant Income", min_value= 0.0, value = 0.0)
            age = st.number_input("Age", min_value=18, max_value=65, value = 25)
            dependents = st.number_input("Dependents", min_value = 0, max_value = 10, value = 0)

        with c2:
            credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value = 650)
            existing_loans = st.number_input("Existing Loans", min_value=0, max_value=10, value = 0)
            dti_ratio = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3)
            savings = st.number_input("Savings ", min_value=0.0, value=10000.0)

        with c3:
            collateral_value = st.number_input("Collateral Value", min_value=1.0, value = 20000.0)
            loan_amount = st.number_input("Loan Amount", min_value=1.0, value=15000.0)
            loan_term = st.number_input("Loan Term (months)", min_value=1.0, value=36.0)

        st.markdown("------")

        c4, c5, c6, c7 = st.columns(4)
        with c4:
            marital_status = st.selectbox("Marital Status", CATEGORIES['Marital_Status'])
            employment_status = st.selectbox("Employment Status", CATEGORIES['Employment_Status'])

        with c5:
            loan_purpose = st.selectbox("Loan Purpose", CATEGORIES['Loan_Purpose'])
            education_level = st.selectbox("Education Level", CATEGORIES['Education_Level'])

        with c6:
            gender = st.selectbox("Gender", CATEGORIES['Gender'])
            employer_category = st.selectbox("Employer Category", CATEGORIES["Employer_Category"])

        with c7:
            property_area = st.selectbox("Proper Area", CATEGORIES['Property_Area'])

        submitted = st.form_submit_button("Check Eligibility", use_container_width=True)

    if submitted:
        inputs = {
            "Applicant_Income" : applicant_income,
            "Coapplicant_Income" : coapplicant_income,
            "Age" : age,
            "Dependents":dependents,
            "Credit_Score" : credit_score,
            "Existing_Loans" : existing_loans,
            "DTI_Ratio" : dti_ratio,
            "Savings": savings,
            "Collateral_Value" : collateral_value,
            "Loan_Amount" : loan_amount,
            "Loan_Term" : loan_term,
            "Marital_Status" : marital_status,
            "Employment_Status": employment_status,
            "Loan_Purpose" : loan_purpose,
            "Education_Level" : education_level,
            "Gender" : gender,
            "Employer_Category": employer_category,
            "Property_Area" : property_area,
        }

        row = build_features_row(inputs)
        scaled = SCALER.transform(row)
        prediction = MODEL.predict(scaled)[0]
        proba = MODEL.predict_proba(scaled)[0]
        confidence = proba[int(prediction)] * 100

        st.markdown("------")

        result_col, detail_col = st.columns([1, 1])
        with result_col:
            if prediction == 1:
                st.success(f"✅ Loan Likely Approved -Confidence: {confidence:.1f}%")
            else:
                st.error(f"❌ Loan Likely Not Approved - Confidence: {confidence:.1f}%")

        with detail_col:
            st.metric("Total Income", f"{row['Total_Income'].iloc[0]:,.0f}")
            st.metric("Monthly Installments", f"{row['Montly_Installments'].iloc[0]:,.2f}")
            st.metric("Collateral / Loan Ratio", f"{row['Collateral_Loan_Ratio'].iloc[0]:.2f}") 


# Sidebar
def render_sidebar():
    with st.sidebar:
        st.markdown("<style>[data-testid='stSidebar'] * {color: black !important;}</style>", unsafe_allow_html=True)
        st.markdown("<style>[data-testid='stSidebar'] button p {color: white !important;}</style>", unsafe_allow_html=True)       
        st.markdown(f"<h2 style='background-color: {COLOR_SIDEBAR};'> LoanLens</h2>", unsafe_allow_html=True)
        st.markdown(f"**Logged in as:** {st.session_state.username}")
        if st.button("Log Out", use_container_width=True):
            supabase.auth.sign_out()
            st.session_state.authenticated = False
            st.rerun()

        st.markdown("-----")
        st.markdown("### How to use this app")
        st.markdown(
            "1. Fill in your details in the **Loan Eligibility** form.\n"
            "2. Click **Check Eligibility** to see your result and confidence score.\n"
            "3. Click the chat bubble (bottom-right) to ask any financial question."
        )
        st.markdown("### What can the Chatbot do?")
        st.markdown(
            "- Answers from our curated financial knowledge base first (with sources shown).\n"
            "- Falls back to a live web search when the knowledge base doesn't have your answer.\n"
            "- Great for questions on loans, savings, investing, financial metrices such as debt-to-income ratio and more."
        )


# chatbot (RAG+MCP)
def render_chatbot():
    toggle_container = st.container(key="chat_toggle_container")
    with toggle_container:
        icon = "\u2715" if st.session_state.chat_open else "\U0001F4AC"
        if st.button(icon, key="chat_toggle_btn"):
            st.session_state.chat_open = not st.session_state.chat_open
            st.rerun()
    toggle_container.float("bottom: 2rem; right: 2rem; z-index: 999;")
    st.markdown(
    f"""
    <style>
    .st-key-chat_toggle_container button {{
        position: fixed !important;
        width: 60px !important;
        height: 60px !important;
        bottom: 2rem; right: 2rem; width: 60px; height: 60px;
        border-radius: 50% !important;
        background-color: {COLOR_HEADER} !important;
        border: none !important;
        background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>');
        background-repeat: no-repeat;
        background-position: center;
        background-size: 26px 26px;
    }}
    .st-key-chat_toggle_container button p {{
        font-size: 0 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

    if st.session_state.chat_open:
        panel = st.container(key="chat_panel")
        with panel:
            st.markdown("**Ask a financial questions**")
            for role, content, extra in st.session_state.chat_history:
                if role == 'user':
                    st.markdown(f"**You:** {content}")
                else:
                    st.markdown(f"**Assistant** _(via {extra.get('source')})_: {content}")
                    if extra.get("sources"):
                        st.caption("Sources: "+", ".join(extra['sources']))

            query = st.text_input("Type your question...", key="chat_input")
            if st.button("Send", key="chat_send"):
                if query.strip():
                    st.session_state.chat_history.append(("user", query, {}))
                    with st.spinner("Thinking...."):
                        result = route_query(query)
                    st.session_state.chat_history.append(
                        ('assistant', result['answer'], {
                            'source': result['source'], 'sources' :result['sources']
                        })
                    )
                    st.rerun()
        panel.float(
            "bottom: 6.5rem; right: 2rem; width: 380px; max-height: 400px;"
            f"overflow-y: auto; background-color: {COLOR_CHATBOT};"
            "border-radius: 16px; padding: 1rem; box-shadow: 0 6px 20px rgba(0,0,0,0.3); z-index: 998;"
        )


def render_main_page():
    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {COLOR_MAIN}; }}
        section[data-testid="stSidebar"] {{ background-color: {COLOR_SIDEBAR}; }}
        .app-header {{
            background-color: {COLOR_HEADER};
            padding: 1.2rem 2rem;
            border-radius: 0 0 12px 12px;
            color: white;
            margin-bottom: 1.5rem;
        }}
        .app-header h1 {{ margin: 0; font-size: 1.8rem; }}
        .app-header p {{ margin: 0.2rem 0 0 0; opacity: 0.9; }}
        .stApp, .stApp p, .stApp label, .stApp label *,
        .stApp h1, .stApp h2, .stApp h3, .stApp span {{
            color: #2C2C2A;
        }}
        .app-header, .app-header * {{
            color: white !important;
        }}
        .stApp button, .stApp button * {{
        color: white !important;
        }}
        </style>
        <div class="app-header">
            <h1>\U0001F4B0 LoanLens</h1>
            <p>Loan Eligibility Checker &amp; Financial Assistant</p>
        </div>
        """,
    unsafe_allow_html=True)

    render_sidebar()
    render_chatbot()
    render_loan_form()


if not st.session_state.authenticated:
    render_auth_page()
else:
    render_main_page() 