import streamlit as st
import sys
import os
import logging
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64

# Set page config at the very beginning
st.set_page_config(
    layout="wide",
    page_title="Airgapped Offline RAG",
    page_icon="assets/airgapped_offline_rag_icon.png",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Custom CSS to inject into the Streamlit app
st.markdown("""
<style>
/* Overall theme */
body {
    color: #e0e0e0;
    background-color: #0a0a0a;
}

/* Adjust title and header */
.main > div:first-child h1 {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
    color: #f4298a;
}

/* Adjust column gap and padding */
.row-widget.stHorizontal {
    gap: 2rem;
    padding: 1rem 0;
}

/* Style for existing documents */
.existing-docs {
    max-height: 200px;
    overflow-y: auto;
    border: 1px solid #333;
    border-radius: 5px;
    padding: 10px;
    margin-bottom: 1rem;
    background-color: #111;
}

/* Footer styling */
.footer {
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #333;
    color: #888;
    font-size: 0.8rem;
}

/* GitHub icon */
.github-icon {
    height: 20px;
    vertical-align: middle;
    margin-left: 5px;
    filter: invert(1);
}

/* Adjust spacing for settings and chat interface */
.stColumn > div {
    padding: 1rem;
    background-color: #111;
    border-radius: 10px;
    margin-bottom: 1rem;
}

/* Adjust font sizes */
body {
    font-size: 16px;
}

.stTextInput > div > div > input,
.stSelectbox > div > div > div,
.stCheckbox > label > div {
    font-size: 16px;
    background-color: #222;
    color: #e0e0e0;
    border-color: #444;
}

/* Remove box above filenames */
.css-1kyxreq {
    display: none;
}

/* Adjust sidebar width when chat is active */
@media (min-width: 768px) {
    .main .block-container {
        max-width: 90%;
        padding-left: 5rem;
        padding-right: 5rem;
    }
}

/* Reduce file name font size */
.existing-docs p {
    font-size: 0.9rem;
    color: #26f6cb;
}

/* Right align Process Documents button */
.stButton > button:last-child {
    float: right;
}

/* Add vertical line between settings and chat */
.main .block-container > div > div > div:nth-child(2) {
    border-left: 1px solid #333;
    padding-left: 2rem;
}

/* Styling for info and success messages */
.stAlert {
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
    background-color: transparent !important;
    border: none !important;
}

/* Adjust button layout */
.settings-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 1rem;
}

/* Style for processing logs */
.processing-logs {
    background-color: #111;
    border: 1px solid #444;
    border-radius: 5px;
    padding: 10px;
    font-family: monospace;
    font-size: 0.9rem;
    white-space: pre-wrap;
    max-height: 200px;
    overflow-y: auto;
    color: #26f6cb;
}

/* Custom styles for Streamlit components */
.stSelectbox > div[data-baseweb="select"] > div {
    background-color: #222;
    border-color: #444;
}

.stCheckbox > label > div[data-testid="stMarkdownContainer"] > p {
    color: #e0e0e0;
}

/* Remove highlight behind checkboxes */
.stCheckbox > label {
    background-color: transparent !important;
}

.stButton > button {
    background-color: #f4298a;
    color: #fff;
    border: none;
    transition: background-color 0.3s ease;
}

.stButton > button:hover {
    background-color: #d54d8e;
    color: #fff;
}

/* Chat message styling */
.stChatMessage {
    background-color: #111;
    border: 1px solid #333;
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
}

.stChatMessage [data-testid="stChatMessageContent"] {
    color: #e0e0e0;
}

/* Expander styling */
.streamlit-expanderHeader {
    background-color: #222;
    color: #f4298a;
    border: 1px solid #444;
}

.streamlit-expanderContent {
    background-color: #111;
    border: 1px solid #444;
    border-top: none;
}

/* File uploader styling */
.stFileUploader > div > div {
    background-color: #222;
    border-color: #444;
}

/* Improve readability of select dropdown */
.stSelectbox > div[data-baseweb="select"] > div > div {
    background-color: #222;
    color: #e0e0e0;
}

.stSelectbox > div[data-baseweb="select"] > div > div:hover {
    background-color: #333;
}

/* Chat container styling */
.chat-container {
    border: 1px solid #333;
    border-radius: 10px;
    padding: 1rem;
    background-color: #111;
}

/* Remove border from chat interface message */
.chat-container > div:first-child {
    border: none !important;
    background-color: transparent !important;
}

/* Remove border and background from info messages */
.stAlert.info {
    border: none !important;
    background-color: transparent !important;
}

/* Remove background from checkboxes */
.stCheckbox {
    background-color: transparent !important;
}

.stCheckbox > label {
    background-color: transparent !important;
}

.stCheckbox > label > div[data-testid="stMarkdownContainer"] {
    background-color: transparent !important;
}

</style>
""", unsafe_allow_html=True)

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
