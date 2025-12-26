import streamlit as st
import pandas as pd

st.title("📄 Dataset Explorer")

st.write("### How Dataset Exploration Works")
st.markdown("""
Before we can build models, we need to **understand the data**.  
Dataset exploration helps us answer questions like:
- What columns are available?
- How many rows do we have?
- What does a sample of the data look like?

For example, in the SMS Spam dataset:
- Each row is a text message.
- The `label` column tells us whether it's **spam** or **ham** (not spam).
- The `message` column contains the actual text.

Exploring the dataset ensures we know what we're working with before preprocessing.
""")

# --- Sidebar dataset selection ---
st.sidebar.header("Choose a dataset")
dataset_choice = st.sidebar.selectbox(
    "Select a dataset",
    ["SMS Spam (default)", "Upload your own"]
)

# --- Load dataset ---
df = None
if dataset_choice == "SMS Spam (default)":
    # assumes you have data/sms_spam.csv in your repo
    try:
        df = pd.read_csv("data/sms_spam.csv", encoding="latin-1")
        # Clean column names if needed
        if "v1" in df.columns and "v2" in df.columns:
            df = df.rename(columns={"v1": "label", "v2": "message"})[["label", "message"]]
    except Exception as e:
        st.error(f"Error loading SMS Spam dataset: {e}")
elif dataset_choice == "Upload your own":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            df = pd.read_csv(uploaded_file, encoding="latin-1")

# --- Show dataset preview ---
if df is not None:
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    st.write("### Dataset Info")
    st.write(f"Shape: {df.shape}")
    st.write("Columns:", df.columns.tolist())

    # Save to session state for later pages
    st.session_state["df"] = df
else:
    st.info("Please select or upload a dataset to continue.")
