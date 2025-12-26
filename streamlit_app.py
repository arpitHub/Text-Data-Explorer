import streamlit as st
import pandas as pd

st.set_page_config(page_title="Text Explorer App", layout="wide")

st.title("📚 Text Explorer App")
st.markdown("""
Welcome to the **Text Explorer App**!  
This app is designed to help students understand how text data is handled in machine learning pipelines.  
Navigate using the sidebar to explore datasets, preprocessing, models, and results.
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
    # Use correct encoding for SMS Spam dataset
    df = pd.read_csv("data/sms_spam.csv", encoding="latin-1").rename(columns={"v1": "label", "v2": "message"})[["label", "message"]]
elif dataset_choice == "Upload your own":
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    # Load default or uploaded dataset
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("data/sms_spam.csv")

    # Store in session state
    st.session_state.df = df

    # Show preview
    st.dataframe(df.head())

# --- Show dataset preview ---
if df is not None:
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    st.write("### Dataset Info")
    st.write(f"Shape: {df.shape}")
    st.write("Columns:", df.columns.tolist())

    st.session_state["df"] = df
else:
    st.info("Please select or upload a dataset to continue.")
