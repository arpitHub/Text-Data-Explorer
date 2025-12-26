import streamlit as st
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

st.title("🔍 Preprocessing Dashboard")

# --- Check if dataset exists ---
if "df" not in st.session_state:
    st.warning("Please upload or select a dataset in the Dataset Explorer first.")
    st.stop()

df = st.session_state["df"]

# ============================================================
# 🧠 SECTION 1 — THEORY: TOKENIZATION
# ============================================================

st.write("### How Tokenization Works")
st.markdown("""
**Tokenization** is the process of splitting text into smaller units called *tokens*.  
These tokens can be words, punctuation marks, or symbols.

Example:  
`"This is a sample sentence."` → `['This', 'is', 'a', 'sample', 'sentence', '.']`

Why this matters:
- Machine learning models **cannot** work directly with raw text.
- Tokenization is the first step in converting text into numerical features.
- It helps models understand structure, meaning, and patterns in language.

👉 **Teaching note:**  
Tokenization is like breaking a paragraph into LEGO bricks — once you have the pieces, you can build anything.
""")

# ============================================================
# 🧪 Tokenization Demo (Regex-based, Cloud-Safe)
# ============================================================

st.write("### Tokenization Example")

sentence = st.text_input(
    "Enter a sentence to tokenize:",
    "This is a sample sentence for tokenization."
)

def regex_tokenize(text):
    """
    A simple regex tokenizer that splits text into words and punctuation.
    This avoids NLTK downloads and works instantly on Streamlit Cloud.
    """
    return re.findall(r"\w+|\S", text)

tokens = regex_tokenize(sentence)
st.write("Tokens:", tokens)

# ============================================================
# 🧠 SECTION 2 — THEORY: TEXT PROCESSING
# ============================================================

st.write("### How Text Processing Works")
st.markdown("""
After tokenization, we convert text into **numerical features** using methods like:

- **Bag of Words (BoW):**  
  Counts how often each word appears.  
  Simple, fast, and surprisingly effective.

- **TF-IDF (Term Frequency–Inverse Document Frequency):**  
  Weighs words by importance.  
  Common words get lower scores; rare but meaningful words get higher scores.

This transforms text into a matrix of numbers that machine learning models can understand.
""")

# ============================================================
# 📘 Worked Example: TF-IDF
# ============================================================

st.write("### Worked Example: How TF-IDF is Calculated")
st.markdown("""
Suppose we have 3 short documents:

1. "I love machine learning"  
2. "Machine learning is fun"  
3. "I love fun"

---

#### **Step 1: Vocabulary**
`["i", "love", "machine", "learning", "is", "fun"]`

#### **Step 2: Term Frequency (TF)**
- Doc1: each word appears once in 4 words → TF = 1/4 = 0.25  
- Doc2: each word appears once in 4 words → TF = 0.25  
- Doc3: each word appears once in 3 words → TF ≈ 0.33  

#### **Step 3: Inverse Document Frequency (IDF)**
Formula: `IDF(t) = log(N / df(t))`  
- N = 3 documents  
- "is" appears in only 1 doc → IDF = log(3/1) ≈ 1.10  
- "love" appears in 2 docs → IDF = log(3/2) ≈ 0.18  

#### **Step 4: TF-IDF = TF × IDF**
- In Doc2, "is" → 0.25 × 1.10 ≈ 0.275 (high score, distinctive word)  
- In Doc1, "love" → 0.25 × 0.18 ≈ 0.045 (low score, common word)

👉 **Teaching note:**  
TF-IDF helps models focus on *meaningful* words instead of common filler words.
""")

# ============================================================
# 🧮 SECTION 3 — Vectorization
# ============================================================

st.write("### Vectorization (Bag of Words vs TF-IDF)")

text_column = st.selectbox("Select text column", df.columns)

# Warn if the user selects the label column
if text_column.lower() == "label":
    st.warning(
        "⚠️ You selected the **label** column. Please choose the **message** column instead, "
        "because labels are targets, not text features."
    )
    st.stop()

# Create vectorizers
bow_vectorizer = CountVectorizer(stop_words="english")
tfidf_vectorizer = TfidfVectorizer(stop_words="english", min_df=2)  # filter rare words

# Fit-transform
bow = bow_vectorizer.fit_transform(df[text_column])
tfidf = tfidf_vectorizer.fit_transform(df[text_column])

# Display shapes
st.write("**Bag of Words shape:**", bow.shape)
st.write("**TF-IDF shape:**", tfidf.shape)

# Show sample vocabulary
st.write("**Sample vocabulary words:**", list(tfidf_vectorizer.vocabulary_.keys())[:20])

# Save for later pages
st.session_state["bow"] = bow
st.session_state["tfidf"] = tfidf
st.session_state["tfidf_vectorizer"] = tfidf_vectorizer
st.session_state["y"] = df["label"] if "label" in df.columns else None

if st.session_state["y"] is None:
    st.warning(
        "The dataset does not contain a 'label' column. "
        "Please ensure your dataset has labels for supervised learning."
    )