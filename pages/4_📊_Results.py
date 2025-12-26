import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.title("📊 Results & Predictions")

st.write("### How Predictions Work")
st.markdown("""
When you type a new message:
1. The text is tokenized and transformed using the **same TF-IDF vocabulary** as training.
2. The model calculates whether the message is more likely spam or ham.
3. The prediction is displayed.

If you use words not seen during training, they won’t influence the prediction —  
this shows why training data coverage is important.
""")

st.write("### Why Probability Scores Matter")
st.markdown("""
Most models don’t just give a label (spam/ham) — they also provide **probability scores**.  
These scores show how confident the model is in its prediction.

For example:
- A message might be predicted as **spam** with 95% probability.  
- Another might be predicted as **ham** with 55% probability.

👉 The higher the probability, the more confident the model is.  
👉 Lower probabilities mean the model is less certain, which can highlight **ambiguous messages**.

This helps students see that classification isn’t always black‑and‑white —  
sometimes the model is very sure, and sometimes it’s making a best guess.
""")

st.write("### Word Clouds")
st.markdown("""
Word clouds visualize the most frequent words in spam vs ham messages.  
- Spam messages often contain words like *free*, *win*, *prize*.  
- Ham messages contain everyday words like *hey*, *tomorrow*, *call*.  

This helps students connect vocabulary frequency with classification outcomes.
""")

if "best_model" not in st.session_state or "tfidf_vectorizer" not in st.session_state:
    st.warning("Please train a model first in the Model Builder page.")
else:
    best_model = st.session_state["best_model"]
    tfidf_vectorizer = st.session_state["tfidf_vectorizer"]
    df = st.session_state["df"]

    st.write("### Try Your Own Text")
    user_input = st.text_area("Enter a sentence to classify:", "Free entry in 2 a weekly competition to win tickets!")

    if user_input.strip():
        X_new = tfidf_vectorizer.transform([user_input])
        prediction = best_model.predict(X_new)[0]

        st.success(f"Prediction: **{prediction}**")

        tokens = user_input.lower().split()
        unseen_words = [w for w in tokens if w not in tfidf_vectorizer.vocabulary_]
        if unseen_words:
            st.info(
                f"Note: The following words were not in the training vocabulary "
                f"and did not influence the prediction: {', '.join(unseen_words)}"
            )

        if hasattr(best_model, "predict_proba"):
            probs = best_model.predict_proba(X_new)[0]
            st.write("### Prediction Probabilities")
            st.bar_chart(probs)

    # --- Word Clouds ---
    st.write("### Word Clouds (Spam vs Ham)")
    if "label" in df.columns and "message" in df.columns:
        spam_text = " ".join(df[df["label"] == "spam"]["message"].astype(str))
        ham_text = " ".join(df[df["label"] == "ham"]["message"].astype(str))

        spam_wc = WordCloud(width=600, height=400, background_color="white").generate(spam_text)
        ham_wc = WordCloud(width=600, height=400, background_color="white").generate(ham_text)

        col1, col2 = st.columns(2)
        with col1:
            st.write("#### Spam Messages")
            fig, ax = plt.subplots()
            ax.imshow(spam_wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

        with col2:
            st.write("#### Ham Messages")
            fig, ax = plt.subplots()
            ax.imshow(ham_wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
