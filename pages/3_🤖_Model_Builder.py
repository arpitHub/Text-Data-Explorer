import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🤖 Model Builder")

if "tfidf" not in st.session_state or "y" not in st.session_state:
    st.warning("Please preprocess the dataset first in the Preprocessing page.")
else:
    X = st.session_state["tfidf"]
    y = st.session_state["y"]

    st.write("### How Models Classify Text")
    st.markdown("""
    Once text is converted into numbers (TF‑IDF features), we can train machine learning models.  
    Here’s how the main ones work:

    - **Logistic Regression:**  
      Learns weights for each word feature. If words like *win* or *free* strongly correlate with spam, their weights push predictions toward spam.

    - **Naive Bayes:**  
      Uses probabilities. It calculates how likely each word is given spam vs ham. For example, if *prize* appears mostly in spam, the probability tilts toward spam.

    - **Support Vector Classifier (SVC):**  
      Finds a boundary (hyperplane) that separates spam and ham messages in the feature space. It tries to maximize the margin between the two classes.

    👉 All models aim to learn patterns from labeled data (spam vs ham) and then apply those patterns to new messages.
    """)

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "Support Vector Classifier": SVC(probability=True)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = (model, acc, preds)

    # --- Display results ---
    st.write("### Model Performance")
    for name, (model, acc, preds) in results.items():
        st.write(f"**{name}** → Accuracy: {acc:.2f}")

    # --- Best model ---
    best_model_name = max(results, key=lambda k: results[k][1])
    best_model, best_acc, best_preds = results[best_model_name]
    st.success(f"Best model: {best_model_name} (Accuracy: {best_acc:.2f})")

    st.session_state["best_model"] = best_model

    # --- Confusion Matrix ---
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, best_preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # --- Feature Importance (Logistic Regression only) ---
    if best_model_name == "Logistic Regression":
        st.write("### Top Features Driving Predictions")

        # Get feature names from TF-IDF vectorizer
        feature_names = np.array(st.session_state["tfidf_vectorizer"].get_feature_names_out())
        coefs = best_model.coef_[0]

        # Create dataframe of words + weights
        coef_df = pd.DataFrame({
            "word": feature_names,
            "weight": coefs
        })

        # Sort by absolute weight
        top_positive = coef_df.sort_values("weight", ascending=False).head(10)
        top_negative = coef_df.sort_values("weight", ascending=True).head(10)

        col1, col2 = st.columns(2)

        with col1:
            st.write("#### Words Most Indicative of Spam")
            st.bar_chart(top_positive.set_index("word"))

        with col2:
            st.write("#### Words Most Indicative of Ham")
            st.bar_chart(top_negative.set_index("word"))

        st.markdown("""
        👉 Positive weights push predictions toward **spam**.  
        👉 Negative weights push predictions toward **ham**.  
        This helps students see which words the model relies on most.
        """)
