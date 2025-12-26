# 📚 Text Explorer App

An interactive **Streamlit app** for learning how text data is processed, transformed, and classified using machine learning.  
Students can explore datasets, preprocess text, build models, and visualize results — all with clear explanations and examples.

---

## 🚀 Features

- **Dataset Explorer (📂)**  
  Preview datasets, inspect rows/columns, and understand the structure of text + labels.

- **Preprocessing (🔍)**  
  - Tokenization demo (split sentences into words).  
  - Bag of Words vs TF‑IDF vectorization.  
  - Worked example showing how TF‑IDF is calculated step‑by‑step.  
  - Vocabulary preview to see which words are included.

- **Model Builder (🤖)**  
  - Train Logistic Regression, Naive Bayes, and Support Vector Classifier.  
  - Compare accuracy across models.  
  - Confusion matrix visualization.  
  - Top Features chart showing which words drive spam vs ham predictions.

- **Results (📊)**  
  - Test new messages against the trained model.  
  - See predictions (spam/ham) with probability scores.  
  - Word clouds for spam vs ham vocabulary.  
  - Explanation of confidence levels in predictions.

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) for interactive UI  
- [scikit-learn](https://scikit-learn.org/) for ML models  
- [NLTK](https://www.nltk.org/) for tokenization  
- [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) for plots  
- [WordCloud](https://amueller.github.io/word_cloud/) for text visualization  

---

## 📂 Project Structure

```text
├── app.py                  # Main entry point and landing page
├── pages/
│   ├── 1_Data_Explorer.py  # Load and preview text datasets
│   ├── 2_Preprocessing.py  # Clean, tokenize, and vectorize text
│   ├── 3_Model_Builder.py  # Train and evaluate ML models on text data
│   └── 4_Results.py        # Display predictions, metrics, and misclassifications
├── requirements.txt        # Dependencies with pinned versions
└── README.md               # Project guide and documentation
```

---

## ⚡ How to Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/text-explorer-app.git
   cd text-explorer-app
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the app:
   ```bash
   streamlit run app.py
   ```
---

### 🌐 Deployment
 - Push your repo to GitHub.
 - Go to Streamlit Cloud.
 - Connect your repo and select app.py as the entry point.
 - Deploy and share the link with students!

---
## 🎓 Learning Outcomes
By using the Text Explorer App, students will:

- **Data Explorer (📂)**  
  Understand how text datasets are structured, preview samples, and recognize the importance of dataset inspection.
- **Preprocessing (🧹)**  
  Learn how to clean text (remove punctuation, stopwords), tokenize words, and convert text into numerical features (e.g., bag‑of‑words, TF‑IDF).
- **Model Builder (🤖)**  
  Train and compare machine learning models (e.g., Logistic Regression, Naive Bayes) for text classification.  
  Explore how different algorithms handle sparse text features.
- **Results (📊)**  
  Interpret predictions, evaluate accuracy, and analyze misclassifications.  
  Gain experience with confusion matrices and probability scores to understand model confidence.

---

## 📸 Screenshots (optional)
Add screenshots of each page here once deployed.

---

## 🙌 Credits
Built with ❤️ by Arpit to make machine learning hands‑on and approachable for everyone.
