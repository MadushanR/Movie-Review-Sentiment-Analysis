# Movie Review Sentiment Analysis

A Streamlit-powered web app and LSTM-based deep learning model for classifying movie reviews as **positive** or **negative**.

**Reported Test Accuracy:** 98%

---

## 🗂️ Project Structure

```
Movie-Review-Sentiment-Analysis/
├── IMDB Dataset.csv          # Raw 50K IMDb reviews (download from Kaggle)
├── user_feedback.csv         # Logged user reviews & corrected labels
├── model.py                  # Training script (uses TextVectorization + LSTM)
├── best_lstm_textvec.keras   # Saved model with vectorization layer
├── app.py                    # Streamlit app for inference & logging
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/MadushanR/Movie-Review-Sentiment-Analysis.git
cd Movie-Review-Sentiment-Analysis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the IMDb dataset  
- Place `IMDB Dataset.csv` in the project root.

### 4. Train the model (optional)
```bash
python model.py
```
This will:
- Combine IMDb + any `user_feedback.csv`
- Adapt a `TextVectorization` layer on the raw text
- Train an LSTM model
- Save the best model to `best_lstm_textvec.keras`
- Print final test accuracy (≈ 98%)

---

## 🖥️ Run the Streamlit App

```bash
streamlit run app.py
```

- Open the browser URL shown (usually `http://localhost:8501`)
- Type a movie review and click **Predict Sentiment**
- See **Positive** or **Negative** with confidence score
- Provide feedback (“Was this prediction correct?”) to collect new data in `user_feedback.csv`

---

## 🔧 File Descriptions

- **model.py**  
  - Loads IMDb + user feedback  
  - Builds a `TextVectorization` → `Embedding` → `LSTM` → `Dense` model  
  - Trains with early stopping & model checkpointing  
  - Evaluates on held-out test set  

- **app.py**  
  - Loads the saved `.keras` model (includes vectorization)  
  - Provides a simple Streamlit UI for input, prediction, and feedback logging  

- **requirements.txt**  
  ```
  pandas
  tensorflow>=2.9
  streamlit
  ```
  (plus any other packages you used)

---

## 📈 Performance

- **Test Accuracy:** ~98%  
- **Loss Function:** Binary cross-entropy  
- **Optimizer:** Adam  
- **Sequence Length:** 300 tokens  
- **Embedding Dim:** 128  
- **LSTM Units:** 64  

---

## 📝 Notes & Next Steps

- **Data Quality:** Skipped malformed CSV lines and consolidated user feedback  
- **Vectorization:** Handled in-model via `TextVectorization`—no external tokenizer  
- **Improvements:**  
  - Experiment with bidirectional LSTM or GRU  
  - Tune `max_tokens` and `maxlen` via cross-validation  
  - Deploy on Streamlit Cloud or Hugging Face Spaces

---

## 📖 References

- IMDb 50K Dataset (Kaggle)  
- TensorFlow `TextVectorization` documentation  
- Streamlit documentation  

---

Feel free to open issues or submit pull requests!  
— Madushan Rajendran
