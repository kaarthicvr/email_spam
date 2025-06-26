
# Spam Mail Detection using Logistic Regression

This project implements a simple machine learning pipeline to detect spam emails using a Logistic Regression classifier. It uses a dataset of labeled emails and extracts features using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.

---

## Features

- Classifies emails as **spam** (0) or **ham** (1)
- Uses natural language processing with **TF-IDF** vectorizer
- Trains a **Logistic Regression** model on text data
- Evaluates performance on both training and test sets
- Accepts real-time input to predict whether a message is spam or not

---

## Dataset

The dataset used is `mail_data.csv` with the following structure:

| Category | Message            |
|----------|--------------------|
| spam     | Free entry in 2 a...|
| ham      | Are we meeting today?|

- `Category`: Label (`spam` or `ham`)
- `Message`: Text content of the email

---

## Libraries Used

- numpy
- pandas
- scikit-learn

---

## Model Workflow

1. **Data Preprocessing**:
   - Handle missing values
   - Convert labels (`spam` = 0, `ham` = 1)

2. **Text Vectorization**:
   - TF-IDF with `stop_words='english'` and `lowercase=True`

3. **Model Training**:
   - Logistic Regression trained on 80% of the dataset

4. **Model Evaluation**:
   - Accuracy on both training and test datasets

5. **Prediction**:
   - Accepts custom email input and classifies as spam or ham

---

## Code Snippet

```python
content = input("enter mail content: ")
input_data_features = feature_extraction.transform([content])
prediction = model.predict(input_data_features)
if prediction[0] == 1:
    print("Ham mail")
else:
    print("Spam mail")
```

---

## Example Output

```
accuracy on training data : 0.9659192825112107
accuracy on testing data : 0.9659192825112107
enter mail content: You have won a $1000 Walmart gift card, click here to claim!
[0]
Spam mail
```

---

## Setup Instructions

1. **Clone the Repository**:

```bash
https://github.com/kaarthicvr/email_spam.git
cd spam-mail-detector
```

2. **Install Requirements**:

```bash
pip install numpy pandas scikit-learn
```

3. **Run the Script**:

Make sure you have the dataset (`mail_data.csv`) in the same directory.

```bash
python spam_classifier.py
```

4. **Provide Email Input** when prompted to test real-time prediction.

---

## Future Improvements

- Use more advanced NLP models like Naive Bayes or transformers
- Build a web interface using Flask or Streamlit
- Add email header analysis for better accuracy

---

## License

This project is for educational purposes only.

---

## Author

Built by Kaarthic VR
