<div align="center">

# 📧 Email Spam Filter

### A packaged machine learning project for predicting whether an email is **Spam** or **Ham**

<p>
  Pretrained spam detection package with a reusable Python API, model training pipeline, automated tests, and an interactive Streamlit UI.
</p>

</div>

---

## 📌 Project Overview

**Email Spam Filter** is a machine learning package that classifies email text into one of two categories:

- **Spam**
- **Ham** (not spam)

This project is designed as a **real Python package**, not just a notebook or script collection.

It includes:

- a **pretrained model** bundled inside the package
- a **clean Python API** for predictions
- a **training pipeline** to retrain the model on new datasets
- **tests** to verify package functionality
- a **Streamlit UI** for interactive use and batch analysis
- **visualizations** to make predictions and model behavior easier to understand

---

## 🎯 What This Project Does

This package allows users to:

- predict whether a single email is spam or ham
- classify multiple emails from a CSV file
- view spam probability scores
- retrain the model using their own labeled dataset
- launch a Streamlit dashboard for interactive testing
- inspect useful visualizations such as:
  - batch class distribution
  - spam probability distribution
  - top indicative words for spam and ham

---

## 🧠 How the Model Works

The model is trained using a machine learning **pipeline** made of three stages:

### 1. Text Preprocessing
The raw email text is cleaned before feature extraction.

Typical preprocessing includes:
- converting text to lowercase
- removing URLs
- removing email addresses
- normalizing whitespace

### 2. TF-IDF Vectorization
The cleaned text is transformed into numerical features using **TF-IDF**  
(Term Frequency–Inverse Document Frequency).

This helps the model understand which words are important in a message.

### 3. Classification
The vectorized text is passed into a **Multinomial Naive Bayes** classifier, which predicts whether the email is:

- `spam`
- `ham`

The entire preprocessing + vectorization + classifier pipeline is saved into a single file:

```text
spam_pipeline.joblib
```

## 🏗️ Project Structure
```
email_spam_filter/
│
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── src/
│   └── email_spam_filter/
│       ├── __init__.py
│       ├── inference.py
│       ├── viz.py
│       ├── schemas.py
│       ├── logging_utils.py
│       │
│       ├── resources/
│       │   ├── spam_pipeline.joblib
│       │   └── metadata.json
│       │
│       └── training/
│           ├── __init__.py
│           ├── data.py
│           ├── features.py
│           ├── evaluate.py
│           └── train.py
│
├── streamlit_ui/
│   ├── app.py
│   └── pages/
│       ├── Batch_Predict.py
│       └── Model_Insights.py
│
└── tests/
    ├── test_inference.py
    ├── test_viz.py
    └── test_training_cli.py
```
### 📂 Folder Explanation

```text
data/
```
Used during model development and training.

- **data/raw/** → original input dataset

- **data/interim/** → optional cleaned/intermediate data

- **data/processed/** → optional transformed datasets

Raw training data should be placed in **data/raw/.**

```text
src/email_spam_filter/
```

Contains the installable Python package.

### Main modules

- **inference.py** → load model and make predictions
- **viz.py** → create charts and plots
- **schemas.py** → shared data structures
- **logging_utils.py** → centralized logging setup

```text
resources/
```

Contains bundled package artifacts:

- **spam_pipeline.joblib** → trained model pipeline
- **metadata.json** → training metadata and metrics

```text
training/
```

Contains the retraining pipeline:

- dataset loading
- preprocessing
- vectorizer setup
- training
- evaluation
- model export

```text
streamlit_ui/
```

Contains the interactive user interface.

- **app.py** → main UI for single email prediction
- **pages/Batch_Predict.py** → classify emails from CSV
- **pages/Model_Insights.py** → view word importance and model metadata

```text
tests/
```

Contains automated tests to verify:

- inference works
- training works
- plotting functions work

## ⚙️ Installation

### 1. Clone the repository

```text
git clone https://github.com/yourusername/email_spam_filter.git
cd email_spam_filter
```

### 2. Create a virtual environment

```text
uv venv
.\.venv\Scripts\Activate
```

### 3. Install the package

Install the package in editable mode with Streamlit UI dependencies:

```text
pip install -e .[ui]
```

This installs:
- **scikit-learn**
- **numpy**
- **pandas**
- **joblib**
- **streamlit**
- **matplotlib**

## 🚀 Using the Package in Python

### Example: classify a single email

```text
from email_spam_filter import classify

result = classify("Congratulations! You have won a free vacation.")

print("Prediction:", result.label)
print("Spam probability:", result.spam_probability)
```

## 🧪 Running Tests

```text
pytest
```

The tests check:

- inference behavior
- training pipeline behavior
- visualization behavior

## 🗂️ Training Data Format

To train the model, place your dataset CSV file inside:

```text
data/raw/
```
### Required columns

- **text** → email content
- **label** → category (spam or ham)

## 🏋️ Training the Model

```text
python -m email_spam_filter.training.train data/raw/your_file_name.csv
```

### What happens during training

The training pipeline will:

- load the CSV
- validate the required columns
- clean the dataset
- split data into training and test sets
- build the text classification pipeline
- train the model
- evaluate the model
- export the trained model and metadata


### Output files after training

Training creates or overwrites these files:

```text
src/email_spam_filter/resources/spam_pipeline.joblib
src/email_spam_filter/resources/metadata.json
```

These files are then used automatically during prediction.

## 📊 Running the Streamlit App

Launch the Streamlit interface with:

```text
streamlit run streamlit_ui/app.py
```

What the Streamlit app includes

### Main Page

- single email prediction
- spam probability display
- threshold control

### Batch Prediction Page

- upload a CSV file with a text column
- run batch classification
- download predictions
- view: class count chart & probability histogram

### Model Insights Page

- top spam-indicative words
- top ham-indicative words
- metadata display

## 📁 CSV Format for Batch Prediction in UI

For the batch prediction page, your uploaded CSV should contain a column named:

```text
text
```

Example:

```text
text
"Congratulations! You have won a free iPhone."
"Hi team, please find the attached report."
"URGENT: Your bank account has been suspended."
"Can we reschedule our meeting?"
```

## 📦 Package Published on TestPyPI

The Email Spam Filter package has been successfully built and published to TestPyPI, allowing users to install and use it like any standard Python library.

This demonstrates that the project is fully packaged, distributable, and reusable outside the development environment.

### Install the Package

You can install the package directly from TestPyPI using:

```text
pip install -i https://test.pypi.org/simple/ email-spam-filter
```

### Example Usage

Once installed, the package can be imported and used in any Python script.

```text

from email_spam_filter import classify

email = "Congratulations! You have won a free vacation."

result = classify(email)

print("Prediction:", result.label)
print("Spam probability:", result.spam_probability)
```