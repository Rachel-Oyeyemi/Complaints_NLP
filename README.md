# CFPB Consumer Complaint Classification

End-to-end NLP project that classifies Consumer Financial Protection Bureau (CFPB) complaint narratives into financial product categories.

This project is designed for a resume-ready analytics portfolio. It includes data collection, preprocessing, TF-IDF modeling, evaluation, topic modeling, and a Streamlit application for live complaint classification.

## Business Problem

Financial institutions receive thousands of customer complaints across products such as credit reporting, debt collection, mortgages, credit cards, bank accounts, and student loans. Manually routing these complaints is slow and inconsistent.

This project builds a machine learning system that automatically predicts the likely product category from the complaint text so teams can route issues faster and identify recurring themes.

## Dataset

Source: CFPB Consumer Complaint Database  
Official download: https://files.consumerfinance.gov/ccdb/complaints.csv.zip

The project focuses on complaints with a public consumer complaint narrative. The default workflow samples a manageable subset so the project can be reproduced on a laptop.

## Methods

- Text cleaning and normalization
- TF-IDF feature extraction
- Logistic Regression classification baseline
- Model evaluation using accuracy, precision, recall, F1-score, and confusion matrix
- Topic modeling with LDA
- Interactive Streamlit prediction app

## Repository Structure

```text
Complaints_NLP/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
│   └── README.md
├── src/
│   ├── download_data.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── topic_modeling.py
│   └── predict.py
├── visuals/
├── .gitignore
├── requirements.txt
└── README.md
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Rachel-Oyeyemi/Complaints_NLP.git
cd Complaints_NLP
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows
```

### 3. Install requirements

```bash
pip install -r requirements.txt
```

### 4. Download a sample of CFPB complaints

```bash
python src/download_data.py --sample-size 50000
```

### 5. Preprocess the text

```bash
python src/preprocess.py
```

### 6. Train the classifier

```bash
python src/train_model.py
```

### 7. Run the app

```bash
streamlit run app/streamlit_app.py
```

## Example App Use Case

Input complaint narrative:

```text
I disputed inaccurate information on my credit report several times, but the bureau keeps verifying accounts that do not belong to me.
```

Expected prediction:

```text
Credit reporting, credit repair services, or other personal consumer reports
```

## Key Deliverables

- Reproducible Python ML pipeline
- Streamlit web app
- Model evaluation report
- Topic modeling script
- Resume-ready README documentation

## Resume Bullets

- Built an NLP classification system using TF-IDF and Logistic Regression to categorize CFPB consumer complaint narratives by financial product type.
- Developed a reproducible machine learning pipeline for complaint text cleaning, feature extraction, model training, evaluation, and deployment.
- Created an interactive Streamlit application that predicts complaint categories and supports faster complaint routing for financial service teams.

## Future Improvements

- Fine-tune BERT for higher classification performance
- Add BERTopic for richer topic discovery
- Deploy the Streamlit app to Streamlit Community Cloud
- Add SHAP or LIME explanations for model interpretability
