import json
from pathlib import Path
import pandas as pd
import streamlit as st
from joblib import load
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, confusion_matrix,
    classification_report, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / 'model'
MODEL_FILES = {
    'Logistic Regression': 'logistic_regression.joblib',
    'Decision Tree': 'decision_tree.joblib',
    'K-Nearest Neighbors': 'knn.joblib',
    'Gaussian Naive Bayes': 'naive_bayes.joblib',
    'Support Vector Machine': 'svm.joblib'
}
TARGET_MAP = {'malignant': 0, 'benign': 1, '0': 0, '1': 1}

st.set_page_config(page_title='Breast Cancer Classifier Evaluation', layout='wide')
st.title('Breast Cancer Classification – Model Evaluation')
st.write('Upload a labelled test CSV to evaluate one of the five previously trained classifiers. The app loads saved models and never retrains them.')

@st.cache_resource
def load_models():
    return {name: load(MODEL_DIR / filename) for name, filename in MODEL_FILES.items()}

@st.cache_data
def get_feature_names():
    with open(MODEL_DIR / 'metadata.json', encoding='utf-8') as file:
        return json.load(file)['feature_names']

models = load_models()
features = get_feature_names()
upload = st.file_uploader('Upload test data (CSV)', type=['csv'])
if upload is not None:
    try:
        data = pd.read_csv(upload)
        missing = [column for column in features if column not in data.columns]
        if missing:
            st.error('The uploaded CSV is missing required feature columns: ' + ', '.join(missing))
            st.stop()
        if 'target' not in data.columns:
            st.error("The uploaded CSV must include the target column named 'target'.")
            st.stop()
        X_test = data[features].copy()
        y_raw = data['target'].astype(str).str.strip().str.lower()
        y_true = y_raw.map(TARGET_MAP)
        if y_true.isna().any():
            st.error("Target values must be 'malignant'/'benign' or 0/1.")
            st.stop()
        if X_test.isna().any().any():
            st.error('The uploaded feature columns contain missing values. Please clean the CSV first.')
            st.stop()
        X_test = X_test.apply(pd.to_numeric, errors='coerce')
        if X_test.isna().any().any():
            st.error('All 30 input features must be numeric.')
            st.stop()
        model_name = st.selectbox('Select classification model', list(MODEL_FILES))
        model = models[model_name]
        predictions = model.predict(X_test)
        positive_index = list(model.classes_).index(1)
        probabilities = model.predict_proba(X_test)[:, positive_index]
        accuracy = accuracy_score(y_true, predictions)
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        f1 = f1_score(y_true, predictions, zero_division=0)
        mcc = matthews_corrcoef(y_true, predictions)
        auc = roc_auc_score(y_true, probabilities) if y_true.nunique() == 2 else float('nan')
        st.subheader(f'Results: {model_name}')
        cols = st.columns(6)
        for col, label, value in zip(cols,
            ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC'],
            [accuracy, auc, precision, recall, f1, mcc]):
            col.metric(label, f'{value:.4f}' if pd.notna(value) else 'N/A')
        st.subheader('Confusion matrix')
        cm = confusion_matrix(y_true, predictions, labels=[0, 1])
        fig, ax = plt.subplots()
        display = ConfusionMatrixDisplay(cm, display_labels=['malignant', 'benign'])
        display.plot(ax=ax, cmap='Blues', colorbar=False)
        st.pyplot(fig)
        plt.close(fig)
        st.subheader('Classification report')
        report = classification_report(y_true, predictions, labels=[0, 1],
            target_names=['malignant', 'benign'], output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)
        st.caption(f'Evaluated {len(data)} uploaded rows. No model training was performed.')

        st.subheader('Comparison of all five models on this uploaded test data')
        comparison_rows = []
        for comparison_name, comparison_model in models.items():
            comparison_pred = comparison_model.predict(X_test)
            comparison_positive_index = list(comparison_model.classes_).index(1)
            comparison_prob = comparison_model.predict_proba(X_test)[:, comparison_positive_index]
            comparison_rows.append({
                'Model': comparison_name,
                'Accuracy': accuracy_score(y_true, comparison_pred),
                'AUC': roc_auc_score(y_true, comparison_prob) if y_true.nunique() == 2 else float('nan'),
                'Precision': precision_score(y_true, comparison_pred, zero_division=0),
                'Recall': recall_score(y_true, comparison_pred, zero_division=0),
                'F1': f1_score(y_true, comparison_pred, zero_division=0),
                'MCC': matthews_corrcoef(y_true, comparison_pred)
            })
        st.dataframe(pd.DataFrame(comparison_rows).set_index('Model').round(4), use_container_width=True)
    except Exception as error:
        st.error(f'Unable to evaluate this CSV: {error}')
else:
    st.info('Upload the included test_data.csv to view evaluation results.')
    st.write('Required columns: all 30 numeric Wisconsin Breast Cancer features and a target column containing malignant or benign.')
