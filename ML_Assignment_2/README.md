# Machine Learning Assignment 2 – Breast Cancer Classification

# BITS ID - 2026AJ05783

## a. Problem Statement

The objective of this assignment is to implement and compare five supervised machine learning classification algorithms to predict whether a breast tumour is malignant or benign using diagnostic measurements.

This is a binary classification problem. The performance of each model is evaluated using six evaluation metrics: Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

The five classification algorithms implemented are:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN) Classifier
4. Gaussian Naive Bayes Classifier
5. Support Vector Machine (SVM)

The objective is to compare their performance on the same dataset and identify the best-performing model.

---

## b. Dataset Description

- **Dataset Name:** Wisconsin Diagnostic Breast Cancer (WDBC)
- **Source:** UCI Machine Learning Repository (Dataset 17)
- **Dataset Access:** `sklearn.datasets.load_breast_cancer`
- **Total Instances:** 569
- **Number of Input Features:** 30
- **Target Variable:** Diagnosis (Malignant / Benign)
- **Number of Classes:** 2
- **Malignant Cases:** 212
- **Benign Cases:** 357
- **Missing Values:** None
- **Training and Testing Split:** 80:20
- **Random State:** 42
- **Splitting Method:** Stratified train-test split

The dataset contains 30 numerical diagnostic features computed from digitized images of fine needle aspirates of breast masses.

The target variable represents two classes:

- Malignant
- Benign

A stratified 80:20 train-test split is used to maintain the class distribution in both training and testing datasets.

**Dataset Reference:**

Wolberg, W., Street, N., & Mangasarian, O. (1993). Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository.

https://doi.org/10.24432/C5DW2B

---

## c. GitHub Repository Link

**GitHub Repository URL:**

https://github.com/SHWETHASENTHIL/ML_ASSIGNMENT_2/tree/main/ML_Assignment_2

The repository contains the complete source code, Jupyter Notebook, trained machine learning models, dataset files, test data, requirements file, and project documentation.

---

## d. Models Used

All five classification models are implemented using the same dataset and the same stratified training and testing split.

StandardScaler is applied within pipelines for Logistic Regression, K-Nearest Neighbors, and Support Vector Machine. Decision Tree and Gaussian Naive Bayes are implemented without feature scaling.

### Evaluation Metrics Comparison

The following table presents the performance of all five classification models on the held-out test dataset.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| K-Nearest Neighbors | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Gaussian Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

**Metric Calculation Convention:**

- Precision, Recall, and F1 Score consider the benign class (class 1) as the positive class.
- AUC is calculated using the predicted probability of the benign class.
- MCC is calculated considering both classes.

### Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---|---|
| Logistic Regression | Achieved 98.25% accuracy, 0.9954 AUC, and 0.9861 F1 Score. It tied for the best F1 Score and MCC, demonstrating strong classification performance and good class separation. |
| Decision Tree | Achieved 91.23% accuracy and 0.9286 F1 Score. It recorded the lowest overall performance among the five models on this test split. Although decision trees can capture nonlinear relationships, its measured generalization performance was comparatively lower. |
| K-Nearest Neighbors | Achieved 95.61% accuracy and 0.9655 F1 Score. Feature scaling was applied before distance calculations. It demonstrated strong classification performance but performed below Logistic Regression and SVM. |
| Gaussian Naive Bayes | Achieved 93.86% accuracy, 0.9878 AUC, and 0.9517 F1 Score. Its high AUC indicates good ranking ability, although its threshold-based classification performance was lower than the top-performing models. |
| Support Vector Machine | Achieved 98.25% accuracy, 0.9950 AUC, and 0.9861 F1 Score. It tied with Logistic Regression in accuracy, F1 Score, and MCC, with a marginally lower AUC. |

### Overall Winner

**Logistic Regression**

Based on the evaluation results, Logistic Regression is selected as the overall winning model.

Both Logistic Regression and Support Vector Machine achieved the same Accuracy (0.9825), F1 Score (0.9861), and MCC (0.9623).

However, Logistic Regression achieved a slightly higher AUC Score (0.9954) compared to Support Vector Machine (0.9950).

Therefore, Logistic Regression is selected as the overall winner based on its slightly higher AUC Score among the tied top-performing models.

This conclusion applies to the current held-out test dataset and does not imply that Logistic Regression will always outperform the other algorithms on different datasets.

---

## e. Streamlit Application

The project includes an interactive Streamlit application (`app.py`) to demonstrate and evaluate the five trained classification models.

**Live Streamlit Application URL:**

https://mlassignment2-ewjdgp7pykozxkj6hcmtis.streamlit.app/

### Application Features

The Streamlit application includes the following features:

1. **Dataset Upload:** Allows users to upload a test dataset in CSV format containing all required input features and the target column.
2. **Model Selection:** Provides a dropdown menu to select one of the five trained classification models.
3. **Evaluation Metrics:** Displays Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient.
4. **Confusion Matrix:** Displays the confusion matrix for the selected classification model.
5. **Classification Report:** Displays the classification report containing precision, recall, F1 Score, and support for each class.
6. **Model Comparison:** Allows users to compare the evaluation results of the five classification models.

The application loads the previously trained and saved models from the `model/` directory. It evaluates the uploaded test dataset without retraining the models.

The uploaded test dataset must contain all 30 required input features and the target column.

---

## f. Project Files

The project repository contains the following files and directories:

| File / Directory | Description |
|---|---|
| `ML_Assignment_2.ipynb` | Complete Jupyter Notebook containing data preparation, model implementation, training, and evaluation. |
| `app.py` | Streamlit application source code. |
| `requirements.txt` | Python dependencies required to run the project. |
| `README.md` | Project documentation and model evaluation results. |
| `test_data.csv` | Held-out test dataset containing input features and target values. |
| `breast_cancer_wisconsin.csv` | Complete dataset used for the assignment. |
| `data/` | Directory containing the dataset. |
| `model/` | Directory containing the five saved trained classification models and metadata. |
| `model_metrics.csv` | Evaluation metrics calculated for all five classification models. |
| `observations.json` | Model evaluation results, confusion matrices, and classification reports. |
| `source_code/` | Directory containing the project Jupyter Notebook. |

### Saved Machine Learning Models

The `model/` directory contains the following trained models:

1. `logistic_regression.joblib`
2. `decision_tree.joblib`
3. `knn.joblib`
4. `naive_bayes.joblib`
5. `svm.joblib`

The saved models are loaded by the Streamlit application for prediction and evaluation.

---

## g. Reproducibility

The following settings are used to maintain reproducibility:

- Train-test split: 80:20.
- Stratified splitting is used to preserve the class distribution.
- Random state: 42.
- StandardScaler is applied within the pipelines for Logistic Regression, KNN, and SVM.
- Decision Tree and Gaussian Naive Bayes are implemented without feature scaling.
- The trained models are saved using Joblib.
- The evaluation metrics are calculated using the held-out test dataset.

The Jupyter Notebook contains the complete implementation, including data preparation, model training, prediction, and evaluation.

### Running the Streamlit Application Locally

To run the application locally, install the required dependencies:

```bash
pip install -r requirements.txt
```

Then execute:

```bash
streamlit run app.py
```

Upload `breast_cancer_wisconsin.csv` through the application to evaluate the selected trained classification model.

---

## h. Deployment Information

- **GitHub Repository:** https://github.com/SHWETHASENTHIL/ML_ASSIGNMENT_2/tree/main/ML_Assignment_2
- **Streamlit Community Cloud Application:** https://mlassignment2-ewjdgp7pykozxkj6hcmtis.streamlit.app/

The GitHub repository contains the complete project source code, trained models, dataset files, test data, dependencies, and documentation.

The Streamlit application provides an interactive interface for selecting the trained models and viewing their evaluation results.
