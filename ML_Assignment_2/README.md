# Machine Learning Assignment 2 – Breast Cancer Classification

## a. Problem statement
Build and compare five supervised classification models to predict whether a breast tumour is malignant or benign using diagnostic measurements. This is a binary classification task.

## b. Dataset description
- **Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC)
- **Source:** UCI Machine Learning Repository, dataset 17; accessed through `sklearn.datasets.load_breast_cancer`.
- **Instances:** 569
- **Input features:** 30 numeric features
- **Target:** malignant / benign
- **Classes:** 212 malignant and 357 benign (full dataset)
- **Split:** stratified 80% training / 20% testing, `random_state=42`.
- **Missing values:** none in the loaded dataset.
- **License/source citation:** Wolberg, W., Street, N., & Mangasarian, O. (1993). Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B

## c. GitHub Repository Link


## d. Models used
All models use the same train/test split. StandardScaler is fitted within pipelines for Logistic Regression, KNN and SVM. Decision Tree and Gaussian Naive Bayes are not scaled.

### Evaluation comparison

| Model                  |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:-----------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression    |     0.9825 | 0.9954 |      0.9861 |   0.9861 | 0.9861 | 0.9623 |
| Decision Tree          |     0.9123 | 0.9157 |      0.9559 |   0.9028 | 0.9286 | 0.8174 |
| K-Nearest Neighbors    |     0.9561 | 0.9788 |      0.9589 |   0.9722 | 0.9655 | 0.9054 |
| Gaussian Naive Bayes   |     0.9386 | 0.9878 |      0.9452 |   0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine |     0.9825 | 0.995  |      0.9861 |   0.9861 | 0.9861 | 0.9623 |

**Metric convention:** Precision, Recall and F1 treat `benign` (class 1) as the positive class. AUC uses the probability of the benign class. MCC is calculated over both classes.

### Observations on model performance

| Model | Observation |
|---|---|
| Logistic Regression | Achieved 98.25% accuracy, 0.9954 AUC and 0.9861 F1. It tied for the best F1 and MCC on this split, showing strong class separation and balanced classification performance. |
| Decision Tree | Achieved 91.23% accuracy and 0.9286 F1, the lowest overall scores among the five models on this split. Its standalone tree may capture nonlinear rules, but its measured generalization was weaker here. |
| KNN | Achieved 95.61% accuracy and 0.9655 F1. Scaling was applied before distance calculations; performance was strong but below Logistic Regression and SVM on this test split. |
| Gaussian Naive Bayes | Achieved 93.86% accuracy, 0.9878 AUC and 0.9517 F1. Its high AUC indicates good ranking ability, while its threshold-based classification scores were lower than the top two models. |
| Support Vector Machine | Achieved 98.25% accuracy, 0.9950 AUC and 0.9861 F1. It tied with Logistic Regression on accuracy, F1 and MCC, with a marginally lower AUC on this split. |

**Overall winner:** Logistic Regression. Selection rule: highest F1 score, with AUC and MCC as tie-breakers. Logistic Regression and SVM tie on F1 and MCC; Logistic Regression has the slightly higher AUC (0.995370 vs. 0.995040), so it wins the tie-break. This conclusion applies only to this held-out test split and is not a claim that the model will be best on every dataset.

## Local application



## Project files

- `ML_Assignment_2.ipynb` – complete executable notebook
- `app.py` – local Streamlit application
- `requirements.txt` – dependencies
- `test_data.csv` – held-out test data with labels
- `data/breast_cancer_wisconsin.csv` – full dataset used
- `model/` – five fitted model pipelines and metadata
- `model_metrics.csv` – measured test-set metrics

## Reproducibility
The notebook uses a stratified 80:20 split with random state 42. Model parameters are explicitly set in the notebook. The test metrics above were calculated from the generated held-out split.

## Deployment / submission status
No GitHub repository was created. No Streamlit Cloud deployment was performed. No files were uploaded or published to any external service. The assignment brief requires GitHub and live Streamlit links in the final submission PDF; those requirements remain outstanding by design under the local-only instruction.
