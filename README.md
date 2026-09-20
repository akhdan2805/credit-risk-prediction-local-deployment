# 💳 Credit Risk Classification using Machine Learning

A machine learning project for classifying customer credit risk categories based on financial and behavioral attributes. The project covers data exploration, preprocessing, feature engineering, model experimentation, cross-validation, model selection, local ML pipeline development, and web deployment using Streamlit.

## Objective

The objective of this project is to develop a machine learning solution for credit risk classification. The project focuses on transforming raw financial data into a reliable machine learning pipeline through data exploration, preprocessing, feature engineering, and model experimentation.

Multiple machine learning algorithms and configurations were evaluated to identify a suitable model based on **Macro F1-Score**. The selected model was then integrated into a reproducible local training pipeline with **MLflow** for experiment tracking and prepared for web-based inference using **Streamlit**.

## Dataset

A provided financial dataset was used for this project. The dataset contains customer-related financial and behavioral information, with `Credit_Score` as the target variable.

The target consists of three categories:

| Credit Score | Label |
|---|---:|
| Poor | 0 |
| Standard | 1 |
| Good | 2 |

The data contains numerical, categorical, binary, and ordinal attributes related to customers' financial conditions, credit usage, loans, payment behavior, and credit history.

After data preparation and cleaning, the dataset was divided into:

- **Training set:** 16,819 records
- **Test set:** 4,205 records

## Methodology

### 1. Data Exploration

The dataset was first inspected to understand its structure, data types, missing values, and target distribution. Exploratory analysis was also performed to identify inconsistent values and potential anomalies that could affect model performance.

### 2. Data Cleaning

Several irrelevant identifier and personal-information columns were removed, including:

- `Unnamed: 0`
- `ID`
- `Customer_ID`
- `Name`
- `SSN`
- `Month`

Additional data quality checks were performed to identify inconsistent financial values and anomalous records before modeling.

### 3. Feature Engineering

Several transformations were applied to convert raw variables into more useful machine learning features.

`Credit_History_Age` was converted from a text representation of years and months into a numerical value in months.

`Type_of_Loan` was parsed into multiple binary features representing whether a customer has specific loan types, including:

- Auto Loan
- Credit-Builder Loan
- Debt Consolidation Loan
- Home Equity Loan
- Mortgage Loan
- Not Specified
- Payday Loan
- Personal Loan
- Student Loan

The target variable was encoded as:

```text
Poor     → 0
Standard → 1
Good     → 2
