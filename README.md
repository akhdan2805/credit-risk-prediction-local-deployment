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
```

### 4. Train-Test Split

The prepared dataset was divided into **80% training data and 20% testing data** using stratified sampling to preserve the original class distribution.

This resulted in:

- **Training set:** 16,819 records
- **Test set:** 4,205 records

A fixed `random_state=42` was used to ensure reproducibility across experiments.

### 5. Preprocessing Pipeline

A structured preprocessing pipeline was built using `ColumnTransformer` and Scikit-learn `Pipeline` to apply different transformations based on feature types.

**Numerical features** were processed using median imputation followed by standard scaling.

**Binary features** were handled using most-frequent imputation and one-hot encoding.

**Categorical features** were processed using most-frequent imputation and one-hot encoding.

**Ordinal features**, particularly `Credit_Mix`, were encoded according to their defined order.

By integrating preprocessing directly into the model pipeline, the same transformations can be consistently applied during training, evaluation, and inference.

## Model Development

### 6. Model Experimentation

Three machine learning algorithms were evaluated for the credit risk classification task:

- **Logistic Regression**
- **Random Forest**
- **LightGBM**

Rather than relying on a single default configuration, multiple configurations were tested for each algorithm by varying their respective hyperparameters.

A total of **15 model configurations** were evaluated:

| Algorithm | Configurations |
|---|---:|
| Logistic Regression | 5 |
| Random Forest | 5 |
| LightGBM | 5 |
| **Total** | **15** |

Class balancing was incorporated into the models using `class_weight='balanced'` to account for differences in the distribution of the target classes.

### 7. Cross-Validation

Model selection was performed using **5-Fold Stratified Cross-Validation** with shuffled folds and `random_state=42`.

The primary evaluation metric was **Macro F1-Score**, which gives equal importance to all target classes regardless of their frequency.

For each model configuration, the mean and standard deviation of the Macro F1-Score across the five folds were recorded and compared.

### 8. Model Selection

Based on the cross-validation results, **Random Forest configuration RF_3** achieved the highest mean Macro F1-Score and was selected for final evaluation.

The selected configuration was:

```text
Model          : Random Forest
n_estimators   : 200
criterion      : entropy
class_weight   : balanced
random_state   : 42
```
