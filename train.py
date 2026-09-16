from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
import re

import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

mlflow.set_tracking_uri("sqlite:///mlflow.db")


class CreditPreprocessor:
    """Handles dataset loading, header formatting, and building pre-processing."""
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state


    def feature_engineering(self, df: pd.DataFrame):
        # Drop Column
        df = df.drop(['Unnamed: 0', 'ID', 'Customer_ID', 'Name', 'SSN', 'Month'],axis=1)

        df['Age'] = df['Age'].str.strip('_')
        df['Age'] = pd.to_numeric(df['Age'])
        df['Annual_Income'] = df['Annual_Income'].str.strip('_')
        df['Annual_Income'] = pd.to_numeric(df['Annual_Income'])
        df['Num_of_Loan'] = df['Num_of_Loan'].str.strip('_')
        df['Num_of_Loan'] = pd.to_numeric(df['Num_of_Loan'])
        df['Num_of_Delayed_Payment'] = df['Num_of_Delayed_Payment'].str.strip('_')
        df['Num_of_Delayed_Payment'] = pd.to_numeric(df['Num_of_Delayed_Payment'])
        df['Changed_Credit_Limit'] = pd.to_numeric(df['Changed_Credit_Limit'], errors='coerce')
        df['Outstanding_Debt'] = df['Outstanding_Debt'].str.strip('_')
        df['Outstanding_Debt'] = pd.to_numeric(df['Outstanding_Debt'])
        df['Amount_invested_monthly'] = df['Amount_invested_monthly'].str.strip('_')
        df['Amount_invested_monthly'] = pd.to_numeric(df['Amount_invested_monthly'])

        # Fix Anomaly Kaetogorik
        df['Occupation'] = df['Occupation'].replace('_______', 'Unknown')
        df['Credit_Mix'] = df['Credit_Mix'].replace('_', 'Unknown')
        df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].replace('NM', 'Unknown')
        df['Payment_Behaviour'] = df['Payment_Behaviour'].replace('!@9#%8', 'Unknown')

        # Fix Anomaly Numerik
        df.loc[(df['Age'] < 18) | (df['Age'] > 100), 'Age'] = np.nan
        df.loc[(df['Num_Bank_Accounts'] < 1) | (df['Num_Bank_Accounts'] > 15), 'Num_Bank_Accounts'] = np.nan
        df = df[(df['Num_of_Loan'] >= 0) & (df['Num_of_Loan'] <= 10)]
        df = df[(df['Delay_from_due_date'] >= 0)]
        df.loc[(df['Num_of_Delayed_Payment'] < 0) | (df['Num_of_Delayed_Payment'] > 30), 'Num_of_Delayed_Payment'] = np.nan
        df = df[(df['Interest_Rate'] <= 35)]
        df = df[(df['Num_Credit_Card'] <= 15)]
        df = df[(df['Num_Credit_Inquiries'] <= 20)]
        df = df[(df['Annual_Income'] <= 200000)]

        # Cek Jumlah Anomali Amount_invested_monthly
        df['monthly_income'] = df['Annual_Income'] / 12
        df['invest_ratio'] = df['Amount_invested_monthly'] / df['monthly_income']
        df.loc[df['invest_ratio'] > 1, 'Amount_invested_monthly'] = np.nan
        df.drop(columns=['monthly_income', 'invest_ratio'], inplace=True)

        # Cek Jumlah Anomali Total_EMI_per_month
        df['monthly_income'] = df['Annual_Income'] / 12
        df['emi_ratio'] = df['Total_EMI_per_month'] / df['monthly_income']
        df = df[(df['emi_ratio'] <= 1)]
        df.drop(columns=['monthly_income', 'emi_ratio'], inplace=True)

        # Drop Column Monthly_Inhand_Salary
        df.drop(columns=['Monthly_Inhand_Salary'], inplace=True)

        # Mengubah Age Years Menjadi Month
        def parse_to_months(text):
            if pd.isna(text):
                return np.nan
            match = re.match(r'(\d+) Years and (\d+) Months', text)
            if match:
                years, months = int(match.group(1)), int(match.group(2))
                return years * 12 + months
            return np.nan
        df['Credit_History_Age'] = df['Credit_History_Age'].apply(parse_to_months)

        # Parsing Type_of_Loan
        def parse_loans(text):
            if pd.isna(text):
                return []
            cleaned = text.replace(', and ', ', ').replace(' and ', ', ')
            return [loan.strip() for loan in cleaned.split(', ')]
        df['Loan_List'] = df['Type_of_Loan'].apply(parse_loans)
        loan_types = ['Auto Loan', 'Credit-Builder Loan', 'Debt Consolidation Loan', 'Home Equity Loan', 'Mortgage Loan', 'Not Specified', 'Payday Loan', 'Personal Loan', 'Student Loan']
        for loan in loan_types:
            col_name = 'Has_' + loan.replace(' ', '_').replace('-', '_')
            df[col_name] = df['Loan_List'].apply(lambda x: 1 if loan in x else 0)
        df.drop(columns=['Type_of_Loan', 'Loan_List'], inplace=True)

        # Encode Target
        score_map = {'Poor': 0, 'Standard': 1, 'Good': 2}
        df['Credit_Score'] = df['Credit_Score'].map(score_map)

        return df


    def clean_and_split(self, data_path: str | Path):
        df = pd.read_csv(Path(data_path), sep=",")
        
        df = self.feature_engineering(df)

        input_df=df.drop(['Credit_Score'],axis=1)
        output_df=df['Credit_Score']

        return train_test_split(input_df, output_df, test_size=self.test_size, random_state=self.random_state, stratify=output_df)


    def get_transformer(self):

        num_cols = ['Age', 'Annual_Income', 'Num_Bank_Accounts', 'Num_Credit_Card',
                    'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
                    'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
                    'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Credit_History_Age',
                    'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance']
        bin_cols = ['Has_Auto_Loan', 'Has_Credit_Builder_Loan', 'Has_Debt_Consolidation_Loan',
                    'Has_Home_Equity_Loan', 'Has_Mortgage_Loan', 'Has_Not_Specified',
                    'Has_Payday_Loan', 'Has_Personal_Loan', 'Has_Student_Loan']
        cat_cols = ['Occupation', 'Payment_of_Min_Amount', 'Payment_Behaviour']
        ord_cols = ['Credit_Mix']
        
        num_preprocess = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('num_scaler', StandardScaler())
        ])

        bin_preprocess = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('bin_encoder', OneHotEncoder(drop='if_binary', handle_unknown='ignore'))
        ])

        cat_preprocess = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])

        order_cremix = ['Bad', 'Standard', 'Good', 'Unknown']
        ord_preprocess = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ord_encoder', OrdinalEncoder(categories=[order_cremix]))
        ])
        
        return ColumnTransformer(transformers=[
            ('numPreprocess', num_preprocess, num_cols),
            ('binPreprocess', bin_preprocess, bin_cols),
            ('catPreprocess', cat_preprocess, cat_cols),
            ('ordPreprocess', ord_preprocess, ord_cols)
        ])



class CreditModelTrainer:
    """Handles feature definition, pipeline building, training, and artifact tracking."""
    
    def __init__(self, experiment_name: str = "Customer Credit Prediction", 
                artifact_path: str = "artifacts", random_state: int = 42):
        self.experiment_name = experiment_name
        self.artifact_dir = Path(artifact_path)
        self.preprocessor = CreditPreprocessor()
        self.random_state = random_state
        
        # Ensure artifact directory exists
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_experiment(self.experiment_name)
        
    def run(self, data_path: str | Path):
        
        x_train, x_test, y_train, y_test = self.preprocessor.clean_and_split(data_path)
        preprocess = self.preprocessor.get_transformer()
        
        models = {
            # --- LOGISTIC REGRESSION ---
            'LR_1': Pipeline([('preprocess', preprocess), ('model', LogisticRegression(C=1.0, solver='lbfgs', class_weight='balanced', random_state=42, max_iter=1000))]),
            'LR_2': Pipeline([('preprocess', preprocess), ('model', LogisticRegression(C=0.1, solver='lbfgs', class_weight='balanced', random_state=42, max_iter=1000))]),
            'LR_3': Pipeline([('preprocess', preprocess), ('model', LogisticRegression(C=1.0, penalty='l1', solver='saga', class_weight='balanced', random_state=42, max_iter=1000))]),
            'LR_4': Pipeline([('preprocess', preprocess), ('model', LogisticRegression(C=0.01, solver='lbfgs', class_weight='balanced', random_state=42, max_iter=1000))]),
            'LR_5': Pipeline([('preprocess', preprocess), ('model', LogisticRegression(C=0.5, penalty='elasticnet', solver='saga', l1_ratio=0.5, class_weight='balanced', random_state=42, max_iter=1000))]),

            # --- RANDOM FOREST ---
            'RF_1': Pipeline([('preprocess', preprocess), ('model', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1))]),
            'RF_2': Pipeline([('preprocess', preprocess), ('model', RandomForestClassifier(n_estimators=300, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1))]),
            'RF_3': Pipeline([('preprocess', preprocess), ('model', RandomForestClassifier(n_estimators=200, criterion='entropy', class_weight='balanced', random_state=42, n_jobs=-1))]),
            'RF_4': Pipeline([('preprocess', preprocess), ('model', RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=4, class_weight='balanced', random_state=42, n_jobs=-1))]),
            'RF_5': Pipeline([('preprocess', preprocess), ('model', RandomForestClassifier(n_estimators=500, max_depth=20, max_features='log2', class_weight='balanced', random_state=42, n_jobs=-1))]),

            # --- LIGHTGBM ---
            'LGBM_1': Pipeline([('preprocess', preprocess), ('model', LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.3, class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1))]),
            'LGBM_2': Pipeline([('preprocess', preprocess), ('model', LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1))]),
            'LGBM_3': Pipeline([('preprocess', preprocess), ('model', LGBMClassifier(n_estimators=200, max_depth=10, learning_rate=0.1, class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1))]),
            'LGBM_4': Pipeline([('preprocess', preprocess), ('model', LGBMClassifier(n_estimators=500, max_depth=4, learning_rate=0.01, class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1))]),
            'LGBM_5': Pipeline([('preprocess', preprocess), ('model', LGBMClassifier(n_estimators=400, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1))]),
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = []

        for name, model in models.items():
            with mlflow.start_run(run_name=name):
                scores = cross_validate(
                    model, x_train, y_train, cv=cv,
                    scoring='f1_macro',
                    n_jobs=-1
                )

                f1_mean = scores['test_score'].mean()
                f1_std = scores['test_score'].std()

                # Log hyperparameter model ini
                model_params = model.named_steps['model'].get_params()
                mlflow.log_params(model_params)
                mlflow.set_tag("Model_Type", name)

                # Log metric CV
                mlflow.log_metric("cv_f1_macro_mean", f1_mean)
                mlflow.log_metric("cv_f1_macro_std", f1_std)

                cv_results.append({
                    'Model': name,
                    'CV F1 Macro Mean': f1_mean,
                    'CV F1 Macro Std': f1_std
                })

        cv_results_df = pd.DataFrame(cv_results).sort_values(
            by=['CV F1 Macro Mean'], ascending=False
        )

        best_model_name = cv_results_df.iloc[0]['Model']
        print(f"Model Terpilih: {best_model_name}")

        best_model = models[best_model_name]

        with mlflow.start_run(run_name=f"{best_model_name}_FINAL") as run:
            best_model.fit(x_train, y_train)

            best_params = best_model.named_steps['model'].get_params()
            mlflow.log_params(best_params)
            mlflow.log_metric("cv_f1_macro_mean", cv_results_df.iloc[0]['CV F1 Macro Mean'])
            mlflow.set_tag("Model_Type", best_model_name)
            mlflow.set_tag("Is_Final_Model", "True")

            model_file_path = self.artifact_dir / "credit_prediction_pipeline.pkl"
            joblib.dump(best_model, model_file_path, compress=('lzma', 9))
            mlflow.sklearn.log_model(best_model, name="model")
            
            print(f"✅ Model {best_model_name} trained & saved locally to {model_file_path}")
            return run.info.run_id, x_test, y_test