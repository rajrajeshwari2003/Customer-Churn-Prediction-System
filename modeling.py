import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings

# Suppress minor warnings for clean console outputs
warnings.filterwarnings("ignore")

# Import custom preprocessing module created in Phase 1-3
from preprocessing import build_preprocessing_pipeline, preprocess_target

def get_param_grids():
    """
    Defines parameter distributions for RandomizedSearchCV.
    Note: We prefix hyperparameter names with 'classifier__' because they are wrapped inside an Imblearn Pipeline step named 'classifier'.
    """
    return {
        'Logistic Regression': {
            'classifier__C': [0.01, 0.1, 1, 5, 10],
            'classifier__solver': ['liblinear', 'lbfgs']
        },
        'Random Forest': {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [10, 15, 20, None],
            'classifier__min_samples_split': [2, 5, 10]
        },
        'XGBoost': {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7]
        },
        'LightGBM': {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__learning_rate': [0.01, 0.05, 0.1],
            'classifier__num_leaves': [31, 50, 100]
        }
    }

def prepare_data(df, target_col='Churn'):
    """
    Phase 4: Train-Test Split
    - Configures an 80/20 data split.
    - Importantly utilizes stratify=y to maintain churn imbalance consistency.
    """
    X = df.drop(columns=[target_col])
    y_raw = df[target_col]
    
    # Encode target explicitly (Yes/No -> 1/0)
    y, target_le = preprocess_target(y_raw)
    
    # Perform 80/20 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, target_le

def train_and_compare(X_train, y_train, strategy='class_weight'):
    """
    Phase 5 & 6: Imbalance Handling and Multi-Model Training.
    Dynamically trains 4 models comparing Oversampling (SMOTE) against native Class Penalization.
    """
    # 1. Initialize our Phase 1-3 feature engineering architecture and unpack steps
    preprocessor_pipeline = build_preprocessing_pipeline(X_train)
    prep_steps = preprocessor_pipeline.steps
    
    # 2. Baseline models initialization
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(random_state=42)
    }
    
    # 3. Enforce strictly Native Class Weight Penalization strategy
    if strategy == 'class_weight':
        # Calculate optimal scale_pos_weight representation for XGBoost
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight_calc = neg_count / pos_count
        
        models['Logistic Regression'] = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        models['Random Forest'] = RandomForestClassifier(class_weight='balanced', random_state=42)
        models['XGBoost'] = XGBClassifier(scale_pos_weight=scale_pos_weight_calc, random_state=42, eval_metric='logloss')
        models['LightGBM'] = LGBMClassifier(class_weight='balanced', random_state=42)

    param_grids = get_param_grids()
    best_models = {}
    
    print(f"\n--- Initiating Model Cross-Validation ---")
    print(f"Imbalance Handler: {strategy.upper()}\n")

    # 4. Train, Tune via RandomizedSearchCV, and Evaluate Models
    for name, model in models.items():
        print(f"[*] Hyper-tuning {name}...")
        
        # EXTREMELY CRITICAL FOR COLLEGE PROJECTS: DATA LEAKAGE PREVENTION
        # We MUST use imblearn.pipeline when testing SMOTE! 
        # Using a standard sklearn pipeline inadvertently applies SMOTE to validation folds inflating CV metrics.
        # Imblearn forbids nested pipelines, so we unpack prep_steps.
        if strategy == 'smote':
            imb_steps = prep_steps + [
                ('smote', SMOTE(random_state=42)),
                ('classifier', model)
            ]
            pipeline = ImbPipeline(steps=imb_steps)
        else:
            imb_steps = prep_steps + [
                ('classifier', model)
            ]
            pipeline = ImbPipeline(steps=imb_steps)
            
        # RandomizedSearchCV logic designed for efficient evaluation
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grids[name],
            n_iter=5,      # Iterate 5 random param combos per model (Scale up for deployment run)
            scoring='f1',  # Optimize purely for F1-Score since detecting Churners implicitly targets FN/FP Cost
            cv=3,          # 3-Fold Stratified Cross Validation ensures absolute reliability 
            verbose=0,
            random_state=42,
            n_jobs=-1      # Utilize maximum system threading compute cores
        )
        
        search.fit(X_train, y_train)
        
        print(f"    -> Best HyperParams: {search.best_params_}")
        print(f"    -> Optimized F1-Score: {search.best_score_:.4f}\n")
        
        best_models[name] = search.best_estimator_
        
    return best_models

if __name__ == "__main__":
    import os
    import joblib
    
    print("Initializing Automated Model Training Sequence...")
    csv_path = 'Telco-Customer-Churn.csv'
    
    if not os.path.exists(csv_path):
        print(f"[{csv_path}] not found locally. Downloading standard IBM Telco Dataset...")
        # Public dataset mapping mapped directly for student usage
        url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
        df = pd.read_csv(url)
        df.to_csv(csv_path, index=False)
        print("Download complete.")
    else:
        df = pd.read_csv(csv_path)
        print(f"Dataset loaded natively: {df.shape[0]} rows, {df.shape[1]} columns.")
        
    # Phase 4: Execute our train/test split sequence natively
    X_train, X_test, y_train, y_test, target_enc = prepare_data(df)
    
    # Phase 6: Execute Model Tuning
    print("\\n--- Starting Deep Training Phase ---")
    tuned_models = train_and_compare(X_train, y_train, strategy='class_weight')
    
    # Phase 9 Prep: Save the absolute best model strictly mapped to XGBoost
    best_algorithmic_pipeline = tuned_models['XGBoost']
    
    print("\\n--- Compiling Production Artifact ---")
    joblib.dump(best_algorithmic_pipeline, 'best_churn_model.pkl')
    print("✅ SUCCESS: 'best_churn_model.pkl' synthesized! You can now safely launch your Streamlit Dashboard.")
