import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, LabelEncoder

class DataCleanerAndFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Phase 1 & 2: Custom Transformer for Data Cleaning and Feature Engineering
    """
    def __init__(self):
        # Save median value for imputation during fit to prevent data leakage from the test set
        self.total_charges_median_ = None
        
        # List of potential active services we want to aggregate based on your requirements
        self.service_features_ = [
            'PhoneService', 'MultipleLines', 'InternetService', 
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]

    def fit(self, X, y=None):
        X_temp = X.copy()
        
        # Calculate median for TotalCharges handling errors for coercion
        if 'TotalCharges' in X_temp.columns:
            X_total_charges = pd.to_numeric(X_temp['TotalCharges'], errors='coerce')
            self.total_charges_median_ = X_total_charges.median()
            
        return self

    def transform(self, X):
        X = X.copy()

        # ==========================================
        # Phase 1: Data Cleaning & Preprocessing
        # ==========================================

        # 1. Remove customerID column
        if 'customerID' in X.columns:
            X.drop('customerID', axis=1, inplace=True)
            
        # 2. Convert TotalCharges to numeric & Impute missing values
        if 'TotalCharges' in X.columns:
            X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
            
            # Impute with median calculated during fit!
            if self.total_charges_median_ is not None:
                X['TotalCharges'] = X['TotalCharges'].fillna(self.total_charges_median_)
            
        # 3. Strip whitespace from categorical columns
        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            X[col] = X[col].astype(str).str.strip()

        # ==========================================
        # Phase 2: Feature Engineering
        # ==========================================

        if 'tenure' in X.columns:
            # 1. tenure_group bucketization
            conditions = [
                (X['tenure'] <= 12),
                (X['tenure'] > 12) & (X['tenure'] <= 24),
                (X['tenure'] > 24) & (X['tenure'] <= 48),
                (X['tenure'] > 48)
            ]
            choices = ['New', 'Mid', 'Loyal', 'Long-term']
            X['tenure_group'] = np.select(conditions, choices, default='Unknown')

            # 2. avg_charge_per_month (TotalCharges / tenure)
            if 'TotalCharges' in X.columns:
                # Avoid division by zero by safely setting 0 tenure to 0 avg charge
                X['avg_charge_per_month'] = np.where(X['tenure'] == 0, 0, X['TotalCharges'] / X['tenure'])

        # 3. services_count
        avail_services = [col for col in self.service_features_ if col in X.columns]
        if avail_services:
            # Count boolean instances where service actively equals "Yes"
            X['services_count'] = (X[avail_services] == 'Yes').sum(axis=1)

        return X


def build_preprocessing_pipeline(X_train):
    """
    Phase 3: Encoding 
    Builds the structural ColumnTransformer Pipeline using sklearn.
    """
    # 1. First, process a small snippet of data to definitively map column types post-feature engineering
    fe_step = DataCleanerAndFeatureEngineer()
    X_sample_transformed = fe_step.fit_transform(X_train.head(10))
    
    binary_cols = []
    multi_cols = []
    numeric_cols = []
    
    # Intelligently bucket columns to their applicable Encoding Strategy
    for col in X_sample_transformed.columns:
        # Check numeric strictly first to avoid Pandas str casting bugs
        if pd.api.types.is_numeric_dtype(X_sample_transformed[col]):
            numeric_cols.append(col)
        else:
            if X_sample_transformed[col].nunique() <= 2:
                binary_cols.append(col)
            else:
                multi_cols.append(col)

    # 2. Construct ColumnTransformer
    # - Binary cols -> OrdinalEncoder maps 0/1 properly for features
    # - Multi-class cols -> OneHotEncoder
    # - Numeric features -> pass-through or optional scaling
    preprocessor = ColumnTransformer(
        transformers=[
            ('binary', OrdinalEncoder(), binary_cols),
            ('multi', OneHotEncoder(drop='first', sparse_output=False), multi_cols),
            ('numeric', StandardScaler(), numeric_cols) # Scaling numeric helps estimators like LR & SVMs
        ],
        remainder='passthrough'
    )

    # 3. Assemble and return the complete Machine Learning Pipeline
    pipeline = Pipeline(steps=[
        ('cleaning_and_engineering', fe_step),
        ('encoding_and_scaling', preprocessor)
    ])

    return pipeline


def preprocess_target(y):
    """
    Binary encoding specifically applied isolated to the label target.
    Expects input shape target (Yes -> 1, No -> 0).
    """
    le = LabelEncoder()
    # Returns encoded array (y) and actual LabelEncoder obj to easily reverse interpret
    return le.fit_transform(y), le
