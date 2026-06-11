import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer

def create_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering basé sur la littérature clinique Alzheimer.
    Injecte des scores composites et des ratios biomédicaux.
    """
    df = df.copy()
    
    # 1. Ratio Amyloid/Tau (Marqueur de la cascade moléculaire)
    df['Amyloid_Tau_Ratio'] = df['Abeta42'] / (df['pTau181'] + 1e-6)
    
    # 2. Score composite cognitif-fonctionnel (Cognition vs Autonomie)
    df['Cog_Functional_Score'] = (df['MMSE'] / 30.0) - (df['FAQ'] / 30.0)
    
    # 3. Atrophie cérébrale relative ajustée sur l'âge
    df['Brain_Age_Ratio'] = df['nWBV'] / (df['Age'] / 100.0)
    
    # 4. Indicateur de risque APOE4 pondéré selon la littérature (Farrer et al., 1997)
    df['APOE4_Risk'] = df['APOE4'].map({0: 1.0, 1: 3.0, 2: 15.0})
    
    # 5. Charge comorbide globale
    df['Comorbidity_Index'] = df['Hypertension'] + df['Diabetes'] + df['Depression_Hx']
    
    # 6. Neurofilaments ajustés sur l'âge
    df['NfL_Age_Adjusted'] = df['NfL'] / (df['Age'] / 70.0)
    
    # 7. Normalisation du score clinique CDR
    df['CDR_Normalized'] = df['CDR'] / 3.0
    
    return df

def build_preprocessing_pipeline(numerical_features, categorical_features):
    """
    Construit un ColumnTransformer standard de production pour l'imputation
    et la mise à l'échelle sans Data Leakage.
    """
    # Biomarqueurs complexes à forte proportion de valeurs manquantes (LCR)
    high_missing_num = ['Abeta42', 'pTau181', 'NfL']
    
    # Features numériques standards
    low_missing_num = [f for f in numerical_features if f not in high_missing_num]
    
    # ── Sous-pipeline numérique standard (Imputation par Médiane + Scaling) ──
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # ── Sous-pipeline pour biomarqueurs (KNN Imputer pour respecter les corrélations + Scaling) ──
    knn_pipeline = Pipeline([
        ('imputer', KNNImputer(n_neighbors=7, weights='distance')),
        ('scaler', StandardScaler())
    ])
    
    # ── Sous-pipeline catégoriel (Imputation Mode + Encodage Ordinal) ──
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    # Assemblage global du préprocesseur de production
    preprocessor = ColumnTransformer(transformers=[
        ('num_standard', num_pipeline,  low_missing_num),
        ('num_knn',      knn_pipeline,  high_missing_num),
        ('cat',          cat_pipeline,  categorical_features),
    ], remainder='drop')
    
    return preprocessor