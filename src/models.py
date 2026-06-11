import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

def get_candidate_models(preprocessor, class_weights):
    """
    Retourne le dictionnaire des pipelines de modèles candidats.
    """
    return {
        'Logistic_Regression': Pipeline([
            ('preprocessor', preprocessor),
            ('clf', LogisticRegression(
                class_weight='balanced', max_iter=1000,
                multi_class='multinomial', C=1.0, random_state=42
            ))
        ]),
        
        'Random_Forest': Pipeline([
            ('preprocessor', preprocessor),
            ('clf', RandomForestClassifier(
                n_estimators=300, class_weight='balanced_subsample',
                random_state=42, n_jobs=-1
            ))
        ]),
        
        'XGBoost': Pipeline([
            ('preprocessor', preprocessor),
            ('clf', XGBClassifier(
                objective='multi:softprob', num_class=3,
                eval_metric='mlogloss', random_state=42, n_jobs=-1
            ))
        ]),
        
        'LightGBM': Pipeline([
            ('preprocessor', preprocessor),
            ('clf', LGBMClassifier(
                objective='multiclass', num_class=3,
                class_weight='balanced', random_state=42,
                n_jobs=-1, verbose=-1
            ))
        ]),
        
        'CatBoost': Pipeline([
            ('preprocessor', preprocessor),
            ('clf', CatBoostClassifier(
                loss_function='MultiClass', class_weights=list(class_weights),
                random_seed=42, verbose=0,
                allow_writing_files=False
            ))
        ]),
    }