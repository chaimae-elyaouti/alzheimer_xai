import shap
import numpy as np

def compute_shap_explainer(model, X_train_transformed):
    """Initialise et retourne le TreeExplainer pour un modèle d'arbres."""
    return shap.TreeExplainer(model)

def get_patient_explanation(explainer, shap_values, patient_idx, pred_class, patient_transformed, feature_names):
    """Génère l'objet Explanation requis pour le tracé de type Waterfall."""
    return shap.Explanation(
        values=shap_values[patient_idx, :, pred_class],
        base_values=explainer.expected_value[pred_class],
        data=patient_transformed[patient_idx],
        feature_names=feature_names
    )