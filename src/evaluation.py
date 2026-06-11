import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

def generate_performance_table(y_test, y_pred):
    """Retourne le dictionnaire des performances sous forme de DataFrame."""
    report = classification_report(y_test, y_pred, target_names=['CN', 'MCI', 'AD'], output_dict=True)
    return pd.DataFrame(report).transpose()