import os
import numpy as np
import pandas as pd

def generate_alzheimer_dataset(n_samples=1500, random_state=42):
    """
    Génère une cohorte clinique synthétique pour la maladie d'Alzheimer
    en respectant les lois physiologiques de la littérature médicale.
    """
    np.random.seed(random_state)
    
    # 1. Définition des distributions de classes (CN, MCI, AD)
    classes = ['CN', 'MCI', 'AD']
    p_classes = [0.45, 0.35, 0.20] # Imbalance Ratio ~ 2.25
    labels = np.random.choice(classes, size=n_samples, p=p_classes)
    
    data = []
    
    for i, label in enumerate(labels):
        subject_id = f"SUB_{i+1:04d}"
        age = np.random.uniform(60, 90)
        sex = np.random.choice(['M', 'F'], p=[0.48, 0.52])
        education = np.random.choice([12, 14, 16, 18, 20], p=[0.3, 0.2, 0.2, 0.2, 0.1])
        
        # Statut génétique APOE4 (Facteur de risque)
        if label == 'CN':
            apoe4 = np.random.choice([0, 1, 2], p=[0.75, 0.22, 0.03])
        elif label == 'MCI':
            apoe4 = np.random.choice([0, 1, 2], p=[0.50, 0.40, 0.10])
        else: # AD
            apoe4 = np.random.choice([0, 1, 2], p=[0.30, 0.50, 0.20])
            
        # Facteurs cardiovasculaires (Comorbidités)
        hypertension = np.random.binomial(1, 0.4 if label == 'AD' else 0.3)
        diabetes = np.random.binomial(1, 0.25 if label == 'AD' else 0.18)
        depression = np.random.binomial(1, 0.35 if label == 'AD' else 0.20)
        
        # Volumes IRM (nWBV : atrophie, eTIV : stable, ASF : stable)
        eTIV = np.random.normal(1450, 120)
        ASF = 1.0 / (eTIV / 1450.0)
        
        if label == 'CN':
            nWBV = np.random.normal(0.78, 0.02)
            mmse = np.clip(np.random.normal(29, 1), 0, 30)
            faq = np.clip(np.random.exponential(0.5), 0, 30)
            cdr = 0.0
            # Biomarqueurs LCR (Sain = Amyloïde haute, Tau/NfL basses)
            abeta = np.random.normal(1000, 150)
            ptau = np.random.normal(20, 5)
            nfl = np.random.normal(800, 150)
        elif label == 'MCI':
            nWBV = np.random.normal(0.73, 0.03)
            mmse = np.clip(np.random.normal(25, 2), 0, 30)
            faq = np.clip(np.random.normal(4, 3), 0, 30)
            cdr = 0.5
            abeta = np.random.normal(600, 120)
            ptau = np.random.normal(45, 12)
            nfl = np.random.normal(1400, 300)
        else: # AD
            nWBV = np.random.normal(0.67, 0.04)
            mmse = np.clip(np.random.normal(16, 4), 0, 30)
            faq = np.clip(np.random.normal(18, 5), 0, 30)
            cdr = np.random.choice([1.0, 2.0], p=[0.7, 0.3])
            abeta = np.random.normal(350, 80)
            ptau = np.random.normal(85, 20)
            nfl = np.random.normal(2400, 500)
            
        data.append({
            'Subject_ID': subject_id, 'Age': round(age, 1), 'Sex': sex, 'Education': education,
            'MMSE': round(mmse, 1), 'CDR': cdr, 'eTIV': round(eTIV, 1), 'nWBV': round(nWBV, 3),
            'ASF': round(ASF, 3), 'Abeta42': round(abeta, 1), 'pTau181': round(ptau, 1),
            'NfL': round(nfl, 1), 'FAQ': round(faq, 1), 'APOE4': apoe4,
            'Hypertension': hypertension, 'Diabetes': diabetes, 'Depression_Hx': depression,
            'Label': label
        })
        
    df = pd.DataFrame(data)
    
    # Injection contrôlée de valeurs manquantes (Missingness clinique réaliste)
    mask_lcr = np.random.rand(len(df)) < 0.12 # 12% de données manquantes sur le LCR
    df.loc[mask_lcr, ['Abeta42', 'pTau181', 'NfL']] = np.nan
    
    return df

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    df = generate_alzheimer_dataset()
    df.to_csv("data/raw/alzheimer_clinical.csv", index=False)
    print("Jeu de données clinique régénéré avec succès dans : data/raw/alzheimer_clinical.csv")