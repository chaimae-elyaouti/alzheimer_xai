import os
import numpy as np
import pandas as pd

def generate_alzheimer_dataset(n_samples=1500):
    """
    Génère un dataset tabulaire cliniquement cohérent
    inspiré des features OASIS-3 / ADNI.
    """
    np.random.seed(42)
    
    n_cn  = int(n_samples * 0.45)   
    n_mci = int(n_samples * 0.35)   
    n_ad  = n_samples - n_cn - n_mci  

    def make_group(n, label, params):
        d = {}
        # Démographie
        d['Age'] = np.random.normal(params['age_mu'], params['age_sd'], n).clip(55, 95)
        d['Sex'] = np.random.choice(['M', 'F'], n, p=params['sex_p'])
        d['Education'] = np.random.normal(params['edu_mu'], 2.0, n).clip(0, 20).astype(int)

        # Score cognitif (MMSE) lié à l'âge
        mmse_base = np.random.normal(params['mmse_mu'], params['mmse_sd'], n)
        age_effect = (d['Age'] - 70) * params['mmse_age_coef']
        d['MMSE'] = (mmse_base + age_effect).clip(0, 30)

        d['CDR'] = np.random.choice(params['cdr_vals'], n, p=params['cdr_probs'])

        # Biomarqueurs cérébraux IRM
        d['eTIV'] = np.random.normal(1480, 180, n).clip(900, 2100)
        nwbv_base = np.random.normal(params['nwbv_mu'], 0.015, n)
        d['nWBV'] = (nwbv_base).clip(0.60, 0.90)
        d['ASF'] = 1764.0 / d['eTIV'] + np.random.normal(0, 0.03, n)

        # Biomarqueurs LCR/Sanguins
        d['Abeta42'] = np.random.normal(params['abeta_mu'], 80, n).clip(100, 800)
        d['pTau181'] = np.random.normal(params['ptau_mu'], 15, n).clip(10, 150)
        d['NfL'] = np.random.normal(params['nfl_mu'], 10, n).clip(5, 120)

        # Perte d'autonomie
        d['FAQ'] = np.random.normal(params['faq_mu'], params['faq_sd'], n).clip(0, 30).round()

        # Génétique & Comorbidités
        d['APOE4'] = np.random.choice([0, 1, 2], n, p=params['apoe4_p'])
        d['Hypertension']   = np.random.binomial(1, params['htn_p'],   n)
        d['Diabetes']       = np.random.binomial(1, params['diab_p'],  n)
        d['Depression_Hx']  = np.random.binomial(1, params['depr_p'],  n)

        d['Label'] = label
        return pd.DataFrame(d)

    params_cn = dict(age_mu=68, age_sd=7, sex_p=[0.45, 0.55], edu_mu=14, mmse_mu=28.5, mmse_sd=1.2, mmse_age_coef=-0.04, cdr_vals=[0, 0.5], cdr_probs=[0.95, 0.05], nwbv_mu=0.78, abeta_mu=580, ptau_mu=22, nfl_mu=18, faq_mu=0.5, faq_sd=1.0, apoe4_p=[0.70, 0.25, 0.05], htn_p=0.28, diab_p=0.10, depr_p=0.12)
    params_mci = dict(age_mu=73, age_sd=7, sex_p=[0.48, 0.52], edu_mu=13, mmse_mu=25.5, mmse_sd=2.5, mmse_age_coef=-0.06, cdr_vals=[0, 0.5, 1], cdr_probs=[0.10, 0.75, 0.15], nwbv_mu=0.73, abeta_mu=420, ptau_mu=42, nfl_mu=35, faq_mu=5, faq_sd=4.0, apoe4_p=[0.50, 0.35, 0.15], htn_p=0.40, diab_p=0.18, depr_p=0.25)
    params_ad = dict(age_mu=78, age_sd=7, sex_p=[0.42, 0.58], edu_mu=11, mmse_mu=17.0, mmse_sd=4.5, mmse_age_coef=-0.10, cdr_vals=[0.5, 1, 2, 3], cdr_probs=[0.05, 0.45, 0.38, 0.12], nwbv_mu=0.67, abeta_mu=250, ptau_mu=88, nfl_mu=72, faq_mu=18, faq_sd=6.0, apoe4_p=[0.30, 0.40, 0.30], htn_p=0.52, diab_p=0.28, depr_p=0.40)

    df = pd.concat([make_group(n_cn, 'CN', params_cn), make_group(n_mci, 'MCI', params_mci), make_group(n_ad, 'AD', params_ad)], ignore_index=True)
    
    missing_pattern = {'Abeta42': 0.15, 'pTau181': 0.15, 'NfL': 0.20, 'MMSE': 0.03, 'FAQ': 0.05}
    for col, rate in missing_pattern.items():
        df.loc[np.random.rand(len(df)) < rate, col] = np.nan

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df['Subject_ID'] = [f'SUB_{i:04d}' for i in range(len(df))]
    return df

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_path = os.path.join(current_dir, "..", "data", "raw", "alzheimer_clinical.csv")
    
    # Sécurité pour créer les dossiers parents s'ils manquent
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    df = generate_alzheimer_dataset(1500)
    df.to_csv(target_path, index=False)
    
    print("[SUCCESS] Jeu de données matérialisé avec succès !")
    print(df.groupby('Label').size())