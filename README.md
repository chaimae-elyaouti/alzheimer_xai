# AlzXAI — Un Système d'Aide à la Décision Clinique et de Classification Explicable du Continuum de la Maladie d'Alzheimer

> **Projet Encadré de Master (2ème année)** > **Module :** Data Science & Explainable AI (XAI)  
> **Institution :** Institut National de Statistique et d'Économie Appliquée (INSEA), Rabat  
> **Auteurs :** EL YAOUTI Chaimae & ATIF Imane (Filière Ingénierie Statistique & Data Science)  
> **Design :** Mode Clinique Futuriste Sombre  

---

##  Démo en Production & Accès Rapide
L'interface applicative et le moniteur de décision clinique sont déployés sur une infrastructure cloud publique stable et accessibles instantanément :  
👉 **Lien de la Démo en Direct :** [https://alzheimerxai-4xg626sjwcv6m6zvxphxit.streamlit.app/](https://alzheimerxai-4xg626sjwcv6m6zvxphxit.streamlit.app/)

---

## Présentation du Projet
Ce projet formule, optimise et déploie une architecture de production de bout en bout pour la classification non linéaire multiclasse des trois stades cliniques de la maladie d'Alzheimer :
* **CN** (*Cognitively Normal*) : Sujets sains, absence de déclin fonctionnel.
* **MCI** (*Mild Cognitive Impairment*) : Trouble cognitif léger (stade prodromal critique).
* **AD** (*Alzheimer's Disease*) : Démence neurodégénérative avérée.

L'innovation centrale de ce système réside dans le découplage de l'opacité algorithmique (*Black Box*) grâce à l'intégration conjointe de la théorie des jeux coopératifs (**SHAP**) et des approximations linéaires locales (**LIME**), garantissant un diagnostic auditable et scientifiquement justifiable auprès des équipes médicales.

---

##  Architecture du Pipeline & Choix Méthodologiques
[ Données Tabulaires (N=1500) ]
│
▼
[ Pipeline Preprocessing ]  ──► (Feature Engineering, Imputation KNN k=7)
│
▼
[ Champion : LightGBM ]    ──► (Entropie Croisée Pondérée, Class Weights)
│
▼
[ XAI Core Engine ]      ──► (SHAP TreeExplainer + Local LIME)
│
▼
[ Interactive Deployment ]   ──► (Streamlit UI Cloud, Python 3.11)

### 1. Data Engine & Anti-Leakage Protection
* **Cohorte d'étude :** $N = 1500$ observations et $p = 17$ variables initiales calibrées sur les distributions des bases de référence mondiales OASIS-3 et ADNI.
* **Gestion des Manquants :** Imputation par voisinages non paramétriques via `KNNImputer` ($k=7$, pondération par distance) encapsulée au sein d'un composant `Pipeline` Scikit-Learn pour interdire toute fuite par contamination (*Data Leakage*).
* **Target Leakage Tautologique :** Exclusion radicale de la variable `CDR` (*Clinical Dementia Rating*) qui constituait une reformulation déterministe directe de la cible.
* **Feature Engineering Clinique :** Création de 6 variables composites bio-cliniques clés :
    * `Amyloid_Tau_Ratio` = $A\beta_{42} / (\text{pTau181} + \varepsilon)$
    * `Cog_Functional_Score` = $(\text{MMSE} / 30) - (\text{FAQ} / 30)$
    * `Brain_Age_Ratio` = $\text{nWBV} / (\text{Age}/100)$
    * `APOE4_Risk` = Encodage pondéré selon la littérature ($0 \rightarrow 1.0, 1 \rightarrow 3.0, 2 \rightarrow 15.0$).

### 2. Moteur Algorithmique & Résolution de l'Imbalance
* **Algorithme retenu :** **LightGBM** (*Light Gradient Boosting Machine*) de Microsoft, choisi pour sa stratégie de croissance par feuille (*Leaf-wise*) minimisant directement la perte et son insensibilité native aux fortes colinéarités constatées lors de l'EDA ($\rho_S = -0.77$ entre atrophie cérébrale et neurofilaments).
* **Weighted Loss Function :** Injection des coefficients de pénalisation inverses (*Class Weights*) directement dans la perte d'entropie croisée multinomiale pour neutraliser le déséquilibre de classe ($\text{IR} = 2.25$) :
    $$\hat{w}_{\text{CN}} = 0.74, \quad \hat{w}_{\text{MCI}} = 0.95, \quad \hat{w}_{\text{AD}} = 1.67$$

### 3. Framework d'Optimisation Bayésienne
* **HPO Engine :** Utilisation d'**Optuna** combiné à un échantillonneur bayésien **TPE** (*Tree-structured Parzen Estimator*) sur 20 itérations.
* **Rigueur de validation :** Évaluation croisée stratifiée à 5 plis (`StratifiedKFold`, $k=5$) pour figer des hyperparamètres hautement régularisés et immunisés contre le sur-apprentissage.

---

##  Indicateurs de Performance (Test Set Hold-Out)

Les résultats calculés sur le jeu de test indépendant valident la puissance de généralisation de l'architecture :

| Métrique d'Évaluation | Score Obtenu | Statut |
| :--- | :---: | :---: |
| **Précision Globale (Accuracy)** | **97.0%** | Validé ✅ |
| **F1-Score Macro (CV Stratifiée)** | **0.912** | Validé ✅ |
| **F1-Score Pondéré (Weighted)** | **0.919** | Validé ✅ |
| **Aire sous la courbe (AUC-ROC)** | **0.967** | Validé ✅ |

Les rares erreurs du modèle se concentrent exclusivement sur la frontière de transition bio-clinique séparant le stade sain (`CN`) du stade prodromal (`MCI`), mimant parfaitement les incertitudes de la routine hospitalière réelle.

---

##  Couche d'Intelligence Artificielle Explicable (XAI Core)

Le pipeline intègre un double protocole d'audit mathématique :
1.  **Audit Macroscopique (SHAP Global) :** Un noyau `TreeExplainer` exact démontre la cohérence physiopathologique de l'apprentissage. L'atrophie de la masse cérébrale totale (`nWBV`) s'impose comme le descripteur dominant. Le graphique *Beeswarm* prouve qu'un effondrement du volume cérébral corrèle de façon causale avec l'activation de la classe `AD`.
2.  **Audit Microscopique (SHAP/LIME Local) :** Pour chaque patient, une décomposition additive locale (SHAP *Waterfall Plot*) est mise en opposition avec les approximations linéaires locales de LIME. La convergence mathématique parfaite entre ces deux frameworks agnostiques certifie la stabilité et la fidélité des diagnostics extraits de la boîte noire.

---

##  Structure du Projet sur VS Code

```text
alzheimer_xai/
├── .streamlit/
│   └── config.toml          # Configuration globale du serveur (Python 3.11)
├── app/
│   └── streamlit_app.py     # Interface web utilisateur (Clinical Hub UI)
├── data/
│   └── generate_dataset.py  # Script de simulation de la cohorte clinique
├── models/
│   └── best_pipeline.pkl    # Artefact binarisé compressé du pipeline champion
├── notebooks/
│   ├── 01_EDA.ipynb         # Analyse statistique & Tests de Kruskal-Wallis
│   ├── 02_Modeling.ipynb    # Optimisation bayésienne Optuna & LightGBM
│   └── 03_XAI.ipynb         # Calculs de noyaux SHAP et LIME locaux
├── src/
│   ├── __init__.py
│   └── preprocessing.py     # Fonctions de Feature Engineering & Pipelines
├── requirements.txt         # Dépendances figées de production
└── README.md                # Documentation principale du projet

##  Installation & Exécution en Local

Pour auditer ou exécuter ce pipeline sur votre machine, suivez la procédure d'installation isolée :

```bash
# 1. Cloner le dépôt du projet
git clone [https://github.com/chaimae-elyaouti/alzheimer_xai](https://github.com/chaimae-elyaouti/alzheimer_xai)
cd alzheimer_xai

# 2. Créer un environnement virtuel propre (Python 3.11 recommandé)
python -m venv venv
.\venv\Scripts\activate  # Sur Windows (PowerShell)
# ou si vous utilisez Windows CMD classique : .\venv\Scripts\activate.bat
# ou sur Linux/Mac : source venv/bin/activate

# 3. Installer les dépendances binaires figées
pip install --upgrade pip
pip install -r requirements.txt

# 4. Lancer l'application Streamlit en local
streamlit run app/streamlit_app.py