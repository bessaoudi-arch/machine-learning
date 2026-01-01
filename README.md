# 🎓 Mini Projet Machine Learning – Vérification de l’apprentissage

Ce projet est un **mini projet de Machine Learning** visant à prédire la réussite (graduation) des étudiants à partir de données académiques.

- **Dataset** : Student Graduation Dataset (Kaggle)
- **Modèle** : Random Forest Classifier
- **Langage** : Python

Le projet couvre l’ensemble du pipeline classique en Machine Learning : chargement des données, prétraitement, entraînement du modèle, évaluation et visualisation des performances.

---

## 📁 Structure du projet

```text
.
├── graduation_dataset.csv      # Dataset (à placer à la racine)
├── main.py                     # Script principal (ou nom équivalent)
├── requirements.txt            # Dépendances Python
├── .gitignore                  # Fichiers ignorés par Git
├── confusion_matrix.png        # Matrice de confusion générée
├── learning_curve.png          # Courbe d’apprentissage générée
└── README.md                   # Documentation du projet
```

---

## ⚙️ Prérequis

- Python **3.8 ou plus**
- pip (gestionnaire de paquets Python)

Vérifiez votre version de Python :

```bash
python --version
```

---

## 🚀 Installation

### 1️⃣ Cloner le dépôt

```bash
git clone https://github.com/votre-username/votre-repo.git
cd votre-repo
```

### 2️⃣ Créer un environnement virtuel (recommandé)

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

### 3️⃣ Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

Le projet utilise le **Student Graduation Dataset** disponible sur Kaggle.

- Téléchargez le dataset depuis Kaggle
- Renommez-le en :

```text
graduation_dataset.csv
```

- Placez-le **à la racine du projet**

Le script vérifiera automatiquement la présence du fichier et lèvera une erreur si le dataset est introuvable.

---

## 🧠 Description du pipeline

### 1. Chargement des données

- Lecture du fichier CSV
- Affichage d’un aperçu (`head`)
- Informations générales (`info`)
- Statistiques descriptives (`describe`)

### 2. Prétraitement

- Séparation des features et de la variable cible (`Target`)
- Encodage de la cible avec `LabelEncoder`
- Normalisation des variables avec `StandardScaler`

### 3. Split des données

- 80 % entraînement
- 20 % test
- Stratification sur la variable cible

### 4. Modélisation

Le modèle utilisé est un **Random Forest Classifier** avec les paramètres suivants :

- `n_estimators = 150`
- `max_depth = 10`
- `min_samples_leaf = 5`
- `random_state = 42`

### 5. Évaluation

- Accuracy sur le jeu de test
- Rapport de classification (precision, recall, F1-score)
- Matrice de confusion

### 6. Visualisation

Deux graphiques sont générés automatiquement :

- 🧩 **Matrice de confusion** → `confusion_matrix.png`
- 📈 **Learning curve** → `learning_curve.png`

---

## ▶️ Exécution du projet

Lancez simplement le script principal :

```bash
python main.py
```

À la fin de l’exécution, les fichiers suivants seront générés :

- `confusion_matrix.png`
- `learning_curve.png`

---

## 📈 Résultats

Les performances du modèle sont évaluées à l’aide :

- de l’accuracy
- du rapport de classification
- de la courbe d’apprentissage pour analyser le biais et la variance

Ces résultats permettent de vérifier la qualité de l’apprentissage du modèle.

---

## 🧪 Technologies utilisées

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## 📌 Remarques

- Le projet est conçu à des fins **pédagogiques**
- Il peut être facilement étendu (tuning des hyperparamètres, autres modèles, validation croisée avancée)

---

## 👨‍🎓 Auteur

Projet réalisé dans le cadre d’un **mini projet Machine Learning**.

---

✨ Bon apprentissage et bonne exploration du Ma