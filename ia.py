"""
 Mini Projet Machine Learning :
 Vérification de l'apprentissage
 Dataset : Student Graduation Dataset (Kaggle)
 Modèle  : Random Forest
"""

# 1. Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 2. Chargement du dataset
print("Chargement du dataset...")

DATA_PATH = os.path.abspath("graduation_dataset.csv")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f" Dataset introuvable : {DATA_PATH}")

data = pd.read_csv(DATA_PATH)

print("\n Aperçu du dataset :")
print(data.head())

print("\n Informations générales :")
print(data.info())

print("\n Statistiques descriptives :")
print(data.describe())


# 3. Prétraitement des données
print("\n Prétraitement des données...")

TARGET_COLUMN = "Target"

X = data.drop(columns=[TARGET_COLUMN])
y = data[TARGET_COLUMN]

# Encodage de la cible
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Normalisation des features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 4. Split Train / Test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print(f"Entraînement : {len(X_train)} échantillons")
print(f"Test        : {len(X_test)} échantillons")


# 5. Modèle Random Forest
print("\n Initialisation du modèle Random Forest...")

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

print(" Entraînement du modèle...")
model.fit(X_train, y_train)


# 6. Évaluation du modèle
print("\n Évaluation du modèle...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n Accuracy (test set) : {accuracy:.3f}")

print("\n Rapport de classification :")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


# 7. Matrice de confusion
print(" Génération de la matrice de confusion...")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.xlabel("Prédiction")
plt.ylabel("Valeur réelle")
plt.title("Matrice de confusion")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

print(" Fichier généré : confusion_matrix.png")

# 8. Learning Curve
print("Calcul de la learning curve...")

train_sizes, train_scores, val_scores = learning_curve(
    model,
    X_scaled,
    y_encoded,
    cv=5,
    scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 5),
    n_jobs=-1
)

plt.figure(figsize=(6, 4))
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label="Train")
plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', label="Validation")

plt.xlabel("Taille du jeu d'entraînement")
plt.ylabel("Accuracy")
plt.title("Learning Curve")
plt.legend()
plt.tight_layout()
plt.savefig("learning_curve.png", dpi=150)
plt.close()

print(" Fichier généré : learning_curve.png")


