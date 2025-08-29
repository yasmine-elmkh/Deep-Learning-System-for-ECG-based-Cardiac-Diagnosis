-------------------------------------English Version---------------------------------------------
# Deep Learning System for ECG-based Cardiac Diagnosis
**Project Overview**

This project aims to design an intelligent system capable of diagnosing cardiac anomalies from ECG (Electrocardiogram) signals using advanced Deep Learning techniques. The model automatically analyzes ECG signals and provides accurate diagnoses for each patient.

**Objective**

Provide fast and precise diagnosis of ECG signals.

Automate cardiac signal analysis to support clinicians and researchers.

Classify cardiac anomalies into different categories (e.g., tachycardia, bradycardia, fibrillation).

**Technologies Used**

Python 3 – for data processing and model development.

NumPy & Pandas – for handling and preprocessing ECG signals.

TensorFlow/Keras – for implementing the Deep Learning model.

Scikit-learn – for data preprocessing and model evaluation.

**Model Architecture and Functioning**

The system leverages a CNN-BiLSTM with attention mechanism trained on MIT-BIH Arrhythmia and PTB Diagnostic ECG datasets.

1. CNN (Convolutional Neural Network)

Extracts local morphological features from ECG signals.

Convolutional layers detect characteristic patterns in heartbeats.

Pooling layers reduce dimensionality and overfitting.

2. BiLSTM (Bidirectional Long Short-Term Memory)

Models temporal dependencies and sequential information of heartbeats.

Processes signals forward and backward to capture full context.

3. Attention Mechanism

Focuses the model on the most informative parts of the sequence.

Generates a context vector representing relevant features for classification.

4. Dense and Output Layers

Context vector is passed through fully connected layers.

Softmax output predicts the probability for each class.

5. Training and Optimization

Loss function: categorical cross-entropy.

Optimizer: Adam.

Trained for multi-class classification (5 heartbeat types) and binary classification (Normal vs Abnormal).

**Model Performance**

Multi-class Classification (5 heartbeat classes):

Class	Precision	Recall	F1-score	Support
Normal	0.99	1.00	0.99	18,118
RBB	0.91	0.76	0.83	556
LBB	0.97	0.96	0.96	1,448
PVC	0.88	0.76	0.81	162
Other	0.98	0.99	0.99	1,608

Weighted average: Precision = 0.98, Recall = 0.92, F1-score = 0.98

Excellent performance in distinguishing between different heartbeat categories.

Binary Classification (Normal vs Abnormal):

Class	Precision	Recall	F1-score	Support
Normal	0.99	1.00	0.99	18,118
Abnormal	0.98	0.95	0.96	3,774

Accuracy: 99%

Strong capability to differentiate between normal and pathological ECG patterns.

**Project Structure**
Deep-Learning-ECG/
│
├─ app.py                  # Streamlit interface
├─ ECG_model.h5            # Trained Deep Learning model
├─ ECG_scaler.gz           # Scaler for normalization
├─ ECG_exemple.py          # Example script for testing ECG signals
├─ ECG_exemple1.csv        # Sample ECG data
├─ ECG_exemple2.csv        # Sample ECG data
├─ README.md               # Project documentation
└─ requirements.txt        # Required Python libraries

**Perspective wor**

Support multi-lead ECG (e.g., 12-lead).

Integration of ECG signal visualization.

Implementation of Explainable AI techniques (e.g., Grad-CAM, attention heatmaps).

Deployment on cloud for online access and real-time monitoring.

Authors

Yasmine El Mkhantar
Zineb Benchekroun
Sara El Mekkaoui

---------------------------------------Version Française-----------------------------------------------
# Système Deep Learning pour le Diagnostic Cardiaque à partir d’ECG
**Présentation du projet**

Ce projet vise à concevoir un système intelligent capable de diagnostiquer les anomalies cardiaques à partir de signaux ECG (Électrocardiogramme) en utilisant des techniques avancées de Deep Learning. Le modèle analyse automatiquement les signaux ECG et fournit un diagnostic précis pour chaque patient.

**Objectifs**

Fournir un diagnostic rapide et précis des signaux ECG.

Automatiser l’analyse des signaux cardiaques pour soutenir les médecins et chercheurs.

Classer les anomalies cardiaques en différentes catégories (ex. tachycardie, bradycardie, fibrillation).

**Technologies utilisées**

Python 3 – traitement des données et développement du modèle.

NumPy & Pandas – manipulation et prétraitement des signaux ECG.

TensorFlow / Keras – implémentation du modèle Deep Learning.

Scikit-learn – prétraitement des données et évaluation des performances.

**Fonctionnement du modèle**

Le système utilise un CNN-BiLSTM avec mécanisme d’attention, entraîné sur les datasets MIT-BIH Arrhythmia et PTB Diagnostic ECG.

1. CNN (Réseau de neurones convolutionnel)

Extrait les caractéristiques locales et morphologiques des signaux ECG.

Les couches convolutionnelles détectent les motifs caractéristiques des battements cardiaques.

Les couches de pooling réduisent la dimensionnalité et préviennent le sur-apprentissage.

2. BiLSTM (Long Short-Term Memory bidirectionnel)

Capture les dépendances temporelles et l’ordre des séquences de battements cardiaques.

Analyse le signal dans les deux directions pour mieux comprendre le contexte global.

3. Mécanisme d’attention

Permet au modèle de se concentrer sur les parties les plus informatives du signal.

Produit un vecteur de contexte représentant les caractéristiques pertinentes pour la classification.

4. Couches denses et sortie

Le vecteur de contexte passe dans des couches fully connected pour la classification finale.

La couche softmax prédit la probabilité pour chaque classe.

5. Entraînement et optimisation

Fonction de perte : categorical cross-entropy.

Optimiseur : Adam.

Entraîné pour la classification multi-classes (5 types de battements) et la classification binaire (Normal vs Anormal).

**Performances du modèle**

Classification multi-classes (5 classes de battements) :

Classe	Précision	Rappel	F1-score	Support
Normal	0.99	1.00	0.99	18,118
RBB	0.91	0.76	0.83	556
LBB	0.97	0.96	0.96	1,448
PVC	0.88	0.76	0.81	162
Autre	0.98	0.99	0.99	1,608

Moyenne pondérée : Précision = 0.98, Rappel = 0.92, F1-score = 0.98

Excellente capacité à distinguer les différentes catégories de battements.

Classification binaire (Normal vs Anormal) :

Classe	Précision	Rappel	F1-score	Support
Normal	0.99	1.00	0.99	18,118
Anormal	0.98	0.95	0.96	3,774

Précision globale : 99%

Forte capacité à différencier les ECG normaux et pathologiques.

**Structure du projet**
Deep-Learning-ECG/
│
├─ app.py                  # Interface utilisateur Streamlit
├─ ECG_model.h5            # Modèle Deep Learning entraîné
├─ ECG_scaler.gz           # Scaler pour normalisation
├─ ECG_exemple.py          # Script d’exemple pour tester les signaux ECG
├─ ECG_exemple1.csv        # Données ECG exemple
├─ ECG_exemple2.csv        # Données ECG exemple
├─ README.md               # Documentation
└─ requirements.txt        # Librairies Python nécessaires

**Perspectives d’amélioration**

Support multi-lead ECG (ex. 12 dérivations).

Intégration de la visualisation graphique des signaux ECG.

Implémentation de techniques Explainable AI (Grad-CAM, attention heatmaps).

Déploiement sur le cloud pour un accès en ligne et suivi en temps réel.

Auteurs

Yasmine El Mkhantar
Zineb Benchekroun
Sara El Mekkaoui