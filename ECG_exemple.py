import pandas as pd

# Paramètres
nombre_fichiers = 2  # Le nombre de patients
colonnes = 187       # Nombre de colonnes (valeurs ECG)

for i in range(1, nombre_fichiers + 1):
    # Ligne avec des valeurs fixes ou générées simplement
    ligne = [round(j * 0.1, 2) for j in range(colonnes)]  # Ex: 0.0, 0.1, ..., 18.6
    
    # Crée un DataFrame avec cette ligne
    df = pd.DataFrame([ligne])
    
    # Enregistre sous forme CSV avec un nom unique
    nom_fichier = f"ECG_exemple{i}.csv"
    df.to_csv(nom_fichier, index=False, header=False)
    print(f"{nom_fichier} généré.")
    