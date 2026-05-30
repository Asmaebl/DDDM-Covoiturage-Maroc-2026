# Viabilité du modèle prix fixe au Maroc face à inDrive — Projet DDDM

> **Module :** Data-Driven Decision Making  
> **Deadline :** 07 Juin 2026  

---

## Question décisionnelle centrale

> **Si un service de covoiturage à prix fixe algorithmique s'implantait au Maroc, pourrait-il être viable et compétitif face à inDrive, compte tenu de la forte tendance des Marocains à utiliser le modèle de négociation ?**

---

## Équipe & Répartition

| Membre | Phases | Responsabilité |
|--------|--------|----------------|
| Coéquipier A | 1 · 2 · 3 | Cadrage · Données · EDA |
| Coéquipier B | 4 · 5 · 6 | Modélisation · Dashboard · Décision |

---

## Structure du projet

```
projet-dddm-prix-fixe-maroc/
│
├── README.md                        ← Ce fichier
├── requirements.txt                 ← Dépendances Python
│
├── notebooks/
│   └── DDDM_Projet.ipynb           ← Notebook complet (6 phases)
│
├── data/
│   ├── uber_final.csv              ← Dataset prix fixe nettoyé (182 136 lignes)
│   └── rides_merged_clustered.csv  ← Rides + Users + Clusters (25 000 lignes)
│
├── assets/                          ← Visualisations générées
│   ├── eda_01_distribution_prix.png
│   ├── eda_02_distributions_temporelles.png
│   ├── eda_03_distributions_rides.png
│   ├── eda_04_outliers.png
│   ├── eda_05_heatmap_correlations.png
│   ├── eda_06_scatter_matrix.png
│   ├── eda_07_tests_statistiques.png
│   ├── eda_08_elbow_silhouette.png
│   ├── eda_09_clusters_visualisation.png
│   └── eda_10_saisonnalite.png
│
└── dashboard/                       ← Application interactive (Coéquipier B)
    ├── app.py
    └── ...
```

---

## Installation & Lancement

### 1. Cloner le dépôt

```bash
git clone https://github.com/<votre-username>/projet-dddm-prix-fixe-maroc.git
cd projet-dddm-prix-fixe-maroc
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Lancer le notebook

```bash
jupyter notebook notebooks/DDDM_Projet.ipynb
```

### 5. Lancer le dashboard (Coéquipier B)

```bash
# Streamlit
streamlit run dashboard/app.py

# ou Dash
python dashboard/app.py
```

---

## Sources de données

| Source | Fichier brut | Lignes | Rôle |
|--------|-------------|--------|------|
| [Uber Fares Dataset — Kaggle](https://www.kaggle.com/datasets/yasserh/uber-fares-dataset) | `uber.csv` | 200 000 | Modélisation comportementale du prix fixe |
| [Ride Hailing Transactions — Kaggle](https://www.kaggle.com/datasets/galihwardiana/ride-hailing-transaction) | `fact_rides.csv` | 25 003 | Comportement utilisateurs ride-hailing |
| Profils utilisateurs | `users.csv` | 11 000 | Segmentation (genre, ancienneté) |

> **Note méthodologique :** Les données Uber proviennent de New York et servent à modéliser les **patterns comportementaux** du modèle prix fixe (heure, distance, segmentation utilisateurs). Ces patterns sont transférables au contexte marocain après adaptation. Les prix absolus sont recontextualisés en MAD dans l'analyse.

> **Note Git :** Les fichiers bruts (`uber.csv`, `fact_rides.csv`, `users.csv`) ne sont pas versionnés (trop volumineux). Téléchargez-les depuis les liens Kaggle. Les fichiers nettoyés (`uber_final.csv`, `rides_merged_clustered.csv`) sont disponibles dans `/data/`.

---

## Pipeline d'analyse

```
[Sources brutes]          [Nettoyage]              [Analyse]
uber.csv ──────────────►  uber_final.csv ─────────► EDA + Modèles
fact_rides.csv ─────────►                            ↓
users.csv ──────────────►  rides_merged_             Dashboard
                           _clustered.csv ──────────► Recommandations
[HCP Maroc Excel] ──────►  (contexte narratif Maroc)
```

---

## Description des fichiers clés

### `uber_final.csv` (182 136 lignes · 11 colonnes)
Dataset prix fixe nettoyé — prix aberrants supprimés (IQR), GPS validés, features temporelles ajoutées.

| Colonne | Type | Description |
|---------|------|-------------|
| `fare_amount` | float | Prix de la course en $ |
| `pickup_datetime` | datetime | Horodatage du départ |
| `pickup_latitude` | float | Latitude point de départ |
| `pickup_longitude` | float | Longitude point de départ |
| `dropoff_latitude` | float | Latitude point d'arrivée |
| `dropoff_longitude` | float | Longitude point d'arrivée |
| `passenger_count` | int | Nombre de passagers (1–6) |
| `hour` | int | Heure de la course (0–23) |
| `day_of_week` | str | Jour de la semaine |
| `month` | int | Mois (1–12) |
| `year` | int | Année |

### `rides_merged_clustered.csv` (25 000 lignes · 19 colonnes)
Rides + profils utilisateurs fusionnés + colonne `cluster` (K-Means, k=5).

| Colonne | Type | Description |
|---------|------|-------------|
| `ride_id` | int | Identifiant unique de la course |
| `customer_id` | int | Identifiant utilisateur |
| `fare_amount` | float | Prix de la course |
| `ride_distance_miles` | float | Distance en miles |
| `ride_duration_minutes` | float | Durée en minutes |
| `payment_method` | str | cash / credit card / mobile wallet |
| `rating` | float | Note donnée (1.0 – 5.0) |
| `gender` | str | Genre de l'utilisateur |
| `cluster` | int | Segment K-Means (0 à 4) |

---

## Tests & Reproductibilité

Toutes les cellules du notebook sont exécutables de haut en bas sans erreur.  
Seed fixée à `random_state=42` pour tous les modèles aléatoires.  
Les graphiques sont sauvegardés automatiquement dans `/assets/`.

---

## Principaux résultats (Phases 1–3)

- **236 003 lignes** brutes collectées — exigence 50 000 min
- **4 tests statistiques** : t-test, ANOVA, Chi-deux, Mann-Whitney
- **5 segments** d'utilisateurs identifiés par K-Means
- **Insight clé :** 40% des utilisateurs (segments Économiques) sont sensibles au prix et risquent de rejeter le modèle prix fixe — cibles prioritaires pour une stratégie d'adaptation tarifaire au Maroc

---

## Liens utiles

- [Kaggle — Uber Fares Dataset](https://www.kaggle.com/datasets/yasserh/uber-fares-dataset)
- [HCP Maroc — Open Data](https://www.hcp.ma)
