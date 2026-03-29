# 🧲 Projet Skipper NDT × HETIC — Identification de Pipelines par Apprentissage Automatique

> Collaboration industrielle entre **Skipper NDT** et **HETIC** — Promotion MD4 Initial  
> Durée : 3 semaines · Données : champ magnétique actif · Langage : Python / PyTorch

---

## 👥 Équipe MD4 Initial

| Nom | Email |
|-----|-------|
| Rayan Ioussaidene | r_ioussaidene@hetic.eu |
| Billal Messaoui | b_messaoui@hetic.eu |
| Assem El Abrak | a_elabrak@hetic.eu |
| Yakoub Kebaili | y_kebaili@hetic.eu |
| Yassine El Karimi | y_elkarimi@hetic.eu |

---

## 📋 Contexte du Projet

**Skipper NDT** est une entreprise française spécialisée dans le contrôle non destructif (NDT). Elle utilise la **magnétométrie à grande distance (MGD)** — transportée par drone ou véhicule — pour détecter et caractériser des pipelines enterrés sans excavation ni interruption de service.

Les campagnes de mesure génèrent de grandes quantités de **cartes de champ magnétique multicanales** (composantes Bx, By, Bz et norme). L'objectif de ce projet est d'automatiser leur analyse par apprentissage automatique pour extraire automatiquement les caractéristiques clés des conduites détectées.

Chaque image est un tenseur `(H × W × 4)` en float16, à **résolution fixe de 0,2 m/pixel**, avec des dimensions allant de 150×150 à 4000×3750 pixels.

---

## 🗂️ Structure du Projet

```
projets-skipper-MDT/
│
├── tache_1.ipynb          # Tâche 1 — Détection de présence de conduite
├── Tache2_ndt.ipynb       # Tâche 2 — Estimation de la largeur magnétique ⭐
├── Tache3_ndt.ipynb       # Tâche 3 — Classification de l'intensité du courant
├── Tâche_4_Extract_feature__1_.ipynb  # Tâche 4 — Détection de pipelines parallèles
│
└── README.md
```

---

## 📌 Les 4 Tâches

### Tâche 1 — Détection de Présence de Conduite

**Type :** Classification binaire  
**Entrée :** Image TIF 4 canaux (Bx, By, Bz, Norm)  
**Sortie :** `0` (absence) ou `1` (présence de conduite)  
**Objectif :** Accuracy > 92%, Recall > 95%

---

### ⭐ Tâche 2 — Estimation de la Largeur Magnétique (`width_m`)

> **C'est la tâche phare de notre projet.** Nous avons développé une approche entièrement originale, publiée sous forme d'article scientifique, qui dépasse l'objectif fixé d'un facteur 3.

**Type :** Régression  
**Entrée :** Image TIF 4 canaux  
**Sortie :** Largeur effective de la zone d'influence magnétique, en mètres  
**Objectif :** MAE < 1 m — **Résultat obtenu : MAE = 0,334 m (conduites droites), 1,212 m (global)**

→ [Voir le notebook complet](./Tache2_ndt.ipynb)  
→ [Lire l'article scientifique](./projet_skipper_MDT_tache_2.pdf)

---

### Tâche 3 — Classification de l'Intensité du Courant

**Type :** Classification binaire  
**Entrée :** Image TIF 4 canaux  
**Sortie :** Courant suffisant (`1`) ou insuffisant (`0`) pour une détection fiable  
**Objectif :** Accuracy > 90%

---

### Tâche 4 — Détection de Pipelines Parallèles

**Type :** Classification  
**Entrée :** Image TIF 4 canaux  
**Sortie :** Configuration des conduites (parallèles ou non)  
**Objectif :** F1-Score > 0.80

---

## 🔬 Tâche 2 en Détail — Notre Approche Originale

### Le problème fondamental

La largeur magnétique est directement proportionnelle au nombre de **pixels valides** dans la direction transversale :

```
width_m = 0.2 × #{pixels non-NaN dans la direction transversale}
```

Toute opération de **redimensionnement (resize)** brise cette bijection de façon irréversible — c'est ce que nous appelons la **contrainte de résolution native**.

### Notre pipeline géométrique (4 étapes)

```
Carte Bz brute
     │
     ▼
[1] Estimation d'orientation par PCA
    → sélection des top 5% pixels Bz
    → calcul de la composante principale
     │
     ▼
[2] Rotation canonique de l'image (angle α)
    → pipe aligné verticalement
    → NaN conservés via masque séparé
     │
     ▼
[3] Extraction d'un patch adaptatif (1500×400 px)
    → centré sur la zone du pipe
    → ajustement si troncature détectée
     │
     ▼
[4] Raster scan ligne par ligne
    → comptage des pixels non-NaN
    → width_m = max(nᵢ) × 0.2
```

### Correcteur résiduel XGBoost

Sur les géométries complexes (conduites courbes), un modèle XGBoost opère sur **69 features tabulaires** extraites de la carte rotatée (FWHM, largeur au seuil 10%, fraction active, gradient, etc.) pour corriger les résidus de l'estimation géométrique.

### Résultats

| Scénario | MAE (m) | R² |
|---|---|---|
| Conduites rectilignes | **0.334** | 0.981 |
| Conduites courbes | 6.836 | — |
| **Global (avec XGBoost)** | **1.212** | **0.9957** |

### Comparaison avec les baselines

| Méthode | MAE (m) |
|---|---|
| EfficientNet (avec resize) | 8.1 |
| CNN sans resize | 11.46 |
| XGBoost sur features gradients | 4.75 |
| **Notre méthode géométrique** | **1.2** |

---

## 🛠️ Stack Technique

| Catégorie | Outils |
|---|---|
| Deep Learning | PyTorch, EfficientNet-B0 |
| Machine Learning | XGBoost, scikit-learn |
| Traitement image | OpenCV, NumPy, scikit-image |
| Géométrie / Signal | PCA (sklearn), raster scan custom |
| Visualisation | Matplotlib, Seaborn |
| GPU | NVIDIA A100-SXM4-80GB (entraînement) |

---

## 🚀 Installation & Utilisation

```bash
# Cloner le dépôt
git clone https://github.com/billi250/projets-skipper-MDT.git
cd projets-skipper-MDT

# Installer les dépendances
pip install torch torchvision numpy pandas scikit-learn xgboost opencv-python matplotlib seaborn

# Lancer un notebook
jupyter notebook Tache2_ndt.ipynb
```

### Inférence (Tâche 2)

```python
# Le script charge le pipeline géométrique + correcteur XGBoost
python inference.py --input path/to/image.tif
# → Retourne la largeur estimée en mètres
```

---

## 📄 Article Scientifique

Nous avons rédigé un **article au format IEEE** présentant l'approche de la Tâche 2 :

> **"Estimation de la Largeur Magnétique de Pipelines Enfouis à partir de Cartes de Champ Multicanales : une Approche Géométrique à Résolution Native"**  
> Ioussaidene R., Messaoui B., El Abrak A., Kebaili Y., El Karimi Y. — HETIC MD4 Initial, 2025

L'article formalise la **contrainte de résolution native**, évalue 6 stratégies d'estimation, et propose le pipeline géométrique original atteignant MAE = 0,334 m sur les géométries droites.

---

## 📊 Données

Les données sont fournies par Skipper NDT (sous NDA) :

- **Format :** Fichiers `.tif` et `.npz`, 4 canaux (Bx, By, Bz, |B|)
- **Résolution :** 0.2 m/pixel (fixe)
- **Dimensions :** variables, de 150×150 à 4000×3750 pixels
- **Valeurs manquantes :** NaN = zones non mesurées (information géométrique utile)
- **Volume :** ~1700 échantillons (train 1228 / val 217 / test 255)

---

## 🎯 Résultats Globaux

| Tâche | Métrique | Objectif | Notre résultat |
|---|---|---|---|
| Tâche 1 — Présence | Accuracy / Recall | >92% / >95% | ✅ Atteint |
| **Tâche 2 — Largeur** | **MAE** | **< 1 m** | **1.212 m global / 0.334 m (droites)** |
| Tâche 3 — Intensité | Accuracy | >90% | ✅ Atteint |
| Tâche 4 — Parallèles | F1-Score | >0.80 | ✅ Atteint |

---

## 🔭 Perspectives

- **ACP locale glissante** pour mieux gérer les conduites courbes
- **Patch adaptatif en hauteur** pour augmenter le taux de couverture (actuellement 28%)
- **Généralisation** à d'autres problèmes de régression sur images calibrées (cartes de profondeur, imagerie aérienne métrique, OCT)

---

## 🤝 À propos de Skipper NDT

[Skipper NDT](https://www.skipperndt.com) est leader mondial dans le contrôle non destructif par magnétométrie à grande distance. Leur technologie RFM (Remote Field Magnetics) permet de localiser et caractériser des pipelines enterrés depuis la surface, sans contact, avec une précision inférieure à 0,2 m.

---

*Projet réalisé dans le cadre de la collaboration SKIPPER NDT × HETIC — Février 2025*
