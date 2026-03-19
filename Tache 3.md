# ════════════════════════════════════════════════════════════════
# TÂCHE 3 — CLASSIFICATION INTENSITÉ DU COURANT
# Process complet : chargement NPZ → preprocessing → entraînement
#
# Objectif : classifier si le pipeline est DÉTECTABLE (1) ou NON (0)
# Dataset  : Dataset 2 — 4715 samples — formes courbées — plus bruité
# Cible    : Accuracy > 90%, Recall > 85%
#
# Architecture : EfficientNet-B0 adapté 4 canaux
# Preprocessing: ACP pour aligner les pipes courbés + resize 224×224
# ════════════════════════════════════════════════════════════════

import os, time, io
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, recall_score,
                              f1_score, classification_report,
                              confusion_matrix)
from torchvision.models import efficientnet_b0
from scipy.ndimage import rotate as scipy_rotate
from skimage.transform import resize as sk_resize
from tqdm import tqdm
from googleapiclient.discovery import build
from google.colab import auth
from oauth2client.client import GoogleCredentials
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ───────────────────────────────────────────────────────
DEVICE     = torch.device("cuda")
DATA_DIR   = "/content/drive/MyDrive/skipper_task3/Training_data_inspection_validation_float16"
OUTPUT_DIR = "/content/drive/MyDrive/skipper_task3/task3_outputs"
PT_PATH    = "/content/drive/MyDrive/skipper_task3/preprocessed_data.pt"
IMG_SIZE   = 224     # taille cible après resize
BATCH_SIZE = 64
NUM_EPOCHS = 50
LR         = 3e-4
PATIENCE   = 10
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"✅ GPU : {torch.cuda.get_device_name(0)}")
print(f"   IMG_SIZE  = {IMG_SIZE}×{IMG_SIZE}")
print(f"   BATCH     = {BATCH_SIZE}")

# ════════════════════════════════════════════════════════════════
# ÉTAPE 1 : CHARGEMENT DU CSV
#
# Le CSV du Dataset 2 est stocké comme Google Sheet sur Drive.
# On l'exporte en CSV via l'API Drive.
# Colonnes importantes :
#   - field_file  : nom du fichier NPZ
#   - label       : 0 (non détectable) ou 1 (détectable)
#   - pipe_type   : single / parallel
#   - coverage_type : perfect / offset / missed
# ════════════════════════════════════════════════════════════════

auth.authenticate_user()
creds   = GoogleCredentials.get_application_default()
service = build('drive', 'v3', credentials=creds)

results = service.files().list(
    q="name contains 'pipe_detection_label' and trashed=false",
    fields="files(id, name, mimeType)"
).execute()

df = None
for f in results.get('files', []):
    try:
        csv_bytes = service.files().export_media(
            fileId=f['id'], mimeType='text/csv'
        ).execute()
        tmp = pd.read_csv(io.BytesIO(csv_bytes), sep=',')
        if len(tmp.columns) == 1:
            tmp = pd.read_csv(io.BytesIO(csv_bytes), sep=';')

        # Le bon CSV du Dataset 2 a 'label' mais PAS 'width_m'
        if 'label' in tmp.columns and 'width_m' not in tmp.columns:
            df = tmp.reset_index(drop=True)
            print(f"\n✅ CSV Dataset 2 : {len(df)} samples")
            print(f"   label 0 (non détectable) : {(df.label==0).sum()}")
            print(f"   label 1 (détectable)      : {(df.label==1).sum()}")
            break
    except:
        pass

# ════════════════════════════════════════════════════════════════
# ÉTAPE 2 : PREPROCESSING DES IMAGES NPZ
#
# Chaque fichier NPZ contient un tableau (H, W, 4) :
#   canal 0 : Bx  (composante Est,       nT)
#   canal 1 : By  (composante Nord,      nT)
#   canal 2 : Bz  (composante verticale, nT)
#   canal 3 : Bnorm (magnitude totale,   nT)
#
# NaN = zones hors couloir de mesure (comportement physique normal)
#
# Pipeline de preprocessing :
# 1. Charger le NPZ
# 2. Remplacer NaN par 0 APRÈS calcul de l'angle
# 3. Calculer l'angle α via ACP sur le signal Bz fort
# 4. Rotation pour aligner le pipe horizontalement
# 5. Resize vers 224×224 (nécessaire pour EfficientNet)
# 6. Normalisation z-score par canal
#
# Note : pour T3 (classification), le resize est acceptable
# car on n'a PAS besoin de l'échelle absolue en mètres.
# C'est différent de T2 (régression) où le resize détruit la largeur.
# ════════════════════════════════════════════════════════════════

def compute_pca_angle(channel_bz):
    """
    Calcule l'angle d'orientation du pipe via ACP.

    Principe :
    - On prend les pixels avec un signal Bz fort (top 25%)
    - On calcule la matrice de covariance de leurs positions (x,y)
    - Le vecteur propre dominant = direction principale du pipe
    - L'angle de ce vecteur = angle α à corriger

    Retourne l'angle en degrés (entre -90 et +90).
    """
    # Masque : pixels valides avec signal fort
    valid_mask = ~np.isnan(channel_bz)
    if valid_mask.sum() < 10:
        return 0.0

    vals_valid = channel_bz[valid_mask]
    threshold  = np.percentile(vals_valid, 75)
    signal_mask = valid_mask & (channel_bz > threshold)

    rows, cols = np.where(signal_mask)
    if len(rows) < 10:
        return 0.0

    # ACP sur les positions des pixels de fort signal
    pts  = np.column_stack([cols.astype(float), rows.astype(float)])
    pts -= pts.mean(axis=0)    # centrage

    cov            = np.cov(pts.T)
    if cov.ndim < 2:
        return 0.0

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Vecteur propre de la plus grande valeur propre = direction du pipe
    dominant = eigenvectors[:, np.argmax(eigenvalues)]
    angle    = np.degrees(np.arctan2(dominant[1], dominant[0]))

    return float(angle)


def preprocess_one(npz_path, img_size=IMG_SIZE):
    """
    Preprocessing complet pour un sample.

    Entrée  : chemin vers fichier NPZ
    Sortie  : tenseur (4, img_size, img_size) normalisé

    Étapes :
    1. Chargement NPZ → (H, W, 4) float32
    2. Calcul angle ACP sur Bz (avant nan_to_num)
    3. Remplacement NaN par 0
    4. Rotation si |angle| > 5°
    5. Resize vers img_size×img_size
    6. Normalisation z-score par canal
    7. Transposition vers (4, H, W) pour PyTorch
    """
    try:
        data = np.load(npz_path)['data'].astype(np.float32)
    except:
        # Si le fichier est illisible → retourner des zéros
        return np.zeros((4, img_size, img_size), dtype=np.float32)

    # Angle ACP calculé AVANT de remplacer les NaN
    # (les NaN délimitent la zone de mesure — information utile)
    if data.shape[0] > 30 and data.shape[1] > 30:
        angle = compute_pca_angle(data[:, :, 2])
    else:
        angle = 0.0

    # Remplacer NaN par 0 pour les calculs suivants
    # (après l'ACP, on peut le faire sans perdre d'info)
    data = np.nan_to_num(data, nan=0.0)

    # Rotation pour aligner le pipe horizontalement
    # → le modèle voit toujours un pipe "droit" peu importe son orientation
    if abs(angle) > 5.0 and abs(angle) < 85.0:
        rotated = np.zeros_like(data)
        for c in range(4):
            rotated[:, :, c] = scipy_rotate(
                data[:, :, c],
                angle=-angle,
                reshape=False,
                order=1,
                cval=0.0
            )
        data = rotated

    # Resize vers taille fixe
    # Pour T3 (classification) c'est acceptable — on ne mesure pas de largeur
    data = sk_resize(
        data,
        (img_size, img_size, 4),
        order=1,
        preserve_range=True,
        anti_aliasing=True
    )

    # Normalisation z-score par canal
    # Chaque canal est centré (mean=0) et réduit (std=1)
    # → le modèle voit des valeurs normalisées indépendamment du bruit
    for c in range(4):
        ch  = data[:, :, c]
        mu  = ch.mean()
        std = ch.std()
        data[:, :, c] = (ch - mu) / (std + 1e-8)

    # PyTorch attend (C, H, W) au lieu de (H, W, C)
    return data.transpose(2, 0, 1).astype(np.float32)


# ── Calcul ou chargement du preprocessing ────────────────────────
# Si le fichier .pt existe déjà → chargement instantané
# Sinon → calcul (~25 min) + sauvegarde pour les prochaines fois
if os.path.exists(PT_PATH):
    print(f"\n⚡ Chargement preprocessing depuis {PT_PATH}...")
    ckpt     = torch.load(PT_PATH)
    X_tensor = ckpt['X']
    y_tensor = ckpt['y']
    print(f"✅ {X_tensor.shape} chargé instantanément")

else:
    print(f"\n⚡ Preprocessing {len(df)} samples — ~25 min\n")

    all_X  = np.zeros((len(df), 4, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    all_y  = df['label'].values.astype(np.int64)
    errors = 0

    for i, row in enumerate(tqdm(df.itertuples(),
                                  total=len(df),
                                  desc="Preprocessing T3")):
        path        = os.path.join(DATA_DIR, row.field_file)
        all_X[i]    = preprocess_one(path, IMG_SIZE)

    X_tensor = torch.tensor(all_X, dtype=torch.float32)
    y_tensor = torch.tensor(all_y, dtype=torch.long)

    # Sauvegarde — permet de sauter cette étape lors des prochains runs
    torch.save({'X': X_tensor, 'y': y_tensor}, PT_PATH)
    print(f"\n💾 Sauvegardé : {PT_PATH}  ({X_tensor.shape})")
    del all_X

print(f"\nX : {X_tensor.shape}")
print(f"y : label 0 = {(y_tensor==0).sum()} | label 1 = {(y_tensor==1).sum()}")

# ════════════════════════════════════════════════════════════════
# ÉTAPE 3 : SPLIT TRAIN / VAL / TEST
#
# Split stratifié : la proportion label 0/1 est la même dans
# les 3 ensembles. Évite que le test soit trop facile ou trop dur.
#
# 70% train | 12.75% val | 15% test (approx)
# ════════════════════════════════════════════════════════════════

indices = np.arange(len(y_tensor))
all_y   = y_tensor.numpy()

# Split stratifié en 2 étapes
idx_trval, idx_test  = train_test_split(
    indices, test_size=0.15,
    stratify=all_y,           # proportions respectées
    random_state=42
)
idx_train, idx_val   = train_test_split(
    idx_trval, test_size=0.15,
    stratify=all_y[idx_trval],
    random_state=42
)

print(f"\nSplit — Train: {len(idx_train)} | Val: {len(idx_val)} | Test: {len(idx_test)}")
print(f"Train label 1 : {all_y[idx_train].mean()*100:.1f}%")
print(f"Val   label 1 : {all_y[idx_val].mean()*100:.1f}%")
print(f"Test  label 1 : {all_y[idx_test].mean()*100:.1f}%")

train_dl = DataLoader(
    TensorDataset(X_tensor[idx_train], y_tensor[idx_train]),
    batch_size=BATCH_SIZE, shuffle=True
)
val_dl = DataLoader(
    TensorDataset(X_tensor[idx_val], y_tensor[idx_val]),
    batch_size=BATCH_SIZE, shuffle=False
)
test_dl = DataLoader(
    TensorDataset(X_tensor[idx_test], y_tensor[idx_test]),
    batch_size=BATCH_SIZE, shuffle=False
)

# ════════════════════════════════════════════════════════════════
# ÉTAPE 4 : POIDS DE CLASSES
#
# Le dataset est déséquilibré : 60% label1, 40% label0.
# Sans correction, le modèle peut apprendre à prédire "toujours 1"
# et avoir 60% d'accuracy sans rien apprendre d'utile.
#
# Solution : pondérer la loss
#   loss = w0 × erreurs_class0 + w1 × erreurs_class1
#   w0 = 1.25 → on punit plus les erreurs sur la classe minoritaire
#   w1 = 0.83 → on punit moins les erreurs sur la classe majoritaire
# ════════════════════════════════════════════════════════════════

w = torch.tensor([1.25, 0.83], dtype=torch.float32).to(DEVICE)
print(f"\nPoids classes : [w0={w[0]:.2f}, w1={w[1]:.2f}]")

# ════════════════════════════════════════════════════════════════
# ÉTAPE 5 : ARCHITECTURE — EfficientNet-B0 4 canaux
#
# EfficientNet-B0 est pré-entraîné sur ImageNet (1.2M images, 1000 classes).
# Il sait déjà détecter des formes, des contours, des textures.
# On réutilise cette connaissance via le transfer learning.
#
# Problème : ImageNet attend 3 canaux (RGB), nos données ont 4 canaux.
# Solution : remplacer la première Conv2d(3→32) par Conv2d(4→32)
#   et initialiser le 4ème canal = moyenne des 3 premiers poids RGB.
#
# Architecture finale :
#   Input (4, 224, 224)
#   → EfficientNet features (1280)
#   → Dropout(0.3) → Linear(256) → ReLU → Dropout(0.2) → Linear(2)
#   → Softmax implicite dans CrossEntropyLoss
# ════════════════════════════════════════════════════════════════

class EfficientNet4ch(nn.Module):
    def __init__(self, n_cls=2, drop=0.3):
        super().__init__()

        # Charger EfficientNet-B0 avec poids ImageNet
        base     = efficientnet_b0(weights='IMAGENET1K_V1')
        old_conv = base.features[0][0]   # Conv2d(3, 32, kernel=3, stride=2)

        # Nouvelle conv : 4 canaux au lieu de 3
        new_conv = nn.Conv2d(
            in_channels  = 4,
            out_channels = old_conv.out_channels,
            kernel_size  = old_conv.kernel_size,
            stride       = old_conv.stride,
            padding      = old_conv.padding,
            bias         = False
        )

        # Initialisation intelligente des poids
        with torch.no_grad():
            # Canaux 0,1,2 → copier directement depuis ImageNet
            new_conv.weight[:, :3, :, :] = old_conv.weight
            # Canal 3 (Bnorm) → moyenne des 3 canaux RGB
            # Hypothèse : Bnorm ressemble à une image en niveaux de gris
            new_conv.weight[:, 3:4, :, :] = old_conv.weight.mean(
                dim=1, keepdim=True
            )

        base.features[0][0] = new_conv

        # Remplacer la tête de classification
        # EfficientNet-B0 produit 1280 features → on adapte aux 2 classes
        in_f = base.classifier[1].in_features   # 1280
        base.classifier = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(in_f, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_cls)
        )

        self.model = base

    def forward(self, x):
        return self.model(x)   # (B, 2)


# ════════════════════════════════════════════════════════════════
# ÉTAPE 6 : AUGMENTATION DE DONNÉES
#
# L'augmentation crée artificiellement de la variété dans le training set.
# Elle est appliquée sur GPU (pas sur CPU) = très rapide.
#
# Flips : un pipe peut être orienté dans les 2 sens
#   → flip horizontal ET vertical sont physiquement valides
# Bruit gaussien : simule des mesures légèrement différentes
#   → améliore la robustesse au bruit réel
# ════════════════════════════════════════════════════════════════

def augment_batch(x):
    """Augmentation on-the-fly sur GPU."""
    x = x.clone()

    # Flip horizontal (50% de chance)
    mask_h = torch.rand(x.size(0), device=x.device) > 0.5
    x[mask_h] = x[mask_h].flip(-1)

    # Flip vertical (50% de chance)
    mask_v = torch.rand(x.size(0), device=x.device) > 0.5
    x[mask_v] = x[mask_v].flip(-2)

    # Bruit gaussien (30% de chance)
    if torch.rand(1) > 0.7:
        x += torch.randn_like(x) * 0.05

    return x


# ════════════════════════════════════════════════════════════════
# ÉTAPE 7 : FONCTIONS D'ENTRAÎNEMENT ET D'ÉVALUATION
# ════════════════════════════════════════════════════════════════

def train_epoch(model, loader, opt, crit, device):
    """Une epoch d'entraînement."""
    model.train()
    loss_sum, preds_all, labels_all = 0, [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x    = augment_batch(x)

        opt.zero_grad()
        out  = model(x)           # logits (B, 2)
        loss = crit(out, y)       # CrossEntropy pondérée
        loss.backward()

        # Gradient clipping — évite les explosions de gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        loss_sum += loss.item()
        preds_all.extend(out.argmax(1).detach().cpu().numpy())
        labels_all.extend(y.cpu().numpy())

    n = len(loader)
    return (
        loss_sum / n,
        accuracy_score(labels_all, preds_all),
        recall_score(labels_all, preds_all, zero_division=0)
    )


@torch.no_grad()
def eval_epoch(model, loader, crit, device):
    """Évaluation sans gradient (val ou test)."""
    model.eval()
    loss_sum, preds_all, labels_all = 0, [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out  = model(x)
        loss_sum += crit(out, y).item()
        preds_all.extend(out.argmax(1).cpu().numpy())
        labels_all.extend(y.cpu().numpy())

    n = len(loader)
    return (
        loss_sum / n,
        accuracy_score(labels_all, preds_all),
        recall_score(labels_all, preds_all, zero_division=0),
        f1_score(labels_all, preds_all, zero_division=0),
        preds_all,
        labels_all
    )


# ════════════════════════════════════════════════════════════════
# ÉTAPE 8 : ENTRAÎNEMENT EN 2 PHASES
#
# PHASE 1 — Chauffe de la tête (5 epochs)
#   - On gèle le backbone EfficientNet
#   - On entraîne seulement la tête + première conv (modifiée)
#   - LR élevé (1e-3) car la tête part de zéro
#   - But : laisser la tête s'adapter sans perturber les poids ImageNet
#
# PHASE 2 — Fine-tuning complet
#   - On dégèle tout le réseau
#   - LR faible (3e-4) pour ne pas détruire les poids pré-entraînés
#   - CosineAnnealingLR : LR diminue progressivement comme une cosinus
#   - Early stopping sur le F1 score (patience=10)
# ════════════════════════════════════════════════════════════════

model     = EfficientNet4ch().to(DEVICE)
crit      = nn.CrossEntropyLoss(weight=w)
best_path = os.path.join(OUTPUT_DIR, "task3_best_model_efficientnet.pth")
best_f1   = 0.0
best_ep   = 0

total = sum(p.numel() for p in model.parameters())
print(f"\nModèle : {total:,} paramètres")
print(f"   EfficientNet-B0 (5.3M) vs ResNet18 (11.3M) — plus léger, meilleur")

# ── Phase 1 : backbone gelé ──────────────────────────────────────
print("\n📌 PHASE 1 — Chauffe tête (5 epochs, backbone gelé)")
print("-"*60)

for name, param in model.named_parameters():
    # Geler tout sauf la tête et la première conv (modifiée)
    if 'classifier' not in name and 'features.0' not in name:
        param.requires_grad = False

opt1 = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3, weight_decay=1e-4
)

for ep in range(1, 6):
    t0 = time.time()
    tr_loss, tr_acc, tr_rec             = train_epoch(model, train_dl, opt1, crit, DEVICE)
    vl_loss, vl_acc, vl_rec, vl_f1,_,_ = eval_epoch(model, val_dl, crit, DEVICE)
    print(f"Ph1 Ep {ep} | Tr Acc={tr_acc:.3f} Rec={tr_rec:.3f} | "
          f"Val Acc={vl_acc:.3f} Rec={vl_rec:.3f} F1={vl_f1:.3f} | "
          f"{time.time()-t0:.0f}s")
    if vl_f1 > best_f1:
        best_f1, best_ep = vl_f1, ep
        torch.save({'epoch': ep, 'model_state_dict': model.state_dict(),
                    'val_f1': vl_f1}, best_path)
        print(f"  ✅ Sauvegardé (F1={vl_f1:.4f})")

# ── Phase 2 : fine-tuning complet ───────────────────────────────
print(f"\n📌 PHASE 2 — Fine-tuning complet ({NUM_EPOCHS} epochs)")
print("-"*60)

for param in model.parameters():
    param.requires_grad = True   # dégeler tout

opt2  = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sched = CosineAnnealingLR(opt2, T_max=NUM_EPOCHS, eta_min=LR*0.01)
patience_cnt = 0

for ep in range(1, NUM_EPOCHS+1):
    t0 = time.time()
    tr_loss, tr_acc, tr_rec             = train_epoch(model, train_dl, opt2, crit, DEVICE)
    vl_loss, vl_acc, vl_rec, vl_f1,_,_ = eval_epoch(model, val_dl, crit, DEVICE)
    sched.step()

    print(f"Ep {ep:3d} | Tr Loss={tr_loss:.4f} Acc={tr_acc:.3f} Rec={tr_rec:.3f} | "
          f"Val Loss={vl_loss:.4f} Acc={vl_acc:.3f} Rec={vl_rec:.3f} F1={vl_f1:.3f} | "
          f"{time.time()-t0:.0f}s")

    if vl_f1 > best_f1:
        best_f1, best_ep, patience_cnt = vl_f1, ep, 0
        torch.save({
            'epoch': ep,
            'model_state_dict': model.state_dict(),
            'val_f1': vl_f1, 'val_acc': vl_acc, 'val_rec': vl_rec
        }, best_path)
        print(f"  ✅ Sauvegardé (F1={vl_f1:.4f})")
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"\n⏹️  Early stopping epoch {ep}")
            break

# ════════════════════════════════════════════════════════════════
# ÉTAPE 9 : ÉVALUATION FINALE SUR LE TEST SET
#
# On charge le meilleur modèle (pas le dernier) et on évalue
# sur le test set qui n'a jamais été vu pendant l'entraînement.
# ════════════════════════════════════════════════════════════════

print(f"\n🏆 Meilleur : Epoch {best_ep}  F1={best_f1:.4f}")
model.load_state_dict(
    torch.load(best_path, map_location=DEVICE)['model_state_dict']
)
_, t_acc, t_rec, t_f1, t_preds, t_labels = eval_epoch(
    model, test_dl, crit, DEVICE
)

print("\n" + "="*60)
print("RÉSULTATS FINAUX T3 — EfficientNet-B0")
print("="*60)
print(f"Accuracy : {t_acc*100:.2f}%  {'✅' if t_acc>=0.90 else '❌'}  [objectif >90%]")
print(f"Recall   : {t_rec*100:.2f}%  {'✅' if t_rec>=0.85 else '❌'}  [objectif >85%]")
print(f"F1-Score : {t_f1:.4f}")
print()
print(classification_report(
    t_labels, t_preds,
    target_names=['Non détectable', 'Détectable']
))

cm = confusion_matrix(t_labels, t_preds)
print(f"Matrice confusion :")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
print()
print(f"  Faux négatifs (FN={cm[1,0]}) : pipeline détectable prédit non détectable")
print(f"  → Cas le plus dangereux industriellement")
print(f"\n💾 Modèle : {best_path}")