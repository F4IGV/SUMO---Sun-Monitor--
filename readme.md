## Screenshot

![SUMO v5.0 Dashboard](screenshots/SUMO_v5_0.png)

# 🌞 SUMO – Sun Monitor  
### 📡 Real-time Solar, HF Propagation & ISS Tracking Dashboard

---

## 🇫🇷 Version Française

### 📖 Présentation

SUMO est une application desktop développée en Python (PySide6) destinée aux radioamateurs.  
Elle centralise en temps réel les données solaires, les conditions de propagation HF et le suivi de l’ISS dans une interface claire, lisible et opérationnelle.

---

### 🚀 Fonctionnalités principales

#### 🌞 Données solaires
- Flux solaire **SFI (F10.7 cm)**
- Nombre de taches solaires **SSN**
- Indice géomagnétique **Kp**
- Vent solaire (vitesse, densité)
- Champ magnétique interplanétaire (**Bz / Bt**)
- Flux X-ray (A/B/C/M/X)
- Flux protons (échelle S0 → S5)
- Activité aurorale

---

#### 📡 Propagation HF
- Barre HF globale (conditions radio)
- Carte **MUF** (fréquence maximale utilisable)
- Détection des ouvertures HF
- Indicateur Radio Blackout

---

#### 🛰️ ISS Tracker
- Position temps réel de l’ISS
- Prédiction des passages :
  - AOS / MAX / LOS
  - Élévation maximale
  - Compte à rebours
- Basé sur la position utilisateur

---

#### 📟 DAPNET
- Envoi de messages POCSAG
- Mode **Emergency**
- Alertes :
  - X-ray
  - Proton
  - ISS
  - Météo solaire

---

#### 🖼️ Imagerie solaire
- SOHO / SDO :
  - EIT (171 / 195 / 284)
  - LASCO C2 / C3
  - HMI continuum

---

#### 📰 RSS dynamique
- NASA Solar System News
- Bandeau défilant configurable

---

#### 🎚️ Audio
- Activation/désactivation
- Volume via sliders
- Sons indépendants

---

#### ⚙️ Configuration
- API NASA
- Paramètres DAPNET
- Position utilisateur
- Rotation automatique des vues

---

### 🧱 Installation

```bash
pip install -r requirements.txt
python main.py
