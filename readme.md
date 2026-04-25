## Screenshot

![SUMO v5.0 Dashboard](screenshots/SUMO_v5_0.png)

# 🌞 SUMO – Sun Monitor  
### 📡 Real-time Solar, HF Propagation & ISS Tracking Dashboard

---

# 🌞 SUMO – Sun Monitor  
### 📡 Real-time Solar, HF Propagation & ISS Tracking Dashboard

---

## 📖 Overview

SUMO is a desktop application written in Python (PySide6) designed for radio amateurs.  
It provides real-time solar data, HF propagation conditions, and ISS tracking in a clean, readable, and operational interface.

---

## 🚀 Main Features

### 🌞 Solar Data
- Solar Flux **SFI (F10.7 cm)**
- Sunspot Number **SSN**
- Geomagnetic index **Kp**
- Solar wind (speed, density)
- Interplanetary magnetic field (**Bz / Bt**)
- X-ray flux (A/B/C/M/X classification)
- Proton flux (S0 → S5 scale)
- Aurora activity (NOAA Ovation)

---

### 📡 HF Propagation
- Global HF condition bar
- **MUF map** (Maximum Usable Frequency)
- HF openings detection
- Radio blackout indicator
- Visualization of usable bands

---

### 🛰️ ISS Tracker
- Real-time ISS position
- Pass prediction:
  - AOS (Acquisition of Signal)
  - MAX (maximum elevation)
  - LOS (Loss of Signal)
- Maximum elevation display
- Countdown to next pass
- Based on user-defined location

---

### 📟 DAPNET Integration
- Send POCSAG messages
- **Emergency mode**
- Alert modules:
  - X-ray
  - Proton
  - ISS
  - Solar weather
- Quick-send interface integrated in UI

---

### 🖼️ Solar Imaging
- SOHO / SDO images:
  - EIT (171 / 195 / 284)
  - LASCO C2 / C3
  - HMI continuum
- Auto-refresh
- Multi-threaded safe loading

---

### 🌍 MUF Map
- Global MUF visualization
- Fixed scale (0–35 MHz)
- Smoothed interpolation
- Geographic overlays

---

### 📰 RSS Feed
- NASA Solar System News
- Scrolling ticker
- Adjustable speed and refresh

---

### 🕒 Clocks
- Main analog clock (UTC or local)
- Secondary clocks (multiple cities)
- Optional hourly chime

---

### 🎚️ Audio System
- Global sound toggle
- Volume control via sliders
- Independent sound levels:
  - Alerts
  - Hourly chime

---

### ⚙️ Settings
- NASA API key
- DAPNET credentials
- User geographic position
- Display rotation settings

---

### 🔄 Dynamic Display System
- 9 main solar panels
- Multiple display modes:
  - Solar dashboard
  - Clocks
  - MUF map
  - SOHO imagery
  - Solar system view
- Automatic rotation

---

### 🧠 Architecture & Stability
- Multi-threaded (QThread)
- Automatic retry (Skyfield / TLE)
- Network throttling
- Advanced logging system
- Anti-crash safeguards


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

