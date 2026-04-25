## Screenshot

![SUMO v5.0 Dashboard](screenshots/SUMO_v5_0.png)

🌞 SUMO – Sun Monitor
📡 Real-time Solar, HF Propagation & ISS Tracking Dashboard

SUMO est une application desktop développée en Python (PySide6) destinée aux radioamateurs.
Elle centralise en temps réel les données solaires, les conditions de propagation HF, et le suivi satellite (ISS) dans une interface claire et opérationnelle.

🚀 Fonctionnalités principales
🌞 Données solaires temps réel
Flux solaire SFI (F10.7 cm)
Nombre de taches solaires SSN
Indice géomagnétique Kp
Vent solaire (vitesse + densité)
Champ magnétique interplanétaire (Bz / Bt)
Flux X-ray (classification A/B/C/M/X)
Flux protons (échelle S0 → S5)
Activité aurorale (NOAA Ovation)

👉 Sources : NOAA SWPC

📡 Propagation HF
HF Bar : indicateur global des conditions radio
MUF (Maximum Usable Frequency) avec carte mondiale
Détection des ouvertures HF
Indicateur Radio Blackout
Visualisation des bandes ouvertes
🛰️ ISS Tracker (fonction avancée)
Position temps réel de l’ISS
Calcul des prochains passages (AOS / MAX / LOS)
Élévation maximale du passage
Compte à rebours avant passage
Basé sur la position utilisateur

👉 Éphémérides via Skyfield + TLE Celestrak

📟 DAPNET Integration
Envoi de messages vers pager (POCSAG)
Mode Emergency
Modules d’alerte :
X-ray
Proton
ISS
Météo solaire
Quick-send intégré à l’UI
🖼️ Imagerie solaire (SOHO / SDO)
EIT (171 / 195 / 284)
LASCO C2 / C3
HMI continuum
Rafraîchissement automatique
Multi-thread sécurisé
🌍 Carte MUF avancée
Interpolation mondiale (GIRO / KC2G style)
Échelle fixe (0–35 MHz)
Lissage intelligent
Contours géographiques
📰 Bandeau RSS dynamique
NASA Solar System News
Défilement configurable
Rafraîchissement automatique
🕒 Horloges analogiques
Horloge principale UTC ou locale
Horloges secondaires (multi villes)
Carillon horaire configurable
🎚️ Gestion audio
Activation/désactivation globale
Volume réglable via sliders
Sons indépendants :
Alertes solaires
Carillon horaire
⚙️ Paramètres configurables
Clé API NASA (DONKI)
RSS personnalisé
Paramètres DAPNET
Position géographique utilisateur
Rotation automatique des vues
🔄 Système d’affichage dynamique
9 panneaux solaires principaux
Modes alternés :
Solar
Clock
MUF
SOHO
Solar System
Rotation automatique configurable
🧠 Architecture & robustesse
Multi-thread (QThread sécurisé)
Retry automatique des dépendances (Skyfield / TLE)
Logs avancés (debug + runtime)
Anti-crash hooks
Gestion réseau optimisée (throttle)
🧱 Installation & build
▶️ Exécution Python
pip install -r requirements.txt
python main.py
🪟 Build Windows (PyInstaller)
python -m PyInstaller --noconfirm --clean --onedir --windowed `
  --name SUMO `
  --add-data "assets;assets" `
  main.py
📂 Structure du projet
SUMO/
 ├── main.py
 ├── assets/
 │    ├── sounds/
 │    ├── images/
 │    └── geojson/
 ├── skyfield_data/
 ├── sumo_config.json
 └── sumo_cache.sqlite
📡 Sources de données
NOAA SWPC → données solaires
NASA DONKI → événements solaires
Celestrak → TLE ISS
SOHO / SDO → imagerie
KC2G / GIRO → MUF
DAPNET → messaging radio
⚠️ Avertissement

SUMO est un outil d’aide à la décision pour les radioamateurs.
Les données sont fournies à titre indicatif et peuvent comporter des retards ou imprécisions.

👨‍💻 Auteur

Yoann – F4IGV
Projet développé pour la communauté radioamateur

📬 Feedback

Version en cours de test.
Les retours sont les bienvenus pour améliorer :

stabilité
lisibilité
fonctionnalités

73 DE F4IGV
⭐ Licence

GPL v3
