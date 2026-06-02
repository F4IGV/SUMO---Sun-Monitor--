## Screenshot

![SUMO v7_9_14 Dashboard](DashBord.png)

☀️ SUMO – Sun Monitor

SUMO (Sun Monitor) est une application de surveillance de la météo spatiale destinée aux radioamateurs, passionnés d'ondes courtes, opérateurs HF, observateurs de l'activité solaire et utilisateurs du réseau DAPNET.

Développé par Yoann Daniel (F4IGV), SUMO regroupe dans une seule interface moderne les principales données NOAA, NASA, DONKI, Météo-France, DAPNET et ISS afin d'offrir une vue en temps réel de l'environnement radioélectrique terrestre.

🚀 Fonctionnalités principales
☀️ Surveillance solaire temps réel

Affichage en temps réel des principaux indicateurs de météo spatiale :

Kp Index
Flux solaire SFI (10.7 cm)
Sunspot Number (SSN)
Flux X-Ray GOES
Flux de protons GOES (>10 MeV)
Vitesse du vent solaire
Champ magnétique interplanétaire :
Bz
Bt
Activité aurorale NOAA Ovation
Prévision et suivi des CME via NASA DONKI

Toutes les données sont récupérées directement depuis les services NOAA SWPC.

📈 Graphiques historiques

Chaque indicateur dispose de son historique :

Axe temporel réel NOAA
Affichage des tendances
Seuils d'alerte colorisés
Conservation des historiques NOAA complets lorsque disponibles
🎨 Système intelligent d'alertes visuelles

SUMO utilise une logique de sévérité uniforme :

Risques de perturbations
🟢 OK
🟡 WATCH
🟠 ALERT
🔴 DANGER
🟣 EXTREME
Conditions HF
🔴 POOR
🟠 FAIR
🟢 GOOD
🔵 EXCELLENT

Les changements d'état déclenchent automatiquement :

changement de couleur
clignotement des panneaux critiques
alertes sonores
📡 Intégration DAPNET

SUMO peut envoyer automatiquement des messages POCSAG via le réseau DAPNET.

Modules disponibles :

Alertes X-Ray

Détection automatique :

M1+
M5+
X1+
X10+

Messages configurables :

début d'événement
fin d'événement
mode urgence
Alertes Protoniques

Notifications :

S1 à S5
dépassement de seuil configurable
anti-spam intégré
Vigilance Météo-France

Envoi automatique des alertes :

Vent violent
Orages
Pluie-Inondation
Crues
Canicule
Neige-Verglas
Grand froid
Avalanches
Vagues-Submersion

Paramètres :

départements sélectionnables
seuils personnalisables
urgence automatique en vigilance rouge
Suivi ISS

Notifications automatiques :

Pré-alerte de passage
Début de passage
Point culminant
Fin de passage

Informations transmises :

élévation
azimut
heure du passage
🛰️ Suivi de l'ISS

Utilisation de :

Skyfield
TLE Celestrak

Fonctions :

calcul des passages
suivi orbital
géolocalisation de l'observateur
prévisions automatiques
🌎 Carte mondiale MUF

## Screenshot

![SUMO v7_9_14 muf](muf.png)

Module expérimental inspiré de KC2G :

Fonctionnalités :

interpolation mondiale des données ionosphériques
estimation MUF en temps réel
carte Europe
carte Monde
lissage avancé
système de confiance des données

Algorithmes :

IDW (Inverse Distance Weighting)
post-traitement HF
interpolation multi-passes
lissage gaussien
🌍 Horloges mondiales

## Screenshot

![SUMO v7_9_14 horloges](horloges.png)

Affichage :

horloge principale analogique
jusqu'à 4 horloges mondiales
fuseaux horaires configurables

Exemples :

Paris
Londres
New York
Tokyo
Sydney
📷 Tableau de bord SOHO

Affichage automatique :

SOHO EIT 171
SOHO EIT 195
SOHO EIT 284
LASCO C2
LASCO C3
HMI Continuum

Images téléchargées directement depuis les serveurs NASA/ESA.

🪐 Vue Système Solaire

Module graphique :

représentation des planètes
positions calculées avec Skyfield
affichage de la Terre et de l'ISS
données orbitales réelles
📰 Flux RSS scientifiques

Lecture automatique :

NASA Solar System News
Bulletins NOAA WWV

Défilement intégré dans l'interface.

🔊 Gestion avancée du son
alertes indépendantes
volume réglable
carillon horaire configurable
fichiers WAV personnalisables
⚙️ Configuration

Paramètres sauvegardés :

API NASA
DAPNET
Météo-France
Horloges
Sons
RSS
Affichages alternés
Position géographique
🛡️ Système Anti-Crash

SUMO intègre :

journalisation avancée
rotation automatique des logs
capture des exceptions non interceptées
dump des erreurs fatales
surveillance des threads

Fichiers générés :

sumo_debug_anti_crash.log
sumo_fault_anti_crash.log
sumo_runtime_state.log
🛠️ Technologies utilisées
Python 3.13+
PySide6
PyQtGraph
NumPy
Requests
SQLite
Skyfield
📡 Sources de données
NOAA SWPC
NASA DONKI
NASA RSS
ESA / SOHO
Celestrak
DAPNET
Météo-France
GIRO
KC2G
👨‍💻 Auteur

Sumo est une idée originale de Yoann Daniel – F4IGV & Eric - F4FAP

Passionnés de radioamateurisme, météo spatiale, propagation HF et développement logiciel.

Projet développé pour fournir aux radioamateurs un outil moderne de surveillance solaire et géomagnétique permettant d'anticiper les conditions de propagation et les événements susceptibles d'impacter les communications radio.

📜 Licence

Ce projet est distribué sous licence :

GNU General Public License v3.0 (GPL-3.0)

Vous êtes libre de :

utiliser
modifier
redistribuer

à condition de conserver la même licence et les mentions de copyright.

⭐ Si SUMO vous est utile, n'hésitez pas à mettre une étoile sur le projet GitHub et à contribuer à son développement.
