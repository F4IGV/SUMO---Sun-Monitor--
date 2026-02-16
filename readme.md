## Screenshot

![SUMO v5.0 Dashboard](screenshots/sumo_v5_0.png)

# SUMO v5.0 – Space Weather & DX Operations Monitor

## 🇬🇧 English

**SUMO (Solar Unified Monitoring Observatory)** is a real-time dashboard designed for **radio amateurs** who want to correlate **space weather conditions** with **on-air radio activity**.

Originally focused on solar and geomagnetic monitoring, **SUMO v5.0** evolves into a **hybrid space weather and DX operations tool**, combining scientific data with live radio spotting sources in a single, operational interface.

---

## 🌞 Space Weather Monitoring

SUMO continuously retrieves and displays data from official sources such as **NOAA SWPC** and **NASA DONKI**, including:

- Solar X-ray flux (GOES)
- Solar wind speed and magnetic parameters
- Geomagnetic activity (Kp index)
- Proton flux
- Sunspot Number (SSN) and Solar Flux Index (SFI)
- Aurora activity and oval evolution
- Coronal Mass Ejection (CME) alerts and probabilities

All charts use **true time-based axes** and provide short-term historical context to help identify trends and rapid changes.

---

## 📡 DX Activity Integration (New in v5.0)

SUMO v5.0 introduces a **left-side DX column**, fully integrated with the main dashboard:

- **DX Cluster (Telnet / DXSpider)** support with configurable server, port and callsign login
- **POTA (Parks On The Air)** spots retrieved from the official POTA API
- Unified DX table displaying frequency, callsign and spot age
- Runtime duplicate protection for POTA spots
- Optional geographic filtering for POTA spots:
  - Worldwide
  - USA
  - Europe
- DX column can be enabled or disabled at runtime without impacting the solar dashboard layout

This allows direct correlation between propagation conditions and real-time radio activity.

---

## 📶 HF Propagation Guidance

SUMO provides **heuristic HF indicators** designed to assist operational decision-making:

- MUF estimation helper
- Radio blackout severity estimation based on X-ray flare class

These indicators are intended as **practical guidance tools**, not as VOACAP or long-term propagation predictions.

---

## ⚙️ Configuration & Usability

- Centralized **Settings dialog** with tabbed sections:
  - RSS feeds
  - API keys
  - Clock mode (UTC / Local)
  - DX configuration
- All parameters are persisted in a human-readable `sumo_config.json`
- HF openings ribbon visibility can be toggled from the main interface
- Dynamic layout adapts automatically when DX features are enabled or disabled

---

## 🔊 Audio & Reliability

- Optional audio alerts for significant space weather events
- Hardened network and audio initialization
- Graceful handling of temporary network or data source outages

---

## 🖥️ Platform & Philosophy

- Developed in **Python / Qt** for Windows
- Designed as a **real-time operational tool**, not a static data viewer
- Focused on clarity, correlation and situational awareness for HF operators

---

## 🇫🇷 Français

**SUMO (Solar Unified Monitoring Observatory)** est un tableau de bord temps réel destiné aux **radioamateurs** souhaitant mettre en relation **la météo spatiale** et **l’activité radio sur les bandes**.

Initialement centré sur la surveillance solaire et géomagnétique, **SUMO v5.0** évolue vers un **outil hybride de veille spatiale et d’exploitation radio**, combinant données scientifiques et sources DX en direct dans une interface unique.

---

## 🌞 Surveillance de la météo spatiale

SUMO récupère et affiche en continu des données issues de sources officielles telles que **NOAA SWPC** et **NASA DONKI**, notamment :

- Flux X-ray solaire (GOES)
- Vent solaire et paramètres magnétiques
- Activité géomagnétique (indice Kp)
- Flux de protons
- Nombre de taches solaires (SSN) et indice de flux solaire (SFI)
- Activité aurorale et ovales
- Alertes et probabilités d’arrivée des CME

Les graphiques utilisent des **axes temporels réels** et conservent un historique court afin de mettre en évidence les tendances.

---

## 📡 Intégration de l’activité DX (nouveauté v5.0)

SUMO v5.0 introduit une **colonne DX latérale** intégrée au tableau principal :

- Support des **DX Cluster (Telnet / DXSpider)** avec configuration du serveur, du port et de l’indicatif
- Intégration des spots **POTA (Parks On The Air)** via l’API officielle
- Tableau DX unifié affichant fréquence, indicatif et âge du spot
- Protection contre les doublons POTA durant l’exécution
- Filtrage géographique optionnel des spots POTA :
  - Worldwide
  - USA
  - Europe
- La colonne DX peut être activée ou désactivée à l’exécution sans perturber l’affichage principal

---

## 📶 Aide à la propagation HF

SUMO fournit des **indicateurs heuristiques HF** destinés à l’aide à la décision opérationnelle :

- Estimation de la MUF
- Estimation du niveau de blackout radio basée sur la classe des éruptions X-ray

Ces indicateurs sont conçus comme une **aide visuelle**, et non comme des prévisions VOACAP.

---

## ⚙️ Configuration & ergonomie

- Fenêtre de réglages centralisée avec **onglets** (RSS, clés API, horloge, DX)
- Paramètres sauvegardés dans un fichier lisible `sumo_config.json`
- Bandeau d’ouverture HF activable ou désactivable depuis l’interface principale
- Mise en page dynamique s’adaptant à l’activation ou non des fonctions DX

---

## 🔊 Audio & fiabilité

- Alertes sonores optionnelles pour les événements significatifs
- Initialisation audio et réseau renforcée
- Gestion robuste des interruptions temporaires des sources de données

---

## 🖥️ Plateforme & philosophie

- Développé en **Python / Qt** pour Windows
- Pensé comme un **outil opérationnel temps réel**, et non comme un simple visualiseur de données
- Orienté clarté, corrélation et conscience de situation pour l’opérateur HF

