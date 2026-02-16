# CHANGELOG

## v5.0

---

## 🇬🇧 English

This release marks a major evolution of **SUMO**.  
While version 4.1 focused exclusively on solar and space weather monitoring, **v5.0
introduces real-time DX activity**, turning SUMO into a hybrid **space weather + radio
operations dashboard**.

### 🚀 New Features

#### DX Column (optional left panel)
- New **left-side DX column**, enabled or disabled at runtime.
- Two supported sources:
  - **DX Cluster (Telnet / DXSpider)** with configurable host, port and callsign login.
  - **POTA (Parks On The Air)** spots via the official `api.pota.app`.
- Unified DX table displaying frequency, callsign and spot age.
- Runtime duplicate protection for POTA spots.
- Seamless integration into the main layout without reducing the solar dashboard.

#### HF Propagation Helpers
- Introduction of **HF heuristic indicators**:
  - MUF estimation helper.
  - Radio blackout severity indicator based on X-ray class.
- These indicators are intended as **operational guidance**, not VOACAP predictions.

#### HF Openings Ribbon Control
- Dedicated top-bar button to **show or hide the HF openings ribbon**.
- Ribbon visibility is persisted in the configuration.

---

### ⚙️ Settings & Configuration

- Settings dialog expanded to include **DX-related configuration**:
  - DX column enable/disable.
  - DX source selection (DX Cluster or POTA).
  - Telnet host, port and login for DX Cluster.
- All new parameters are stored in `sumo_config.json`.
- Configuration remains backward compatible with v4.1.

---

### 🖥️ UI & Layout Improvements

- Dynamic layout reflow:
  - Enabling or disabling the DX column no longer leaves empty grid columns.
- Data connection/status indicator moved to a **bottom status bar**, freeing space in the top bar.
- Improved balance between information density and readability.

---

### 🔊 Audio & Runtime Improvements

- Hardened audio system initialization:
  - Explicit sound enable/disable handling.
  - Safer fallback behavior when audio backends are unavailable.
- Improved DX Cluster network handling:
  - Cleaner connect/disconnect lifecycle.
  - Safer stop and retry logic.

---

### 🛠️ Fixes & Stability

- Fixed layout inconsistencies when toggling the DX column at runtime.
- Improved resilience against temporary network or data source failures.

---

### ℹ️ Notes

- v5.0 is the **first SUMO release to merge solar activity monitoring and real-time DX activity**
  into a single operational interface.
- All solar data acquisition, caching and rendering features introduced in v4.1 remain unchanged
  and fully compatible.

---

## 🇫🇷 Français

Cette version marque une **évolution majeure de SUMO**.  
Alors que la version 4.1 était exclusivement dédiée à la surveillance de la météo
solaire et spatiale, **la v5.0 introduit l’activité DX en temps réel**, faisant de SUMO
un tableau de bord hybride **météo spatiale + exploitation radio**.

### 🚀 Nouvelles fonctionnalités

#### Colonne DX (panneau gauche optionnel)
- Nouvelle **colonne DX sur la gauche**, activable ou désactivable à l’exécution.
- Deux sources disponibles :
  - **DX Cluster (Telnet / DXSpider)** avec configuration du serveur, du port et de l’indicatif.
  - **POTA (Parks On The Air)** via l’API officielle `api.pota.app`.
- Tableau DX unifié affichant fréquence, indicatif et âge du spot.
- Protection contre les doublons POTA durant l’exécution.
- Intégration complète dans la mise en page sans réduire le tableau solaire principal.

#### Aides à la propagation HF
- Introduction d’**indicateurs heuristiques HF** :
  - Estimation de la MUF.
  - Indication du niveau de blackout radio basée sur la classe X-ray.
- Ces indicateurs sont conçus comme une **aide opérationnelle**, et non comme des prévisions VOACAP.

#### Contrôle du bandeau d’ouvertures HF
- Bouton dédié dans la barre supérieure pour **afficher ou masquer le bandeau HF**.
- L’état de visibilité est sauvegardé dans la configuration.

---

### ⚙️ Réglages & configuration

- Extension de la fenêtre de réglages avec la **configuration DX** :
  - Activation/désactivation de la colonne DX.
  - Sélection de la source DX (DX Cluster ou POTA).
  - Paramètres Telnet (hôte, port, indicatif).
- Tous les nouveaux paramètres sont stockés dans `sumo_config.json`.
- Le format de configuration reste compatible avec la v4.1.

---

### 🖥️ Interface & mise en page

- Recalcul dynamique de la mise en page :
  - L’activation ou la désactivation de la colonne DX ne laisse plus de colonne vide.
- Indicateur d’état des données déplacé dans une **barre de statut inférieure**,
  libérant de l’espace dans la barre supérieure.
- Interface mieux équilibrée pour l’affichage d’informations denses.

---

### 🔊 Audio & améliorations d’exécution

- Initialisation audio renforcée :
  - Gestion explicite de l’activation/désactivation du son.
  - Comportement de secours plus sûr en cas d’absence de backend audio.
- Amélioration de la gestion réseau du DX Cluster :
  - Cycle de connexion/déconnexion plus propre.
  - Arrêt et reprise plus sûrs en cas d’erreur.

---

### 🛠️ Correctifs & stabilité

- Correction des incohérences de mise en page lors du basculement de la colonne DX.
- Meilleure tolérance aux pannes réseau ou aux indisponibilités temporaires des sources de données.

---

### ℹ️ Notes

- La v5.0 est la **première version de SUMO combinant météo solaire et activité DX en temps réel**
  dans une interface unique.
- Toutes les fonctionnalités de récupération, de cache et d’affichage des données solaires
  introduites en v4.1 sont conservées et entièrement compatibles.

---
