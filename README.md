# SUMO – Sun Monitor

**SUMO (Sun Monitor)** is a desktop application dedicated to **space weather monitoring** and **solar activity awareness**.

It is designed primarily for **radio amateurs**, **space weather enthusiasts**, and technically curious users who want a clear and reliable overview of solar and geomagnetic conditions that may impact radio propagation and technological systems.

SUMO relies exclusively on **official data sources** from NOAA and NASA.

---

## 🚀 Features

- Real-time monitoring of solar and geomagnetic activity
- CME arrival probability using **NASA DONKI**
- GOES **X-ray flux**
- **Proton flux (P10 / PFU)**
- **Solar wind speed**
- **Interplanetary Magnetic Field (Bz / Bt)**
- **Kp geomagnetic index**
- **Sunspot Number (SSN)** and **Solar Flux Index (SFI)**
- Color-coded visual alerts for fast situation awareness
- Optional sound alerts
- UTC or local time display
- Clean, dark interface optimized for continuous monitoring

---

## 🛠 Data sources

SUMO uses publicly available and authoritative data from:
- **NOAA Space Weather Prediction Center (SWPC)**
- **NASA DONKI (Space Weather Database of Notifications, Knowledge, Information)**

No third-party or unofficial sources are used.

---

## 💻 Platform

- **Windows** (compiled executable provided in Releases)
- Built with **Python** and **Qt (PySide6)**

---

## 📦 Installation

1. Go to the **Releases** section of this repository
2. Download the latest Windows distribution (`.zip`)
3. Extract the archive
4. Run `SUMO.exe`

No installation is required.

> ⚠️ On first launch, some antivirus software may display a warning.  
> This is a common false positive for unsigned Python executables.

---

## ⚙️ Configuration

SUMO provides a built-in **Settings** panel allowing you to:
- Enable or disable sound alerts
- Configure RSS information feeds
- Select time display mode (UTC / Local time)
- Adjust visual behavior of the interface

All settings are stored locally.

---

## 🧠 Philosophy

SUMO is a **technical tool**, built with clarity and accuracy in mind.  
It is also a project created by a developer who enjoys exploring, experimenting, and sometimes hiding small details in the code.

Some features are intentionally **not documented**.  
Users who explore the source code may discover more than what is described here.

---

## 📄 License

This project is released under the **GPL-3.0 License**.  
You are free to study, modify, and redistribute the code under the terms of this license.

---

## 🙌 Acknowledgments

- NOAA SWPC for providing open and reliable space weather data
- NASA DONKI for CME and solar event predictions
- The amateur radio and space weather communities for inspiration and feedback
