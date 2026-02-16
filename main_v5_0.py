#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SUMO - Sun Monitor
# Copyright (C) 2026 Yoann Daniel (F4IGV)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

"""
SUMO - Sun Monitor (Prototype)
UI: PySide6 + pyqtgraph

✅ UI:
- Bouton Fullscreen (vert/rouge) + ESC pour sortir
- Bouton Settings (jaune):
  - RSS feed URL configurable
  - NASA API key configurable (masqué)
  - Bouton "Get NASA API key" (ouvre https://api.nasa.gov/)
  - Sauvegarde dans sumo_config.json
  - Application immédiate (restart threads RSS + DataWorker)

✅ Time axis (cohérence échelle temps):
- Les courbes utilisent les timestamps NOAA/GOES (axe temps réel)
- Les séries ne sont plus tronquées à 60 points : on affiche l'historique complet du JSON récupéré

✅ FIX BT:
- Parsing MAG via le header NOAA (mapping par nom de colonne) pour éviter d'afficher un angle (lon/phi) à la place de Bt.
- Fallback: si la colonne Bt n'existe pas, Bt est calculé par sqrt(bx^2 + by^2 + bz^2) si possible.
"""

APP_VERSION = "v5.0"

import sys
import math
import time
import sqlite3
import xml.etree.ElementTree as ET
import json
import socket
import re
from dataclasses import dataclass
from datetime import datetime, timezone, date, timedelta
from pathlib import Path
from collections import deque

import numpy as np
import requests
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtMultimedia import QSoundEffect
import pyqtgraph as pg


# =====================================================
# RSS (NASA Solar System News)
# =====================================================
RSS_SOLAR_SYSTEM_URL = "https://www.nasa.gov/news-release/feed/"
NOAA_WWV_URL = "https://services.swpc.noaa.gov/text/wwv.txt"
RSS_REFRESH_SECONDS = 15 * 60  # refresh RSS every 15 minutes
RSS_SCROLL_FPS = 30
RSS_SCROLL_PX_PER_TICK = 2

NASA_API_KEY_REQUEST_URL = "https://api.nasa.gov/"  # official API key page

# PayPal donation link (replace with your own PayPal.me or donate link if desired)
PAYPAL_DONATE_URL = "https://www.paypal.me/yoanndaniel1"
GPL3_LICENSE_URL = "https://www.gnu.org/licenses/gpl-3.0.html"


# =====================================================
# STYLE
# =====================================================
BG_APP = "#0b0f12"
BG_PANEL = "#0e141a"
FG_TEXT = "#d7dde6"
BORDER = "#2a3440"

ACCENT_GREEN = "#44d16e"
ACCENT_GREY = "#7f8c8d"
ACCENT_BLUE = "#4aa3ff"  # BT in blue

STATUS_OK = "#44d16e"
STATUS_WARN = "#ffd34d"
STATUS_ERR = "#ff4d4d"

BTN_GREEN = "#44d16e"
BTN_RED = "#ff4d4d"
BTN_TEXT = "#0b0f12"

BTN_SETTINGS_YELLOW = "#ffd34d"


# =====================================================
# NOAA + DONKI ENDPOINTS
# =====================================================
NOAA_BASE = "https://services.swpc.noaa.gov"
NOAA_URLS = {
    "kp": f"{NOAA_BASE}/products/noaa-planetary-k-index.json",
    "sw_mag_6h": f"{NOAA_BASE}/products/solar-wind/mag-6-hour.json",
    "sw_plasma_6h": f"{NOAA_BASE}/products/solar-wind/plasma-6-hour.json",
    "goes_xray_7d": f"{NOAA_BASE}/json/goes/primary/xrays-7-day.json",
    "goes_protons_7d": f"{NOAA_BASE}/json/goes/primary/integral-protons-7-day.json",
    "daily_solar_indices": f"{NOAA_BASE}/text/daily-solar-indices.txt",
    "ovation_aurora_latest": f"{NOAA_BASE}/json/ovation_aurora_latest.json",
}
DONKI_BASE = "https://api.nasa.gov/DONKI"


# =====================================================
# PATHS
# =====================================================
def resource_path(rel: str) -> Path:
    """Return a resource path that works both in dev and in a PyInstaller bundle."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / rel  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent / rel


APP_DIR = Path.cwd()  # folder where the exe/script is launched
ASSETS_DIR = resource_path("assets")
DB_PATH = APP_DIR / "sumo_cache.sqlite"

CONFIG_PATH = APP_DIR / "sumo_config.json"
DEFAULT_RSS_URL = RSS_SOLAR_SYSTEM_URL


# =====================================================
# CONFIG HELPERS
# =====================================================
def load_config(path: Path) -> dict:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def save_config(path: Path, cfg: dict) -> None:
    try:
        path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


# =====================================================
# TIME PARSING (robuste)
# =====================================================
def _to_epoch_seconds(dt: datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).timestamp()


def parse_time_to_epoch(value) -> float:
    """
    Parse plusieurs formats NOAA/GOES possibles en seconds epoch (UTC).
    Retourne NaN si non parsable.
    """
    try:
        if value is None:
            return float("nan")
        if isinstance(value, (int, float)):
            return float(value)

        s = str(value).strip()
        if not s:
            return float("nan")

        # ISO 8601 (GOES): 2026-02-08T12:34:56Z / ...+00:00
        if "T" in s:
            s2 = s.replace("Z", "+00:00")
            try:
                dt = datetime.fromisoformat(s2)
                return _to_epoch_seconds(dt)
            except Exception:
                pass

        # NOAA: "YYYY-MM-DD HH:MM:SS" or with milliseconds
        for fmt in (
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
        ):
            try:
                dt = datetime.strptime(s, fmt)
                return _to_epoch_seconds(dt)
            except Exception:
                continue

        return float("nan")
    except Exception:
        return float("nan")


def align_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Aligne x/y (même longueur), enlève les NaN de x, garde l'ordre."""
    if x is None or y is None:
        return np.array([], dtype=float), np.array([], dtype=float)

    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    n = int(min(len(x), len(y)))
    if n <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    x = x[-n:]
    y = y[-n:]

    m = np.isfinite(x)
    if not np.any(m):
        return np.array([], dtype=float), np.array([], dtype=float)

    x = x[m]
    y = y[m]
    return x, y


# =====================================================
# MODELS
# =====================================================
@dataclass
class PanelConfig:
    title: str
    unit: str = ""
    mode: str = "line"
    big_value_fmt: str = "{:.1f}"


# =====================================================
# HELPERS
# =====================================================
def proton_s_scale(p10_pfu: float) -> str:
    if p10_pfu is None or (isinstance(p10_pfu, float) and math.isnan(p10_pfu)):
        return "S?"
    if p10_pfu >= 100000:
        return "S5"
    if p10_pfu >= 10000:
        return "S4"
    if p10_pfu >= 1000:
        return "S3"
    if p10_pfu >= 100:
        return "S2"
    if p10_pfu >= 10:
        return "S1"
    return "S0"


def s_level_color(s_level: str) -> str:
    return {
        "S0": "#44d16e",
        "S1": "#ffd34d",
        "S2": "#ff9f43",
        "S3": "#ff4d4d",
        "S4": "#c0392b",
        "S5": "#8e44ad",
        "S?": ACCENT_GREY,
    }.get(s_level, ACCENT_GREY)


def xray_flare_class(flux_w_m2: float) -> tuple[str, float]:
    if flux_w_m2 is None or (isinstance(flux_w_m2, float) and math.isnan(flux_w_m2)) or flux_w_m2 <= 0:
        return "?", float("nan")
    if flux_w_m2 >= 1e-4:
        return "X", flux_w_m2 / 1e-4
    if flux_w_m2 >= 1e-5:
        return "M", flux_w_m2 / 1e-5
    if flux_w_m2 >= 1e-6:
        return "C", flux_w_m2 / 1e-6
    if flux_w_m2 >= 1e-7:
        return "B", flux_w_m2 / 1e-7
    return "A", flux_w_m2 / 1e-8


def xray_class_color(letter: str) -> str:
    return {
        "A": "#44d16e",
        "B": "#8be28b",
        "C": "#ffd34d",
        "M": "#ff9f43",
        "X": "#ff4d4d",
        "?": ACCENT_GREY,
    }.get(letter, ACCENT_GREY)


def xray_class_label(letter: str, magnitude: float) -> str:
    if letter == "?" or (isinstance(magnitude, float) and math.isnan(magnitude)):
        return "?"
    return f"{letter}{magnitude:.1f}"


def sfi_color(val: float) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ACCENT_GREY
    if val < 80:
        return "#4aa3ff"
    if val < 100:
        return "#44d16e"
    if val < 150:
        return "#ff9f43"
    if val < 170:
        return "#ff4d4d"
    if val < 200:
        return "#9b59b6"
    return "#9b59b6"


def ssn_color(val: float) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ACCENT_GREY
    if val < 20:
        return "#4aa3ff"
    if val < 60:
        return "#ffd34d"
    if val < 120:
        return "#44d16e"
    if val < 150:
        return "#ff9f43"
    if val < 180:
        return "#ff4d4d"
    return "#ff4d4d"


def solar_wind_color(speed: float) -> str:
    """Color scale for Solar Wind Speed (km/s) accent."""
    if speed is None or (isinstance(speed, float) and math.isnan(speed)):
        return ACCENT_GREY
    if speed < 350:
        return "#4aa3ff"   # blue (slow)
    if speed < 450:
        return "#44d16e"   # green (normal)
    if speed < 550:
        return "#ff9f43"   # orange (fast)
    return "#ff4d4d"       # red (very fast)


def kp_g_scale(kp: float) -> str:
    if kp is None or (isinstance(kp, float) and math.isnan(kp)):
        return "G?"
    if kp >= 8:
        return "G5"
    if kp >= 7:
        return "G4"
    if kp >= 6:
        return "G3"
    if kp >= 5:
        return "G2"
    if kp >= 4:
        return "G1"
    return "G0"


def kp_color(kp: float) -> str:
    if kp is None or (isinstance(kp, float) and math.isnan(kp)):
        return ACCENT_GREY
    if kp >= 8:
        return "#8e44ad"
    if kp >= 7:
        return "#9b59b6"
    if kp >= 6:
        return "#ff4d4d"
    if kp >= 5:
        return "#ff9f43"
    if kp >= 4:
        return "#ffd34d"
    return "#44d16e"


def cme_color(level: str) -> str:
    level = (level or "").upper().strip()
    return {
        "NONE": "#44d16e",
        "LOW":  "#ffd34d",
        "MED":  "#ff9f43",
        "HIGH": "#ff4d4d",
        "N/A":  ACCENT_GREY,
        "":     ACCENT_GREY,
    }.get(level, ACCENT_GREY)


# =====================================================
# RSS: fetch + parse
# =====================================================
def fetch_rss_titles(url: str, timeout: int = 15, max_items: int = 12) -> list[str]:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "SUMO-SunMonitor/0.1"})
    r.raise_for_status()

    root = ET.fromstring(r.content)

    titles: list[str] = []
    for item in root.findall(".//item"):
        t = item.findtext("title")
        if t:
            titles.append(" ".join(t.split()))
        if len(titles) >= max_items:
            break

    if not titles:
        for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry"):
            t = entry.findtext("{http://www.w3.org/2005/Atom}title")
            if t:
                titles.append(" ".join(t.split()))
            if len(titles) >= max_items:
                break

    return titles


def _is_text_banner_url(url: str) -> bool:
    """Heuristique simple: une source 'texte brut' (ex: NOAA WWV) au lieu d'un flux RSS/Atom."""
    u = (url or "").strip().lower()
    return u.endswith(".txt") or "/text/" in u


def fetch_text_lines(url: str, timeout: int = 15, max_lines: int = 50) -> list[str]:
    """Récupère un bulletin texte brut (ex: NOAA WWV) et renvoie une liste de lignes 'propres'."""
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "SUMO-SunMonitor/0.1"})
    r.raise_for_status()
    raw = r.text or ""
    lines: list[str] = []
    for ln in raw.splitlines():
        ln = " ".join(ln.strip().split())
        if ln:
            lines.append(ln)
        if len(lines) >= max_lines:
            break
    return lines


def fetch_banner_text(url: str) -> tuple[str, str]:
    """Retourne (label, text) pour le bandeau défilant, selon le type de source."""
    u = (url or "").strip()
    if not u:
        return ("RSS", "(no url)")
    if _is_text_banner_url(u):
        lines = fetch_text_lines(u)
        text = "   •   ".join(lines) if lines else "(no data)"
        label = "NOAA WWV Geophysical Alert"
        return (label, text)

    titles = fetch_rss_titles(u)
    text = "   •   ".join(titles) if titles else "(no items)"
    label = "NASA Solar System News"
    return (label, text)


class RssWorker(QtCore.QObject):
    rss_ready = QtCore.Signal(str)
    rss_error = QtCore.Signal(str)

    def __init__(self, url: str, refresh_seconds: int):
        super().__init__()
        self.url = url
        self.refresh_seconds = refresh_seconds
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        while not self._stop:
            try:
                label, body = fetch_banner_text(self.url)
                text = f"{label}   •   {body}"
                self.rss_ready.emit(text)
            except Exception as e:
                self.rss_error.emit(str(e))

            for _ in range(self.refresh_seconds):
                if self._stop:
                    break
                time.sleep(1)



# =====================================================
# DX CLUSTER (Telnet) - Worker + Panel
# =====================================================

def _dx_band_color(freq_val: float) -> str:
    """Return a background color for a frequency.

    DX clusters usually emit the frequency in **kHz** (e.g. 14074.0, 144200.0),
    while our UI logic wants **MHz**. This helper accepts either:
      - MHz (e.g. 14.074, 144.200)
      - kHz (e.g. 14074.0, 144200.0)  -> auto-converted to MHz
    """
    try:
        f = float(freq_val)
    except Exception:
        return "#121a22"

    # Auto-detect kHz vs MHz
    # Anything above ~1000 is almost certainly kHz (since 1000 MHz = 1 GHz).
    if f >= 1000.0:
        f = f / 1000.0  # kHz -> MHz

    # HF amateur bands (MHz) — each band gets its own distinct color
    # (freq ranges are intentionally broad to match typical DX spot frequencies)
    if 1.8   <= f <= 2.0:     return "#2dd4bf"  # 160m (teal)
    if 3.5   <= f <= 4.0:     return "#22c55e"  # 80m  (green)
    if 5.3   <= f <= 5.5:     return "#60a5fa"  # 60m  (blue)
    if 7.0   <= f <= 7.3:     return "#f59e0b"  # 40m  (amber)
    if 10.1  <= f <= 10.15:   return "#a78bfa"  # 30m  (violet)
    if 14.0  <= f <= 14.35:   return "#f472b6"  # 20m  (pink)
    if 18.068<= f <= 18.168:  return "#34d399"  # 17m  (mint)
    if 21.0  <= f <= 21.45:   return "#fb7185"  # 15m  (rose)
    if 24.89 <= f <= 24.99:   return "#93c5fd"  # 12m  (light blue)
    if 28.0  <= f <= 29.7:    return "#f97316"  # 10m  (orange)
    if 50.0  <= f <= 54.0:    return "#ef4444"  # 6m   (red)
    if 70.0  <= f <= 70.5:    return "#e879f9"  # 4m   (magenta, region dependent)

    # VHF/UHF/SHF (MHz)
    # Note: allocations vary by ITU region; these ranges are broad enough for most spots.
    # EU/ITU1 common: 2m = 144–146, 70cm = 430–440 (we keep wider to be safe).
    if 144.0 <= f <= 148.0: return "#38bdf8"  # 2m (cyan)
    if 219.0 <= f <= 225.0: return "#60a5fa"  # 1.25m (US) (blue)
    if 420.0 <= f <= 450.0: return "#facc15"  # 70cm (yellow)
    if 902.0 <= f <= 928.0: return "#2dd4bf"  # 33cm (US) (teal)
    if 1240.0 <= f <= 1300.0: return "#c084fc"  # 23cm (purple)

    # Fallback: bucket by frequency range so it's never "all violet"
    if f < 30.0:   return "#8bd3dd"  # generic HF
    if f < 300.0:  return "#caffbf"  # generic VHF
    if f < 1000.0: return "#ffd6a5"  # generic UHF
    return "#b36bff"  # generic SHF


class DxClusterWorker(QtCore.QObject):
    """Simple DXCluster (DXSpider-like) telnet reader (Python 3.13 safe).

    Notes:
    - Many DXSpider nodes use TELNET negotiation (IAC ...). We strip it.
    - Most nodes require a callsign login after a prompt ("login:", "call:", "callsign").
      Sending the callsign too early can cause immediate disconnect.
    Emits parsed spots as dicts: {"freq": float, "call": str, "raw": str, "ts": float}.
    """
    spot = QtCore.Signal(dict)
    status = QtCore.Signal(str)
    error = QtCore.Signal(str)

    def __init__(self, host: str, port: int, login: str = ""):
        super().__init__()
        self.host = host
        self.port = int(port)
        self.login = (login or "").strip()
        self._stop = False

        # Incremental TELNET filter state.
        # This avoids the classic bug where keeping a raw tail (for split IAC sequences)
        # causes already-decoded text to be decoded again, producing duplicated spots.
        self._tn_state = {
            "iac": False,          # last byte was IAC
            "cmd": None,           # DO/DONT/WILL/WONT waiting for opt
            "sb": False,           # inside subnegotiation
            "sb_iac": False,       # saw IAC inside SB, waiting for SE or IAC
        }

    def _telnet_filter(self, chunk: bytes) -> bytes:
        """Filter a TELNET stream incrementally and return printable payload bytes."""
        IAC = 255
        DO, DONT, WILL, WONT = 253, 254, 251, 252
        SB, SE = 250, 240

        st = self._tn_state
        out = bytearray()

        for b in chunk:
            # Waiting for option after DO/DONT/WILL/WONT
            if st["cmd"] is not None:
                # consume option byte, ignore
                st["cmd"] = None
                continue

            # Inside subnegotiation
            if st["sb"]:
                if st["sb_iac"]:
                    # We previously saw IAC inside SB
                    if b == SE:
                        st["sb"] = False
                        st["sb_iac"] = False
                        continue
                    if b == IAC:
                        # escaped IAC inside SB
                        st["sb_iac"] = False
                        continue
                    # some other command; stay in SB
                    st["sb_iac"] = False
                    continue

                if b == IAC:
                    st["sb_iac"] = True
                # ignore SB payload
                continue

            # Not in SB
            if st["iac"]:
                st["iac"] = False

                if b == IAC:
                    # Escaped 255 -> literal 255
                    out.append(IAC)
                    continue
                if b in (DO, DONT, WILL, WONT):
                    st["cmd"] = b
                    continue
                if b == SB:
                    st["sb"] = True
                    st["sb_iac"] = False
                    continue
                # Other TELNET commands are ignored
                continue

            if b == IAC:
                st["iac"] = True
                continue

            # Normal printable byte
            out.append(b)

        return bytes(out)

    def stop(self):
        self._stop = True

    def _emit_status(self, s: str):
        try:
            self.status.emit(str(s))
        except Exception:
            pass

    @staticmethod
    def _strip_telnet_iac(buf: bytes) -> bytes:
        """Remove TELNET IAC negotiation sequences from a byte buffer."""
        IAC = 255
        DO, DONT, WILL, WONT = 253, 254, 251, 252
        SB, SE = 250, 240

        out = bytearray()
        i = 0
        n = len(buf)
        while i < n:
            b = buf[i]
            if b != IAC:
                out.append(b)
                i += 1
                continue

            # IAC encountered
            if i + 1 >= n:
                # incomplete IAC at end; drop it
                break
            cmd = buf[i + 1]

            # Escaped 255 (IAC IAC) => literal 255
            if cmd == IAC:
                out.append(IAC)
                i += 2
                continue

            # WILL/WONT/DO/DONT <opt>
            if cmd in (DO, DONT, WILL, WONT):
                if i + 2 < n:
                    i += 3
                else:
                    break
                continue

            # Subnegotiation: IAC SB ... IAC SE
            if cmd == SB:
                i += 2
                # consume until IAC SE
                while i < n:
                    if buf[i] == IAC and (i + 1) < n and buf[i + 1] == SE:
                        i += 2
                        break
                    i += 1
                continue

            # Other 2-byte commands: skip IAC <cmd>
            i += 2

        return bytes(out)

    def run(self):
        # Best-effort reconnect loop
        while not self._stop:
            sock = None
            try:
                self._emit_status(f"connecting to {self.host}:{self.port}…")
                sock = socket.create_connection((self.host, int(self.port)), timeout=10)
                sock.settimeout(1.0)

                connected_at = time.time()
                logged = False
                last_rx = time.time()

                text_buf = ""  # decoded, telnet-filtered (incremental)

                self._emit_status(f"connected to {self.host}:{self.port} (waiting login prompt)")

                while not self._stop:
                    try:
                        chunk = sock.recv(4096)
                    except (socket.timeout, TimeoutError):
                        # normal idle: keep connection
                        chunk = b""

                    if chunk:
                        last_rx = time.time()
                        # TELNET negotiation can arrive split across recv() calls.
                        # Use the incremental filter to avoid re-decoding old bytes
                        # (which was causing each spot line to appear twice).
                        clean = self._telnet_filter(chunk)
                        try:
                            text_buf += clean.decode("utf-8", errors="ignore")
                        except Exception:
                            pass

                    # If server closed connection
                    if chunk == b"":
                        # no data: only treat as closed if recv returned 0 bytes *without* timeout
                        # In our code, timeout sets chunk=b"" too, so we can't tell here.
                        # Instead, use a longer "no RX" window after having been connected.
                        pass

                    # 1) Login handling: wait for a prompt
                    if self.login and not logged and text_buf:
                        low = text_buf.lower()
                        # Common prompts; be permissive
                        if ("login" in low) or ("call:" in low) or ("callsign" in low) or ("enter your call" in low):
                            try:
                                sock.sendall((self.login + "\r\n").encode("utf-8", errors="ignore"))
                                logged = True
                                self._emit_status(f"DX: logged as {self.login}")
                                # reset buffers to avoid parsing banner as spots
                                text_buf = ""
                                continue
                            except Exception as e:
                                raise ConnectionError(f"failed to send login: {e}")

                        # Fallback: if a prompt isn't explicit, some nodes show '>' after banner
                        if (time.time() - connected_at) > 2.0 and (">" in low or "dxspider" in low):
                            try:
                                sock.sendall((self.login + "\r\n").encode("utf-8", errors="ignore"))
                                logged = True
                                self._emit_status(f"DX: logged as {self.login}")
                                text_buf = ""
                                continue
                            except Exception as e:
                                raise ConnectionError(f"failed to send login: {e}")

                    # 2) If we haven't received anything for a while, treat as disconnect
                    # (some nodes accept TCP then immediately close on bad login)
                    if (time.time() - last_rx) > 20.0 and not logged:
                        raise ConnectionError("no banner/prompt received (node closed or filtered)")

                    # 3) Parse complete lines for spots
                    if "\n" in text_buf:
                        lines = text_buf.splitlines()
                        # keep the last partial line (if any)
                        if not text_buf.endswith("\n"):
                            text_buf = lines.pop() if lines else ""
                        else:
                            text_buf = ""

                        for line in lines:
                            line = line.strip()
                            if not line:
                                continue

                            # DXSpider classic: "DX de CALL-#: 14074.0  STATION  ..."
                            if line.startswith("DX de"):
                                m1 = re.search(r":\s*([0-9]+\.[0-9]+)\s+(\S+)", line)
                                if m1:
                                    try:
                                        freq = float(m1.group(1))
                                    except Exception:
                                        freq = float("nan")
                                    call = (m1.group(2) or "").strip()
                                    self.spot.emit({"freq": freq, "call": call, "raw": line, "ts": time.time()})
                                continue

                            # Some nodes print: "<freq> <call> ..." (rare) — keep best effort
                            m2 = re.match(r"^\s*([0-9]+\.[0-9]+)\s+(\S+)\s+(.+)$", line)
                            if m2:
                                try:
                                    freq = float(m2.group(1))
                                except Exception:
                                    freq = float("nan")
                                call = (m2.group(2) or "").strip()
                                self.spot.emit({"freq": freq, "call": call, "raw": line, "ts": time.time()})
                                continue

                    # 4) Gentle loop pacing
                    QtCore.QThread.msleep(20)

                # stop requested
            except Exception as e:
                try:
                    self.error.emit(str(e))
                except Exception:
                    pass
                self._emit_status(f"DX: offline ({e})")
                # Backoff (max ~10s) without blocking stop
                for _ in range(10):
                    if self._stop:
                        break
                    time.sleep(1)
            finally:
                try:
                    if sock:
                        sock.close()
                except Exception:
                    pass



# =====================================================
# POTA SPOTS (HTTP JSON) - Worker
# =====================================================

POTA_SPOTS_URL = "https://api.pota.app/spot/activator"  # public JSON list of current activators

def _pota_parse_time_iso(ts: str) -> float:
    """Parse POTA spotTime ISO string to epoch seconds (best-effort)."""
    s = (ts or "").strip()
    if not s:
        return time.time()
    try:
        # Example: 2023-12-29T16:31:57  (no timezone)
        # We assume UTC if tz info is missing.
        if s.endswith("Z"):
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return time.time()


class PotaSpotsWorker(QtCore.QObject):
    """Poll POTA spots (current activators) and emit them as 'spot' dicts.

    Emits spot dicts compatible with DxClusterPanel.add_spot():
      {"freq": float, "call": str, "raw": str, "ts": float}

    Source: https://api.pota.app/spot/activator (JSON array). citeturn3search3turn3search9
    """
    spot = QtCore.Signal(dict)
    status = QtCore.Signal(str)
    error = QtCore.Signal(str)

    def __init__(self, url: str = POTA_SPOTS_URL, refresh_seconds: int = 30):
        super().__init__()
        self.url = url
        self.refresh_seconds = int(refresh_seconds)
        self._stop = False
        self._seen_ids: set[str] = set()

    def stop(self):
        self._stop = True

    def _emit_status(self, s: str):
        try:
            self.status.emit(str(s))
        except Exception:
            pass

    def run(self):
        while not self._stop:
            try:
                self._emit_status("POTA: loading…")
                r = requests.get(self.url, timeout=15, headers={"User-Agent": "SUMO-SunMonitor/0.1"})
                r.raise_for_status()
                data = r.json()

                if not isinstance(data, list):
                    raise ValueError("unexpected POTA response (not a list)")

                # Sort newest first by spotTime
                def _key(x):
                    try:
                        return _pota_parse_time_iso(str(x.get("spotTime") or ""))
                    except Exception:
                        return 0.0

                data_sorted = sorted(data, key=_key, reverse=False)  # oldest first; panel inserts new spots at top

                emitted = 0
                now = time.time()

                for it in data_sorted:
                    if self._stop:
                        break
                    try:
                        freq_s = str(it.get("frequency") or "").strip()
                        call = str(it.get("activator") or it.get("callsign") or it.get("call") or "").strip()
                        ref = str(it.get("reference") or it.get("park") or "").strip()
                        mode = str(it.get("mode") or "").strip()
                        spot_time = str(it.get("spotTime") or "").strip()

                        try:
                            freq = float(freq_s) if freq_s else float("nan")
                        except Exception:
                            freq = float("nan")

                        ts = _pota_parse_time_iso(spot_time) if spot_time else now

                        # Build a stable-ish id to avoid duplicates across refresh cycles
                        sid = f"{call}|{ref}|{mode}|{freq_s}|{int(ts)}"
                        if sid in self._seen_ids:
                            continue
                        self._seen_ids.add(sid)

                        raw = f"POTA {ref} {mode} {freq_s} {call}".strip()
                        loc = str(it.get("locationDesc") or it.get("location") or "").strip()
                        self.spot.emit({"freq": freq, "call": call, "raw": raw, "ts": ts, "reference": ref, "locationDesc": loc})
                        emitted += 1

                        # Keep the seen set bounded
                        if len(self._seen_ids) > 4000:
                            # drop random-ish old entries by recreating from tail
                            self._seen_ids = set(list(self._seen_ids)[-2000:])

                        if emitted >= 60:
                            break
                    except Exception:
                        continue

                self._emit_status(f"POTA: ok ({len(data)} active)")

            except Exception as e:
                try:
                    self.error.emit(str(e))
                except Exception:
                    pass
                self._emit_status(f"POTA: offline ({e})")

            # Sleep (interruptible)
            for _ in range(max(5, self.refresh_seconds)):
                if self._stop:
                    break
                time.sleep(1)


class DxClusterPanel(QtWidgets.QFrame):
    """A compact left-side DX Cluster panel (list of latest spots)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("dxPanel")
        self.setFrameShape(QtWidgets.QFrame.NoFrame)

        self._spots: list[dict] = []
        self._max_spots = 40

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        head = QtWidgets.QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(8)

        self.lbl_title = QtWidgets.QLabel("Cluster")
        self.lbl_title.setStyleSheet("color:#44d16e; font-size: 28px; font-weight: 900;")
        head.addWidget(self.lbl_title, 0)

        head.addStretch(1)

        self.lbl_count = QtWidgets.QLabel("▲0")
        self.lbl_count.setStyleSheet("color:#44d16e; font-size: 18px; font-weight: 900;")
        head.addWidget(self.lbl_count, 0)

        root.addLayout(head)

        self.lbl_sub = QtWidgets.QLabel("offline")
        self.lbl_sub.setStyleSheet("color:#44d16e; font-size: 16px; font-family: Consolas; font-weight: 800;")
        root.addWidget(self.lbl_sub, 0)

        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Freq", "Call", "Age"])
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setFocusPolicy(QtCore.Qt.NoFocus)
        self.table.setAlternatingRowColors(False)
        self.table.setStyleSheet("""
            QTableWidget {
                background: #0f1720;
                color: #d7dde6;
                border: 1px solid #2a3440;
                border-radius: 10px;
                font-family: Consolas;
                font-size: 14px;
            }
            QHeaderView::section {
                background: #0e141a;
                color: #aab6c5;
                font-weight: 900;
                border: 0px;
                padding: 6px 6px;
            }
        """)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.table.setColumnWidth(2, 52)

        root.addWidget(self.table, 1)

        # Age updater
        self._age_timer = QtCore.QTimer(self)
        self._age_timer.timeout.connect(self._refresh_age)
        self._age_timer.start(1000)

        self.setStyleSheet("""
            QFrame#dxPanel {
                background: #0f1720;
                border: 2px solid #2a3440;
                border-radius: 14px;
            }
        """)

    def set_connection_label(self, host: str, port: int):
        self.lbl_sub.setText(f"{host}:{int(port)}")

    def set_status(self, s: str):
        # keep one-line status
        self.lbl_sub.setText(s)

    def set_title(self, t: str):
        try:
            self.lbl_title.setText(str(t))
        except Exception:
            pass

    def clear_spots(self):
        self._spots.clear()
        self.table.setRowCount(0)
        self.lbl_count.setText("▲0")

    def add_spot(self, spot: dict):
        self._spots.insert(0, spot)
        if len(self._spots) > self._max_spots:
            self._spots = self._spots[: self._max_spots]

        self._rebuild_table()

    def _rebuild_table(self):
        self.table.setRowCount(len(self._spots))
        now = time.time()

        for r, s in enumerate(self._spots):
            freq = s.get("freq", float("nan"))
            call = (s.get("call") or "").strip()
            ts = float(s.get("ts") or now)

            # Freq cell with colored background
            freq_txt = "--.--"
            try:
                if isinstance(freq, float) and not math.isnan(freq):
                    freq_txt = f"{freq:0.1f}"
            except Exception:
                pass

            it0 = QtWidgets.QTableWidgetItem(freq_txt)
            it0.setTextAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            it0.setForeground(QtGui.QBrush(QtGui.QColor("#0b0f12")))
            it0.setBackground(QtGui.QBrush(QtGui.QColor(_dx_band_color(freq))))
            it0.setFont(QtGui.QFont("Consolas", 14, QtGui.QFont.Bold))

            it1 = QtWidgets.QTableWidgetItem(call)
            it1.setTextAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            it1.setForeground(QtGui.QBrush(QtGui.QColor("#d7dde6")))
            it1.setFont(QtGui.QFont("Consolas", 14, QtGui.QFont.Bold))

            age = max(0.0, now - ts)
            age_txt = f"{int(age/60)}m" if age >= 60 else f"{int(age)}s"
            it2 = QtWidgets.QTableWidgetItem(age_txt)
            it2.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            it2.setForeground(QtGui.QBrush(QtGui.QColor("#d7dde6")))
            it2.setFont(QtGui.QFont("Consolas", 12, QtGui.QFont.Bold))

            self.table.setItem(r, 0, it0)
            self.table.setItem(r, 1, it1)
            self.table.setItem(r, 2, it2)

        self.lbl_count.setText(f"▲{len(self._spots)}")

    def _refresh_age(self):
        # Update the "Age" column only (cheap)
        try:
            now = time.time()
            for r, s in enumerate(self._spots):
                ts = float(s.get("ts") or now)
                age = max(0.0, now - ts)
                age_txt = f"{int(age/60)}m" if age >= 60 else f"{int(age)}s"
                it = self.table.item(r, 2)
                if it:
                    it.setText(age_txt)
        except Exception:
            pass
class RssTicker(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("rssTicker")
        self._text = "NASA Solar System News   •   loading…"
        self._offset = 0
        self._text_width = 0
        self._px_per_tick = RSS_SCROLL_PX_PER_TICK

        font = self.font()
        base = max(10, font.pointSize())
        font.setPointSize(int(base * 3.0))
        font.setBold(True)
        self.setFont(font)

        self.setMinimumHeight(36)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(int(1000 / RSS_SCROLL_FPS))

    def setSpeed(self, px_per_tick: int):
        """Set scroll speed in pixels per tick (clamped 1..10)."""
        try:
            v = int(px_per_tick)
        except Exception:
            v = RSS_SCROLL_PX_PER_TICK
        self._px_per_tick = max(1, min(10, v))

    def setText(self, text: str):
        if not text:
            return
        self._text = text
        self._offset = 0
        self.update()

    def _tick(self):
        fm = QtGui.QFontMetrics(self.font())
        self._text_width = fm.horizontalAdvance(self._text)
        if self._text_width <= 0:
            return

        self._offset -= self._px_per_tick
        if self._offset < -self._text_width - 60:
            self._offset = self.width() + 60
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        painter.fillRect(self.rect(), QtGui.QColor("#0f1720"))

        pen = QtGui.QPen(QtGui.QColor(BORDER))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))

        painter.setClipRect(self.rect().adjusted(8, 0, -8, 0))

        painter.setPen(QtGui.QColor("#c7d1df"))
        x = self._offset
        y = int(self.height() * 0.75)
        painter.drawText(x, y, self._text)

        if self._text_width > 0 and x + self._text_width < self.width():
            painter.drawText(x + self._text_width + 80, y, self._text)


# =====================================================
# UI PANEL
# =====================================================
class KpLikePanel(QtWidgets.QFrame):
    def __init__(self, cfg: PanelConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.setObjectName("panel")
        self.setFrameShape(QtWidgets.QFrame.NoFrame)

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(6)

        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)

        self.lbl_title = QtWidgets.QLabel(cfg.title)
        self.lbl_title.setObjectName("panelTitle")

        self.lbl_big = QtWidgets.QLabel("--")
        self.lbl_big.setObjectName("panelBig")
        self.lbl_big.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.lbl_big.setTextFormat(QtCore.Qt.RichText)

        header.addWidget(self.lbl_title, 1)
        header.addWidget(self.lbl_big, 0)
        v.addLayout(header)

        axis = pg.graphicsItems.DateAxisItem.DateAxisItem(orientation="bottom")
        self.plot = pg.PlotWidget(axisItems={"bottom": axis})
        self.plot.setBackground(BG_PANEL)
        self.plot.showGrid(x=True, y=True, alpha=0.25)

        axL = self.plot.getPlotItem().getAxis("left")
        axB = self.plot.getPlotItem().getAxis("bottom")
        axL.setPen(pg.mkPen(FG_TEXT))
        axB.setPen(pg.mkPen(FG_TEXT))
        axL.setTextPen(pg.mkPen(FG_TEXT))
        axB.setTextPen(pg.mkPen(FG_TEXT))
        axB.setStyle(tickTextOffset=6)
        axL.setStyle(tickTextOffset=6)

        self.plot.getPlotItem().setContentsMargins(6, 6, 6, 6)
        v.addWidget(self.plot, 1)

        self._accent_color = ACCENT_GREEN          # currently applied (may toggle during blink)
        self._accent_target = ACCENT_GREEN         # desired state color
        self._accent_initialized = False  # avoid blink on first accent set
        self._blink_active = False
        self._blink_until = 0.0
        self._blink_state = False
        self._blink_timer = QtCore.QTimer(self)
        self._blink_timer.setInterval(500)  # blink ~2Hz
        self._blink_timer.timeout.connect(self._blink_tick)

        # Grace period: do not blink on early startup accent changes
        self._startup_ts = time.time()
        self._blink_grace_seconds = 6.0

        self._apply_accent(ACCENT_GREEN)
        self.set_data(np.zeros(2, dtype=float), float("nan"), x=np.array([time.time() - 60, time.time()], dtype=float))

    def _apply_accent(self, color: str):
        """Apply accent immediately (no blink logic)."""
        self._accent_color = color
        self.setStyleSheet(f"""
            QFrame#panel {{
                background: {BG_PANEL};
                border: 2px solid {color};
                border-radius: 6px;
            }}
        """)
        self.lbl_big.setStyleSheet(f"color: {color};")
        try:
            if getattr(self, "_plot_item", None) is not None:
                self._plot_item.setPen(pg.mkPen(color, width=2))
        except Exception:
            pass

    
    def _apply_border(self, color: str):
        """Apply only the panel border accent (keep big text + plot color unchanged)."""
        self.setStyleSheet(f"""
            QFrame#panel {{
                background: {BG_PANEL};
                border: 2px solid {color};
                border-radius: 6px;
            }}
        """)
    def start_blink(self, duration_seconds: int = 300, border_only: bool = False):
        """Blink the panel border/text to indicate a state/color change."""
        self._blink_active = True
        self._blink_border_only = bool(border_only)
        self._blink_until = time.time() + float(duration_seconds)
        self._blink_state = False
        if not self._blink_timer.isActive():
            self._blink_timer.start()

    def stop_blink(self):
        self._blink_active = False
        if self._blink_timer.isActive():
            self._blink_timer.stop()
        # ensure we end on the real target color
        self._apply_accent(self._accent_target)

    def _blink_tick(self):
        if not self._blink_active:
            return
        if time.time() >= self._blink_until:
            self.stop_blink()
            return
        self._blink_state = not self._blink_state
        # alternate between target accent and a bright highlight
        highlight = "#ffffff"
        if getattr(self, "_blink_border_only", False):
            self._apply_border(highlight if self._blink_state else self._accent_target)
        else:
            self._apply_accent(highlight if self._blink_state else self._accent_target)

    def set_accent(self, color: str, blink: bool = True):
        """
        Set the state accent. If the accent changed, start a 5-minute blink.
        """
        color = (str(color) if color is not None else "").strip() or ACCENT_GREY

        # First real accent application should NOT blink (startup / first refresh)
        if not getattr(self, '_accent_initialized', False):
            self._accent_initialized = True
            self._accent_target = color
            self._apply_accent(color)
            return

        # Also avoid blinking during a short startup grace window
        try:
            if (time.time() - float(getattr(self, '_startup_ts', 0.0))) < float(getattr(self, '_blink_grace_seconds', 0.0)):
                self._accent_target = color
                self._apply_accent(color)
                return
        except Exception:
            pass

        if not hasattr(self, "_accent_target"):
            # backward-safety if object loaded differently
            self._accent_target = color
            self._apply_accent(color)
            return

        if color == self._accent_target:
            return

        self._accent_target = color
        self._apply_accent(color)
        if blink:
            self.start_blink(300)

    def set_big_text(self, text: str):
        self.lbl_big.setText(text)

    def set_data(self, series: np.ndarray, big_value: float, y_range=None, color: str = None, x: np.ndarray | None = None):
        if color is None:
            color = self._accent_color

        if isinstance(big_value, float) and math.isnan(big_value):
            self.lbl_big.setText("--")
        else:
            txt = self.cfg.big_value_fmt.format(big_value)
            if self.cfg.unit:
                txt = f"{txt} {self.cfg.unit}"
            self.lbl_big.setText(txt)

        self.plot.clear()
        self._plot_item = None

        if series is None:
            series = np.array([], dtype=float)
        else:
            series = np.array(series, dtype=float)

        if x is None:
            x = np.arange(len(series), dtype=float)
        else:
            x = np.array(x, dtype=float)

        x, series = align_xy(x, series)

        if len(series) <= 0:
            now = time.time()
            x = np.array([now - 60, now], dtype=float)
            series = np.array([np.nan, np.nan], dtype=float)

        # --- draw line or bars ---
        if getattr(self.cfg, "mode", "line") == "bar":
            # width based on timestamp spacing (kp is often ~3h cadence)
            if len(x) > 1:
                dx = float(np.median(np.diff(x)))
                w = max(60.0, dx * 0.8)
            else:
                w = 60.0 * 60.0  # 1h fallback

            # One bar per point to allow per-bar color (KP severity coloring)
            for xi, yi in zip(x, series):
                if not np.isfinite(xi) or not np.isfinite(yi):
                    continue
                c = kp_color(float(yi))  # reuse existing mapping
                bar = pg.BarGraphItem(
                    x=[float(xi)],
                    height=[float(yi)],
                    width=w,
                    brush=pg.mkBrush(c),
                    pen=pg.mkPen(c),
                )
                self.plot.addItem(bar)
        else:
            self._plot_item = self.plot.plot(x, series, pen=pg.mkPen(color, width=2))

        if len(x):
            self.plot.setXRange(float(x[0]), float(x[-1]))

        if y_range is not None:
            self.plot.setYRange(y_range[0], y_range[1])
        else:
            yy = series[np.isfinite(series)]
            if yy.size:
                y_max = float(np.max(yy))
                y_min = float(np.min(yy))
                pad = (y_max - y_min) * 0.15 if y_max != y_min else 1.0
                self.plot.setYRange(y_min - pad, y_max + pad)
            else:
                self.plot.setYRange(0, 1)


# =====================================================
# SQLITE (CME HISTORY)
# =====================================================
def db_init(db_path: Path):
    con = sqlite3.connect(db_path)
    try:
        con.execute("""
            CREATE TABLE IF NOT EXISTS cme_history (
                ts_utc TEXT PRIMARY KEY,
                prob REAL NOT NULL,
                level TEXT NOT NULL
            );
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_cme_ts ON cme_history(ts_utc);")

        # Aurora local history (rolling window, persisted)
        con.execute("""
            CREATE TABLE IF NOT EXISTS aurora_history (
                ts_epoch REAL PRIMARY KEY,
                aur_max REAL,
                aur_mean REAL,
                aur_area REAL
            );
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_aur_ts ON aurora_history(ts_epoch);")

        con.commit()
    finally:
        con.close()


def db_insert_cme(db_path: Path, ts_utc: str, prob: float, level: str, keep_rows: int = 2000):
    con = sqlite3.connect(db_path)
    try:
        con.execute(
            "INSERT OR REPLACE INTO cme_history(ts_utc, prob, level) VALUES(?,?,?)",
            (ts_utc, float(prob), str(level)),
        )
        con.execute(f"""
            DELETE FROM cme_history
            WHERE ts_utc NOT IN (
                SELECT ts_utc FROM cme_history
                ORDER BY ts_utc DESC
                LIMIT {int(keep_rows)}
            );
        """)
        con.commit()
    finally:
        con.close()


def db_load_cme_series(db_path: Path, limit: int = 60) -> np.ndarray:
    con = sqlite3.connect(db_path)
    try:
        cur = con.execute(
            "SELECT prob FROM cme_history ORDER BY ts_utc DESC LIMIT ?",
            (int(limit),),
        )
        rows = cur.fetchall()
        if not rows:
            return np.array([], dtype=float)
        vals = [float(r[0]) for r in rows][::-1]
        return np.array(vals, dtype=float)
    finally:
        con.close()


def db_load_cme_last(db_path: Path):
    con = sqlite3.connect(db_path)
    try:
        cur = con.execute(
            "SELECT prob, level, ts_utc FROM cme_history ORDER BY ts_utc DESC LIMIT 1"
        )
        row = cur.fetchone()
        if not row:
            return None
        return float(row[0]), str(row[1]), str(row[2])
    finally:
        con.close()


# --- Aurora history helpers (persisted in SQLite) ---
def _nan_to_none(v):
    try:
        fv = float(v)
        return fv if math.isfinite(fv) else None
    except Exception:
        return None


def db_insert_aurora(
    db_path: Path,
    ts_epoch: float,
    aur_max: float,
    aur_mean: float,
    aur_area: float,
    keep_seconds: int = 26 * 3600,
):
    """Insert one aurora sample and prune old samples."""
    con = sqlite3.connect(db_path)
    try:
        con.execute(
            "INSERT OR REPLACE INTO aurora_history(ts_epoch, aur_max, aur_mean, aur_area) VALUES(?,?,?,?)",
            (float(ts_epoch), _nan_to_none(aur_max), _nan_to_none(aur_mean), _nan_to_none(aur_area)),
        )
        cutoff = float(ts_epoch) - float(keep_seconds)
        con.execute("DELETE FROM aurora_history WHERE ts_epoch < ?", (cutoff,))
        con.commit()
    finally:
        con.close()


def db_load_aurora_since(db_path: Path, since_epoch: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load aurora samples newer than since_epoch (ascending by time)."""
    con = sqlite3.connect(db_path)
    try:
        cur = con.execute(
            "SELECT ts_epoch, aur_max, aur_mean, aur_area FROM aurora_history WHERE ts_epoch >= ? ORDER BY ts_epoch ASC",
            (float(since_epoch),),
        )
        rows = cur.fetchall()
        if not rows:
            return (
                np.array([], dtype=float),
                np.array([], dtype=float),
                np.array([], dtype=float),
                np.array([], dtype=float),
            )

        x = np.array([float(r[0]) for r in rows], dtype=float)
        y_max = np.array([(float(r[1]) if r[1] is not None else float("nan")) for r in rows], dtype=float)
        y_mean = np.array([(float(r[2]) if r[2] is not None else float("nan")) for r in rows], dtype=float)
        y_area = np.array([(float(r[3]) if r[3] is not None else float("nan")) for r in rows], dtype=float)
        return x, y_max, y_mean, y_area
    finally:
        con.close()



# =====================================================
# SETTINGS DIALOG
# =====================================================
class SettingsDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent=None,
        rss_url: str = "",
        nasa_api_key: str = "",
        time_mode: str = "utc",
        rss_speed: int = RSS_SCROLL_PX_PER_TICK,
        dx_enabled: bool = False,
        dx_source: str = "dx",
        dx_host: str = "dxspider.co.uk",
        dx_port: int = 7300,
        dx_login: str = "",
        pota_zone: str = "worldwide",
    ):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.resize(760, 420)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        # --- Tabs ---
        tabs = QtWidgets.QTabWidget()
        tabs.setDocumentMode(True)
        tabs.setMovable(False)
        tabs.setStyleSheet("""
            QTabWidget::pane {
                border-left: 1px solid #2a3440;
                border-right: 1px solid #2a3440;
                border-bottom: 1px solid #2a3440;
                border-top: 0px;
                border-radius: 10px;
                background: #0e141a;
                top: -1px;
            }
            QTabBar::tab {
                background: #121a22;
                color: #d7dde6;
                border: 1px solid #2a3440;
                padding: 8px 14px;
                margin-right: 6px;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                font-weight: 800;
            }
            QTabBar::tab:selected {
                background: #0f1720;
                border-bottom: 1px solid #0f1720;
            }
        """)

        # ========== TAB 1: RSS ==========
        tab_rss = QtWidgets.QWidget()
        rss_l = QtWidgets.QVBoxLayout(tab_rss)
        rss_l.setContentsMargins(14, 14, 14, 14)
        rss_l.setSpacing(10)

        form_rss = QtWidgets.QFormLayout()
        form_rss.setHorizontalSpacing(10)
        form_rss.setVerticalSpacing(8)

        self.ed_rss = QtWidgets.QLineEdit(rss_url or "")
        self.ed_rss.setPlaceholderText(
            "Ex: https://www.nasa.gov/news-release/feed/  (RSS)  ou  https://services.swpc.noaa.gov/text/wwv.txt  (NOAA WWV)"
        )
        self.ed_rss.setClearButtonEnabled(True)
        form_rss.addRow("RSS feed URL:", self.ed_rss)

        # RSS shortcuts
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_use_nasa = QtWidgets.QPushButton("Use NASA RSS")
        self.btn_use_nasa.setToolTip("Remplit l'URL avec le feed RSS NASA (Solar System News)")
        self.btn_use_nasa.clicked.connect(lambda: self.ed_rss.setText(RSS_SOLAR_SYSTEM_URL))
        btn_row.addWidget(self.btn_use_nasa)

        self.btn_use_wwv = QtWidgets.QPushButton("Use NOAA Alert")
        self.btn_use_wwv.setToolTip("Remplit l'URL avec le bulletin NOAA WWV (texte brut)")
        self.btn_use_wwv.clicked.connect(lambda: self.ed_rss.setText(NOAA_WWV_URL))
        btn_row.addWidget(self.btn_use_wwv)

        btn_row.addStretch(1)
        form_rss.addRow("", btn_row)

        # RSS scroll speed presets
        SPEED_NORMAL = 2
        SPEED_FAST = 3
        SPEED_TURBO = 5

        self._rss_speed_value = int(rss_speed) if rss_speed is not None else SPEED_NORMAL
        if self._rss_speed_value not in (SPEED_NORMAL, SPEED_FAST, SPEED_TURBO):
            self._rss_speed_value = SPEED_NORMAL

        speed_row = QtWidgets.QHBoxLayout()
        speed_row.setContentsMargins(0, 0, 0, 0)
        speed_row.setSpacing(8)

        self.btn_speed_normal = QtWidgets.QPushButton("Normal")
        self.btn_speed_fast = QtWidgets.QPushButton("Fast")
        self.btn_speed_turbo = QtWidgets.QPushButton("Turbo")

        for b in (self.btn_speed_normal, self.btn_speed_fast, self.btn_speed_turbo):
            b.setCheckable(True)
            b.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
            b.setFixedHeight(28)
            b.setStyleSheet("""
                QPushButton {
                    background: #121a22;
                    color: #d7dde6;
                    border: 1px solid #2a3440;
                    border-radius: 6px;
                    padding: 4px 10px;
                    font-weight: 800;
                }
                QPushButton:hover { background: #16202a; }
            """)

        # checked variants (SUMO vibe)
        self.btn_speed_normal.setStyleSheet(self.btn_speed_normal.styleSheet() + """
            QPushButton:checked { background: #44d16e; color:#0b0f12; border:0px; }
        """)
        self.btn_speed_fast.setStyleSheet(self.btn_speed_fast.styleSheet() + """
            QPushButton:checked { background: #ff9f43; color:#0b0f12; border:0px; }
        """)
        self.btn_speed_turbo.setStyleSheet(self.btn_speed_turbo.styleSheet() + """
            QPushButton:checked { background: #9b59b6; color:#0b0f12; border:0px; }
        """)

        def set_speed(v: int):
            self._rss_speed_value = int(v)
            self.btn_speed_normal.setChecked(v == SPEED_NORMAL)
            self.btn_speed_fast.setChecked(v == SPEED_FAST)
            self.btn_speed_turbo.setChecked(v == SPEED_TURBO)

        self.btn_speed_normal.clicked.connect(lambda: set_speed(SPEED_NORMAL))
        self.btn_speed_fast.clicked.connect(lambda: set_speed(SPEED_FAST))
        self.btn_speed_turbo.clicked.connect(lambda: set_speed(SPEED_TURBO))

        speed_row.addWidget(self.btn_speed_normal)
        speed_row.addWidget(self.btn_speed_fast)
        speed_row.addWidget(self.btn_speed_turbo)
        speed_row.addStretch(1)
        form_rss.addRow("Scroll speed:", speed_row)

        set_speed(self._rss_speed_value)

        rss_l.addLayout(form_rss)
        rss_l.addStretch(1)

        # ========== TAB 2: API Keys ==========
        tab_api = QtWidgets.QWidget()
        api_l = QtWidgets.QVBoxLayout(tab_api)
        api_l.setContentsMargins(14, 14, 14, 14)
        api_l.setSpacing(10)

        form_api = QtWidgets.QFormLayout()
        form_api.setHorizontalSpacing(10)
        form_api.setVerticalSpacing(8)

        self.ed_nasa = QtWidgets.QLineEdit(nasa_api_key or "")
        self.ed_nasa.setPlaceholderText("NASA API key (optionnel, recommandé)")
        self.ed_nasa.setClearButtonEnabled(True)
        self.ed_nasa.setEchoMode(QtWidgets.QLineEdit.Password)
        form_api.addRow("NASA API key:", self.ed_nasa)

        api_l.addLayout(form_api)

        link_row = QtWidgets.QHBoxLayout()
        link_row.setContentsMargins(0, 0, 0, 0)
        link_row.setSpacing(8)

        self.btn_get_key = QtWidgets.QPushButton("Get NASA API key")
        self.btn_get_key.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btn_get_key.setFixedHeight(28)
        self.btn_get_key.clicked.connect(self._open_nasa_key_page)
        self.btn_get_key.setStyleSheet("""
            QPushButton {
                background: #121a22;
                color: #d7dde6;
                border: 1px solid #2a3440;
                border-radius: 6px;
                padding: 4px 10px;
                font-weight: 700;
            }
            QPushButton:hover { background: #16202a; }
        """)
        link_row.addStretch(1)
        api_l.addLayout(link_row)

        hint_api = QtWidgets.QLabel("• La clé NASA réduit fortement les erreurs 429 (rate limit) sur DONKI.")
        hint_api.setStyleSheet("color: #aab6c5; font-size: 11px;")
        hint_api.setWordWrap(True)
        api_l.addWidget(hint_api)
        api_l.addStretch(1)

        # ========== TAB 3: Clock ==========
        tab_time = QtWidgets.QWidget()
        time_l = QtWidgets.QVBoxLayout(tab_time)
        time_l.setContentsMargins(14, 14, 14, 14)
        time_l.setSpacing(10)

        form_time = QtWidgets.QFormLayout()
        form_time.setHorizontalSpacing(10)
        form_time.setVerticalSpacing(8)

        self.cb_time_mode = QtWidgets.QComboBox()
        self.cb_time_mode.addItem("UTC", "utc")
        self.cb_time_mode.addItem("Local", "local")
        tm = (time_mode or "utc").strip().lower()
        self.cb_time_mode.setCurrentIndex(0 if tm != "local" else 1)

        form_time.addRow("Clock:", self.cb_time_mode)
        time_l.addLayout(form_time)

        hint_time = QtWidgets.QLabel("• Change l’affichage de l’horloge (UTC / heure locale).")
        hint_time.setStyleSheet("color: #aab6c5; font-size: 11px;")
        hint_time.setWordWrap(True)
        time_l.addWidget(hint_time)
        time_l.addStretch(1)

        # ========== TAB 4: DX Cluster ==========
        tab_dx = QtWidgets.QWidget()
        dx_l = QtWidgets.QVBoxLayout(tab_dx)
        dx_l.setContentsMargins(14, 14, 14, 14)
        dx_l.setSpacing(10)

        form_dx = QtWidgets.QFormLayout()
        form_dx.setHorizontalSpacing(10)
        form_dx.setVerticalSpacing(8)

        self.cb_dx_enabled = QtWidgets.QCheckBox("Enable DX Cluster panel")
        self.cb_dx_enabled.setChecked(bool(dx_enabled))
        self.cb_dx_enabled.setToolTip(
            "Adds a left-side DX Cluster column (telnet). Can also be toggled from main UI button."
        )
        form_dx.addRow("DX Cluster:", self.cb_dx_enabled)

        self.cb_dx_source = QtWidgets.QComboBox()
        self.cb_dx_source.addItem("DX Spider (telnet)", "dx")
        self.cb_dx_source.addItem("POTA spots (api.pota.app)", "pota")
        ds = (dx_source or "dx").strip().lower()
        if ds not in ("dx", "pota"):
            ds = "dx"
        self.cb_dx_source.setCurrentIndex(0 if ds == "dx" else 1)
        self.cb_dx_source.setToolTip("Choose the content of the left DX column.")
        form_dx.addRow("DX column source:", self.cb_dx_source)

        dx_row = QtWidgets.QHBoxLayout()
        dx_row.setContentsMargins(0, 0, 0, 0)
        dx_row.setSpacing(8)

        self.ed_dx_host = QtWidgets.QLineEdit(dx_host or "dxspider.co.uk")
        self.ed_dx_host.setPlaceholderText("Host (ex: dxspider.co.uk)")
        self.ed_dx_host.setClearButtonEnabled(True)
        dx_row.addWidget(self.ed_dx_host, 1)

        self.ed_dx_port = QtWidgets.QSpinBox()
        self.ed_dx_port.setRange(1, 65535)
        self.ed_dx_port.setValue(int(dx_port) if dx_port else 7300)
        self.ed_dx_port.setFixedWidth(110)
        dx_row.addWidget(self.ed_dx_port, 0)

        form_dx.addRow("DX host:port:", dx_row)

        self.ed_dx_login = QtWidgets.QLineEdit((dx_login or "").strip())
        self.ed_dx_login.setPlaceholderText("Your callsign (ex: F4IGV)")
        self.ed_dx_login.setClearButtonEnabled(True)
        self.ed_dx_login.setToolTip(
            "DXSpider requires a login callsign right after connect. SUMO will send this callsign automatically."
        )
        form_dx.addRow("DX login:", self.ed_dx_login)


        # POTA zone filter
        self.cb_pota_zone = QtWidgets.QComboBox()
        self.cb_pota_zone.addItem("Worldwide", "worldwide")
        self.cb_pota_zone.addItem("USA", "usa")
        self.cb_pota_zone.addItem("Europe", "europe")
        pz = (pota_zone or "worldwide").strip().lower()
        if pz not in ("worldwide", "usa", "europe"):
            pz = "worldwide"
        # set current
        for i in range(self.cb_pota_zone.count()):
            if str(self.cb_pota_zone.itemData(i)) == pz:
                self.cb_pota_zone.setCurrentIndex(i)
                break
        self.cb_pota_zone.setToolTip("Filter POTA spots by region (applies only when DX source is POTA).")
        form_dx.addRow("POTA zone:", self.cb_pota_zone)

        def _sync_pota_zone_enabled():
            try:
                srcv = str(self.cb_dx_source.currentData() or "dx").strip().lower()
                self.cb_pota_zone.setEnabled(srcv == "pota")
            except Exception:
                self.cb_pota_zone.setEnabled(False)

        self.cb_dx_source.currentIndexChanged.connect(_sync_pota_zone_enabled)
        _sync_pota_zone_enabled()

        dx_l.addLayout(form_dx)

        hint_dx = QtWidgets.QLabel("• Le contenu de la colonne DX peut être DXSpider (telnet) ou POTA (HTTP).")
        hint_dx.setStyleSheet("color: #aab6c5; font-size: 11px;")
        hint_dx.setWordWrap(True)
        dx_l.addWidget(hint_dx)
        dx_l.addStretch(1)

        # Add tabs
        tabs.addTab(tab_rss, "RSS")
        tabs.addTab(tab_api, "API Keys")
        tabs.addTab(tab_time, "Clock")
        tabs.addTab(tab_dx, "DX Cluster")

        root.addWidget(tabs, 1)

        # --- Save/Cancel ---
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns, 0)

        # --- Bottom status bar ---
        self.sb = QtWidgets.QStatusBar()
        self.sb.setSizeGripEnabled(False)
        self.sb.setFixedHeight(22)
        self.sb.setStyleSheet("""
            QStatusBar {
                background: #0f1720;
                border-top: 1px solid #2a3440;
                padding-left: 8px;
                padding-right: 8px;
            }
        """)
        root.addWidget(self.sb, 0)

    # ===== getters (same API as before) =====
    def rss_url(self) -> str:
        return self.ed_rss.text().strip()

    def nasa_api_key(self) -> str:
        return self.ed_nasa.text().strip()

    def time_mode(self) -> str:
        data = self.cb_time_mode.currentData()
        return (str(data) if data else "utc").strip().lower()

    def rss_speed(self) -> int:
        return int(getattr(self, "_rss_speed_value", RSS_SCROLL_PX_PER_TICK))

    def dx_enabled(self) -> bool:
        return bool(self.cb_dx_enabled.isChecked())

    def dx_source(self) -> str:
        try:
            data = self.cb_dx_source.currentData()
            return (str(data) if data else "dx").strip().lower()
        except Exception:
            return "dx"

    def dx_host(self) -> str:
        return self.ed_dx_host.text().strip() or "dxspider.co.uk"

    def dx_port(self) -> int:
        try:
            return int(self.ed_dx_port.value())
        except Exception:
            return 7300

    def dx_login(self) -> str:
        return self.ed_dx_login.text().strip()

    def pota_zone(self) -> str:
        try:
            data = self.cb_pota_zone.currentData() if hasattr(self, "cb_pota_zone") else "worldwide"
            v = (str(data) if data else "worldwide").strip().lower()
            return v if v in ("worldwide", "usa", "europe") else "worldwide"
        except Exception:
            return "worldwide"

    @QtCore.Slot()
    def _open_nasa_key_page(self):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(NASA_API_KEY_REQUEST_URL))

class AboutDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About SUMO")
        self.setModal(True)
        self.resize(720, 420)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        # Header
        head = QtWidgets.QHBoxLayout()
        head.setSpacing(12)

        logo = QtWidgets.QLabel()
        logo.setFixedSize(64, 64)
        logo.setScaledContents(True)
        pm = QtGui.QPixmap(str(ASSETS_DIR / "logo.png"))
        if pm.isNull():
            pm = QtGui.QPixmap(64, 64)
            pm.fill(QtGui.QColor("#333"))
        logo.setPixmap(pm)
        head.addWidget(logo, 0)

        title_col = QtWidgets.QVBoxLayout()
        title_col.setSpacing(2)

        lbl_name = QtWidgets.QLabel("SUMO — Sun Monitor")
        lbl_name.setStyleSheet("font-size: 22px; font-weight: 900; color: #d7dde6;")
        lbl_ver = QtWidgets.QLabel(f"Version: {APP_VERSION}")
        lbl_ver.setStyleSheet("font-size: 12px; color: #aab6c5;")
        lbl_by = QtWidgets.QLabel("By F4IGV and F4FAP")
        lbl_by.setStyleSheet("font-size: 12px; color: #aab6c5;")

        title_col.addWidget(lbl_name)
        title_col.addWidget(lbl_ver)
        title_col.addWidget(lbl_by)
        head.addLayout(title_col, 1)

        root.addLayout(head)

        # Body text
        info = QtWidgets.QTextBrowser()
        info.setOpenExternalLinks(True)
        info.setReadOnly(True)
        info.setStyleSheet("""
            QTextBrowser {
                background: #0e141a;
                color: #d7dde6;
                border: 1px solid #2a3440;
                border-radius: 8px;
                padding: 10px;
                font-size: 12px;
            }
            a { color: #4aa3ff; text-decoration: none; font-weight: 700; }
            a:hover { text-decoration: underline; }
        """)
        info.setHtml(f"""
            <p><b>SUMO</b> is a solar-weather monitoring dashboard (NOAA + NASA DONKI) designed for radio amateurs.</p>

            <p style="margin-top:10px;"><b>License</b></p>
            <p>
              This program is distributed under the <b>GNU General Public License v3.0 (GPLv3)</b>.
              You may use, study, share and modify it under the terms of the GPLv3.
            </p>
            <p>Full license text is available via the <b>“Open GPLv3”</b> button below.</p>

            <p style="margin-top:10px;"><b>Support the project</b></p>
            <p>
              If you like SUMO, you can support development using the button just below.
            </p>
        """)

        root.addWidget(info, 1)

        # --- Section buttons (aligned with their content) ---
        # License button
        lic_row = QtWidgets.QHBoxLayout()
        lic_row.setContentsMargins(0, 0, 0, 0)
        lic_row.setSpacing(8)

        btn_gpl = QtWidgets.QPushButton("Open GPLv3")
        btn_gpl.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        btn_gpl.setFixedHeight(28)
        btn_gpl.clicked.connect(lambda: self._open_url(GPL3_LICENSE_URL))
        btn_gpl.setStyleSheet("""
            QPushButton {
                background: #121a22;
                color: #d7dde6;
                border: 1px solid #2a3440;
                border-radius: 6px;
                padding: 4px 10px;
                font-weight: 800;
            }
            QPushButton:hover { background: #16202a; }
        """)

        lic_row.addWidget(btn_gpl, 0)
        lic_row.addStretch(1)
        root.addLayout(lic_row)

        # Support button
        sup_row = QtWidgets.QHBoxLayout()
        sup_row.setContentsMargins(0, 0, 0, 0)
        sup_row.setSpacing(8)

        btn_support = QtWidgets.QPushButton("Support SUMO ❤️")
        btn_support.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        btn_support.setFixedHeight(28)
        btn_support.clicked.connect(lambda: self._open_url(PAYPAL_DONATE_URL))
        btn_support.setStyleSheet("""
            QPushButton {
                background: #121a22;
                color: #d7dde6;
                border: 1px solid #2a3440;
                border-radius: 6px;
                padding: 4px 10px;
                font-weight: 800;
            }
            QPushButton:hover { background: #16202a; }
        """)

        sup_row.addWidget(btn_support, 0)
        sup_row.addStretch(1)
        root.addLayout(sup_row)

        # Close button (bottom right)
        close_row = QtWidgets.QHBoxLayout()
        close_row.addStretch(1)

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        btn_close.setFixedHeight(28)
        btn_close.clicked.connect(self.accept)
        btn_close.setStyleSheet("""
            QPushButton {
                background: #121a22;
                color: #d7dde6;
                border: 1px solid #2a3440;
                border-radius: 6px;
                padding: 4px 10px;
                font-weight: 800;
            }
            QPushButton:hover { background: #16202a; }
        """)

        close_row.addWidget(btn_close)
        root.addLayout(close_row)

    def _open_url(self, url: str):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(url))


# =====================================================
# WORKER (NOAA + DONKI)
# =====================================================
class DataWorker(QtCore.QObject):
    data_ready = QtCore.Signal(dict)
    error = QtCore.Signal(str)

    def __init__(self, refresh_seconds: int = 60, nasa_api_key: str = ""):
        super().__init__()
        self.refresh_seconds = refresh_seconds
        self.nasa_api_key = (nasa_api_key or "").strip()

        self._stop = False
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "SUMO-SunMonitor/0.1"})

        self._last_donki_fetch = 0.0
        self._donki_interval = 60 * 60
        self._donki_backoff_429 = 2 * 60 * 60

        self._last_good: dict[str, object] = {}

    def stop(self):
        self._stop = True

    def _get_json(self, url: str, retries: int = 3, retry_delay: float = 0.8):
        last_err: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                r = self._session.get(url, timeout=20)
                r.raise_for_status()
                return r.json()
            except (json.JSONDecodeError, ValueError) as e:
                last_err = e
            except requests.RequestException as e:
                last_err = e
            if attempt < retries:
                time.sleep(retry_delay)
        raise RuntimeError(f"Fetch/JSON failed after {retries} tries: {url} ({last_err})")

    def _get_text(self, url: str, retries: int = 3, retry_delay: float = 0.8) -> str:
        last_err: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                r = self._session.get(url, timeout=20)
                r.raise_for_status()
                return r.text
            except requests.RequestException as e:
                last_err = e
            if attempt < retries:
                time.sleep(retry_delay)
        raise RuntimeError(f"Fetch text failed after {retries} tries: {url} ({last_err})")

    @staticmethod
    def _noaa_rows(data):
        if not isinstance(data, list) or len(data) < 2:
            return []
        return data[1:]

    def _parse_kp(self, data):
        t, vals = [], []
        for row in self._noaa_rows(data):
            try:
                t_epoch = parse_time_to_epoch(row[0])
                v = float(row[1])
                if math.isnan(t_epoch):
                    continue
                t.append(t_epoch)
                vals.append(v)
            except Exception:
                continue
        return t, vals

    def _parse_sw_mag(self, data):
        """
        NOAA mag-6-hour.json: first row is header.
        Typical header: ["time_tag","bx_gsm","by_gsm","bz_gsm","lon_gsm","lat_gsm","bt"]
        We map by name to avoid wrong indexes (Bt accidentally reading lon/phi).
        """
        if not isinstance(data, list) or len(data) < 2:
            return [], [], []

        header = data[0]
        rows = data[1:]

        idx = {}
        if isinstance(header, list):
            for i, name in enumerate(header):
                try:
                    idx[str(name).strip().lower()] = i
                except Exception:
                    pass

        def col(*names, default=None):
            for n in names:
                k = str(n).strip().lower()
                if k in idx:
                    return idx[k]
            return default

        i_time = col("time_tag", "time", "timestamp", default=0)
        i_bx = col("bx_gsm", "bx", default=None)
        i_by = col("by_gsm", "by", default=None)
        i_bz = col("bz_gsm", "bz", default=None)
        i_bt = col("bt", "bt_gsm", default=None)

        t, bz_vals, bt_vals = [], [], []

        for row in rows:
            if not isinstance(row, list):
                continue
            try:
                t_epoch = parse_time_to_epoch(row[i_time])
                if math.isnan(t_epoch):
                    continue

                bz = float(row[i_bz]) if i_bz is not None else float("nan")

                if i_bt is not None:
                    bt = float(row[i_bt])
                else:
                    if i_bx is None or i_by is None or i_bz is None:
                        bt = float("nan")
                    else:
                        bx = float(row[i_bx])
                        by = float(row[i_by])
                        bt = math.sqrt(bx * bx + by * by + bz * bz)

                t.append(t_epoch)
                bz_vals.append(bz)
                bt_vals.append(bt)
            except Exception:
                continue

        return t, bz_vals, bt_vals

    def _parse_sw_speed(self, data):
        t, spd_vals = [], []
        for row in self._noaa_rows(data):
            try:
                t_epoch = parse_time_to_epoch(row[0])
                spd = float(row[2])
                if math.isnan(t_epoch):
                    continue
                t.append(t_epoch)
                spd_vals.append(spd)
            except Exception:
                continue
        return t, spd_vals

    @staticmethod
    def _parse_daily_solar_indices(text: str):
        toks = text.replace("\n", " ").split()
        out_t, out_sfi, out_ssn = [], [], []

        def is_int(s: str) -> bool:
            try:
                int(s)
                return True
            except Exception:
                return False

        i = 0
        while i + 5 < len(toks):
            if (
                len(toks[i]) == 4 and is_int(toks[i]) and
                is_int(toks[i + 1]) and is_int(toks[i + 2]) and
                is_int(toks[i + 3]) and is_int(toks[i + 4])
            ):
                y = int(toks[i])
                m = int(toks[i + 1])
                d = int(toks[i + 2])
                if 1970 <= y <= 2100 and 1 <= m <= 12 and 1 <= d <= 31:
                    try:
                        sfi = float(toks[i + 3])
                        ssn = float(toks[i + 4])
                        dt = datetime(y, m, d, 0, 0, 0, tzinfo=timezone.utc)
                        out_t.append(_to_epoch_seconds(dt))
                        out_sfi.append(sfi)
                        out_ssn.append(ssn)
                        i += 6
                        continue
                    except Exception:
                        pass
            i += 1

        return out_t, out_sfi, out_ssn

    def _ovation_global_activity(self, ovation_json: dict) -> dict:
        coords = ovation_json.get("coordinates")
        if not isinstance(coords, list) or not coords:
            return {"max": float("nan"), "mean_active": float("nan"), "area_gt10": float("nan")}

        vals = []
        for p in coords:
            if not isinstance(p, (list, tuple)) or len(p) < 3:
                continue
            try:
                vals.append(float(p[2]))
            except Exception:
                continue

        if not vals:
            return {"max": float("nan"), "mean_active": float("nan"), "area_gt10": float("nan")}

        arr = np.array(vals, dtype=float)
        maxv = float(np.nanmax(arr))
        mean_active = float(np.nanmean(arr[arr > 0])) if np.any(arr > 0) else 0.0
        area_gt10 = float(np.sum(arr > 10) / arr.size * 100.0)

        return {
            "max": max(0.0, min(100.0, maxv)) if not math.isnan(maxv) else float("nan"),
            "mean_active": max(0.0, min(100.0, mean_active)) if not math.isnan(mean_active) else float("nan"),
            "area_gt10": max(0.0, min(100.0, area_gt10)) if not math.isnan(area_gt10) else float("nan"),
        }

    def _donki_cme_probability_strict(self) -> tuple[float, str]:
        end = date.today()
        start = end - timedelta(days=7)

        api_key = self.nasa_api_key or "DEMO_KEY"
        url = (
            f"{DONKI_BASE}/CME"
            f"?startDate={start.isoformat()}"
            f"&endDate={end.isoformat()}"
            f"&api_key={api_key}"
        )

        try:
            events = self._get_json(url)
        except requests.HTTPError as e:
            resp = getattr(e, "response", None)
            if resp is not None and resp.status_code == 429:
                self._last_donki_fetch = time.time()
                self._donki_interval = self._donki_backoff_429
                raise RuntimeError("DONKI rate limit (429) - retry later") from e
            raise

        if not isinstance(events, list) or not events:
            return float("nan"), "N/A"


        def extract_event_epoch(ev: dict) -> float:
            """
            DONKI CME events can contain several timestamp fields depending on the event.
            We try a few keys safely. Returns NaN if not found.
            """
            for k in ("startTime", "time21_5", "activityID", "submissionTime"):
                v = ev.get(k)
                if not v:
                    continue
                # activityID can look like "2026-02-08T12:34:00-CME-001"
                s = str(v).split("-CME-")[0].strip()
                t = parse_time_to_epoch(s)
                if not (isinstance(t, float) and math.isnan(t)):
                    return float(t)
            return float("nan")


        def age_factor_from_epoch(t_event: float, now_epoch: float) -> float:
            """
            Age weighting: 1.0 for very recent, decreasing to 0.4 after ~5 days.
            Prevents old events from dominating too long.
            """
            if t_event is None or (isinstance(t_event, float) and math.isnan(t_event)):
                return 1.0
            age_hours = max(0.0, (now_epoch - float(t_event)) / 3600.0)
            # 0h => 1.0 ; 120h (~5 days) => 0.4 ; beyond => stays 0.4
            return max(0.4, 1.0 - (age_hours / 120.0) * 0.6)

        def score_event(ev: dict) -> float:
            now_epoch = time.time()

            note_l = (ev.get("note") or "").lower()

            analyses = ev.get("cmeAnalyses") or []
            has_enlil = False
            has_earth_impact = False
            has_earth_mention = ("earth" in note_l)

            # Global text for lightweight "likely/may" nuance (no heavy NLP)
            global_txt = note_l

            for a in analyses:
                if not isinstance(a, dict):
                    continue

                enlil_list = a.get("enlilList") or []
                if isinstance(enlil_list, list) and len(enlil_list) > 0:
                    has_enlil = True

                impact_list = a.get("impactList") or []
                if isinstance(impact_list, list):
                    for imp in impact_list:
                        if isinstance(imp, dict):
                            txt = " ".join(str(v).lower() for v in imp.values() if v is not None)
                            global_txt += " " + txt
                            if "earth" in txt:
                                has_earth_impact = True

                txt = " ".join(
                    str(a.get(k, "")).lower()
                    for k in ("type", "analysisLevel", "note")
                    if a.get(k) is not None
                )
                global_txt += " " + txt
                if "earth" in txt:
                    has_earth_mention = True

            # --- base heuristic score (conservative) ---
            score = 0.0
            if has_earth_mention:
                score = max(score, 30.0)

            if analyses:
                score = max(score, 60.0 if has_earth_mention else 20.0)

            if has_enlil and has_earth_impact:
                score = max(score, 85.0)
            elif has_enlil and has_earth_mention:
                score = max(score, 70.0)

            # --- wording nuance: "likely" > "may" (small bonus, never huge) ---
            if "likely" in global_txt:
                score += 8.0
            elif "may" in global_txt:
                score += 4.0

            # --- recency weighting ---
            t_event = extract_event_epoch(ev)
            score *= age_factor_from_epoch(t_event, now_epoch)

            # --- hard cap ---
            return min(score, 95.0)

        best = 0.0
        for ev in events[-40:]:
            if isinstance(ev, dict):
                best = max(best, score_event(ev))

        if best <= 0:
            return 0.0, "NONE"
        if best < 40:
            return best, "LOW"
        if best < 70:
            return best, "MED"
        return best, "HIGH"

    def run(self):
        while not self._stop:
            try:
                payload: dict = {"partial_error": ""}

                # KP
                try:
                    kp_data = self._get_json(NOAA_URLS["kp"])
                    kp_t, kp_vals = self._parse_kp(kp_data)
                    payload["kp_x"] = np.array(kp_t, dtype=float)
                    payload["kp_series"] = np.array(kp_vals, dtype=float)
                    payload["kp_now"] = float(kp_vals[-1]) if kp_vals else float("nan")

                    self._last_good["kp_x"] = payload["kp_x"]
                    self._last_good["kp_series"] = payload["kp_series"]
                    self._last_good["kp_now"] = payload["kp_now"]
                except Exception as e:
                    payload["kp_x"] = self._last_good.get("kp_x", np.array([], dtype=float))
                    payload["kp_series"] = self._last_good.get("kp_series", np.array([], dtype=float))
                    payload["kp_now"] = self._last_good.get("kp_now", float("nan"))
                    payload["partial_error"] += f" KP:{e}"

                # MAG (BZ/BT) - 6h JSON
                try:
                    mag_data = self._get_json(NOAA_URLS["sw_mag_6h"])
                    t, bz_vals, bt_vals = self._parse_sw_mag(mag_data)
                    payload["mag_x"] = np.array(t, dtype=float)
                    payload["bz_series"] = np.array(bz_vals, dtype=float)
                    payload["bt_series"] = np.array(bt_vals, dtype=float)
                    payload["bz_now"] = float(bz_vals[-1]) if bz_vals else float("nan")
                    payload["bt_now"] = float(bt_vals[-1]) if bt_vals else float("nan")

                    self._last_good["mag_x"] = payload["mag_x"]
                    self._last_good["bz_series"] = payload["bz_series"]
                    self._last_good["bt_series"] = payload["bt_series"]
                    self._last_good["bz_now"] = payload["bz_now"]
                    self._last_good["bt_now"] = payload["bt_now"]
                except Exception as e:
                    payload["mag_x"] = self._last_good.get("mag_x", np.array([], dtype=float))
                    payload["bz_series"] = self._last_good.get("bz_series", np.array([], dtype=float))
                    payload["bt_series"] = self._last_good.get("bt_series", np.array([], dtype=float))
                    payload["bz_now"] = self._last_good.get("bz_now", float("nan"))
                    payload["bt_now"] = self._last_good.get("bt_now", float("nan"))
                    payload["partial_error"] += f" MAG:{e}"

                # SW speed - 6h JSON
                try:
                    plasma_data = self._get_json(NOAA_URLS["sw_plasma_6h"])
                    t, spd_vals = self._parse_sw_speed(plasma_data)
                    payload["sw_speed_x"] = np.array(t, dtype=float)
                    payload["sw_speed_series"] = np.array(spd_vals, dtype=float)
                    payload["sw_speed_now"] = float(spd_vals[-1]) if spd_vals else float("nan")

                    self._last_good["sw_speed_x"] = payload["sw_speed_x"]
                    self._last_good["sw_speed_series"] = payload["sw_speed_series"]
                    self._last_good["sw_speed_now"] = payload["sw_speed_now"]
                except Exception as e:
                    payload["sw_speed_x"] = self._last_good.get("sw_speed_x", np.array([], dtype=float))
                    payload["sw_speed_series"] = self._last_good.get("sw_speed_series", np.array([], dtype=float))
                    payload["sw_speed_now"] = self._last_good.get("sw_speed_now", float("nan"))
                    payload["partial_error"] += f" SW:{e}"

                # X-RAY - 7 days JSON
                try:
                    xray = self._get_json(NOAA_URLS["goes_xray_7d"])
                    t_list, xr_vals = [], []
                    if isinstance(xray, list):
                        for x in xray:
                            if (
                                isinstance(x, dict)
                                and x.get("energy") == "0.1-0.8nm"
                                and x.get("flux") is not None
                                and x.get("time_tag") is not None
                            ):
                                try:
                                    t_epoch = parse_time_to_epoch(x.get("time_tag"))
                                    if math.isnan(t_epoch):
                                        continue
                                    t_list.append(t_epoch)
                                    xr_vals.append(float(x["flux"]))
                                except Exception:
                                    pass

                    payload["xray_x"] = np.array(t_list, dtype=float)
                    payload["xray_series"] = np.array(xr_vals, dtype=float)
                    payload["xray_now"] = float(xr_vals[-1]) if xr_vals else float("nan")
                    cls, mag = xray_flare_class(payload["xray_now"])
                    payload["xray_class"] = cls
                    payload["xray_mag"] = mag
                    payload["xray_label"] = xray_class_label(cls, mag)

                    self._last_good["xray_x"] = payload["xray_x"]
                    self._last_good["xray_series"] = payload["xray_series"]
                    self._last_good["xray_now"] = payload["xray_now"]
                    self._last_good["xray_class"] = payload["xray_class"]
                    self._last_good["xray_mag"] = payload["xray_mag"]
                    self._last_good["xray_label"] = payload["xray_label"]
                except Exception as e:
                    payload["xray_x"] = self._last_good.get("xray_x", np.array([], dtype=float))
                    payload["xray_series"] = self._last_good.get("xray_series", np.array([], dtype=float))
                    payload["xray_now"] = self._last_good.get("xray_now", float("nan"))
                    payload["xray_class"] = self._last_good.get("xray_class", "?")
                    payload["xray_mag"] = self._last_good.get("xray_mag", float("nan"))
                    payload["xray_label"] = self._last_good.get("xray_label", "?")
                    payload["partial_error"] += f" XRAY:{e}"

                # P10 - 7 days JSON
                try:
                    prot = self._get_json(NOAA_URLS["goes_protons_7d"])
                    t_list, p10_vals = [], []
                    if isinstance(prot, list):
                        for x in prot:
                            if not isinstance(x, dict):
                                continue
                            en = (x.get("energy") or "").strip()
                            if "10" in en and x.get("flux") is not None and x.get("time_tag") is not None:
                                try:
                                    t_epoch = parse_time_to_epoch(x.get("time_tag"))
                                    if math.isnan(t_epoch):
                                        continue
                                    t_list.append(t_epoch)
                                    p10_vals.append(float(x["flux"]))
                                except Exception:
                                    pass

                    payload["p10_x"] = np.array(t_list, dtype=float)
                    payload["p10_series"] = np.array(p10_vals, dtype=float)
                    payload["p10_now"] = float(p10_vals[-1]) if p10_vals else float("nan")
                    payload["s_level"] = proton_s_scale(payload["p10_now"])

                    self._last_good["p10_x"] = payload["p10_x"]
                    self._last_good["p10_series"] = payload["p10_series"]
                    self._last_good["p10_now"] = payload["p10_now"]
                    self._last_good["s_level"] = payload["s_level"]
                except Exception as e:
                    payload["p10_x"] = self._last_good.get("p10_x", np.array([], dtype=float))
                    payload["p10_series"] = self._last_good.get("p10_series", np.array([], dtype=float))
                    payload["p10_now"] = self._last_good.get("p10_now", float("nan"))
                    payload["s_level"] = self._last_good.get("s_level", "S?")
                    payload["partial_error"] += f" P10:{e}"

                # SFI/SSN (text)
                try:
                    dsi_txt = self._get_text(NOAA_URLS["daily_solar_indices"])
                    t_dsi, sfi_vals, ssn_vals = self._parse_daily_solar_indices(dsi_txt)
                    payload["dsi_x"] = np.array(t_dsi, dtype=float)
                    payload["sfi_series"] = np.array(sfi_vals, dtype=float)
                    payload["ssn_series"] = np.array(ssn_vals, dtype=float)
                    payload["sfi_now"] = float(sfi_vals[-1]) if sfi_vals else float("nan")
                    payload["ssn_now"] = float(ssn_vals[-1]) if ssn_vals else float("nan")

                    self._last_good["dsi_x"] = payload["dsi_x"]
                    self._last_good["sfi_series"] = payload["sfi_series"]
                    self._last_good["ssn_series"] = payload["ssn_series"]
                    self._last_good["sfi_now"] = payload["sfi_now"]
                    self._last_good["ssn_now"] = payload["ssn_now"]
                except Exception as e:
                    payload["dsi_x"] = self._last_good.get("dsi_x", np.array([], dtype=float))
                    payload["sfi_series"] = self._last_good.get("sfi_series", np.array([], dtype=float))
                    payload["ssn_series"] = self._last_good.get("ssn_series", np.array([], dtype=float))
                    payload["sfi_now"] = self._last_good.get("sfi_now", float("nan"))
                    payload["ssn_now"] = self._last_good.get("ssn_now", float("nan"))
                    payload["partial_error"] += f" DSI:{e}"

                # AURORA (latest snapshot)
                try:
                    ov = self._get_json(NOAA_URLS["ovation_aurora_latest"])
                    aur = self._ovation_global_activity(ov)
                    payload["aur_now"] = aur["max"]
                    payload["aur_mean_active"] = aur["mean_active"]
                    payload["aur_area_gt10"] = aur["area_gt10"]
                    payload["aur_ts"] = time.time()

                    self._last_good["aur_now"] = payload["aur_now"]
                    self._last_good["aur_mean_active"] = payload["aur_mean_active"]
                    self._last_good["aur_area_gt10"] = payload["aur_area_gt10"]
                    self._last_good["aur_ts"] = payload["aur_ts"]
                except Exception as e:
                    payload["aur_now"] = self._last_good.get("aur_now", float("nan"))
                    payload["aur_mean_active"] = self._last_good.get("aur_mean_active", float("nan"))
                    payload["aur_area_gt10"] = self._last_good.get("aur_area_gt10", float("nan"))
                    payload["aur_ts"] = self._last_good.get("aur_ts", time.time())
                    payload["partial_error"] += f" AUR:{e}"

                # DONKI CME (cache DB)
                now = time.time()
                if now - self._last_donki_fetch > self._donki_interval:
                    try:
                        prob, level = self._donki_cme_probability_strict()
                        payload["cme_prob"] = prob
                        payload["cme_level"] = level

                        ts_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                        if not (isinstance(prob, float) and math.isnan(prob)):
                            db_insert_cme(DB_PATH, ts_utc, prob, level)
                        self._last_donki_fetch = now
                    except Exception as e:
                        payload["partial_error"] += f" CME:{e}"

                self.data_ready.emit(payload)

            except Exception as e:
                self.error.emit(str(e))

            for _ in range(self.refresh_seconds):
                if self._stop:
                    break
                time.sleep(1)



# =====================================================
# EASTER EGG - MINI GAME (BLACK & WHITE)
# =====================================================
class SumoEasterInvaders(QtWidgets.QDialog):
    """
    Mini jeu rétro noir & blanc (Space Invader-like)
    - Vaisseau = logo SUMO (assets/logo.png) rendu en blanc
    - Soleil en haut, émet des CME (cercles) qui descendent
    - Laser (trait) avec SPACE
    Controls: ←/→ (ou A/D), SPACE, ESC
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SUMO Defense (B/W)")
        self.setModal(True)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        # Fenêtre simple (pas fullscreen forcé) pour rester discret; tu peux l'agrandir si tu veux
        self.setFixedSize(900, 600)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        # --- Game state ---
        self._w = self.width()
        self._h = self.height()

        self._rng = np.random.default_rng()

        self._ship_speed = 7
        self._ship_x = self._w // 2
        self._ship_y = self._h - 70
        self._left = False
        self._right = False

        self._shots = []   # list[dict(x,y)]
        self._cmes = []    # list[dict(x,y,vy,r)]
        self._score = 0
        self._lives = 3
        self._cooldown = 0

        self._spawn_every = 110  # frames (encore plus lent)
        self._frame = 0

        # --- Load logo and make it white silhouette ---
        pm = QtGui.QPixmap(str(ASSETS_DIR / "logo.png"))
        if pm.isNull():
            pm = QtGui.QPixmap(64, 64)
            pm.fill(QtGui.QColor("white"))

        pm = pm.scaled(64, 64, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self._ship_pix = self._to_white_pixmap(pm)

        # --- Loop (~60 fps) ---
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def _to_white_pixmap(self, pm: QtGui.QPixmap) -> QtGui.QPixmap:
        # Convertit tous les pixels opaques en blanc (conserve l'alpha)
        img = pm.toImage().convertToFormat(QtGui.QImage.Format_ARGB32)
        w, h = img.width(), img.height()
        for y in range(h):
            for x in range(w):
                c = QtGui.QColor(img.pixelColor(x, y))
                if c.alpha() > 0:
                    img.setPixelColor(x, y, QtGui.QColor(255, 255, 255, c.alpha()))
        return QtGui.QPixmap.fromImage(img)

    def keyPressEvent(self, e: QtGui.QKeyEvent):
        # Game-over handling
        if self._lives <= 0:
            if e.key() == QtCore.Qt.Key_Escape:
                self.close()
                return
            if e.key() in (QtCore.Qt.Key_R, QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
                self._restart_game()
                return
            return

        if e.key() == QtCore.Qt.Key_Escape:
            self.close()
            return
        if e.key() in (QtCore.Qt.Key_Left, QtCore.Qt.Key_A):
            self._left = True
        elif e.key() in (QtCore.Qt.Key_Right, QtCore.Qt.Key_D):
            self._right = True
        elif e.key() == QtCore.Qt.Key_Space:
            self._fire()
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e: QtGui.QKeyEvent):
        if e.key() in (QtCore.Qt.Key_Left, QtCore.Qt.Key_A):
            self._left = False
        elif e.key() in (QtCore.Qt.Key_Right, QtCore.Qt.Key_D):
            self._right = False
        super().keyReleaseEvent(e)

    def _fire(self):
        if self._cooldown > 0:
            return
        self._cooldown = 10
        self._shots.append({"x": float(self._ship_x), "y": float(self._ship_y - 18)})

    def _spawn_cme(self):
        # Spawn depuis le "soleil" (haut), léger spread horizontal
        x = float(self._rng.integers(40, self._w - 40))
        r = float(self._rng.integers(10, 18))
        vy = float(self._rng.uniform(0.6, 1.1))
        self._cmes.append({"x": x, "y": 95.0, "vy": vy, "r": r})

    
    def _restart_game(self):
        # Reset state and restart timer after GAME OVER
        self._score = 0
        self._lives = 3
        self._frame = 0
        self._cooldown = 0
        self._left = False
        self._right = False
        self._shots.clear()
        self._cmes.clear()
        self._ship_x = self._w // 2
        if not self._timer.isActive():
            self._timer.start()
        self.update()

    def _tick(self):
        self._frame += 1
        if self._cooldown > 0:
            self._cooldown -= 1

        # Ship move
        if self._left:
            self._ship_x -= self._ship_speed
        if self._right:
            self._ship_x += self._ship_speed
        halfw = self._ship_pix.width() / 2
        self._ship_x = int(max(halfw + 10, min(self._w - halfw - 10, self._ship_x)))

        # Shots move
        for s in self._shots:
            s["y"] -= 9.0
        self._shots = [s for s in self._shots if s["y"] > 40]

        # Spawn CME (accélère avec le score)
        spawn_every = self._spawn_every  # difficulté fixe (pas d'accélération automatique)
        if (self._frame % spawn_every) == 0:
            self._spawn_cme()

        # CME move
        for c in self._cmes:
            c["y"] += c["vy"] + min(0.25, self._score * 0.00025)

        # Collisions shot <-> CME
        new_cmes = []
        for c in self._cmes:
            hit = False
            cx, cy, cr = c["x"], c["y"], c["r"]
            for s in self._shots:
                dx = s["x"] - cx
                dy = s["y"] - cy
                if (dx * dx + dy * dy) <= (cr * cr):
                    hit = True
                    s["y"] = -9999  # supprime le tir
                    self._score += 10
                    break
            if not hit:
                new_cmes.append(c)
        self._cmes = new_cmes
        self._shots = [s for s in self._shots if s["y"] > 0]

        # CME reaches bottom -> lose life
        still = []
        for c in self._cmes:
            if c["y"] > (self._h - 40):
                self._lives -= 1
            else:
                still.append(c)
        self._cmes = still

        if self._lives <= 0:
            self._timer.stop()

        self.update()

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        p.fillRect(self.rect(), QtGui.QColor("black"))

        pen = QtGui.QPen(QtGui.QColor("white"))
        pen.setWidth(2)
        p.setPen(pen)
        p.setBrush(QtCore.Qt.NoBrush)

        # Sun (top center)
        sun_r = 26
        sun_x = self._w // 2
        sun_y = 60
        p.drawEllipse(QtCore.QPoint(sun_x, sun_y), sun_r, sun_r)
        for a in range(0, 360, 30):
            rad = math.radians(a)
            x1 = sun_x + int(math.cos(rad) * (sun_r + 6))
            y1 = sun_y + int(math.sin(rad) * (sun_r + 6))
            x2 = sun_x + int(math.cos(rad) * (sun_r + 16))
            y2 = sun_y + int(math.sin(rad) * (sun_r + 16))
            p.drawLine(x1, y1, x2, y2)

        # HUD
        p.setFont(QtGui.QFont("Consolas", 14, QtGui.QFont.Bold))
        p.drawText(14, 28, f"SCORE {self._score:05d}")
        p.drawText(self._w - 160, 28, f"LIVES {self._lives}")

        # CME
        for c in self._cmes:
            p.drawEllipse(QtCore.QPointF(c["x"], c["y"]), c["r"], c["r"])

        # Shots
        for s in self._shots:
            p.drawLine(int(s["x"]), int(s["y"]), int(s["x"]), int(s["y"] - 14))

        # Ship (logo)
        ship_left = int(self._ship_x - self._ship_pix.width() / 2)
        ship_top = int(self._ship_y - self._ship_pix.height() / 2)
        p.drawPixmap(ship_left, ship_top, self._ship_pix)

        if self._lives <= 0:
            p.setFont(QtGui.QFont("Consolas", 30, QtGui.QFont.Bold))
            p.drawText(self.rect(), QtCore.Qt.AlignCenter, "GAME OVER\n\nR to restart  |  ESC to quit")

        p.end()



# =====================================================
# HF BAND OPENINGS RIBBON (0–30 MHz)
# =====================================================

HF_BANDS = [
    ("160m", 1.8),
    ("80m",  3.5),
    ("40m",  7.0),
    ("30m", 10.1),
    ("20m", 14.0),
    ("17m", 18.1),
    ("15m", 21.0),
    ("12m", 24.9),
    ("10m", 28.0),
]

def _hf_muf_estimate_mhz(sfi: float, kp: float) -> float:
    """Heuristic MUF estimate (MHz) from SFI, degraded by Kp."""
    if sfi is None or (isinstance(sfi, float) and math.isnan(sfi)):
        return float("nan")
    if kp is None or (isinstance(kp, float) and math.isnan(kp)):
        kp = 0.0

    muf = 0.14 * float(sfi) + 3.0
    kp_excess = max(0.0, float(kp) - 3.0)
    muf *= max(0.60, 1.0 - 0.055 * kp_excess)  # floor at 60%

    return float(max(1.0, muf))

def _hf_absorption_factor(xray_class: str) -> float:
    """Simple D-layer absorption factor from flare class."""
    c = (xray_class or "").strip().upper()
    if not c or c == "?":
        return 1.0
    if c.startswith("X"):
        return 0.45
    if c.startswith("M"):
        return 0.70
    if c.startswith("C"):
        return 0.90
    return 1.0

def _hf_band_state(freq_mhz: float, muf_mhz: float, absorption: float) -> tuple[str, str, int]:
    """Return (label, color, score 0..100) for a band."""
    if muf_mhz is None or (isinstance(muf_mhz, float) and math.isnan(muf_mhz)):
        return ("N/A", ACCENT_GREY, 0)

    # Low bands more affected by absorption
    if freq_mhz <= 7.0:
        abs_eff = absorption
    elif freq_mhz <= 14.0:
        abs_eff = min(1.0, 0.85 + 0.15 * absorption)
    else:
        abs_eff = min(1.0, 0.92 + 0.08 * absorption)

    eff_muf = muf_mhz * abs_eff
    ratio = eff_muf / float(freq_mhz)

    if ratio >= 1.20:
        return ("OPEN", ACCENT_GREEN, 90)
    if ratio >= 1.05:
        return ("FAIR", "#ffd34d", 70)
    if ratio >= 0.90:
        return ("POOR", "#ff9f43", 45)
    return ("CLOSED", "#ff4d4d", 15)

def _muf_color(muf_mhz: float) -> tuple[str, str]:
    """Return (bg_color, text) for MUF."""
    if muf_mhz is None or (isinstance(muf_mhz, float) and math.isnan(muf_mhz)):
        return (ACCENT_GREY, "--.- MHz")
    m = float(muf_mhz)
    if m < 7:
        return ("#ff4d4d", f"{m:.1f} MHz")
    if m < 10:
        return ("#ff9f43", f"{m:.1f} MHz")
    if m < 14:
        return ("#ffd34d", f"{m:.1f} MHz")
    if m < 18:
        return (ACCENT_GREEN, f"{m:.1f} MHz")
    if m < 24:
        return ("#4aa3ff", f"{m:.1f} MHz")
    return ("#b36bff", f"{m:.1f} MHz")

def _parse_xray_magnitude(xray_class: str) -> tuple[str, float]:
    """Parse 'M5.2' -> ('M', 5.2). Unknown -> ('', nan)."""
    c = (xray_class or "").strip().upper()
    if not c or c == "?":
        return ("", float("nan"))
    letter = c[0]
    try:
        val = float(c[1:])
    except Exception:
        return (letter, float("nan"))
    return (letter, val)

def _radio_blackout_level(xray_class: str) -> tuple[str, str]:
    """Return (R-level label, color) from GOES flare class (heuristic NOAA R-scale mapping)."""
    letter, mag = _parse_xray_magnitude(xray_class)
    # Defaults
    if not letter:
        return ("R0", ACCENT_GREEN)

    if letter in ("A", "B", "C"):
        return ("R0", ACCENT_GREEN)

    if letter == "M":
        if isinstance(mag, float) and not math.isnan(mag):
            if mag < 5.0:
                return ("R1", "#ffd34d")  # minor
            return ("R2", "#ff9f43")     # moderate
        return ("R1", "#ffd34d")

    if letter == "X":
        if isinstance(mag, float) and not math.isnan(mag):
            if mag < 5.0:
                return ("R3", "#ff4d4d")  # strong
            if mag < 10.0:
                return ("R4", "#ff4dd2")  # severe (magenta)
            return ("R5", "#b36bff")      # extreme (violet)
        return ("R3", "#ff4d4d")

    return ("R0", ACCENT_GREEN)


class HfBandBar(QtWidgets.QFrame):
    """Compact HF (0–30 MHz) openings ribbon shown under the header."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("hfBandBar")
        self.setFixedHeight(48)

        self._muf_hist: list[tuple[float, float]] = []

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(12, 6, 12, 6)
        lay.setSpacing(8)

        self.lbl_title = QtWidgets.QLabel("HF openings (0–30 MHz):")
        self.lbl_title.setStyleSheet("color: #aab6c5; font-size: 12px; font-weight: 900;")
        lay.addWidget(self.lbl_title, 0)

        # Band chips
        self._chips: dict[str, QtWidgets.QLabel] = {}
        for name, _f in HF_BANDS:
            chip = QtWidgets.QLabel(name)
            chip.setAlignment(QtCore.Qt.AlignCenter)
            chip.setFixedHeight(26)
            chip.setMinimumWidth(52)
            chip.setStyleSheet(self._chip_style("#121a22"))
            self._chips[name] = chip
            lay.addWidget(chip, 0)

        lay.addStretch(1)

        # MUF badge
        self.lbl_muf = QtWidgets.QLabel("MUF: --.- MHz")
        self.lbl_muf.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_muf.setFixedHeight(26)
        self.lbl_muf.setMinimumWidth(190)
        self.lbl_muf.setStyleSheet(self._badge_style(ACCENT_GREY))
        lay.addWidget(self.lbl_muf, 0)

        # Radio blackout badge
        self.lbl_blackout = QtWidgets.QLabel("RADIO BLACKOUT R0")
        self.lbl_blackout.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_blackout.setFixedHeight(26)
        self.lbl_blackout.setMinimumWidth(170)
        self.lbl_blackout.setStyleSheet(self._badge_style(ACCENT_GREEN))
        lay.addWidget(self.lbl_blackout, 0)

        self.setStyleSheet(f"""
            QFrame#hfBandBar {{
                background: {BG_PANEL};
                border: 1px solid {BORDER};
                border-radius: 10px;
            }}
        """)

        self.update_from_indices(float("nan"), float("nan"), "", float("nan"))

    @staticmethod
    def _badge_style(bg: str) -> str:
        return f"""
            QLabel {{
                background: {bg};
                color: #0b0f12;
                border-radius: 8px;
                padding: 2px 10px;
                font-weight: 900;
                font-size: 12px;
            }}
        """

    @staticmethod
    def _chip_style(bg: str) -> str:
        # If bg is dark, keep border. If bg is accent, no border.
        is_dark = (bg.lower() == "#121a22") or (bg.lower() == "#0f1720")
        if is_dark:
            return """
                QLabel {
                    background: #121a22;
                    color: #d7dde6;
                    border: 1px solid #2a3440;
                    border-radius: 8px;
                    padding: 2px 8px;
                    font-weight: 900;
                    font-size: 12px;
                }
            """
        return f"""
            QLabel {{
                background: {bg};
                color: #0b0f12;
                border: 0px;
                border-radius: 8px;
                padding: 2px 8px;
                font-weight: 900;
                font-size: 12px;
            }}
        """

    def update_from_indices(self, sfi_now: float, kp_now: float, xray_class: str, sw_speed: float):
        muf = _hf_muf_estimate_mhz(sfi_now, kp_now)
        absorption = _hf_absorption_factor(xray_class)

        # --- MUF trend over last 10 minutes ---
        now = time.time()
        self._muf_hist.append((now, muf))
        self._muf_hist = [(t, v) for t, v in self._muf_hist if now - t <= 600]

        muf_delta = 0.0
        trend = "→"
        if len(self._muf_hist) >= 2:
            v0 = self._muf_hist[0][1]
            v1 = self._muf_hist[-1][1]
            if not (isinstance(v0, float) and math.isnan(v0)) and not (isinstance(v1, float) and math.isnan(v1)):
                muf_delta = v1 - v0
                if muf_delta > 0.3:
                    trend = "▲"
                elif muf_delta < -0.3:
                    trend = "▼"

        muf_color, muf_text = _muf_color(muf)
        delta_txt = f"{trend} {muf_delta:+.1f} MHz"
        self.lbl_muf.setText(f"MUF: {muf_text}  {delta_txt}")
        self.lbl_muf.setStyleSheet(self._badge_style(muf_color))

        r_label, r_color = _radio_blackout_level(xray_class)
        self.lbl_blackout.setText(f"RADIO BLACKOUT {r_label}")
        self.lbl_blackout.setStyleSheet(self._badge_style(r_color))

        # Band chips
        for band, freq in HF_BANDS:
            state, color, score = _hf_band_state(freq, muf, absorption)
            chip = self._chips.get(band)
            if chip:
                chip.setText(band)
                chip.setStyleSheet(self._chip_style(color))
                chip.setToolTip(
                    f"{band} ({freq:.1f} MHz)\n"
                    f"Status: {state} — score {score}/100\n"
                    f"MUF est.: {muf_text}\n"
                    f"Kp: {kp_now if not (isinstance(kp_now,float) and math.isnan(kp_now)) else '?'}\n"
                    f"SFI: {sfi_now if not (isinstance(sfi_now,float) and math.isnan(sfi_now)) else '?'}\n"
                    f"X-ray: {xray_class or '?'}\n"
                    "Note: heuristic indicator (not path/location specific)."
                )

# =====================================================
# MAIN WINDOW
# =====================================================
class MainWindow(QtWidgets.QMainWindow):
    def _play_startup_sound(self):
        # Play a short startup sound if sound is enabled
        if not getattr(self, "_sound_enabled", True):
            return
        path = str(self._sounds_dir / "startup.wav")
        if not QtCore.QFile.exists(path):
            return
        try:
            s = QSoundEffect(self)
            s.setSource(QtCore.QUrl.fromLocalFile(path))
            s.setVolume(0.6)
            self._startup_sfx = s  # keep reference
            QtCore.QTimer.singleShot(250, s.play)
        except Exception:
            pass

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SUMO - Sun Monitor")
        self.resize(1300, 780)

        self._cfg = load_config(CONFIG_PATH)

        # --- One-shot 5-minute border blink when panels ENTER red state ---
        self._kp_last_is_red = None  # type: bool | None
        self._sw_last_is_red = None  # type: bool | None


        # --- Sound alerts ---
        # Persisted in sumo_config.json ("sound_enabled")
        self._sound_enabled = bool(self._cfg.get("sound_enabled", True))
        self._alert_state = {"cme_high": False, "xray_m99": False, "proton_s4": False}
        QtCore.QTimer.singleShot(0, self._play_startup_sound)
        self._sounds_dir = (APP_DIR / "sounds")
        try:
            self._sounds_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        self._sound_paths = {
            "cme_high": str(self._sounds_dir / "cme_high.wav"),
            "xray_m99": str(self._sounds_dir / "xray_m99.wav"),
            "proton_s4": str(self._sounds_dir / "proton_s4.wav"),
        }
        self._sfx = {}
        # Initialize QSoundEffect objects if possible (avoid relying on a method that may be missing if code is edited)
        try:
            for key, path in self._sound_paths.items():
                s = QSoundEffect(self)
                s.setLoopCount(1)
                s.setVolume(0.9)
                s.setSource(QtCore.QUrl.fromLocalFile(path))
                self._sfx[key] = s
        except Exception:
            # If QtMultimedia backend is missing, we'll fall back to QApplication.beep() where needed.
            self._sfx = {}
        self._rss_url = str(self._cfg.get("rss_url") or DEFAULT_RSS_URL).strip() or DEFAULT_RSS_URL
        self._nasa_api_key = str(self._cfg.get("nasa_api_key") or "").strip()
        self._rss_speed = int(self._cfg.get("rss_speed", RSS_SCROLL_PX_PER_TICK))
        self._rss_speed = max(1, min(10, self._rss_speed))

        self._time_mode = str(self._cfg.get("time_mode") or "utc").strip().lower()
        if self._time_mode not in ("utc", "local"):
            self._time_mode = "utc"


        # --- DX Column (Cluster / POTA) settings ---
        self._dx_enabled = bool(self._cfg.get("dx_enabled", False))
        self._dx_source = str(self._cfg.get("dx_source") or "dx").strip().lower()
        if self._dx_source not in ("dx", "pota"):
            self._dx_source = "dx"
        self._dx_host = str(self._cfg.get("dx_host") or "dxspider.co.uk").strip() or "dxspider.co.uk"
        self._dx_login = str(self._cfg.get("dx_login") or "").strip()
        self._pota_zone = str(self._cfg.get("pota_zone") or "worldwide").strip().lower()
        if self._pota_zone not in ("worldwide", "usa", "europe"):
            self._pota_zone = "worldwide"
        try:
            self._dx_port = int(self._cfg.get("dx_port", 7300))
        except Exception:
            self._dx_port = 7300

        # --- HF openings ribbon visibility ---
        self._hf_bar_visible = bool(self._cfg.get("hf_bar_visible", True))

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        top = QtWidgets.QFrame()
        top.setObjectName("topbar")
        top.setFixedHeight(96)
        top_l = QtWidgets.QHBoxLayout(top)
        top_l.setContentsMargins(12, 10, 12, 10)
        top_l.setSpacing(12)

        self.logo = QtWidgets.QLabel()
        self.logo.setFixedSize(78, 78)
        self.logo.setScaledContents(True)
        if hasattr(self, "_load_logo"):
            self._load_logo(str(ASSETS_DIR / "logo.png"))
        else:
            pm = QtGui.QPixmap(str(ASSETS_DIR / "logo.png"))
            if pm.isNull():
                pm = QtGui.QPixmap(58, 58)
                pm.fill(QtGui.QColor("#333"))
            self.logo.setPixmap(pm)

        # --- Easter egg: triple-click on logo to launch mini-game ---
        self.logo.installEventFilter(self)
        self._egg_clicks = 0
        self._egg_last_click_ts = 0.0

        title_box = QtWidgets.QVBoxLayout()
        title_box.setSpacing(2)
        self.lbl_app = QtWidgets.QLabel("SUMO")
        self.lbl_app.setObjectName("appTitle")
        self.lbl_sub = QtWidgets.QLabel("Sun Monitor")
        self.lbl_sub.setObjectName("appSubTitle")
        self.lbl_version = QtWidgets.QLabel(APP_VERSION)
        self.lbl_version.setObjectName("appVersion")
        self.lbl_version.setAlignment(QtCore.Qt.AlignHCenter)
        title_box.addWidget(self.lbl_app)
        title_box.addWidget(self.lbl_sub)
        title_box.addWidget(self.lbl_version)
        title_box.setAlignment(self.lbl_version, QtCore.Qt.AlignHCenter)

        self.rss_ticker = RssTicker()
        self.rss_ticker.setToolTip("RSS/Atom ticker")
        self.rss_ticker.setSpeed(self._rss_speed)

        right = QtWidgets.QVBoxLayout()
        right.setSpacing(2)

        # ---- UTC BADGE (visible, a gauche du RSS) ----
        self.lbl_utc_badge = QtWidgets.QLabel("UTC\n--:--:--")
        self.lbl_utc_badge.setAlignment(QtCore.Qt.AlignCenter)

        font = QtGui.QFont("Consolas", 20, QtGui.QFont.Bold)
        self.lbl_utc_badge.setFont(font)

        self.lbl_utc_badge.setStyleSheet("""
        QLabel {
            color: #4aa3ff;
            background: #0e141a;
            border: 1px solid #2a3440;
            border-radius: 8px;
            padding: 6px 12px;
        }
        """)

        self.lbl_status = QtWidgets.QLabel("Data: connecting…")
        self.lbl_status.setObjectName("statusText")
        self._set_data_status("connecting")

        # Put the data status in the bottom status bar (saves space in the top-right)
        try:
            self.statusBar().addPermanentWidget(self.lbl_status, 1)
        except Exception:
            pass


        btn_grid = QtWidgets.QGridLayout()
        btn_grid.setContentsMargins(0, 0, 0, 0)
        btn_grid.setHorizontalSpacing(8)
        btn_grid.setVerticalSpacing(6)

        self.btn_fullscreen = QtWidgets.QPushButton("Full screen")
        self.btn_fullscreen.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btn_fullscreen.setFixedHeight(24)
        self.btn_fullscreen.clicked.connect(self._toggle_fullscreen)

        self.btn_settings = QtWidgets.QPushButton("Settings")
        self.btn_settings.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btn_settings.setFixedHeight(24)
        self.btn_settings.clicked.connect(self._open_settings)
        self.btn_settings.setStyleSheet(f"""
            QPushButton {{
                background: {BTN_SETTINGS_YELLOW};
                color: {BTN_TEXT};
                border: 0px;
                border-radius: 6px;
                padding: 4px 10px;
                font-weight: 800;
            }}
            QPushButton:hover {{ opacity: 0.95; }}
        """)

        self.btn_dx = QtWidgets.QPushButton("")
        self.btn_dx.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btn_dx.setFixedHeight(24)
        self.btn_dx.clicked.connect(self._toggle_dx_panel)
        self._update_dx_button()

        self.btn_hfbar = QtWidgets.QPushButton("")
        self.btn_hfbar.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btn_hfbar.setFixedHeight(24)
        self.btn_hfbar.clicked.connect(self._toggle_hf_bar)
        self._update_hfbar_button()

        self.btn_about = QtWidgets.QPushButton("About")

        self.btn_sound = QtWidgets.QPushButton("")
        self.btn_sound.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btn_sound.setFixedHeight(24)
        self.btn_sound.clicked.connect(self._toggle_sound)
        self._update_sound_button()
        self.btn_about.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btn_about.setFixedHeight(24)
        self.btn_about.clicked.connect(self._open_about)
        self.btn_about.setStyleSheet("""
            QPushButton {
                background: #4aa3ff;
                color: #0b0f12;
                border: 0px;
                border-radius: 6px;
                padding: 4px 10px;
                font-weight: 800;
            }
            QPushButton:hover { opacity: 0.95; }
        """)


        btn_grid.addWidget(self.btn_fullscreen, 0, 0)
        btn_grid.addWidget(self.btn_settings,   1, 0)
        btn_grid.addWidget(self.btn_dx,         2, 0)
        btn_grid.addWidget(self.btn_hfbar,      2, 1)
        btn_grid.addWidget(self.btn_sound,      0, 1)
        btn_grid.addWidget(self.btn_about,      1, 1)

        right.addLayout(btn_grid)
        right.addStretch(1)

        self._update_fullscreen_button()

        top_l.addWidget(self.logo, 0)
        top_l.addLayout(title_box, 0)
        top_l.addWidget(self.lbl_utc_badge, 0)
        top_l.addWidget(self.rss_ticker, 1)
        top_l.addLayout(right, 0)
        root.addWidget(top)

        # --- HF openings ribbon (0–30 MHz) ---
        self.hf_bar = HfBandBar()
        root.addWidget(self.hf_bar)
        self.hf_bar.setVisible(bool(getattr(self, "_hf_bar_visible", True)))

        self.main_grid = QtWidgets.QGridLayout()
        self.main_grid.setContentsMargins(0, 0, 0, 0)
        self.main_grid.setSpacing(10)
        root.addLayout(self.main_grid, 1)

        self.panels: dict[str, KpLikePanel] = {}

        configs = [
            ("CME",  PanelConfig("CME ARRIVAL PROBABILITY [N/A]", unit="%", big_value_fmt="{:.0f}")),
            ("XRAY", PanelConfig("X-RAYS (GOES 0.1-0.8nm)", unit="", big_value_fmt="{:.2e}")),
            ("KP",   PanelConfig("K-INDEX (Planetary)", unit="", mode="bar", big_value_fmt="{:.1f}")),

            ("SSN",  PanelConfig("SSN (Sunspot Number)", unit="", big_value_fmt="{:.0f}")),
            ("SFI",  PanelConfig("SFI (F10.7 cm Flux)", unit="SFU", big_value_fmt="{:.0f}")),
            ("BZBT", PanelConfig("BZ / BT (Solar Wind MAG)", unit="nT", big_value_fmt="{:.1f}")),

            ("P10",  PanelConfig("PROTONS P10 (>=10MeV)", unit="pfu", big_value_fmt="{:.2f}")),
            ("AUR",  PanelConfig("AURORA", unit="%", big_value_fmt="{:.0f}")),
            ("SW",   PanelConfig("SOLAR WIND SPEED", unit="km/s", big_value_fmt="{:.0f}")),
        ]
        positions = [(0, 0), (0, 1), (0, 2),
                     (1, 0), (1, 1), (1, 2),
                     (2, 0), (2, 1), (2, 2)]

        for (key, cfg), (r, c) in zip(configs, positions):
            p = KpLikePanel(cfg)
            p.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
            self.panels[key] = p
            # Widgets are laid out later (to support optional DX column)
            pass

        # Build the grid layout (9 panels, optional DX column)
        self.dx_panel = DxClusterPanel()
        self.dx_panel.setMinimumWidth(320)
        self.dx_panel.setMaximumWidth(380)
        self._layout_main_grid()

        if self._dx_enabled:
            self._start_dx_worker()

        # Aurora: keep a rolling timeline of the last 25 hours (no extra network calls)
        # One sample per refresh (default: 60s) -> ~1500 points over 25h.
        self._aur_window_seconds = 25 * 3600
        self._aur_max_series = deque(maxlen=5000)
        self._aur_mean_series = deque(maxlen=5000)
        self._aur_area_series = deque(maxlen=5000)
        self._aur_ui_x = deque(maxlen=5000)
        self._load_aurora_from_db()
        self._refresh_cme_from_db()
        self._utc_timer = QtCore.QTimer(self)
        self._utc_timer.timeout.connect(self._tick_utc)
        self._utc_timer.start(1000)
        self._tick_utc()

        self._start_data_worker()
        self._start_rss_worker()

    def _start_data_worker(self):
        self._thread = QtCore.QThread(self)
        self._worker = DataWorker(refresh_seconds=60, nasa_api_key=self._nasa_api_key)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.data_ready.connect(self._on_data)
        self._worker.error.connect(self._on_error)
        self._thread.start()

    def _restart_data_worker(self):
        try:
            if hasattr(self, "_worker") and self._worker:
                self._worker.stop()
            if hasattr(self, "_thread") and self._thread:
                self._thread.quit()
                self._thread.wait(2000)
        except Exception:
            pass
        self._start_data_worker()

    def _start_rss_worker(self):
        self._rss_thread = QtCore.QThread(self)
        self._rss_worker = RssWorker(self._rss_url, RSS_REFRESH_SECONDS)
        self._rss_worker.moveToThread(self._rss_thread)
        self._rss_thread.started.connect(self._rss_worker.run)
        self._rss_worker.rss_ready.connect(self._on_rss_text)
        self._rss_worker.rss_error.connect(self._on_rss_error)
        self._rss_thread.start()

    def _restart_rss_worker(self, new_url: str):
        try:
            if hasattr(self, "_rss_worker") and self._rss_worker:
                self._rss_worker.stop()
            if hasattr(self, "_rss_thread") and self._rss_thread:
                self._rss_thread.quit()
                self._rss_thread.wait(2000)
        except Exception:
            pass

        self._rss_thread = QtCore.QThread(self)
        self._rss_worker = RssWorker(new_url, RSS_REFRESH_SECONDS)
        self._rss_worker.moveToThread(self._rss_thread)
        self._rss_thread.started.connect(self._rss_worker.run)
        self._rss_worker.rss_ready.connect(self._on_rss_text)
        self._rss_worker.rss_error.connect(self._on_rss_error)
        self._rss_thread.start()

    def _open_settings(self):
        dlg = SettingsDialog(self, rss_url=self._rss_url, nasa_api_key=self._nasa_api_key, time_mode=self._time_mode, rss_speed=self._rss_speed,
                          dx_enabled=self._dx_enabled, dx_source=self._dx_source, dx_host=self._dx_host, dx_port=self._dx_port, dx_login=self._dx_login, pota_zone=getattr(self, "_pota_zone", "worldwide"))
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return

        new_rss = dlg.rss_url().strip() or DEFAULT_RSS_URL
        new_key = dlg.nasa_api_key().strip()
        new_mode = dlg.time_mode().strip().lower() or "utc"
        new_speed = dlg.rss_speed()
        new_dx_enabled = dlg.dx_enabled()
        new_dx_source = dlg.dx_source().strip().lower() or "dx"
        if new_dx_source not in ("dx", "pota"):
            new_dx_source = "dx"
        new_dx_host = dlg.dx_host().strip() or "dxspider.co.uk"
        new_dx_port = dlg.dx_port()
        new_dx_login = dlg.dx_login().strip()
        new_pota_zone = dlg.pota_zone().strip().lower() or "worldwide"
        if new_pota_zone not in ("worldwide", "usa", "europe"):
            new_pota_zone = "worldwide"

        changed_rss = (new_rss != self._rss_url)
        changed_key = (new_key != self._nasa_api_key)
        changed_mode = (new_mode != self._time_mode)
        changed_speed = (int(new_speed) != int(self._rss_speed))
        changed_dx = (bool(new_dx_enabled) != bool(self._dx_enabled)) or (str(new_dx_source) != str(self._dx_source)) or (new_dx_host != self._dx_host) or (int(new_dx_port) != int(self._dx_port)) or (new_dx_login != self._dx_login) or (str(new_pota_zone) != str(getattr(self, "_pota_zone", "worldwide")) )

        self._rss_url = new_rss
        self._nasa_api_key = new_key
        self._time_mode = new_mode
        self._rss_speed = max(1, min(10, int(new_speed)))

        self._dx_enabled = bool(new_dx_enabled)
        self._dx_source = str(new_dx_source).strip().lower() if str(new_dx_source).strip() else "dx"
        if self._dx_source not in ("dx", "pota"):
            self._dx_source = "dx"
        self._dx_host = str(new_dx_host).strip() or "dxspider.co.uk"
        try:
            self._dx_port = int(new_dx_port)
        except Exception:
            self._dx_port = 7300
        self._dx_login = str(new_dx_login).strip()
        self._pota_zone = str(new_pota_zone).strip().lower() or "worldwide"
        if self._pota_zone not in ("worldwide", "usa", "europe"):
            self._pota_zone = "worldwide"

        self._cfg["rss_url"] = self._rss_url
        self._cfg["nasa_api_key"] = self._nasa_api_key
        self._cfg["time_mode"] = self._time_mode
        self._cfg["rss_speed"] = int(self._rss_speed)
        self._cfg["dx_enabled"] = bool(self._dx_enabled)
        self._cfg["dx_source"] = str(self._dx_source)
        self._cfg["dx_host"] = self._dx_host
        self._cfg["dx_port"] = int(self._dx_port)
        self._cfg["dx_login"] = self._dx_login
        self._cfg["pota_zone"] = str(getattr(self, "_pota_zone", "worldwide"))
        save_config(CONFIG_PATH, self._cfg)

        if changed_rss:
            self.rss_ticker.setText("NASA Solar System News   •   updating RSS…")
            self._restart_rss_worker(self._rss_url)

        if changed_key:
            self._restart_data_worker()

        if changed_mode:
            self._tick_utc()

        if changed_speed:
            self.rss_ticker.setSpeed(self._rss_speed)

        if changed_dx:
            self._update_dx_button()
            self._layout_main_grid()
            if self._dx_enabled:
                self.dx_panel.clear_spots()
                self._start_dx_worker()
            else:
                self._stop_dx_worker()


    def _open_about(self):
        dlg = AboutDialog(self)
        dlg.exec()

    def _is_fullscreen(self) -> bool:
        return bool(self.windowState() & QtCore.Qt.WindowFullScreen)

    
    def _update_dx_button(self):
        if getattr(self, "_dx_enabled", False):
            self.btn_dx.setText("DX ON")
            self.btn_dx.setStyleSheet(
                f"""
                QPushButton {{
                    background: {BTN_GREEN};
                    color: {BTN_TEXT};
                    border: 0px;
                    border-radius: 6px;
                    padding: 4px 10px;
                    font-weight: 900;
                }}
                """
            )
        else:
            self.btn_dx.setText("DX OFF")
            self.btn_dx.setStyleSheet(
                """
                QPushButton {
                    background: #121a22;
                    color: #d7dde6;
                    border: 1px solid #2a3440;
                    border-radius: 6px;
                    padding: 4px 10px;
                    font-weight: 900;
                }
                QPushButton:hover { background: #16202a; }
                """
            )


    def _update_hfbar_button(self):
        """Update the HF ribbon toggle button in the top bar."""
        if getattr(self, "_hf_bar_visible", True):
            self.btn_hfbar.setText("HF BAR ON")
            self.btn_hfbar.setStyleSheet(
                f"""
                QPushButton {{
                    background: {BTN_GREEN};
                    color: {BTN_TEXT};
                    border: 0px;
                    border-radius: 6px;
                    padding: 4px 10px;
                    font-weight: 900;
                }}
                """
            )
        else:
            self.btn_hfbar.setText("HF BAR OFF")
            self.btn_hfbar.setStyleSheet(
                """
                QPushButton {
                    background: #121a22;
                    color: #d7dde6;
                    border: 1px solid #2a3440;
                    border-radius: 6px;
                    padding: 4px 10px;
                    font-weight: 900;
                }
                QPushButton:hover { background: #16202a; }
                """
            )

    @QtCore.Slot()
    def _toggle_hf_bar(self):
        """Show/hide the HF openings ribbon under the top bar."""
        self._hf_bar_visible = not getattr(self, "_hf_bar_visible", True)
        self._cfg["hf_bar_visible"] = bool(self._hf_bar_visible)
        save_config(CONFIG_PATH, self._cfg)

        try:
            if hasattr(self, "hf_bar") and self.hf_bar is not None:
                self.hf_bar.setVisible(bool(self._hf_bar_visible))
        except Exception:
            pass

        self._update_hfbar_button()

    def _clear_layout(self, layout: QtWidgets.QLayout):
        while layout.count():
            it = layout.takeAt(0)
            w = it.widget()
            l = it.layout()
            if w is not None:
                w.setParent(None)
            if l is not None:
                self._clear_layout(l)

    def _layout_main_grid(self):
        """Lay out the 9 solar panels, optionally with a left DX column."""
        g = self.main_grid
        self._clear_layout(g)

        if getattr(self, "_dx_enabled", False):
            # DX column spans all 3 rows on the left (col 0)
            g.addWidget(self.dx_panel, 0, 0, 3, 1)

            # Shift solar panels to cols 1..3
            positions = [(0, 1), (0, 2), (0, 3),
                         (1, 1), (1, 2), (1, 3),
                         (2, 1), (2, 2), (2, 3)]

            # Column sizing: fixed DX + 3 equal columns
            g.setColumnMinimumWidth(0, 320)
            g.setColumnStretch(0, 0)
            for cc in (1, 2, 3):
                g.setColumnStretch(cc, 1)

        else:
            positions = [(0, 0), (0, 1), (0, 2),
                         (1, 0), (1, 1), (1, 2),
                         (2, 0), (2, 1), (2, 2)]

            # Reset any previous DX-layout column sizing (important when toggling DX ON/OFF at runtime)
            # - Column 0 should behave like a normal solar column again (no fixed min width)
            g.setColumnMinimumWidth(0, 0)

            # - Column 3 must not keep stretch from the DX layout, otherwise it stays as an empty column
            g.setColumnMinimumWidth(3, 0)
            g.setColumnStretch(3, 0)

            for cc in (0, 1, 2):
                g.setColumnStretch(cc, 1)


        # Always 3 rows equal
        for rr in range(3):
            g.setRowStretch(rr, 1)

        order = ["CME", "XRAY", "KP", "SSN", "SFI", "BZBT", "P10", "AUR", "SW"]
        for key, (r, c) in zip(order, positions):
            w = self.panels.get(key)
            if w is not None:
                g.addWidget(w, r, c)

    def _start_dx_worker(self):
        """Start the left DX column worker (DX Cluster telnet OR POTA spots)."""
        self._stop_dx_worker()

        # Decide source
        src = str(getattr(self, "_dx_source", "dx") or "dx").strip().lower()
        if src not in ("dx", "pota"):
            src = "dx"
        self._dx_source = src

        self._dx_thread = QtCore.QThread(self)

        if src == "pota":
            # POTA spots (HTTP JSON)
            self.dx_panel.set_title("POTA")
            self.dx_panel.set_status("POTA: loading…")
            self._dx_worker = PotaSpotsWorker(POTA_SPOTS_URL, refresh_seconds=30)
            # Buffering to avoid UI freezes when many spots arrive at once
            self._pota_spot_buffer = []
            self._pota_flush_scheduled = False
        else:
            # DX Cluster (telnet)
            self.dx_panel.set_title("Cluster")
            self.dx_panel.set_connection_label(self._dx_host, self._dx_port)
            self.dx_panel.set_status(f"{self._dx_host}:{self._dx_port}")
            self._dx_worker = DxClusterWorker(self._dx_host, self._dx_port, login=self._dx_login)

        self._dx_worker.moveToThread(self._dx_thread)
        self._dx_thread.started.connect(self._dx_worker.run)
        self._dx_worker.spot.connect(self._on_dx_spot)
        self._dx_worker.status.connect(self.dx_panel.set_status)
        self._dx_worker.error.connect(self._on_dx_error)
        self._dx_thread.start()

    def _stop_dx_worker(self):
        try:
            if hasattr(self, "_dx_worker") and self._dx_worker:
                self._dx_worker.stop()
        except Exception:
            pass
        try:
            if hasattr(self, "_dx_thread") and self._dx_thread:
                self._dx_thread.quit()
                self._dx_thread.wait(1500)
        except Exception:
            pass
        try:
            self._pota_spot_buffer = []
            self._pota_flush_scheduled = False
        except Exception:
            pass
        self._dx_thread = None
        self._dx_worker = None

    @QtCore.Slot()
    def _flush_pota_spots(self):
        """Apply buffered POTA spots in one UI update to avoid freezes."""
        try:
            buf = getattr(self, "_pota_spot_buffer", None)
            if not buf:
                self._pota_flush_scheduled = False
                return
            self._pota_spot_buffer = []
            self._pota_flush_scheduled = False

            if not getattr(self, "_dx_enabled", False):
                return

            # Bulk insert then rebuild once (much cheaper than rebuilding for every spot)
            try:
                self.dx_panel.table.setUpdatesEnabled(False)
            except Exception:
                pass

            for s in buf:
                try:
                    self.dx_panel._spots.insert(0, s)
                except Exception:
                    continue

            # keep bounded
            try:
                if len(self.dx_panel._spots) > self.dx_panel._max_spots:
                    self.dx_panel._spots = self.dx_panel._spots[: self.dx_panel._max_spots]
            except Exception:
                pass

            self.dx_panel._rebuild_table()

        except Exception:
            pass
        finally:
            try:
                self.dx_panel.table.setUpdatesEnabled(True)
            except Exception:
                pass

    @QtCore.Slot(dict)
    def _on_dx_spot(self, spot: dict):
        try:
            if not getattr(self, "_dx_enabled", False):
                return

            # If POTA source, buffer spots and flush in a single repaint
            if str(getattr(self, "_dx_source", "dx") or "dx").strip().lower() == "pota":
                if not hasattr(self, "_pota_spot_buffer") or self._pota_spot_buffer is None:
                    self._pota_spot_buffer = []

                # Apply POTA zone filter (Worldwide / USA / Europe)
                try:
                    z = str(getattr(self, "_pota_zone", "worldwide") or "worldwide").strip().lower()
                    if z in ("usa", "europe"):
                        # Get POTA fields (worker provides them; fallback to parsing 'raw')
                        ref = str(spot.get("reference") or "").strip().upper()
                        loc = str(spot.get("locationDesc") or "").strip().upper()
                        if not ref:
                            try:
                                raw = str(spot.get("raw") or "")
                                parts = raw.split()
                                # expected: "POTA <REF> <MODE> <FREQ> <CALL>"
                                if len(parts) >= 2 and parts[0].upper() == "POTA":
                                    ref = parts[1].strip().upper()
                            except Exception:
                                pass
                        prog = ref.split("-", 1)[0] if "-" in ref else ""
                        if z == "usa":
                            if not (loc.startswith("US-") or prog == "K"):
                                return
                        else:
                            # Europe: prefer locationDesc ISO country code when available
                            country = loc.split("-", 1)[0] if "-" in loc else ""
                            EUROPE_ISO = {
                                "AD","AL","AT","BA","BE","BG","BY","CH","CY","CZ","DE","DK","EE","ES","FI","FR","GB","GE","GR","HR","HU","IE","IS","IT","LI","LT","LU","LV","MC","MD","ME","MK","MT","NL","NO","PL","PT","RO","RS","RU","SE","SI","SK","SM","TR","UA","VA"
                            }
                            # Fallback to reference program prefix for common European programs
                            EU_PROG = {"F","DL","EA","CT","I","OE","HB","ON","PA","LX","G","GM","GW","GI","EI","OK","OM","SP","LA","SM","OH","OZ","SV","LZ","YO","HA","9A","S5","YU","TF","IS"}
                            if country:
                                if country not in EUROPE_ISO:
                                    return
                            else:
                                if prog not in EU_PROG:
                                    return
                except Exception:
                    pass

                self._pota_spot_buffer.append(spot)

                if not getattr(self, "_pota_flush_scheduled", False):
                    self._pota_flush_scheduled = True
                    QtCore.QTimer.singleShot(120, self._flush_pota_spots)
                return

            # DXSpider: low rate, safe to update directly
            self.dx_panel.add_spot(spot)
        except Exception:
            pass

    @QtCore.Slot(str)
    def _on_dx_error(self, msg: str):
        try:
            self.dx_panel.set_status(f"DX: offline ({msg})")
        except Exception:
            pass

    @QtCore.Slot()
    def _toggle_dx_panel(self):
        self._dx_enabled = not bool(getattr(self, "_dx_enabled", False))
        self._cfg["dx_enabled"] = bool(self._dx_enabled)
        self._cfg["dx_source"] = str(getattr(self, "_dx_source", "dx"))
        self._cfg["dx_host"] = self._dx_host
        self._cfg["dx_port"] = int(self._dx_port)
        save_config(CONFIG_PATH, self._cfg)

        if self._dx_enabled:
            self._layout_main_grid()
            self.dx_panel.clear_spots()
            self._start_dx_worker()
        else:
            self._stop_dx_worker()
            self._layout_main_grid()

        self._update_dx_button()
    def _update_fullscreen_button(self):
        if self._is_fullscreen():
            self.btn_fullscreen.setText("Exit full screen")
            self.btn_fullscreen.setStyleSheet(
                f"""
                QPushButton {{
                    background: {BTN_RED};
                    color: {BTN_TEXT};
                    border: 0px;
                    border-radius: 6px;
                    padding: 4px 10px;
                    font-weight: 800;
                }}
                """
            )
        else:
            self.btn_fullscreen.setText("Full screen")
            self.btn_fullscreen.setStyleSheet(
                f"""
                QPushButton {{
                    background: {BTN_GREEN};
                    color: {BTN_TEXT};
                    border: 0px;
                    border-radius: 6px;
                    padding: 4px 10px;
                    font-weight: 800;
                }}
                """
            )

    @QtCore.Slot()
    def _toggle_fullscreen(self):
        if self._is_fullscreen():
            self.showNormal()
        else:
            self.showFullScreen()
        self._update_fullscreen_button()

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() == QtCore.Qt.Key_Escape and self._is_fullscreen():
            self.showNormal()
            self._update_fullscreen_button()
            event.accept()
            return
        super().keyPressEvent(event)

    # ----------------- Sound alert helpers -----------------
    def _init_sound_effects(self):
        """Initialize QSoundEffect objects if audio backend is available."""
        self._sfx = {}
        try:
            for key, path in self._sound_paths.items():
                s = QSoundEffect(self)
                s.setLoopCount(1)
                s.setVolume(0.9)
                # QSoundEffect expects a file URL
                s.setSource(QtCore.QUrl.fromLocalFile(path))
                self._sfx[key] = s
        except Exception:
            # If QtMultimedia backend is missing, we will fall back to QApplication.beep().
            self._sfx = {}

    def _update_sound_button(self):
        if getattr(self, "_sound_enabled", True):
            self.btn_sound.setText("Sound ON")
            bg = BTN_GREEN
        else:
            self.btn_sound.setText("Sound OFF")
            bg = BTN_RED
        self.btn_sound.setStyleSheet(f"""
            QPushButton {{
                background: {bg};
                color: {BTN_TEXT};
                border: 0px;
                border-radius: 6px;
                padding: 4px 10px;
                font-weight: 900;
            }}
            QPushButton:hover {{ opacity: 0.95; }}
        """)

    @QtCore.Slot()
    def _toggle_sound(self):
        self._sound_enabled = not getattr(self, "_sound_enabled", True)
        self._cfg["sound_enabled"] = bool(self._sound_enabled)
        save_config(CONFIG_PATH, self._cfg)
        self._update_sound_button()
        # quick feedback
        if self._sound_enabled:
            QtWidgets.QApplication.beep()

    def _play_sound(self, key: str):
        if not getattr(self, "_sound_enabled", True):
            return
        s = self._sfx.get(key)
        try:
            if s is not None and s.source().isValid():
                s.stop()
                s.play()
            else:
                QtWidgets.QApplication.beep()
        except Exception:
            QtWidgets.QApplication.beep()

    def _handle_cme_sound(self, level: str):
        is_high = (str(level).strip().upper() == "HIGH")
        prev = self._alert_state.get("cme_high", False)
        if is_high and not prev:
            self._play_sound("cme_high")
        self._alert_state["cme_high"] = is_high

    def _handle_xray_sound(self, flux_w_m2: float):
        try:
            is_m99 = (isinstance(flux_w_m2, (int, float)) and math.isfinite(flux_w_m2) and flux_w_m2 >= 9.9e-5)
        except Exception:
            is_m99 = False
        prev = self._alert_state.get("xray_m99", False)
        if is_m99 and not prev:
            self._play_sound("xray_m99")
        self._alert_state["xray_m99"] = is_m99

    def _handle_proton_sound(self, s_level: str):
        s = str(s_level).strip().upper()
        n = -1
        if len(s) >= 2 and s[0] == "S" and s[1].isdigit():
            n = int(s[1])
        is_s4 = (n >= 4)
        prev = self._alert_state.get("proton_s4", False)
        if is_s4 and not prev:
            self._play_sound("proton_s4")
        self._alert_state["proton_s4"] = is_s4

    def _set_data_status(self, state: str, details: str | None = None):
        if state == "ok":
            self.lbl_status.setText("Data: OK")
            color = STATUS_OK
        elif state == "error":
            if details:
                short = details[:67] + "..." if len(details) > 70 else details
                self.lbl_status.setText(f"Data: ERROR ({short})")
            else:
                self.lbl_status.setText("Data: ERROR")
            color = STATUS_ERR
        else:
            if details:
                short = details[:67] + "..." if len(details) > 70 else details
                self.lbl_status.setText(f"Data: connecting… ({short})")
            else:
                self.lbl_status.setText("Data: connecting…")
            color = STATUS_WARN

        self.lbl_status.setStyleSheet(f"color: {color}; font-size: 12px;")

    def closeEvent(self, event):
        try:
            self._stop_dx_worker()
        except Exception:
            pass

        try:
            self._worker.stop()

            self._thread.quit()
            self._thread.wait(2000)
        except Exception:
            pass

        try:
            self._rss_worker.stop()
            self._rss_thread.quit()
            self._rss_thread.wait(2000)
        except Exception:
            pass

        super().closeEvent(event)

    def _load_logo(self, path: str):
        pm = QtGui.QPixmap(path)
        if pm.isNull():
            pm = QtGui.QPixmap(58, 58)
            pm.fill(QtGui.QColor("#333"))
        self.logo.setPixmap(pm)


    def eventFilter(self, obj, ev):
        # Easter egg trigger: triple-click on the SUMO logo within ~1.2s
        try:
            if obj is self.logo and ev.type() == QtCore.QEvent.MouseButtonPress:
                now = time.time()
                if (now - float(getattr(self, "_egg_last_click_ts", 0.0))) > 1.2:
                    self._egg_clicks = 0
                self._egg_last_click_ts = now
                self._egg_clicks = int(getattr(self, "_egg_clicks", 0)) + 1
                if self._egg_clicks >= 3:
                    self._egg_clicks = 0
                    self._open_easter_egg()
                    return True
        except Exception:
            pass
        return super().eventFilter(obj, ev)

    def _open_easter_egg(self):
        dlg = SumoEasterInvaders(self)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
    def _tick_utc(self):
        if getattr(self, "_time_mode", "utc") == "local":
            now = datetime.now().astimezone()
            label = "LOCAL"
        else:
            now = datetime.now(timezone.utc)
            label = "UTC"

        t = now.strftime("%H:%M:%S")
        d = now.strftime("%Y-%m-%d")
        self.lbl_utc_badge.setText(f"{label} {t}\n{d}")

    def _style_panel_title_default(self, panel: KpLikePanel):
        panel.lbl_title.setStyleSheet("")

    def _style_panel_title_color(self, panel: KpLikePanel, color: str):
        panel.lbl_title.setStyleSheet(
            f"color: {color}; font-size: 20px; font-weight: 700; letter-spacing: 0.5px;"
        )

    def _refresh_cme_from_db(self):
        series = db_load_cme_series(DB_PATH, limit=60)
        last = db_load_cme_last(DB_PATH)

        if last is None:
            level = "N/A"
            accent = cme_color(level)
            self._handle_cme_sound(level)
            self.panels["CME"].lbl_title.setText("CME ARRIVAL PROBABILITY [N/A]")
            self._style_panel_title_color(self.panels["CME"], accent)
            self.panels["CME"].lbl_title.setToolTip("NASA DONKI: no cached data yet.\nX axis: rolling 6-hour window (display-only).")
            self.panels["CME"].set_accent(accent)
            now = time.time()
            x = np.linspace(now - 60 * 60 * 6, now, num=max(2, len(series) if series.size else 2))
            y = series if series.size else np.array([np.nan, np.nan], dtype=float)
            self.panels["CME"].set_data(y, float("nan"), y_range=(0, 100), color=accent, x=x)
            return

        prob, level, ts_utc = last
        accent = cme_color(level)
        self._handle_cme_sound(level)
        self.panels["CME"].lbl_title.setText(f"CME ARRIVAL PROBABILITY [{level}]")
        self._style_panel_title_color(self.panels["CME"], accent)
        self.panels["CME"].lbl_title.setToolTip(f"""DONKI last update: {ts_utc}
Heuristic: Earth mention + analyses + ENLIL/impactList
Optimized: age-weighted (~5 days) + 'likely/may' nuance
Display: X axis is a rolling 6-hour window (evenly spaced points; DONKI event times are in the tooltip header)
Note: this is an index, not a physical probability.""")
        self.panels["CME"].set_accent(accent)

        now = time.time()
        x = np.linspace(now - 60 * 60 * 6, now, num=max(2, len(series) if series.size else 2))
        y = series if series.size else np.array([np.nan, np.nan], dtype=float)
        self.panels["CME"].set_data(y, prob, y_range=(0, 100), color=accent, x=x)

    @QtCore.Slot(str)
    def _on_rss_text(self, text: str):
        self.rss_ticker.setText(text)

    @QtCore.Slot(str)
    def _on_rss_error(self, msg: str):
        self.rss_ticker.setText(f"RSS error: {msg}")


    def _load_aurora_from_db(self):
        """Seed the in-memory Aurora history from SQLite (last 25h)."""
        try:
            now = time.time()
            win = float(getattr(self, "_aur_window_seconds", 25 * 3600))
            x, y_max, y_mean, y_area = db_load_aurora_since(DB_PATH, now - win)

            self._aur_ui_x.clear()
            self._aur_max_series.clear()
            self._aur_mean_series.clear()
            self._aur_area_series.clear()

            for xi, amax, amean, aarea in zip(x, y_max, y_mean, y_area):
                self._aur_ui_x.append(float(xi))
                self._aur_max_series.append(float(amax))
                self._aur_mean_series.append(float(amean))
                self._aur_area_series.append(float(aarea))
        except Exception:
            pass



    def _format_x_span(self, x: np.ndarray | None) -> str:
        """Human readable X-axis span for tooltips (UTC)."""
        try:
            if x is None:
                return "X axis: (no timeline data)"
            xx = np.array(x, dtype=float)
            xx = xx[np.isfinite(xx)]
            if xx.size < 2:
                return "X axis: (no timeline data)"
            xmin = float(np.min(xx))
            xmax = float(np.max(xx))
            dt = max(0.0, xmax - xmin)

            if dt < 3600:
                span = f"{dt/60.0:.0f} min"
            elif dt < 172800:
                span = f"{dt/3600.0:.1f} h"
            else:
                span = f"{dt/86400.0:.1f} d"

            s0 = datetime.fromtimestamp(xmin, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            s1 = datetime.fromtimestamp(xmax, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            return f"X axis: {span} (UTC, {s0} → {s1})"
        except Exception:
            return "X axis: (no timeline data)"

    def _apply_panel_tooltips(self, d: dict):
        """Update tooltips on panel titles (hover)."""
        # KP (NOAA planetary K-index)
        self.panels["KP"].lbl_title.setToolTip(
            "Planetary Kp index (0..9).\n"
            "Bars are time-stamped values from NOAA's planetary K-index product (typical cadence: 3h).\n"
            "G-scale (G0..G5) shown at right is derived from the latest Kp.\n"
            f"{self._format_x_span(d.get('kp_x', None))}"
        )

        # X-RAY (GOES)
        self.panels["XRAY"].lbl_title.setToolTip(
            "GOES X-ray flux (0.1–0.8 nm) in W/m².\n"
            "Flare class A/B/C/M/X label is computed from the latest flux value.\n"
            f"{self._format_x_span(d.get('xray_x', None))}"
        )

        # Solar wind speed (NOAA plasma 6-hour)
        self.panels["SW"].lbl_title.setToolTip(
            "Solar wind speed in km/s (NOAA plasma 6-hour product).\n"
            f"{self._format_x_span(d.get('sw_speed_x', None))}"
        )

        # BZ / BT (NOAA mag 6-hour)
        self.panels["BZBT"].lbl_title.setToolTip(
            "Interplanetary Magnetic Field (IMF) in nT (GSM).\n"
            "Bz: north/south component (negative = southward, more geoeffective).\n"
            "Bt: total magnetic field magnitude (may be computed from Bx/By/Bz if missing).\n"
            f"{self._format_x_span(d.get('mag_x', None))}"
        )

        # SFI / SSN (daily solar indices text)
        self.panels["SFI"].lbl_title.setToolTip(
            "Solar Flux Index (F10.7 cm) in SFU (Solar Flux Units).\n"
            "Daily value parsed from NOAA daily-solar-indices.txt.\n"
            "Higher SFI usually means better HF conditions.\n"
            f"{self._format_x_span(d.get('dsi_x', None))}"
        )
        self.panels["SSN"].lbl_title.setToolTip(
            "Sunspot Number (SSN).\n"
            "Daily value parsed from NOAA daily-solar-indices.txt.\n"
            f"{self._format_x_span(d.get('dsi_x', None))}"
        )

        # Protons P10 (GOES)
        self.panels["P10"].lbl_title.setToolTip(
            "Integral proton flux ≥10 MeV (P10) in PFU = particles / (cm²·s·sr).\n"
            "S-scale (S0..S5) shown at right is derived from the latest PFU value.\n"
            f"{self._format_x_span(d.get('p10_x', None))}"
        )

        # Aurora (OVATION snapshot + local rolling history)
        aur_now = d.get("aur_now", float("nan"))
        aur_mean = d.get("aur_mean_active", float("nan"))
        aur_area = d.get("aur_area_gt10", float("nan"))
        parts = []
        if not (isinstance(aur_now, float) and math.isnan(aur_now)):
            parts.append(f"Max: {aur_now:.1f}%")
        if not (isinstance(aur_mean, float) and math.isnan(aur_mean)):
            parts.append(f"Mean(active>0): {aur_mean:.1f}%")
        if not (isinstance(aur_area, float) and math.isnan(aur_area)):
            parts.append(f"Area(cells>10): {aur_area:.1f}%")

        self.panels["AUR"].lbl_title.setToolTip(
            "NOAA OVATION global auroral activity proxy.\n"
            + (("Current: " + " • ".join(parts) + "\n") if parts else "")
            + "The panel displays a local rolling history (last 60 refreshes, ~1 point/min).\n"
            + "Series: Max (%), Mean(active>0) (%), Area(cells>10) (%).\n"
            + f"{self._format_x_span(np.array(list(self._aur_ui_x), dtype=float) if hasattr(self, '_aur_ui_x') else None)}"
        )

        # CME tooltip is handled in _refresh_cme_from_db(), but keep a short hint here too
        # (so it exists even before DB has data)
        if not self.panels["CME"].lbl_title.toolTip():
            self.panels["CME"].lbl_title.setToolTip(
                "NASA DONKI CME-based arrival risk index (heuristic).\n"
                "Values are cached in the local SQLite DB.\n"
                "X axis: rolling 6-hour window (display-only)."
            )

    @QtCore.Slot(dict)
    def _on_data(self, d: dict):
        pe = (d.get("partial_error") or "").strip()
        if pe:
            self._set_data_status("connecting", pe)
        else:
            self._set_data_status("ok")

        kp_now = d.get("kp_now", float("nan"))
        kp_g = kp_g_scale(kp_now)
        kp_accent = kp_color(kp_now)

        self.panels["KP"].lbl_title.setText("K-INDEX (Planetary)")
        self._style_panel_title_default(self.panels["KP"])
        self.panels["KP"].set_accent(kp_accent, blink=False)
        kp_is_red = (kp_accent == "#ff4d4d")
        if self._kp_last_is_red is None:
            self._kp_last_is_red = kp_is_red
        elif (kp_is_red and not self._kp_last_is_red):
            # Entered red: blink BORDER ONLY for 5 minutes
            self.panels["KP"].start_blink(300, border_only=True)
            self._kp_last_is_red = True
        else:
            self._kp_last_is_red = kp_is_red
        self.panels["KP"].set_data(
            d.get("kp_series", np.array([])),
            kp_now,
            y_range=(0, 9),
            color=kp_accent,
            x=d.get("kp_x", None)
        )

        if isinstance(kp_now, float) and math.isnan(kp_now):
            self.panels["KP"].set_big_text("--")
        else:
            kp_txt = f"Kp {kp_now:.1f}"
            self.panels["KP"].lbl_big.setStyleSheet("")
            self.panels["KP"].lbl_big.setText(
                f'<span style="color:{FG_TEXT};">{kp_txt}</span>&nbsp;&nbsp;'
                f'<span style="color:{kp_accent}; font-weight:900;">{kp_g}</span>'
            )

        flare_label = d.get("xray_label", "?")
        flare_cls = d.get("xray_class", "?")
        xray_now = d.get("xray_now", float("nan"))
        self._handle_xray_sound(xray_now)
        xray_accent = xray_class_color(flare_cls)

        self.panels["XRAY"].lbl_title.setText("X-RAYS (GOES 0.1-0.8nm)")
        self._style_panel_title_default(self.panels["XRAY"])
        self.panels["XRAY"].set_accent(xray_accent)
        self.panels["XRAY"].set_data(
            d.get("xray_series", np.array([])),
            xray_now,
            color=xray_accent,
            x=d.get("xray_x", None)
        )

        if isinstance(xray_now, float) and math.isnan(xray_now):
            self.panels["XRAY"].set_big_text("--")
        else:
            flux_txt = f"{xray_now:.2e}"
            self.panels["XRAY"].lbl_big.setStyleSheet("")
            self.panels["XRAY"].lbl_big.setText(
                f'<span style="color:{FG_TEXT};">{flux_txt}</span>&nbsp;&nbsp;'
                f'<span style="color:{xray_accent}; font-weight:900;">{flare_label}</span>'
            )

        self.panels["SW"].lbl_title.setText("SOLAR WIND SPEED")
        self._style_panel_title_default(self.panels["SW"])
        sw_now = d.get("sw_speed_now", float("nan"))
        sw_accent = solar_wind_color(sw_now)
        self.panels["SW"].set_accent(sw_accent, blink=False)
        sw_is_red = (sw_accent == "#ff4d4d")
        if self._sw_last_is_red is None:
            self._sw_last_is_red = sw_is_red
        elif (sw_is_red and not self._sw_last_is_red):
            # Entered red: blink BORDER ONLY for 5 minutes
            self.panels["SW"].start_blink(300, border_only=True)
            self._sw_last_is_red = True
        else:
            self._sw_last_is_red = sw_is_red
        self.panels["SW"].set_data(
            d.get("sw_speed_series", np.array([])),
            sw_now,
            y_range=(0, 1200),
            color=sw_accent,
            x=d.get("sw_speed_x", None),
        )

        ssn_now = d.get("ssn_now", float("nan"))
        ssn_accent = ssn_color(ssn_now)
        self.panels["SSN"].lbl_title.setText("SSN (Sunspot Number) daily index")
        self._style_panel_title_default(self.panels["SSN"])
        self.panels["SSN"].set_accent(ssn_accent)
        self.panels["SSN"].set_data(
            d.get("ssn_series", np.array([])),
            ssn_now,
            y_range=(0, 300),
            color=ssn_accent,
            x=d.get("dsi_x", None),
        )

        sfi_now = d.get("sfi_now", float("nan"))

        # --- HF openings ribbon update (MUF + blackout + per-band chips) ---
        try:
            if hasattr(self, "hf_bar") and self.hf_bar is not None and getattr(self, "_hf_bar_visible", True) and self.hf_bar.isVisible():
                self.hf_bar.update_from_indices(sfi_now, kp_now, flare_cls, sw_now)
        except Exception:
            pass

        sfi_accent = sfi_color(sfi_now)
        self.panels["SFI"].lbl_title.setText("SFI (F10.7 cm Flux) daily index")
        self._style_panel_title_default(self.panels["SFI"])
        self.panels["SFI"].set_accent(sfi_accent)
        self.panels["SFI"].set_data(
            d.get("sfi_series", np.array([])),
            sfi_now,
            y_range=(50, 350),
            color=sfi_accent,
            x=d.get("dsi_x", None),
        )

        bz_s = np.array(d.get("bz_series", np.array([])), dtype=float)
        bt_s = np.array(d.get("bt_series", np.array([])), dtype=float)
        bz_now = d.get("bz_now", float("nan"))
        bt_now = d.get("bt_now", float("nan"))
        mag_x = np.array(d.get("mag_x", np.array([])), dtype=float)

        p = self.panels["BZBT"]
        p.lbl_title.setText("BZ / BT (Solar Wind MAG)")
        self._style_panel_title_default(p)
        p.set_accent(ACCENT_GREEN)

        if (isinstance(bz_now, float) and math.isnan(bz_now)) and (isinstance(bt_now, float) and math.isnan(bt_now)):
            p.set_big_text("--")
        else:
            bz_txt = "--" if (isinstance(bz_now, float) and math.isnan(bz_now)) else f"{bz_now:.1f}"
            bt_txt = "--" if (isinstance(bt_now, float) and math.isnan(bt_now)) else f"{bt_now:.1f}"
            p.lbl_big.setStyleSheet("")
            p.lbl_big.setText(
                f'<span style="color:{ACCENT_GREEN}; font-weight:900;">Bz {bz_txt}</span>'
                f'&nbsp;&nbsp;'
                f'<span style="color:{ACCENT_BLUE}; font-weight:900;">Bt {bt_txt}</span>'
            )

        p.plot.clear()
        x_bz, y_bz = align_xy(mag_x, bz_s)
        x_bt, y_bt = align_xy(mag_x, bt_s)

        if len(x_bz) <= 0 and len(x_bt) <= 0:
            now = time.time()
            p.plot.plot([now - 60, now], [np.nan, np.nan], pen=pg.mkPen(ACCENT_GREEN, width=2))
            p.plot.plot([now - 60, now], [np.nan, np.nan], pen=pg.mkPen(ACCENT_BLUE, width=2))
            p.plot.setXRange(now - 60, now)
            p.plot.setYRange(-1, 1)
        else:
            if len(x_bz):
                p.plot.plot(x_bz, y_bz, pen=pg.mkPen(ACCENT_GREEN, width=2))
            if len(x_bt):
                p.plot.plot(x_bt, y_bt, pen=pg.mkPen(ACCENT_BLUE, width=2))

            x_all = np.concatenate([x_bz, x_bt]) if (len(x_bz) and len(x_bt)) else (x_bz if len(x_bz) else x_bt)
            if len(x_all):
                p.plot.setXRange(float(np.min(x_all)), float(np.max(x_all)))

            yy = []
            if len(y_bz):
                yy.append(y_bz[np.isfinite(y_bz)])
            if len(y_bt):
                yy.append(y_bt[np.isfinite(y_bt)])
            if yy:
                yy = np.concatenate(yy) if len(yy) > 1 else yy[0]
                if yy.size:
                    y_min = float(np.min(yy))
                    y_max = float(np.max(yy))
                    pad = (y_max - y_min) * 0.15 if y_max != y_min else 1.0
                    p.plot.setYRange(y_min - pad, y_max + pad)
                else:
                    p.plot.setYRange(-1, 1)
            else:
                p.plot.setYRange(-1, 1)

        p10_now = d.get("p10_now", float("nan"))
        s_level = d.get("s_level", "S?")
        self._handle_proton_sound(s_level)
        p_accent = s_level_color(s_level)

        self.panels["P10"].lbl_title.setText("PROTONS P10 (>=10MeV)")
        self._style_panel_title_default(self.panels["P10"])
        self.panels["P10"].set_accent(p_accent)
        self.panels["P10"].set_data(
            d.get("p10_series", np.array([])),
            p10_now,
            y_range=(0, 200),
            color=p_accent,
            x=d.get("p10_x", None),
        )

        if isinstance(p10_now, float) and math.isnan(p10_now):
            self.panels["P10"].set_big_text("--")
        else:
            p10_txt = f"{p10_now:.2f} PFU"
            self.panels["P10"].lbl_big.setStyleSheet("")
            self.panels["P10"].lbl_big.setText(
                f'<span style="color:{FG_TEXT};">{p10_txt}</span>&nbsp;&nbsp;'
                f'<span style="color:{p_accent}; font-weight:900;">{s_level}</span>'
            )

        aur_now = d.get("aur_now", float("nan"))
        aur_mean = d.get("aur_mean_active", float("nan"))
        aur_area = d.get("aur_area_gt10", float("nan"))
        aur_ts = float(d.get("aur_ts", time.time()))

        p_aur = self.panels["AUR"]
        p_aur.lbl_title.setText("AURORA")
        self._style_panel_title_default(p_aur)
        p_aur.set_accent(ACCENT_GREEN)

        # Tooltip: show the 3 metrics (Max / Mean(active>0) / Area>10)
        tip_parts = []
        if not (isinstance(aur_now, float) and math.isnan(aur_now)):
            tip_parts.append(f"Max: {aur_now:.1f}%")
        if not (isinstance(aur_mean, float) and math.isnan(aur_mean)):
            tip_parts.append(f"Mean(active>0): {aur_mean:.1f}%")
        if not (isinstance(aur_area, float) and math.isnan(aur_area)):
            tip_parts.append(f"Area(cells>10): {aur_area:.1f}%")
        p_aur.lbl_title.setToolTip(" summary" + (" • " + " • ".join(tip_parts) if tip_parts else ""))

        # Keep a small in-UI history (no extra network calls)# Keep a rolling in-UI history over the last 25 hours (no extra network calls)
        if not (isinstance(aur_now, float) and math.isnan(aur_now)):
            self._aur_ui_x.append(aur_ts)
            self._aur_max_series.append(float(aur_now))
            self._aur_mean_series.append(float(aur_mean) if not (isinstance(aur_mean, float) and math.isnan(aur_mean)) else float("nan"))
            self._aur_area_series.append(float(aur_area) if not (isinstance(aur_area, float) and math.isnan(aur_area)) else float("nan"))
            # Persist aurora sample (for 25h rolling history across restarts)
            try:
                db_insert_aurora(DB_PATH, aur_ts, aur_now, aur_mean, aur_area)
            except Exception:
                pass


        # Purge samples older than the rolling window
        try:
            win = float(getattr(self, "_aur_window_seconds", 25 * 3600))
            while self._aur_ui_x and (aur_ts - float(self._aur_ui_x[0])) > win:
                self._aur_ui_x.popleft()
                if self._aur_max_series: self._aur_max_series.popleft()
                if self._aur_mean_series: self._aur_mean_series.popleft()
                if self._aur_area_series: self._aur_area_series.popleft()
        except Exception:
            pass
        # --- Multi-series plot (Solution 2) + dynamic Y zoom (Solution 1) ---
        p_aur.plot.clear()

        x_aur = np.array(list(self._aur_ui_x), dtype=float)
        y_max = np.array(list(self._aur_max_series), dtype=float)
        y_mean = np.array(list(self._aur_mean_series), dtype=float)
        y_area = np.array(list(self._aur_area_series), dtype=float)

        x_max, y_max = align_xy(x_aur, y_max)
        x_mean, y_mean = align_xy(x_aur, y_mean)
        x_area, y_area = align_xy(x_aur, y_area)

        # Plot each series (different colors)
        if len(x_max):
            p_aur.plot.plot(x_max, y_max, pen=pg.mkPen(ACCENT_GREEN, width=2))   # Max
        if len(x_mean):
            p_aur.plot.plot(x_mean, y_mean, pen=pg.mkPen(ACCENT_BLUE, width=2))  # Mean(active>0)
        if len(x_area):
            p_aur.plot.plot(x_area, y_area, pen=pg.mkPen(STATUS_WARN, width=2))  # Area(cells>10)

        # X range# X range: force last 25 hours (rolling window)
        now = time.time()
        win = float(getattr(self, "_aur_window_seconds", 25 * 3600))
        x0 = now - win
        x_all = np.concatenate([x_max, x_mean, x_area]) if (len(x_max) or len(x_mean) or len(x_area)) else np.array([], dtype=float)
        if x_all.size:
            # Keep the view pinned to the window even if data span is smaller
            xmin = float(np.min(x_all))
            xmax = float(np.max(x_all))
            p_aur.plot.setXRange(min(x0, xmin), max(now, xmax))
        else:
            p_aur.plot.plot([now - 60, now], [np.nan, np.nan], pen=pg.mkPen(ACCENT_GREEN, width=2))
            p_aur.plot.setXRange(x0, now)
# Dynamic Y zoom (based on recent finite values, clamped 0..100)
        yy_parts = []
        if y_max.size:
            yy_parts.append(y_max[np.isfinite(y_max)])
        if y_mean.size:
            yy_parts.append(y_mean[np.isfinite(y_mean)])
        if y_area.size:
            yy_parts.append(y_area[np.isfinite(y_area)])

        if yy_parts and any(arr.size for arr in yy_parts):
            yy = np.concatenate([arr for arr in yy_parts if arr.size])
            y_min = float(np.min(yy))
            y_maxv = float(np.max(yy))

            # Ensure we still see something even if it's almost flat
            span = max(8.0, (y_maxv - y_min))
            center = (y_maxv + y_min) / 2.0
            low = center - span / 2.0
            high = center + span / 2.0

            # Add some padding
            pad = span * 0.15
            low -= pad
            high += pad

            # Clamp to valid percent range
            low = max(0.0, low)
            high = min(100.0, high)
            if high - low < 4.0:
                # last safety
                low = max(0.0, center - 2.0)
                high = min(100.0, center + 2.0)

            p_aur.plot.setYRange(low, high)
        else:
            p_aur.plot.setYRange(0, 100)

        # Big label: show Max / Mean / Area in one glance
        def _fmt_pct(v: float) -> str:
            return "--" if (isinstance(v, float) and math.isnan(v)) else f"{v:.0f}%"

        p_aur.lbl_big.setStyleSheet("")
        p_aur.lbl_big.setText(
            f'<span style="color:{ACCENT_GREEN}; font-size:21px; font-weight:900;">Max {_fmt_pct(aur_now)}</span>'
            f'&nbsp;&nbsp;'
            f'<span style="color:{ACCENT_BLUE}; font-size:21px; font-weight:900;">Mean {_fmt_pct(aur_mean)}</span>'
            f'&nbsp;&nbsp;'
            f'<span style="color:{STATUS_WARN}; font-size:21px; font-weight:900;">Area {_fmt_pct(aur_area)}</span>'
        )

        if "cme_prob" in d and "cme_level" in d:
            self._refresh_cme_from_db()

        # Update tooltips (hover on panel titles)
        self._apply_panel_tooltips(d)

    @QtCore.Slot(str)
    def _on_error(self, msg: str):
        low = msg.lower()
        if "donki" in low or "429" in low or "rate limit" in low:
            self._set_data_status("connecting", msg)
            self._refresh_cme_from_db()
        else:
            self._set_data_status("error", msg)
        print("Worker error:", msg)


# =====================================================
# STYLES
# =====================================================
def apply_styles(app: QtWidgets.QApplication):
    app.setStyle("Fusion")
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(BG_APP))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(FG_TEXT))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(BG_PANEL))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(BG_APP))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor(FG_TEXT))
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor("#121a22"))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(FG_TEXT))
    app.setPalette(palette)

    app.setStyleSheet(f"""
        QMainWindow {{
            background: {BG_APP};
        }}
        #topbar {{
            background: #101820;
            border: 1px solid {BORDER};
            border-radius: 6px;
        }}
        QLabel#appTitle {{
            color: {FG_TEXT};
            font-size: 26px;
            font-weight: 800;
            letter-spacing: 1px;
        }}
        QLabel#appSubTitle {{
            color: #aab6c5;
            font-size: 13px;
        }}
        QLabel#appVersion {{
            color: #7f8c8d;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 1px;
        }}
        QLabel#statusText {{
            color: #c7d1df;
            font-size: 12px;
        }}
        QFrame#panel {{
            background: {BG_PANEL};
            border: 1px solid {BORDER};
            border-radius: 6px;
        }}
        QLabel#panelTitle {{
            color: #cfd7e3;
            font-size: 20px;
            font-weight: 700;
            letter-spacing: 0.5px;
        }}
        QLabel#panelBig {{
            color: {FG_TEXT};
            font-size: 28px;
            font-weight: 900;
        }}
    """)


# =====================================================
# MAIN
# =====================================================
def main():
    if not getattr(sys, "frozen", False):
        ASSETS_DIR.mkdir(exist_ok=True)
    db_init(DB_PATH)

    app = QtWidgets.QApplication(sys.argv)
    apply_styles(app)

    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
