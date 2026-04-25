#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import traceback
import faulthandler
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

APP_VERSION = "v7.9.7"

SUMO_DEBUG_LOG_PATH = Path.cwd() / "sumo_debug_anti_crash.log"
SUMO_DEBUG_FAULT_PATH = Path.cwd() / "sumo_fault_anti_crash.log"
SUMO_RUNTIME_LOG_PATH = Path.cwd() / "sumo_runtime_state.log"
SUMO_DEFAULT_DEBUG_MODE = True
SUMO_DEFAULT_DEBUG_HARDCORE = True

def safe_debug_enabled(cfg: Optional[dict] = None) -> bool:
    try:
        if isinstance(cfg, dict) and "debug_mode" in cfg:
            return bool(cfg.get("debug_mode", True))
    except Exception:
        pass
    return SUMO_DEFAULT_DEBUG_MODE

def safe_debug_hardcore(cfg: Optional[dict] = None) -> bool:
    try:
        if isinstance(cfg, dict) and "debug_hardcore" in cfg:
            return bool(cfg.get("debug_hardcore", True))
    except Exception:
        pass
    return SUMO_DEFAULT_DEBUG_HARDCORE

def rotate_log(path: Path, max_size: int = 2_000_000) -> None:
    """Rotate a log file before it grows too large. Keeps one .old.log backup."""
    try:
        if path.exists() and path.stat().st_size > int(max_size):
            backup = path.with_suffix(".old.log")
            try:
                if backup.exists():
                    backup.unlink()
            except Exception:
                pass
            path.rename(backup)
    except Exception:
        pass


def runtime_log(message: str) -> None:
    try:
        rotate_log(SUMO_RUNTIME_LOG_PATH)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {message}"
        with SUMO_RUNTIME_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def debug_log(message: str) -> None:
    try:
        rotate_log(SUMO_DEBUG_LOG_PATH)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {message}"
        print(line)
        with SUMO_DEBUG_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


_SUMO_FAULT_FILE = None

def install_debug_hooks() -> None:
    global _SUMO_FAULT_FILE
    try:
        _SUMO_FAULT_FILE = open(SUMO_DEBUG_FAULT_PATH, "a", encoding="utf-8")
        faulthandler.enable(file=_SUMO_FAULT_FILE, all_threads=True)
    except Exception:
        pass

    def _excepthook(exc_type, exc_value, exc_tb):
        try:
            debug_log("UNCAUGHT EXCEPTION START")
            for line in traceback.format_exception(exc_type, exc_value, exc_tb):
                for sub in str(line).rstrip().splitlines():
                    debug_log(sub)
            debug_log("UNCAUGHT EXCEPTION END")
        except Exception:
            pass
        try:
            sys.__excepthook__(exc_type, exc_value, exc_tb)
        except Exception:
            pass

    sys.excepthook = _excepthook


import sys
import math
import time
import sqlite3
import xml.etree.ElementTree as ET
import json
import base64
import zlib
import socket
import re
import statistics
import threading
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timezone, date, timedelta
from pathlib import Path
from collections import deque
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from urllib.parse import urljoin
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional, Tuple


def parse_http_date(value: str | None) -> datetime | None:
    try:
        if not value:
            return None
        dt = parsedate_to_datetime(str(value).strip())
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None

import numpy as np
import requests
from PySide6 import QtCore, QtGui, QtWidgets, QtSvg
from PySide6.QtMultimedia import QSoundEffect
import pyqtgraph as pg

try:
    import shiboken6
except Exception:
    shiboken6 = None

def _qt_obj_alive(obj) -> bool:
    try:
        if obj is None:
            return False
        if shiboken6 is not None:
            try:
                if not shiboken6.isValid(obj):
                    return False
            except Exception:
                return False
        return True
    except Exception:
        return False

try:
    from skyfield.api import Loader as SkyfieldLoader, EarthSatellite, wgs84
    from skyfield.framelib import ecliptic_frame
    SKYFIELD_AVAILABLE = True
    SKYFIELD_IMPORT_ERROR = ""
except Exception as _skyfield_exc:
    SkyfieldLoader = None  # type: ignore[assignment]
    EarthSatellite = None  # type: ignore[assignment]
    wgs84 = None  # type: ignore[assignment]
    ecliptic_frame = None  # type: ignore[assignment]
    SKYFIELD_AVAILABLE = False
    SKYFIELD_IMPORT_ERROR = str(_skyfield_exc)

def safe_zoneinfo(name: str, fallback: timezone | None = None):
    fallback = fallback or timezone.utc
    try:
        return ZoneInfo(str(name or "UTC"))
    except Exception:
        return fallback


def strict_zoneinfo(name: str):
    try:
        tz_name = str(name or "").strip()
        if not tz_name:
            return None
        return ZoneInfo(tz_name)
    except Exception:
        return None


def convert_clock_datetime(dt: datetime, timezone_name: str | None) -> datetime:
    try:
        base_dt = dt if isinstance(dt, datetime) else datetime.now(timezone.utc)
        if base_dt.tzinfo is None:
            base_dt = base_dt.replace(tzinfo=timezone.utc)
    except Exception:
        base_dt = datetime.now(timezone.utc)

    tz_name = str(timezone_name or "").strip()
    if not tz_name:
        return base_dt

    try:
        tz = strict_zoneinfo(tz_name)
        if tz is not None:
            return base_dt.astimezone(tz)
    except Exception:
        pass

    try:
        qtz = QtCore.QTimeZone(bytes(tz_name, "utf-8"))
        if qtz.isValid():
            epoch = int(base_dt.astimezone(timezone.utc).timestamp())
            qdt_utc = QtCore.QDateTime.fromSecsSinceEpoch(epoch, QtCore.QTimeZone(b"UTC"))
            qdt_city = qdt_utc.toTimeZone(qtz)
            py = qdt_city.toPython()
            if isinstance(py, datetime):
                if py.tzinfo is None:
                    return py.replace(tzinfo=timezone.utc)
                return py
    except Exception:
        pass

    return base_dt.astimezone(timezone.utc)


class KC2GMufMapBuilderV2:
    """
    V2.3:
    - interpolation IDW un peu plus tolérante
    - second pass de gap fill avec rayon élargi
    - lissage gaussien léger mais plus efficace
    - confidence grid
    - export JSON + preview PNG optionnelle
    """

    def __init__(
        self,
        input_points_path: str,
        output_grid_path: str,
        output_png_path: Optional[str] = None,
    ) -> None:
        self.input_points_path = Path(input_points_path)
        self.output_grid_path = Path(output_grid_path)
        self.output_png_path = Path(output_png_path) if output_png_path else None

    def load_points(self) -> List[Dict]:
        if not self.input_points_path.exists():
            raise FileNotFoundError(f"Fichier introuvable: {self.input_points_path}")

        data = json.loads(self.input_points_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("Le fichier points doit contenir une liste JSON.")

        points = []
        for item in data:
            lat = item.get("lat")
            lon = item.get("lon")
            muf = item.get("muf")
            if lat is None or lon is None or muf is None:
                continue

            points.append(
                {
                    "ursi": item.get("ursi"),
                    "station_name": item.get("station_name"),
                    "lat": float(lat),
                    "lon": float(lon),
                    "muf": float(muf),
                    "cs": float(item.get("cs")) if item.get("cs") is not None else None,
                    "age_minutes": float(item.get("age_minutes")) if item.get("age_minutes") is not None else None,
                    "score": float(item.get("score")) if item.get("score") is not None else None,
                }
            )
        return points

    @staticmethod
    def normalize_lon(lon: float) -> float:
        while lon > 180:
            lon -= 360
        while lon < -180:
            lon += 360
        return lon

    @staticmethod
    def great_circle_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        r = 6371.0
        lat1r = math.radians(lat1)
        lon1r = math.radians(lon1)
        lat2r = math.radians(lat2)
        lon2r = math.radians(lon2)

        dlat = lat2r - lat1r
        dlon = lon2r - lon1r

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return r * c

    @staticmethod
    def point_weight(point: Dict) -> float:
        score = point.get("score")
        cs = point.get("cs")
        age = point.get("age_minutes")

        base = float(score) if score is not None else float(cs) if cs is not None else 50.0
        freshness_bonus = 0.0
        if age is not None:
            freshness_bonus = max(0.0, 45.0 - float(age)) * 0.45

        return max(1.0, base + freshness_bonus)

    @staticmethod
    def _nanmean(values: List[float]) -> float:
        valid = [float(v) for v in values if not np.isnan(v)]
        if not valid:
            return np.nan
        return float(sum(valid) / len(valid))

    def interpolate_idw(
        self,
        points: List[Dict],
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        lat_step: float,
        lon_step: float,
        power: float = 2.25,
        max_distance_km: float = 3200.0,
        min_points: int = 2,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        lats = np.arange(lat_min, lat_max + 1e-9, lat_step)
        lons = np.arange(lon_min, lon_max + 1e-9, lon_step)

        muf_grid = np.full((len(lats), len(lons)), np.nan, dtype=float)
        contrib_grid = np.zeros((len(lats), len(lons)), dtype=int)
        confidence_grid = np.full((len(lats), len(lons)), np.nan, dtype=float)

        for i, grid_lat in enumerate(lats):
            for j, grid_lon in enumerate(lons):
                weighted_sum = 0.0
                weight_sum = 0.0
                contributor_count = 0
                direct_value = None
                direct_conf = None
                conf_acc = 0.0

                for pt in points:
                    dist_km = self.great_circle_distance_km(
                        grid_lat,
                        grid_lon,
                        pt["lat"],
                        self.normalize_lon(pt["lon"]),
                    )

                    if dist_km > max_distance_km:
                        continue

                    station_weight = self.point_weight(pt)
                    local_conf = min(100.0, station_weight)

                    if dist_km < 25.0:
                        direct_value = pt["muf"]
                        direct_conf = local_conf
                        contributor_count = 1
                        break

                    w = station_weight / max(1.0, dist_km ** power)
                    weighted_sum += pt["muf"] * w
                    weight_sum += w
                    conf_acc += local_conf * w
                    contributor_count += 1

                if direct_value is not None:
                    muf_grid[i, j] = direct_value
                    contrib_grid[i, j] = contributor_count
                    confidence_grid[i, j] = direct_conf
                    continue

                if contributor_count >= min_points and weight_sum > 0.0:
                    muf_grid[i, j] = weighted_sum / weight_sum
                    contrib_grid[i, j] = contributor_count
                    confidence_grid[i, j] = conf_acc / weight_sum

        return lats, lons, muf_grid, contrib_grid, confidence_grid

    def fill_gaps_local(
        self,
        grid: np.ndarray,
        passes: int = 3,
        min_neighbors: int = 2,
    ) -> np.ndarray:
        out = grid.copy()
        rows, cols = out.shape

        for _ in range(passes):
            updated = out.copy()
            changed = False
            for i in range(rows):
                for j in range(cols):
                    if not np.isnan(out[i, j]):
                        continue

                    neighbors = []
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            if di == 0 and dj == 0:
                                continue
                            ni = i + di
                            nj = j + dj
                            if 0 <= ni < rows and 0 <= nj < cols and not np.isnan(out[ni, nj]):
                                neighbors.append(float(out[ni, nj]))

                    if len(neighbors) >= min_neighbors:
                        updated[i, j] = float(sum(neighbors) / len(neighbors))
                        changed = True
            out = updated
            if not changed:
                break

        return out

    def fill_gaps_radius(
        self,
        grid: np.ndarray,
        passes: int = 2,
        radius: int = 3,
        min_neighbors: int = 4,
        sigma: float = 1.8,
    ) -> np.ndarray:
        out = grid.copy()
        rows, cols = out.shape

        for _ in range(passes):
            updated = out.copy()
            changed = False

            for i in range(rows):
                for j in range(cols):
                    if not np.isnan(out[i, j]):
                        continue

                    weighted_sum = 0.0
                    weight_sum = 0.0
                    contributors = 0

                    for di in range(-radius, radius + 1):
                        for dj in range(-radius, radius + 1):
                            if di == 0 and dj == 0:
                                continue
                            ni = i + di
                            nj = (j + dj) % cols  # wrap longitude
                            if not (0 <= ni < rows):
                                continue
                            value = out[ni, nj]
                            if np.isnan(value):
                                continue

                            dist2 = float(di * di + dj * dj)
                            if dist2 <= 0.0:
                                continue
                            w = math.exp(-dist2 / max(0.25, sigma))
                            weighted_sum += float(value) * w
                            weight_sum += w
                            contributors += 1

                    if contributors >= min_neighbors and weight_sum > 0.0:
                        updated[i, j] = weighted_sum / weight_sum
                        changed = True

            out = updated
            if not changed:
                break

        return out

    def smooth_grid(
        self,
        grid: np.ndarray,
        kernel_size: int = 5,
        passes: int = 1,
    ) -> np.ndarray:
        if kernel_size not in (3, 5):
            raise ValueError("kernel_size doit être 3 ou 5")

        if kernel_size == 3:
            kernel = np.array(
                [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
                dtype=float,
            )
        else:
            kernel = np.array(
                [
                    [1.0, 4.0, 6.0, 4.0, 1.0],
                    [4.0, 16.0, 24.0, 16.0, 4.0],
                    [6.0, 24.0, 36.0, 24.0, 6.0],
                    [4.0, 16.0, 24.0, 16.0, 4.0],
                    [1.0, 4.0, 6.0, 4.0, 1.0],
                ],
                dtype=float,
            )

        out = grid.copy()
        pad = kernel_size // 2

        for _ in range(passes):
            padded = np.pad(out, ((pad, pad), (0, 0)), mode="edge")
            padded = np.concatenate([padded[:, -pad:], padded, padded[:, :pad]], axis=1)
            smoothed = out.copy()

            for i in range(out.shape[0]):
                for j in range(out.shape[1]):
                    window = padded[i : i + kernel_size, j : j + kernel_size]
                    valid_mask = ~np.isnan(window)
                    if not np.any(valid_mask):
                        continue
                    k = kernel * valid_mask
                    denom = float(k.sum())
                    if denom <= 0.0:
                        continue
                    smoothed[i, j] = float(np.nansum(window * k) / denom)

            out = smoothed

        return out


    def _local_nanmean_radius(self, grid: np.ndarray, i: int, j: int, radius: int = 1) -> float:
        values = []
        rows, cols = grid.shape
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                ni = i + di
                nj = (j + dj) % cols
                if not (0 <= ni < rows):
                    continue
                v = grid[ni, nj]
                if not np.isnan(v):
                    values.append(float(v))
        if not values:
            return np.nan
        return float(sum(values) / len(values))

    def postprocess_visual_hf(
        self,
        grid: np.ndarray,
        points: List[Dict],
        lats: np.ndarray,
        lons: np.ndarray,
        *,
        region: str,
        confidence_grid: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        V2.3:
        - limite les plateaux saturés quand peu de points sont disponibles
        - réduit l'influence très longue distance
        - resserre légèrement les hauts MUF autour des zones les mieux contraintes
        """
        if grid.size == 0:
            return grid

        out = grid.copy()
        if not points:
            return out

        density_factor = min(1.0, len(points) / (18.0 if region == "europe" else 40.0))
        max_delta = 7.5 + 3.5 * density_factor
        blend_strength = 0.14 if region == "world" else 0.18
        near_km = 1600.0 if region == "world" else 700.0
        falloff_km = 5200.0 if region == "world" else 1400.0

        norm_points = [
            {
                **p,
                "lon": self.normalize_lon(float(p["lon"])),
                "muf": float(p["muf"]),
            }
            for p in points
        ]

        rows, cols = out.shape
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                value = out[i, j]
                if np.isnan(value):
                    continue

                local_mean = self._local_nanmean_radius(out, i, j, radius=1)
                if np.isnan(local_mean):
                    local_mean = float(value)

                nearest_km = min(
                    self.great_circle_distance_km(float(lat), float(lon), p["lat"], p["lon"])
                    for p in norm_points
                )

                # Atténuation douce quand on extrapole loin de toute station.
                if nearest_km > near_km:
                    attenuation = math.exp(-(nearest_km - near_km) / falloff_km)
                else:
                    attenuation = 1.0

                # Réduit un peu les amplitudes quand la densité globale est faible.
                density_scale = 0.96 + 0.04 * density_factor

                # Clamp haut pour casser les grands plateaux rouges uniformes.
                upper_cap = local_mean + max_delta
                new_value = min(float(value), upper_cap)

                # Léger recentrage vers la moyenne locale pour redonner du relief.
                new_value = (1.0 - blend_strength) * new_value + blend_strength * local_mean

                # Si on a une grille de confiance, on atténue légèrement les zones peu contraintes.
                if confidence_grid is not None and confidence_grid.shape == out.shape:
                    conf = confidence_grid[i, j]
                    if not np.isnan(conf):
                        conf_scale = 0.96 + 0.04 * max(0.0, min(1.0, float(conf) / 100.0))
                    else:
                        conf_scale = 0.97
                else:
                    conf_scale = 1.0

                out[i, j] = max(0.0, new_value * attenuation * density_scale * conf_scale)

        return out

    def _build_payload(
        self,
        region: str,
        points: List[Dict],
        lats: np.ndarray,
        lons: np.ndarray,
        muf_grid_raw: np.ndarray,
        muf_grid_filled: np.ndarray,
        muf_grid_smoothed: np.ndarray,
        contributors_grid: np.ndarray,
        confidence_grid: np.ndarray,
        lat_step: float,
        lon_step: float,
        power: float,
        max_distance_km: float,
        min_points: int,
        gap_fill_passes: int,
        smoothing_passes: int,
        smoothing_kernel_size: int,
    ) -> Dict:
        return {
            "region": region,
            "generated_at": datetime.now(UTC).isoformat(),
            "point_count": len(points),
            "lat_step": lat_step,
            "lon_step": lon_step,
            "power": power,
            "max_distance_km": max_distance_km,
            "min_points": min_points,
            "gap_fill_passes": gap_fill_passes,
            "smoothing_passes": smoothing_passes,
            "smoothing_kernel_size": smoothing_kernel_size,
            "source_points_path": str(self.input_points_path),
            "stations": points,
            "latitudes": lats.tolist(),
            "longitudes": lons.tolist(),
            "muf_grid_raw": np.round(muf_grid_raw, 2).tolist(),
            "muf_grid_filled": np.round(muf_grid_filled, 2).tolist(),
            "muf_grid_smoothed": np.round(muf_grid_smoothed, 2).tolist(),
            "contributors_grid": contributors_grid.tolist(),
            "confidence_grid": np.round(confidence_grid, 2).tolist(),
        }

    def _build_region_grid(
        self,
        *,
        region: str,
        points: List[Dict],
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        lat_step: float,
        lon_step: float,
        power: float,
        max_distance_km: float,
        min_points: int,
        gap_fill_passes: int,
        smoothing_passes: int,
        smoothing_kernel_size: int,
    ) -> Dict:
        lats, lons, raw_grid, contrib_grid, conf_grid = self.interpolate_idw(
            points=points,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_step=lat_step,
            lon_step=lon_step,
            power=power,
            max_distance_km=max_distance_km,
            min_points=min_points,
        )

        filled_local = self.fill_gaps_local(raw_grid, passes=gap_fill_passes, min_neighbors=2)
        filled_radius = self.fill_gaps_radius(
            filled_local,
            passes=max(2, gap_fill_passes // 2),
            radius=3,
            min_neighbors=4,
            sigma=2.3 if region == "europe" else 2.5,
        )
        smoothed = self.smooth_grid(
            filled_radius,
            kernel_size=smoothing_kernel_size,
            passes=smoothing_passes,
        )
        smoothed = self.postprocess_visual_hf(
            smoothed,
            points,
            lats,
            lons,
            region=region,
            confidence_grid=conf_grid,
        )

        payload = self._build_payload(
            region=region,
            points=points,
            lats=lats,
            lons=lons,
            muf_grid_raw=raw_grid,
            muf_grid_filled=filled_radius,
            muf_grid_smoothed=smoothed,
            contributors_grid=contrib_grid,
            confidence_grid=conf_grid,
            lat_step=lat_step,
            lon_step=lon_step,
            power=power,
            max_distance_km=max_distance_km,
            min_points=min_points,
            gap_fill_passes=gap_fill_passes,
            smoothing_passes=smoothing_passes,
            smoothing_kernel_size=smoothing_kernel_size,
        )
        self.output_grid_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return payload

    def build_world_grid(
        self,
        lat_step: float = 4.0,
        lon_step: float = 4.0,
        power: float = 2.0,
        max_distance_km: float = 5200.0,
        min_points: int = 2,
        gap_fill_passes: int = 4,
        smoothing_passes: int = 2,
        smoothing_kernel_size: int = 3,
    ) -> Dict:
        points = self.load_points()
        return self._build_region_grid(
            region="world",
            points=points,
            lat_min=-60.0,
            lat_max=80.0,
            lon_min=-180.0,
            lon_max=180.0,
            lat_step=lat_step,
            lon_step=lon_step,
            power=power,
            max_distance_km=max_distance_km,
            min_points=min_points,
            gap_fill_passes=gap_fill_passes,
            smoothing_passes=smoothing_passes,
            smoothing_kernel_size=smoothing_kernel_size,
        )

    def build_europe_grid(
        self,
        lat_step: float = 1.5,
        lon_step: float = 1.5,
        power: float = 2.35,
        max_distance_km: float = 2200.0,
        min_points: int = 2,
        gap_fill_passes: int = 3,
        smoothing_passes: int = 3,
        smoothing_kernel_size: int = 5,
    ) -> Dict:
        points = self.load_points()
        europe_points = [
            p for p in points if 30.0 <= p["lat"] <= 72.0 and -15.0 <= self.normalize_lon(p["lon"]) <= 40.0
        ]
        return self._build_region_grid(
            region="europe",
            points=europe_points,
            lat_min=30.0,
            lat_max=72.0,
            lon_min=-15.0,
            lon_max=40.0,
            lat_step=lat_step,
            lon_step=lon_step,
            power=power,
            max_distance_km=max_distance_km,
            min_points=min_points,
            gap_fill_passes=gap_fill_passes,
            smoothing_passes=smoothing_passes,
            smoothing_kernel_size=smoothing_kernel_size,
        )

    def save_preview_png(self, payload: Dict, layer: str = "muf_grid_smoothed") -> Optional[Path]:
        if self.output_png_path is None:
            return None

        try:
            import matplotlib.pyplot as plt
        except Exception:
            return None

        try:
            # Validate required keys in payload
            if not isinstance(payload, dict):
                return None
            if "latitudes" not in payload or "longitudes" not in payload:
                return None
            if layer not in payload:
                return None
            if "region" not in payload:
                return None

            lats = np.array(payload["latitudes"], dtype=float)
            lons = np.array(payload["longitudes"], dtype=float)
            grid = np.array(payload[layer], dtype=float)

            from matplotlib.colors import LinearSegmentedColormap

            cmap = LinearSegmentedColormap.from_list(
                "muf_thermal",
                [
                    (0.0, "#07172f"),
                    (0.15, "#1f5b80"),
                    (0.35, "#3fb98b"),
                    (0.55, "#f7f066"),
                    (0.75, "#ff9e38"),
                    (1.0, "#e63900"),
                ],
            )

            fig, ax = plt.subplots(figsize=(12, 6))
            im = ax.imshow(
                grid,
                origin="lower",
                aspect="auto",
                extent=[lons.min(), lons.max(), lats.min(), lats.max()],
                cmap=cmap,
                vmin=0.0,
                vmax=35.0,
            )
            ax.set_title(f"SUMO MUF Map V2.3 ({payload['region']})")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

            stations = payload.get("stations", [])
            xs = [self.normalize_lon(s["lon"]) for s in stations if isinstance(s, dict) and "lon" in s and "lat" in s]
            ys = [s["lat"] for s in stations if isinstance(s, dict) and "lon" in s and "lat" in s]
            ax.scatter(xs, ys, s=18)

            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Estimated MUF (MHz)")

            self.output_png_path.parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout()
            fig.savefig(self.output_png_path, dpi=140)
            plt.close(fig)
            return self.output_png_path
        except Exception:
            try:
                plt.close("all")
            except Exception:
                pass
            return None


# KC2G grid builder debug bootstrap removed from app startup.
# Use a dedicated external/debug script if you want to regenerate giro_debug grids manually.



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

KC2G_SITE_URL = "https://prop.kc2g.com/"
KC2G_MUF_RENDER_URL = "https://prop.kc2g.com/renders/current/mufd-normal-now.svg"
KC2G_RENDER_REFRESH_SECONDS = 15 * 60

MUF_FIXED_SCALE_MIN = 0.0
MUF_FIXED_SCALE_MAX = 35.0

SOHO_SITE_URL = "https://soho.nascom.nasa.gov/data/realtime-images.html"
SOHO_REFRESH_SECONDS = 10 * 60
SOHO_IMAGE_URLS = {
    "eit_171": "https://soho.nascom.nasa.gov/data/realtime/eit_171/512/latest.jpg",
    "eit_195": "https://soho.nascom.nasa.gov/data/realtime/eit_195/512/latest.jpg",
    "eit_284": "https://soho.nascom.nasa.gov/data/realtime/eit_284/512/latest.jpg",
    "hmi_continuum": "https://soho.nascom.nasa.gov/data/realtime/hmi_igr/512/latest.jpg",
    "lasco_c2": "https://soho.nascom.nasa.gov/data/realtime/c2/512/latest.jpg",
    "lasco_c3": "https://soho.nascom.nasa.gov/data/realtime/c3/512/latest.jpg",
}
SOHO_IMAGE_LABELS = {
    "eit_171": "SOHO EIT 171",
    "eit_195": "SOHO EIT 195",
    "eit_284": "SOHO EIT 284",
    "hmi_continuum": "SDO/HMI Continuum",
    "lasco_c2": "SOHO LASCO C2",
    "lasco_c3": "SOHO LASCO C3",
}
SOHO_DASHBOARD_INSTRUMENTS = [
    "eit_171",
    "eit_195",
    "eit_284",
    "hmi_continuum",
    "lasco_c2",
    "lasco_c3",
]


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
# PANEL SEVERITY (coherent color grammar)
# =====================================================
# Two families:
# - RISK: OK → WATCH → ALERT → DANGER → EXTREME
# - QUALITY (HF conditions): POOR → FAIR → GOOD → EXCELLENT
SEV_UNKNOWN = "UNKNOWN"

# Risk severities
SEV_OK = "OK"
SEV_WATCH = "WATCH"
SEV_ALERT = "ALERT"
SEV_DANGER = "DANGER"
SEV_EXTREME = "EXTREME"

# Quality severities
SEV_POOR = "POOR"
SEV_FAIR = "FAIR"
SEV_GOOD = "GOOD"
SEV_EXCELLENT = "EXCELLENT"

RISK_ORDER = [SEV_UNKNOWN, SEV_OK, SEV_WATCH, SEV_ALERT, SEV_DANGER, SEV_EXTREME]
QUAL_ORDER = [SEV_UNKNOWN, SEV_POOR, SEV_FAIR, SEV_GOOD, SEV_EXCELLENT]

RISK_COLORS = {
    SEV_UNKNOWN: ACCENT_GREY,
    SEV_OK: ACCENT_GREEN,
    SEV_WATCH: STATUS_WARN,
    SEV_ALERT: "#ff9f1a",
    SEV_DANGER: STATUS_ERR,
    SEV_EXTREME: "#b44dff",
}
QUAL_COLORS = {
    SEV_UNKNOWN: ACCENT_GREY,
    SEV_POOR: STATUS_ERR,
    SEV_FAIR: "#ff9f1a",
    SEV_GOOD: ACCENT_GREEN,
    SEV_EXCELLENT: ACCENT_BLUE,
}

def _is_nan(v) -> bool:
    try:
        return isinstance(v, float) and math.isnan(v)
    except Exception:
        return True

def _sev_index(order: list[str], sev: str) -> int:
    try:
        return order.index(sev)
    except Exception:
        return 0

def _clamp_sev(order: list[str], sev: str) -> str:
    if sev in order:
        return sev
    return order[0]

def _downgrade_sev(order: list[str], sev: str, steps: int = 1) -> str:
    i = _sev_index(order, sev)
    return order[max(0, i - int(steps))]




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

# ============================================================
# Tooltip helpers: show color scale / thresholds in each tile
# ============================================================
def _fmt_scale_lines(title, lines):
    out = [f"{title}"]
    out.extend([f"• {ln}" for ln in lines])
    return "\n".join(out)

RISK_PALETTE_TEXT = _fmt_scale_lines("Palette RISK (perturbations)", ["🟢 OK", "🟡 WATCH", "🟠 ALERT", "🔴 DANGER", "🟣 EXTREME"])
QUALITY_PALETTE_TEXT = _fmt_scale_lines("Palette QUALITY (HF propagation)", ["🔴 POOR", "🟠 FAIR", "🟢 GOOD", "🔵 EXCELLENT"])

def tooltip_thresholds(case_name, refresh_s, palette_text, thresholds_lines, extra_lines=None):
    th = _fmt_scale_lines("Seuils", thresholds_lines)
    extra = ""
    if extra_lines:
        extra = "\n\n" + _fmt_scale_lines("Notes", extra_lines)
    return f"{case_name}\nRefresh: {refresh_s}s\n\n{palette_text}\n\n{th}{extra}"

# =====================================================
def resource_path(rel: str) -> Path:
    """Return a resource path that works both in dev and in a PyInstaller bundle."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / rel  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent / rel


install_debug_hooks()
debug_log(f"SUMO starting {APP_VERSION}")

APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
ASSETS_DIR = resource_path("assets")

SKYFIELD_CACHE_DIR = APP_DIR / "skyfield_data"  # stable next to script/exe, not current working directory
SKYFIELD_EPHEMERIS_FILE = "de421.bsp"
SKYFIELD_BUNDLED_DIR = ASSETS_DIR / "skyfield"
SKYFIELD_BUNDLED_PATH = SKYFIELD_BUNDLED_DIR / SKYFIELD_EPHEMERIS_FILE
ISS_TLE_CACHE_PATH = APP_DIR / "iss_stations.tle"
ISS_TLE_URL = "https://celestrak.org/NORAD/elements/stations.txt"

# Network concurrency limiter to reduce concurrent SSL handshakes on Windows.
NETWORK_FETCH_SEMAPHORE = None
try:
    import threading as _sumo_threading
    NETWORK_FETCH_SEMAPHORE = _sumo_threading.BoundedSemaphore(2)
except Exception:
    NETWORK_FETCH_SEMAPHORE = None


def safe_get_text(url: str, timeout: int = 10, headers: dict | None = None) -> str:
    headers = headers or {"User-Agent": "SUMO-SunMonitor/0.1"}
    if NETWORK_FETCH_SEMAPHORE is not None:
        with NETWORK_FETCH_SEMAPHORE:
            r = requests.get(url, timeout=timeout, headers=headers)
    else:
        r = requests.get(url, timeout=timeout, headers=headers)
    r.raise_for_status()
    return r.text or ""


def soho_safe_get(image_url: str, timeout: int = 20, headers: dict | None = None):
    headers = headers or {
        "User-Agent": "SUMO-SunMonitor/0.1",
        "Accept": "image/jpeg,image/jpg,image/png,*/*;q=0.8",
        "Referer": SOHO_SITE_URL,
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Connection": "close",
    }
    if NETWORK_FETCH_SEMAPHORE is not None:
        with NETWORK_FETCH_SEMAPHORE:
            return requests.get(image_url, timeout=timeout, headers=headers)
    return requests.get(image_url, timeout=timeout, headers=headers)



def _skyfield_ephemeris_candidates() -> list[Path]:
    """Return possible local locations for de421.bsp.

    SUMO first tries existing local/bundled ephemerides, then falls back to a
    controlled download into ./skyfield_data next to the script/exe. This avoids
    Skyfield's own runtime download quirks in PyInstaller builds.
    """
    candidates: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        try:
            key = str(path.resolve())
        except Exception:
            key = str(path)
        if key not in seen:
            seen.add(key)
            candidates.append(path)

    add(resource_path("skyfield") / SKYFIELD_EPHEMERIS_FILE)
    add(resource_path("skyfield_data") / SKYFIELD_EPHEMERIS_FILE)
    add(resource_path("assets/skyfield") / SKYFIELD_EPHEMERIS_FILE)
    add(APP_DIR / "skyfield" / SKYFIELD_EPHEMERIS_FILE)
    add(APP_DIR / "skyfield_data" / SKYFIELD_EPHEMERIS_FILE)
    return candidates


def _download_skyfield_ephemeris(target_path: Path) -> Path:
    """Download de421.bsp in a controlled way, using a temporary file first."""
    url = "https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de421.bsp"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".download")

    try:
        if tmp_path.exists():
            tmp_path.unlink()
    except Exception:
        pass

    debug_log(f"Skyfield downloading {SKYFIELD_EPHEMERIS_FILE} to {target_path}")
    with requests.get(
        url,
        stream=True,
        timeout=(15, 120),
        headers={"User-Agent": "SUMO-SunMonitor/7.9.4"},
    ) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if chunk:
                    fh.write(chunk)
            fh.flush()

    if not tmp_path.exists() or tmp_path.stat().st_size < 1_000_000:
        raise RuntimeError(f"Downloaded ephemeris looks incomplete: {tmp_path}")

    tmp_path.replace(target_path)
    debug_log(f"Skyfield download OK: {target_path} ({target_path.stat().st_size} bytes)")
    return target_path


def get_skyfield_loader_and_ephemeris():
    """Return (loader, eph, source_text) with local-first + auto-download fallback.

    Search order:
    1) bundled/local de421.bsp if present;
    2) controlled first-run download into APP_DIR/skyfield_data/de421.bsp;
    3) clear RuntimeError without crashing the application.
    """
    if not SKYFIELD_AVAILABLE or SkyfieldLoader is None:
        raise RuntimeError(f"Skyfield unavailable: {SKYFIELD_IMPORT_ERROR}")

    checked: list[str] = []

    # 1) Try already available local/bundled ephemerides.
    for eph_path in _skyfield_ephemeris_candidates():
        try:
            checked.append(str(eph_path))
            if not eph_path.exists():
                continue
            if eph_path.stat().st_size < 1_000_000:
                raise RuntimeError(f"Local ephemeris looks incomplete: {eph_path}")
            loader = SkyfieldLoader(str(eph_path.parent))
            eph = loader(SKYFIELD_EPHEMERIS_FILE)
            debug_log(f"Skyfield loaded from local ephemeris: {eph_path}")
            return loader, eph, f"local:{eph_path}"
        except Exception as exc:
            debug_log(f"Skyfield local ephemeris candidate failed: {eph_path} • {exc}")

    # 2) If missing, download ourselves to a stable writable folder next to exe/script.
    download_path = APP_DIR / "skyfield_data" / SKYFIELD_EPHEMERIS_FILE
    try:
        checked.append(str(download_path))
        eph_path = _download_skyfield_ephemeris(download_path)
        loader = SkyfieldLoader(str(eph_path.parent))
        eph = loader(SKYFIELD_EPHEMERIS_FILE)
        debug_log(f"Skyfield loaded from downloaded ephemeris: {eph_path}")
        return loader, eph, f"downloaded:{eph_path}"
    except Exception as exc:
        debug_log(f"Skyfield automatic download failed: {exc}")

    # 3) Clean error. MainWindow catches this and SUMO continues with fallback.
    raise RuntimeError(
        "Skyfield unavailable: de421.bsp missing and automatic download failed. "
        "Checked: " + " | ".join(checked)
    )
# ============================================================
# Unified sound alert system (color/severity change)
# - Plays ONE sound file (assets/alert.wav) on any color/severity change
# - Also used as feedback when user enables sound via the UI button
# ============================================================
from PySide6.QtMultimedia import QSoundEffect
from PySide6.QtCore import QUrl


# ============================================================
# Palette tooltips (for right-side values in panels)
# ============================================================
PALETTE_RISK_TOOLTIP = (
    "Palette RISK (perturbations)\n"
    "🟢 OK\n"
    "🟡 WATCH\n"
    "🟠 ALERT\n"
    "🔴 DANGER\n"
    "🟣 EXTREME"
)

PALETTE_QUALITY_TOOLTIP = (
    "Palette QUALITY (HF propagation)\n"
    "🔴 POOR\n"
    "🟠 FAIR\n"
    "🟢 GOOD\n"
    "🔵 EXCELLENT"
)


# ============================================================
# Palette tooltips (shown on right-side value in each panel)
# ============================================================
PALETTE_TOOLTIP_RISK = (
    "Palette RISK (perturbations)\n"
    "🟢 OK\n"
    "🟡 WATCH\n"
    "🟠 ALERT\n"
    "🔴 DANGER\n"
    "🟣 EXTREME"
)

PALETTE_TOOLTIP_QUALITY = (
    "Palette QUALITY (HF propagation)\n"
    "🔴 POOR\n"
    "🟠 FAIR\n"
    "🟢 GOOD\n"
    "🔵 EXCELLENT"
)

ALERT_WAV_PATH = (ASSETS_DIR / "sounds" / "alert.wav")
HOUR_CHIME_WAV_PATH = (ASSETS_DIR / "sounds" / "hour.wav")

def _get_mainwindow(obj):
    try:
        w = obj.window()
        return w
    except Exception:
        return None

def play_alert_sound(obj):
    """Play assets/alert.wav if sound is enabled in the main window."""
    try:
        mw = _get_mainwindow(obj)
        if mw is not None and not getattr(mw, "_sound_enabled", True):
            return
        if not ALERT_WAV_PATH.exists():
            return
        volume = 0.9
        if mw is not None:
            try:
                volume = max(0.0, min(1.0, float(getattr(mw, "_alert_volume", 0.9))))
            except Exception:
                volume = 0.9
        # Reuse a single QSoundEffect instance stored on the main window when possible
        if mw is not None:
            sfx = getattr(mw, "_alert_sfx", None)
            if sfx is None:
                sfx = QSoundEffect(mw)
                sfx.setSource(QUrl.fromLocalFile(str(ALERT_WAV_PATH)))
                mw._alert_sfx = sfx
            sfx.setVolume(volume)
            # Restart sound cleanly
            try:
                sfx.stop()
            except Exception:
                pass
            sfx.play()
        else:
            sfx = QSoundEffect(obj)
            sfx.setSource(QUrl.fromLocalFile(str(ALERT_WAV_PATH)))
            sfx.setVolume(volume)
            sfx.play()
    except Exception:
        pass

def play_hour_chime(obj):
    """Play assets/sounds/hour.wav for the configured hour chime."""
    try:
        mw = _get_mainwindow(obj)
        if mw is not None and str(getattr(mw, "_clock_hour_chime_mode", "off")).strip().lower() == "off":
            return
        if not HOUR_CHIME_WAV_PATH.exists():
            return
        volume = 0.9
        if mw is not None:
            try:
                volume = max(0.0, min(1.0, float(getattr(mw, "_hour_chime_volume", 0.9))))
            except Exception:
                volume = 0.9
        if mw is not None:
            sfx = getattr(mw, "_hour_chime_sfx", None)
            if sfx is None:
                sfx = QSoundEffect(mw)
                sfx.setSource(QUrl.fromLocalFile(str(HOUR_CHIME_WAV_PATH)))
                mw._hour_chime_sfx = sfx
            sfx.setVolume(volume)
            try:
                sfx.stop()
            except Exception:
                pass
            sfx.play()
        else:
            sfx = QSoundEffect(obj)
            sfx.setSource(QUrl.fromLocalFile(str(HOUR_CHIME_WAV_PATH)))
            sfx.setVolume(volume)
            sfx.play()
    except Exception:
        pass

DB_PATH = APP_DIR / "sumo_cache.sqlite"

CONFIG_PATH = APP_DIR / "sumo_config.json"
DEFAULT_RSS_URL = RSS_SOLAR_SYSTEM_URL

CLOCK_CITY_OPTIONS = [
    ("New York", "America/New_York"),
    ("London", "Europe/London"),
    ("Paris", "Europe/Paris"),
    ("Tokyo", "Asia/Tokyo"),
    ("Sydney", "Australia/Sydney"),
    ("Moscow", "Europe/Moscow"),
    ("Dubai", "Asia/Dubai"),
    ("Delhi", "Asia/Kolkata"),
    ("Hong Kong", "Asia/Hong_Kong"),
    ("Singapore", "Asia/Singapore"),
    ("Los Angeles", "America/Los_Angeles"),
    ("Chicago", "America/Chicago"),
    ("Denver", "America/Denver"),
    ("Sao Paulo", "America/Sao_Paulo"),
    ("Johannesburg", "Africa/Johannesburg"),
]
CLOCK_CITY_MAP = {k: v for k, v in CLOCK_CITY_OPTIONS}

VALID_DISPLAY_VIEWS = ("solar", "clock", "muf", "soho", "solarsystem", "widgetdemo")

LEGACY_DISPLAY_MODE_MAP = {
    "solar": ["solar"],
    "clock": ["clock"],
    "muf": ["muf"],
    "soho": ["soho"],
    "solarsystem": ["solarsystem"],
    "alternate": ["solar", "clock"],
    "alternate_solar_muf": ["solar", "muf"],
    "alternate_clock_muf": ["clock", "muf"],
    "alternate_solar_soho": ["solar", "soho"],
    "alternate_clock_soho": ["clock", "soho"],
    "alternate_muf_soho": ["muf", "soho"],
    "alternate_solar_solarsystem": ["solar", "solarsystem"],
    "alternate_clock_solarsystem": ["clock", "solarsystem"],
    "alternate_muf_solarsystem": ["muf", "solarsystem"],
    "alternate_soho_solarsystem": ["soho", "solarsystem"],
    "alternate_solar_clock_solarsystem": ["solar", "clock", "solarsystem"],
    "alternate_all": ["solar", "clock", "muf", "soho", "solarsystem", "widgetdemo"],
    "alternate_custom": ["solar", "clock"],
    "widgetdemo": ["widgetdemo"],
}

def normalize_display_views(display_views=None, display_mode: str = "solar") -> list[str]:
    seq = []
    if isinstance(display_views, str):
        seq = [x.strip().lower() for x in display_views.split(",")]
    elif isinstance(display_views, (list, tuple)):
        seq = [str(x).strip().lower() for x in display_views]
    if not seq:
        seq = list(LEGACY_DISPLAY_MODE_MAP.get(str(display_mode or "solar").strip().lower(), ["solar"]))
    out = []
    for item in seq:
        if item in VALID_DISPLAY_VIEWS and item not in out:
            out.append(item)
    return out or ["solar"]



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
# DAPNET CONFIG + CLIENT
# V7.5.2 CLEAN ARCHITECTURE:
# - centralized latest-value coercion for NOAA/SUMO payloads
# - robust A-index formatting
# - safer numeric guards for pager message assembly
# =====================================================
DAPNET_API_URL = "https://hampager.de/api/calls"
DAPNET_HTTP_TIMEOUT = 10
DAPNET_STATE_PATH = APP_DIR / "dapnet_state.json"
DAPNET_GLOBAL_COOLDOWN = 120  # seconds, global anti-spam guard for all DAPNET modules
_last_dapnet_global = 0.0

def default_dapnet_config() -> dict:
    return {
        "enabled": False,
        "username": "",
        "password": "",
        "tx_group": "f-53",
        "callsigns": ["f4igv"],
        "quick_ui_enabled": True,
        "xray": {
            "enabled": False,
            "threshold": "M1.0",
            "send_start": True,
            "send_end": True,
            "emergency_on_start": True,
        },
        "proton": {
            "enabled": False,
            "threshold": "S1",
            "cooldown_minutes": 30,
            "include_bz_bt": True,
        },
        "solar_summary": {
            "enabled": False,
            "interval_minutes": 60,
        },
        "iss": {
            "enabled": False,
            "prealert_enabled": True,
            "start_enabled": True,
            "peak_enabled": True,
            "end_enabled": True,
            "prealert_minutes": 15,
            "min_elevation": 5.0,
            "observer_lat": 48.1173,
            "observer_lon": -1.6778,
            "observer_alt_m": 60,
        },
    }

def _deep_update_dict(dst: dict, src: dict) -> dict:
    for key, value in (src or {}).items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update_dict(dst[key], value)
        else:
            dst[key] = value
    return dst

def normalize_callsigns(value) -> list[str]:
    if isinstance(value, str):
        raw = re.split(r"[,;\s]+", value.strip())
    elif isinstance(value, (list, tuple)):
        raw = [str(x).strip() for x in value]
    else:
        raw = []
    out = []
    for item in raw:
        item = str(item or "").strip()
        if not item:
            continue
        if item not in out:
            out.append(item.lower())
    return out

def normalize_dapnet_config(cfg: Optional[dict]) -> dict:
    base = default_dapnet_config()
    if isinstance(cfg, dict):
        _deep_update_dict(base, cfg)
    base["enabled"] = bool(base.get("enabled", False))
    base["username"] = str(base.get("username") or "").strip()
    base["password"] = str(base.get("password") or "")
    base["tx_group"] = str(base.get("tx_group") or "f-53").strip() or "f-53"
    base["quick_ui_enabled"] = bool(base.get("quick_ui_enabled", True))
    callsigns = normalize_callsigns(base.get("callsigns"))
    base["callsigns"] = callsigns or ["f4igv"]

    xray = base.get("xray", {})
    xray["enabled"] = bool(xray.get("enabled", False))
    xray["threshold"] = str(xray.get("threshold") or "M1.0").strip().upper() or "M1.0"
    xray["send_start"] = bool(xray.get("send_start", True))
    xray["send_end"] = bool(xray.get("send_end", True))
    xray["emergency_on_start"] = bool(xray.get("emergency_on_start", True))

    proton = base.get("proton", {})
    proton["enabled"] = bool(proton.get("enabled", False))
    proton["threshold"] = str(proton.get("threshold") or "S1").strip().upper() or "S1"
    try:
        proton["cooldown_minutes"] = max(5, int(proton.get("cooldown_minutes", 30)))
    except Exception:
        proton["cooldown_minutes"] = 30
    proton["include_bz_bt"] = bool(proton.get("include_bz_bt", True))

    solar_summary = base.get("solar_summary", {})
    solar_summary["enabled"] = bool(solar_summary.get("enabled", False))
    try:
        solar_summary["interval_minutes"] = max(5, int(solar_summary.get("interval_minutes", 60)))
    except Exception:
        solar_summary["interval_minutes"] = 60

    iss = base.get("iss", {})
    iss["enabled"] = bool(iss.get("enabled", False))
    iss["prealert_enabled"] = bool(iss.get("prealert_enabled", True))
    iss["start_enabled"] = bool(iss.get("start_enabled", True))
    iss["peak_enabled"] = bool(iss.get("peak_enabled", True))
    iss["end_enabled"] = bool(iss.get("end_enabled", True))
    try:
        iss["prealert_minutes"] = max(1, int(iss.get("prealert_minutes", 15)))
    except Exception:
        iss["prealert_minutes"] = 15
    try:
        iss["min_elevation"] = float(iss.get("min_elevation", 5.0))
    except Exception:
        iss["min_elevation"] = 5.0
    try:
        iss["observer_lat"] = float(iss.get("observer_lat", 48.1173))
    except Exception:
        iss["observer_lat"] = 48.1173
    try:
        iss["observer_lon"] = float(iss.get("observer_lon", -1.6778))
    except Exception:
        iss["observer_lon"] = -1.6778
    try:
        iss["observer_alt_m"] = int(float(iss.get("observer_alt_m", 60)))
    except Exception:
        iss["observer_alt_m"] = 60
    return base


def _coerce_scalar_last(value):
    """Return the last scalar value from scalar/list/tuple/numpy-like containers."""
    try:
        if value is None:
            return None
        if isinstance(value, str):
            s = value.strip()
            return s if s != "" else None
        if isinstance(value, (list, tuple)):
            if not value:
                return None
            return _coerce_scalar_last(value[-1])
        if hasattr(value, "shape") and hasattr(value, "__len__") and not isinstance(value, (bytes, bytearray)):
            try:
                ln = len(value)
            except Exception:
                ln = None
            if ln == 0:
                return None
            if ln is not None:
                try:
                    return _coerce_scalar_last(value[-1])
                except Exception:
                    pass
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        return value
    except Exception:
        return None

def _coerce_float_last(value):
    """Return last numeric value as float from scalar/list/tuple/numpy-like containers."""
    try:
        v = _coerce_scalar_last(value)
        if v in (None, ""):
            return float("nan")
        return float(v)
    except Exception:
        return float("nan")

def _fmt_int_from_any(value):
    try:
        v = _coerce_float_last(value)
        if math.isnan(v):
            return None
        return str(int(round(v)))
    except Exception:
        return None


def _dapnet_is_valid_number(value) -> bool:
    try:
        return value is not None and not math.isnan(float(value))
    except Exception:
        return False

def _dapnet_tail_float(payload: dict, *keys: str) -> float:
    """Read the first matching payload key and coerce its latest scalar numeric value."""
    try:
        for key in keys:
            if key in payload:
                return _coerce_float_last(payload.get(key))
    except Exception:
        pass
    return float("nan")

def _dapnet_tail_text_int(payload: dict, *keys: str):
    """Read the first matching payload key and return int text or None."""
    try:
        for key in keys:
            if key in payload:
                return _fmt_int_from_any(payload.get(key))
    except Exception:
        pass
    return None


def _dapnet_sanitize_a_index(value):
    """Return plausible A-index integer text or None; reject timestamp-like values."""
    try:
        v = _coerce_float_last(value)
        if math.isnan(v):
            return None
        if v < 0 or v > 999:
            return None
        return str(int(round(v)))
    except Exception:
        return None

def _dapnet_compact_text(parts):
    out = []
    for p in parts:
        s = str(p or "").strip()
        if s:
            out.append(s)
    return " ".join(out)[:80]

class DapnetClient:
    def __init__(self, config: Optional[dict] = None, logger=None):
        self.logger = logger
        self.session = requests.Session()  # Connection pooling
        self.update_config(config or {})

    def update_config(self, config: Optional[dict]) -> None:
        self.config = normalize_dapnet_config(config)
        # Update session auth when config changes
        self.session.auth = (self.config.get("username", ""), self.config.get("password", ""))

    def _log(self, message: str) -> None:
        try:
            if callable(self.logger):
                self.logger(f"[DAPNET] {message}")
            else:
                debug_log(f"[DAPNET] {message}")
        except Exception:
            pass

    def can_send(self) -> tuple[bool, str]:
        if not self.config.get("enabled", False):
            return False, "DAPNET disabled"
        if not self.config.get("username"):
            return False, "Missing username"
        if not self.config.get("password"):
            return False, "Missing password"
        callsigns = normalize_callsigns(self.config.get("callsigns"))
        if not callsigns:
            return False, "Missing recipient callsigns"
        return True, "OK"

    def build_payload(self, text: str, emergency: bool = False, recipients: Optional[list[str]] = None, tx_group: Optional[str] = None) -> dict:
        targets = normalize_callsigns(recipients) if recipients is not None else normalize_callsigns(self.config.get("callsigns"))
        group_name = str(tx_group or self.config.get("tx_group") or "f-53").strip() or "f-53"
        return {
            "text": str(text or "").strip()[:80],
            "callSignNames": targets,
            "transmitterGroupNames": [group_name],
            "emergency": bool(emergency),
        }

    def send_message(self, text: str, emergency: bool = False, force: bool = False, recipients: Optional[list[str]] = None, tx_group: Optional[str] = None) -> tuple[bool, str]:
        if not force:
            ok, reason = self.can_send()
            if not ok:
                return False, reason
        else:
            cfg = normalize_dapnet_config(self.config)
            final_recipients = normalize_callsigns(recipients) if recipients is not None else normalize_callsigns(cfg.get("callsigns"))
            if not cfg.get("username") or not cfg.get("password") or not final_recipients:
                return False, "Missing credentials or recipient callsigns"
        payload = self.build_payload(text, emergency=emergency, recipients=recipients, tx_group=tx_group)
        if not payload.get("text"):
            return False, "Empty message"

        # Global anti-spam guard: prevents multiple modules from flooding DAPNET.
        # Test sends keep force=True, but still respect this guard once credentials are valid.
        global _last_dapnet_global
        now_ts = time.time()
        if now_ts - float(_last_dapnet_global or 0.0) < DAPNET_GLOBAL_COOLDOWN:
            remaining = int(DAPNET_GLOBAL_COOLDOWN - (now_ts - float(_last_dapnet_global or 0.0)))
            return False, f"Global cooldown ({remaining}s)"

        # Retry logic with exponential backoff (3 attempts)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                r = self.session.post(
                    DAPNET_API_URL,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(payload),
                    timeout=DAPNET_HTTP_TIMEOUT,
                )
                r.raise_for_status()
                _last_dapnet_global = time.time()
                self._log(f"message sent to {','.join(payload.get('callSignNames', []))}: {payload['text']}")
                return True, str(r.text).strip() or f"HTTP {r.status_code}"
            except (requests.RequestException, Exception) as exc:
                error_msg = f"Attempt {attempt + 1}/{max_retries} failed: {str(exc)}"
                self._log(error_msg)
                if attempt == max_retries - 1:  # Last attempt
                    return False, f"Failed after {max_retries} attempts: {str(exc)}"
                # Exponential backoff: 1s, 2s, 4s
                time.sleep(2 ** attempt)


# =====================================================
# TIME PARSING (robuste)
# =====================================================

def load_dapnet_state(path: Path) -> dict:
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {
        "xray": {"active": False, "last_start_label": "", "last_end_label": ""},
        "proton": {"last_sent_ts": 0.0, "last_level": "S0"},
        "solar_summary": {"last_sent_ts": 0.0},
        "iss": {"sent_keys": {}},
    }

def save_dapnet_state(path: Path, state: dict) -> None:
    try:
        path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

def xray_label_to_flux(label: str) -> float:
    s = str(label or "").strip().upper()
    m = re.fullmatch(r"([ABCMX])\s*([0-9]+(?:\.[0-9]+)?)", s)
    if not m:
        return float("nan")
    letter = m.group(1)
    magnitude = float(m.group(2))
    base = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}[letter]
    return magnitude * base

def xray_flux_meets_threshold(flux: float, threshold_label: str) -> bool:
    try:
        thr = xray_label_to_flux(threshold_label)
        return bool(math.isfinite(float(flux)) and math.isfinite(thr) and float(flux) >= thr)
    except Exception:
        return False

def s_level_to_int(level: str) -> int:
    s = str(level or "").strip().upper()
    m = re.fullmatch(r"S\s*([0-5])", s)
    if not m:
        return -1
    try:
        return int(m.group(1))
    except Exception:
        return -1

def s_level_meets_threshold(level: str, threshold_level: str) -> bool:
    try:
        return s_level_to_int(level) >= s_level_to_int(threshold_level)
    except Exception:
        return False

def format_local_dt(dt: datetime, local_tz: timezone | None = None) -> str:
    try:
        local_tz = local_tz or safe_zoneinfo("Europe/Paris")
    except Exception:
        local_tz = timezone.utc
    try:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(local_tz).strftime("%d/%m %H:%M")
    except Exception:
        try:
            return dt.strftime("%d/%m %H:%M")
        except Exception:
            return "?"

def short_float_text(v, fmt: str = "{:.1f}") -> str:
    try:
        vv = float(v)
        if math.isnan(vv):
            return "--"
        return fmt.format(vv)
    except Exception:
        return "--"


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
    y_axis_kind: str = "linear"


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


def smooth_xray_series_for_plot(x: np.ndarray, y: np.ndarray, max_gap_seconds: float = 30.0 * 60.0) -> tuple[np.ndarray, np.ndarray]:
    """Light cleanup for GOES X-ray plotting."""
    try:
        xx = np.array(x, dtype=float)
        yy = np.array(y, dtype=float)
        n = int(min(len(xx), len(yy)))
        if n <= 0:
            return np.array([], dtype=float), np.array([], dtype=float)
        xx = xx[:n]
        yy = yy[:n]
        m = np.isfinite(xx)
        xx = xx[m]
        yy = yy[m]
        if xx.size <= 0:
            return np.array([], dtype=float), np.array([], dtype=float)
        out = yy.copy()
        for i in range(len(out)):
            bad = (not np.isfinite(out[i])) or (out[i] <= 0.0)
            if not bad:
                continue
            prev_i = next((j for j in range(i - 1, -1, -1) if np.isfinite(out[j]) and out[j] > 0.0), None)
            next_i = next((j for j in range(i + 1, len(out)) if np.isfinite(out[j]) and out[j] > 0.0), None)
            if prev_i is None or next_i is None:
                continue
            gap = float(xx[next_i] - xx[prev_i])
            if gap <= 0.0 or gap > max_gap_seconds:
                continue
            a = (xx[i] - xx[prev_i]) / gap
            log_prev = math.log10(out[prev_i])
            log_next = math.log10(out[next_i])
            out[i] = 10 ** (log_prev + a * (log_next - log_prev))
        for i in range(1, len(out) - 1):
            cur = out[i]
            left = out[i - 1]
            right = out[i + 1]
            if not (np.isfinite(cur) and np.isfinite(left) and np.isfinite(right)):
                continue
            if cur <= 0.0 or left <= 0.0 or right <= 0.0:
                continue
            dt_lr = float(xx[i + 1] - xx[i - 1])
            if dt_lr <= 0.0 or dt_lr > max_gap_seconds:
                continue
            log_cur = math.log10(cur)
            log_left = math.log10(left)
            log_right = math.log10(right)
            log_ref = 0.5 * (log_left + log_right)
            if abs(log_left - log_right) <= 0.35 and (log_ref - log_cur) >= 0.85:
                out[i] = 10 ** log_ref
        return xx, out
    except Exception:
        return np.array(x, dtype=float), np.array(y, dtype=float)


def latest_xray_peak_label(x: np.ndarray, y: np.ndarray) -> str:
    """Return the most recent local peak as flare class label (e.g. X1.1)."""
    try:
        xx = np.array(x, dtype=float)
        yy = np.array(y, dtype=float)
        n = int(min(len(xx), len(yy)))
        if n <= 0:
            return "?"
        xx = xx[:n]
        yy = yy[:n]
        valid = np.isfinite(xx) & np.isfinite(yy) & (yy > 0.0)
        yy = yy[valid]
        if yy.size <= 0:
            return "?"
        if yy.size == 1:
            c, m = xray_flare_class(float(yy[0]))
            return xray_class_label(c, m)
        for i in range(len(yy) - 2, 0, -1):
            if yy[i] >= yy[i - 1] and yy[i] >= yy[i + 1]:
                c, m = xray_flare_class(float(yy[i]))
                return xray_class_label(c, m)
        c, m = xray_flare_class(float(np.nanmax(yy)))
        return xray_class_label(c, m)
    except Exception:
        return "?"


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

    Source: https://api.pota.app/spot/activator (JSON array).
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
class AnalogClockWidget(QtWidgets.QFrame):
    def __init__(self, title: str = "", timezone_name: str | None = None, compact: bool = False, parent=None):
        super().__init__(parent)
        self._title = str(title or "")
        self._timezone_name = timezone_name
        self._compact = bool(compact)
        self._current_dt = datetime.now(timezone.utc)
        self._planet_hitboxes = []
        self._hover_planet_name = None
        self.setMouseTracking(True)
        self.setMinimumSize(150 if compact else 260, 150 if compact else 260)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setStyleSheet("background: transparent; border: 0px;")

    def set_title(self, title: str):
        self._title = str(title or "")
        self.update()

    def set_timezone_name(self, timezone_name: str | None):
        self._timezone_name = timezone_name
        self.update()

    def set_datetime(self, dt: datetime):
        self._current_dt = dt
        self.update()


    def _planet_tooltip_text(self, row: dict) -> str:
        try:
            planet = row.get("planet", {}) or {}
            name = str(planet.get("name", "Planet"))
            distance_au = float(row.get("distance_au", planet.get("radius_au", 0.0)) or 0.0)
            lat_deg = float(row.get("lat_deg", 0.0) or 0.0)
            period_days = float(planet.get("period_days", 0.0) or 0.0)
            source = str(row.get("source", "approximate") or "approximate")
            angle_deg = (math.degrees(float(row.get("angle_rad", 0.0) or 0.0)) + 360.0) % 360.0

            lines = [
                f"<b>{name}</b>",
                f"Source: {'Skyfield/JPL' if source == 'skyfield' else 'Approximate fallback'}",
                f"Distance from Sun: {distance_au:.3f} AU",
                f"Orbital period: {period_days:.1f} days",
                f"Ecliptic longitude: {angle_deg:.1f}°",
                f"Ecliptic latitude: {lat_deg:+.2f}°",
            ]

            if source == "skyfield":
                x_au = float(row.get("x_au", 0.0) or 0.0)
                y_au = float(row.get("y_au", 0.0) or 0.0)
                z_au = float(row.get("z_au", 0.0) or 0.0)
                lines.extend([
                    f"Heliocentric X: {x_au:+.3f} AU",
                    f"Heliocentric Y: {y_au:+.3f} AU",
                    f"Heliocentric Z: {z_au:+.3f} AU",
                ])

            return "<br>".join(lines)
        except Exception:
            return "<b>Planet</b>"

    def _update_planet_tooltip(self, global_pos, local_pos) -> bool:
        try:
            point = QtCore.QPointF(local_pos)
            for item in reversed(self._planet_hitboxes):
                center = item.get("center")
                radius = float(item.get("radius", 0.0) or 0.0)
                if center is None or radius <= 0.0:
                    continue
                dx = point.x() - center.x()
                dy = point.y() - center.y()
                if (dx * dx) + (dy * dy) <= (radius * radius):
                    name = str(item.get("name", ""))
                    if self._hover_planet_name != name:
                        self._hover_planet_name = name
                        QtWidgets.QToolTip.showText(global_pos, item.get("tooltip", ""), self)
                    return True
        except Exception:
            pass

        if self._hover_planet_name is not None:
            self._hover_planet_name = None
            QtWidgets.QToolTip.hideText()
        return False

    def mouseMoveEvent(self, event):
        try:
            global_pos = event.globalPosition().toPoint()
            local_pos = event.position()
            self._update_planet_tooltip(global_pos, local_pos)
        except Exception:
            pass
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self._hover_planet_name = None
        QtWidgets.QToolTip.hideText()
        super().leaveEvent(event)

    def paintEvent(self, event):
        self._planet_hitboxes = []
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        rect = self.rect().adjusted(4, 4, -4, -4)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor("#0f1720"))
        painter.drawRoundedRect(rect, 16, 16)

        margin = 18 if not self._compact else 12
        title_h = 28 if not self._compact else 22
        subtitle_h = 18 if not self._compact else 14
        face_rect = QtCore.QRectF(
            rect.left() + margin,
            rect.top() + margin + title_h,
            rect.width() - 2 * margin,
            rect.height() - 2 * margin - title_h - subtitle_h,
        )
        side = max(40.0, min(face_rect.width(), face_rect.height()))
        face_rect = QtCore.QRectF(
            face_rect.center().x() - side / 2.0,
            face_rect.center().y() - side / 2.0,
            side,
            side,
        )

        painter.setPen(QtGui.QPen(QtGui.QColor("#2a3440"), 2))
        painter.setBrush(QtGui.QColor("#111922"))
        painter.drawEllipse(face_rect)

        cx = face_rect.center().x()
        cy = face_rect.center().y()
        radius = face_rect.width() / 2.0

        for i in range(60):
            angle_deg = i * 6.0 - 90.0
            angle = math.radians(angle_deg)
            outer = QtCore.QPointF(cx + math.cos(angle) * (radius - 7), cy + math.sin(angle) * (radius - 7))
            inner_len = 16 if i % 5 == 0 else 8
            inner = QtCore.QPointF(cx + math.cos(angle) * (radius - inner_len), cy + math.sin(angle) * (radius - inner_len))
            pen = QtGui.QPen(QtGui.QColor("#d7dde6" if i % 5 == 0 else "#5e6b78"), 2 if i % 5 == 0 else 1)
            painter.setPen(pen)
            painter.drawLine(inner, outer)

        font_num = QtGui.QFont("Segoe UI", 10 if self._compact else 12, QtGui.QFont.Bold)
        painter.setFont(font_num)
        painter.setPen(QtGui.QColor("#d7dde6"))
        if not self._compact:
            for hour in range(1, 13):
                angle_deg = hour * 30.0 - 90.0
                angle = math.radians(angle_deg)
                tx = cx + math.cos(angle) * (radius - 34)
                ty = cy + math.sin(angle) * (radius - 34)
                tr = QtCore.QRectF(tx - 12, ty - 10, 24, 20)
                painter.drawText(tr, QtCore.Qt.AlignCenter, str(hour))

        dt = convert_clock_datetime(self._current_dt, self._timezone_name)
        sec = dt.second + dt.microsecond / 1_000_000.0
        minute = dt.minute + sec / 60.0
        hour = (dt.hour % 12) + minute / 60.0

        def draw_hand(value, max_value, length_ratio, color, width):
            angle_deg = (value / max_value) * 360.0 - 90.0
            angle = math.radians(angle_deg)
            end = QtCore.QPointF(cx + math.cos(angle) * (radius * length_ratio), cy + math.sin(angle) * (radius * length_ratio))
            painter.setPen(QtGui.QPen(QtGui.QColor(color), width, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
            painter.drawLine(QtCore.QPointF(cx, cy), end)

        draw_hand(hour, 12.0, 0.46, "#d7dde6", 5 if not self._compact else 4)
        draw_hand(minute, 60.0, 0.68, "#4aa3ff", 4 if not self._compact else 3)
        draw_hand(sec, 60.0, 0.78, "#ff4d4d", 2)

        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor("#d7dde6"))
        painter.drawEllipse(QtCore.QPointF(cx, cy), 5, 5)

        title_font = QtGui.QFont("Segoe UI", 12 if self._compact else 16, QtGui.QFont.Bold)
        painter.setFont(title_font)
        painter.setPen(QtGui.QColor("#44d16e"))
        title_rect = QtCore.QRectF(rect.left() + 10, rect.top() + 8, rect.width() - 20, title_h)
        painter.drawText(title_rect, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter, self._title)

        sub_font = QtGui.QFont("Consolas", 9 if self._compact else 11, QtGui.QFont.Bold)
        painter.setFont(sub_font)
        painter.setPen(QtGui.QColor("#aab6c5"))
        subtitle = dt.strftime("%H:%M:%S")
        sub_rect = QtCore.QRectF(rect.left() + 10, rect.bottom() - subtitle_h - 4, rect.width() - 20, subtitle_h)
        painter.drawText(sub_rect, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter, subtitle)


class ClockDashboard(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("clockDashboard")
        self.setStyleSheet("""
            QFrame#clockDashboard {
                background: #0e141a;
                border: 2px solid #2a3440;
                border-radius: 14px;
            }
            QFrame#clockSideRail {
                background: transparent;
                border: none;
            }
        """)
        self._city_map = {
            "tl": ("New York", "America/New_York"),
            "tr": ("London", "Europe/London"),
            "bl": ("Tokyo", "Asia/Tokyo"),
            "br": ("Sydney", "Australia/Sydney"),
        }

        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        self.clock_center = AnalogClockWidget("UTC", None, compact=False)
        self.clock_center.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        root.addWidget(self.clock_center, 2)

        self.side_rail = QtWidgets.QFrame()
        self.side_rail.setObjectName("clockSideRail")
        side = QtWidgets.QVBoxLayout(self.side_rail)
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(12)

        self.clock_tl = AnalogClockWidget("New York", "America/New_York", compact=True)
        self.clock_tr = AnalogClockWidget("London", "Europe/London", compact=True)
        self.clock_bl = AnalogClockWidget("Tokyo", "Asia/Tokyo", compact=True)
        self.clock_br = AnalogClockWidget("Sydney", "Australia/Sydney", compact=True)

        for w in (self.clock_tl, self.clock_tr, self.clock_bl, self.clock_br):
            w.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
            side.addWidget(w, 1)

        root.addWidget(self.side_rail, 1)

    def set_city(self, position: str, timezone_name: str):
        timezone_name = str(timezone_name or "").strip()
        city_name = next((city for city, tz in CLOCK_CITY_OPTIONS if tz == timezone_name), timezone_name.split("/")[-1].replace("_", " "))
        self._city_map[position] = (city_name, timezone_name)
        widget = getattr(self, f"clock_{position}", None)
        if widget is not None:
            widget.set_title(city_name)
            widget.set_timezone_name(timezone_name)

    def set_center_mode(self, mode: str):
        mode = (str(mode) or "utc").strip().lower()
        self.clock_center.set_title("LOCAL" if mode == "local" else "UTC")
        self.clock_center.set_timezone_name(None)

    def update_times(self, center_dt: datetime):
        self.clock_center.set_datetime(center_dt)
        for pos in ("tl", "tr", "bl", "br"):
            city_name, tz_name = self._city_map.get(pos, ("", "UTC"))
            widget = getattr(self, f"clock_{pos}", None)
            if widget is not None:
                widget.set_title(city_name)
                widget.set_timezone_name(tz_name)
                widget.set_datetime(center_dt)



class SumoSafeWidgetBase(QtWidgets.QFrame):
    """Base légère pour ajouter des widgets sans refactorer l'architecture existante."""

    widget_id = "base"
    widget_title = "Safe widget"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName(f"safeWidget_{self.widget_id}")

    def widget_id_str(self) -> str:
        return str(getattr(self, "widget_id", "base") or "base").strip().lower()

    def widget_title_str(self) -> str:
        return str(getattr(self, "widget_title", "Safe widget") or "Safe widget").strip()

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def apply_config(self, cfg: dict) -> None:
        pass


class WidgetDemoPanel(SumoSafeWidgetBase):
    widget_id = "widgetdemo"
    widget_title = "Widget demo"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("widgetDemoPanel")
        self.setStyleSheet("""
            QFrame#widgetDemoPanel {
                background: #0e141a;
                border: 2px solid #2a3440;
                border-radius: 14px;
            }
            QLabel {
                color: #d7dde6;
            }
        """)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(28, 28, 28, 28)
        root.setSpacing(16)

        self.lbl_title = QtWidgets.QLabel("SUMO Safe Widget")
        self.lbl_title.setStyleSheet("color:#44d16e; font-size: 30px; font-weight: 900;")
        root.addWidget(self.lbl_title, 0, QtCore.Qt.AlignHCenter)

        self.lbl_subtitle = QtWidgets.QLabel("Base de test pour ajouter de nouveaux widgets sans modifier le coeur des dashboards existants.")
        self.lbl_subtitle.setWordWrap(True)
        self.lbl_subtitle.setStyleSheet("font-size: 16px; color:#d7dde6; font-weight: 700;")
        root.addWidget(self.lbl_subtitle, 0, QtCore.Qt.AlignHCenter)

        info_box = QtWidgets.QFrame()
        info_box.setStyleSheet("background:#111922; border:1px solid #2a3440; border-radius:12px;")
        info_layout = QtWidgets.QVBoxLayout(info_box)
        info_layout.setContentsMargins(18, 18, 18, 18)
        info_layout.setSpacing(10)

        bullet_lines = [
            "• aucune modification du SolarDashboard / ClockDashboard / MUF / SOHO / Solar System",
            "• vue activable depuis Settings > Display",
            "• rotation compatible avec le système existant",
            "• prêt à servir de base pour un futur widget radio, satellites ou propagation",
        ]
        for line in bullet_lines:
            lbl = QtWidgets.QLabel(line)
            lbl.setWordWrap(True)
            lbl.setStyleSheet("font-size: 15px; color:#aab6c5; font-weight: 700;")
            info_layout.addWidget(lbl)

        root.addWidget(info_box, 0)

        self.lbl_status = QtWidgets.QLabel("Status: idle")
        self.lbl_status.setStyleSheet("font-family: Consolas; font-size: 15px; color:#4aa3ff; font-weight: 800;")
        root.addWidget(self.lbl_status, 0, QtCore.Qt.AlignHCenter)

        root.addStretch(1)

    def start(self) -> None:
        self.lbl_status.setText("Status: active")

    def stop(self) -> None:
        self.lbl_status.setText("Status: idle")

    def apply_config(self, cfg: dict) -> None:
        try:
            current = str(cfg.get("display_current_view") or "").strip().lower()
        except Exception:
            current = ""
        self.lbl_status.setText(f"Status: {'active' if current == self.widget_id else 'ready'}")






class SolarSystemDashboard(QtWidgets.QFrame):
    """Heliocentric dashboard with an Earth side panel."""

    skyfield_state_changed = QtCore.Signal()

    _PLANETS = [
        {"name": "Mercury", "skyfield_key": "mercury barycenter", "period_days": 87.969,   "radius_au": 0.387,  "phase_deg": 75.0,  "size": 7.5,  "body": "#b7b7b7", "ring": None, "kind": "planet", "subtitle": "Rocky inner planet"},
        {"name": "Venus",   "skyfield_key": "venus barycenter",   "period_days": 224.701,  "radius_au": 0.723,  "phase_deg": 120.0, "size": 10.5, "body": "#d9c59a", "ring": None, "kind": "planet", "subtitle": "Dense atmosphere"},
        {"name": "Earth",   "skyfield_key": "earth barycenter",   "period_days": 365.256,  "radius_au": 1.000,  "phase_deg": 200.0, "size": 11.0, "body": "#4aa3ff", "ring": None, "kind": "planet", "subtitle": "Home world"},
        {"name": "Mars",    "skyfield_key": "mars barycenter",    "period_days": 686.980,  "radius_au": 1.524,  "phase_deg": 15.0,  "size": 9.2,  "body": "#d96b43", "ring": None, "kind": "planet", "subtitle": "Cold desert world"},
        {"name": "Jupiter", "skyfield_key": "jupiter barycenter", "period_days": 4332.59,  "radius_au": 5.203,  "phase_deg": 260.0, "size": 18.5, "body": "#d7b28a", "ring": None, "kind": "planet", "subtitle": "Gas giant"},
        {"name": "Saturn",  "skyfield_key": "saturn barycenter",  "period_days": 10759.22, "radius_au": 9.537,  "phase_deg": 320.0, "size": 16.5, "body": "#e0c982", "ring": "#b99b5f", "kind": "planet", "subtitle": "Ringed giant"},
        {"name": "Uranus",  "skyfield_key": "uranus barycenter",  "period_days": 30688.5,  "radius_au": 19.191, "phase_deg": 40.0,  "size": 14.5, "body": "#82d7e8", "ring": None, "kind": "planet", "subtitle": "Ice giant"},
        {"name": "Neptune", "skyfield_key": "neptune barycenter", "period_days": 60182.0,  "radius_au": 30.07,  "phase_deg": 140.0, "size": 14.2, "body": "#4c7fe5", "ring": None, "kind": "planet", "subtitle": "Outer ice giant"},
    ]

    _MOONS = {
        "Earth": [
            {"name": "", "period_days": 27.321, "phase_deg": 50.0, "orbit_r": 15.0, "size": 2.8, "body": "#d7dde6"},
        ],
        "Mars": [
            {"name": "Phobos", "period_days": 0.319, "phase_deg": 85.0, "orbit_r": 11.0, "size": 2.1, "body": "#bba891"},
            {"name": "Deimos", "period_days": 1.263, "phase_deg": 205.0, "orbit_r": 16.0, "size": 1.7, "body": "#9f8d79"},
        ],
        "Jupiter": [
            {"name": "Io", "period_days": 1.769, "phase_deg": 15.0, "orbit_r": 13.0, "size": 2.6, "body": "#f1d26a"},
            {"name": "Europa", "period_days": 3.551, "phase_deg": 120.0, "orbit_r": 19.0, "size": 2.5, "body": "#dfe9f1"},
            {"name": "Ganymede", "period_days": 7.155, "phase_deg": 240.0, "orbit_r": 25.0, "size": 2.8, "body": "#a69788"},
            {"name": "Callisto", "period_days": 16.689, "phase_deg": 320.0, "orbit_r": 33.0, "size": 2.7, "body": "#85796d"},
        ],
        "Saturn": [
            {"name": "Titan", "period_days": 15.945, "phase_deg": 58.0, "orbit_r": 24.0, "size": 2.8, "body": "#d6c594"},
            {"name": "Enceladus", "period_days": 1.370, "phase_deg": 140.0, "orbit_r": 14.0, "size": 2.0, "body": "#ecf7ff"},
            {"name": "Rhea", "period_days": 4.518, "phase_deg": 250.0, "orbit_r": 18.0, "size": 2.2, "body": "#cdc9c0"},
            {"name": "Iapetus", "period_days": 79.321, "phase_deg": 330.0, "orbit_r": 37.0, "size": 2.0, "body": "#b8aa92"},
        ],
        "Uranus": [
            {"name": "Titania", "period_days": 8.706, "phase_deg": 70.0, "orbit_r": 17.0, "size": 2.4, "body": "#d1d7e0"},
            {"name": "Oberon", "period_days": 13.463, "phase_deg": 220.0, "orbit_r": 22.0, "size": 2.3, "body": "#babec7"},
            {"name": "Ariel", "period_days": 2.520, "phase_deg": 150.0, "orbit_r": 12.0, "size": 2.1, "body": "#edf3ff"},
        ],
        "Neptune": [
            {"name": "Triton", "period_days": 5.877, "phase_deg": 34.0, "orbit_r": 17.0, "size": 2.5, "body": "#dce8f5"},
        ],
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("solarSystemDashboard")
        self.setStyleSheet(
            "QFrame#solarSystemDashboard { background: #0b1016; border: 2px solid #243141; border-radius: 16px; }"
        )

        self._epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self._title = "Solar System"
        self._subtitle = "Solar system left / Earth system right"
        self._status = "Preparing Skyfield ephemerides…" if SKYFIELD_AVAILABLE else "Skyfield not installed • premium fallback in use"

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(5000)

        self._anim_timer = QtCore.QTimer(self)
        self._anim_timer.timeout.connect(self._tick_animation)
        self._anim_timer.start(40)

        self.setMinimumSize(420, 260)
        self.setMouseTracking(True)

        self._skyfield_loader = None
        self._skyfield_ts = None
        self._skyfield_eph_source = ""
        self._skyfield_eph = None
        self._skyfield_thread = None
        self._skyfield_ready = False
        self._skyfield_loading = False
        self._skyfield_error = ""
        self._skyfield_local_only = True
        self._iss_tle_url = ISS_TLE_URL
        self._iss_tle_cache_path = ISS_TLE_CACHE_PATH
        self._iss_satellite = None
        self._iss_tle_loading = False
        self._iss_tle_error = ""
        self._iss_tle_epoch_text = ""
        self._iss_tle_loaded_from = ""
        self._iss_waiting_for_skyfield_retry = False
        self._iss_refresh_attempts = 0
        self._iss_observer_lat = 48.1173
        self._iss_observer_lon = -1.6778
        self._iss_observer_alt_m = 60.0
        self._iss_min_elevation_deg = 10.0
        self._iss_pass_cache_key = ""
        self._iss_pass_cache = None
        self._earth_panel_cache_key = None
        self._earth_panel_cache = None
        self._earth_track_minutes = 28
        self._earth_track_step_seconds = 90

        self._hover_items = []
        self._hover_name = None
        self._position_cache_key = None
        self._position_cache = []
        self._zoom_factor = 1.0
        self._target_zoom_factor = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._target_pan_x = 0.0
        self._target_pan_y = 0.0
        self._focused_planet = None
        self._focus_progress = 0.0
        self._target_focus_progress = 0.0
        self._dragging = False
        self._last_mouse_pos = None
        self._solar_scene_rect = QtCore.QRectF()
        self._earth_panel_rect = QtCore.QRectF()
        self._earth_texture = None
        self._earth_texture_size = (0, 0)
        self._earth_globe_cache = None
        self._earth_globe_cache_key = None
        self._earth_globe_cache_max_age_s = 1.2
        self._earth_globe_render_scale = 0.72

        self._star_field = self._build_star_field()
        self.skyfield_state_changed.connect(self.update)
        QtCore.QTimer.singleShot(50, self._start_skyfield_prepare)
        QtCore.QTimer.singleShot(400, self._prepare_iss_tle)

    def status_text(self) -> str:
        return self._status

    def set_iss_observer(self, lat: float, lon: float, alt_m: float = 60.0, min_elevation_deg: float = 10.0) -> None:
        """Set observer location used for ISS pass prediction.

        This keeps orbital/ISS information independent from solar data. The values
        normally come from Settings -> DAPNET/ISS observer position.
        """
        try:
            self._iss_observer_lat = max(-90.0, min(90.0, float(lat)))
            self._iss_observer_lon = max(-180.0, min(180.0, float(lon)))
            self._iss_observer_alt_m = float(alt_m)
            self._iss_min_elevation_deg = max(0.0, min(80.0, float(min_elevation_deg)))
            self._iss_pass_cache_key = ""
            self._iss_pass_cache = None
            self.update()
        except Exception as exc:
            try:
                debug_log(f"ISS observer config ignored: {exc}")
            except Exception:
                pass

    def _tick_animation(self):
        try:
            changed = False
            for attr, target_attr, factor, eps in (
                ("_zoom_factor", "_target_zoom_factor", 0.18, 0.02),
                ("_pan_x", "_target_pan_x", 0.22, 0.5),
                ("_pan_y", "_target_pan_y", 0.22, 0.5),
                ("_focus_progress", "_target_focus_progress", 0.18, 0.01),
            ):
                cur = float(getattr(self, attr))
                target = float(getattr(self, target_attr))
                nxt = cur + (target - cur) * factor
                if abs(target - nxt) <= eps:
                    nxt = target
                if abs(nxt - cur) > 1e-9:
                    setattr(self, attr, nxt)
                    changed = True
            if self._target_focus_progress <= 0.0 and self._focus_progress <= 0.0:
                self._focused_planet = None
            if changed:
                self.update()
        except Exception:
            self.update()

    def wheelEvent(self, event):
        try:
            event.accept()
            return
        except Exception:
            pass
        super().wheelEvent(event)

    def mousePressEvent(self, event):
        try:
            if event.button() == QtCore.Qt.LeftButton and self._solar_scene_rect.contains(event.position()):
                self._dragging = True
                self._last_mouse_pos = event.position()
                event.accept()
                return
        except Exception:
            pass
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        try:
            if self._dragging and self._last_mouse_pos is not None:
                delta = event.position() - self._last_mouse_pos
                self._last_mouse_pos = event.position()
                factor = max(0.18, 1.0 / max(1.0, self._zoom_factor))
                self._pan_x += float(delta.x()) * factor
                self._pan_y += float(delta.y()) * factor
                self.update()
            self._update_hover_tooltip(event.globalPosition().toPoint(), event.position())
        except Exception:
            pass
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        try:
            if event.button() == QtCore.Qt.LeftButton:
                self._dragging = False
                self._last_mouse_pos = None
                event.accept()
                return
        except Exception:
            pass
        super().mouseReleaseEvent(event)

    def _hit_test_item(self, pos):
        try:
            point = QtCore.QPointF(pos)
            for item in reversed(self._hover_items):
                center = item.get("center")
                radius = float(item.get("radius", 0.0) or 0.0)
                if center is None or radius <= 0.0:
                    continue
                dx = point.x() - center.x()
                dy = point.y() - center.y()
                if (dx * dx) + (dy * dy) <= (radius * radius):
                    return item
        except Exception:
            pass
        return None

    def _focus_zoom_for_planet(self, name: str) -> float:
        moons = self._MOONS.get(name, [])
        moon_extent = max([float(m.get("orbit_r", 0.0) or 0.0) for m in moons] + [0.0])
        if name == "Earth":
            return 16.0
        if name in ("Jupiter", "Saturn"):
            return 12.0 if moon_extent > 30.0 else 14.0
        if name in ("Uranus", "Neptune"):
            return 15.0
        if name == "Mars":
            return 18.0
        return 20.0

    def _set_focus_planet(self, name: str | None):
        if name:
            self._focused_planet = str(name)
            self._target_focus_progress = 1.0
            self._target_zoom_factor = self._focus_zoom_for_planet(str(name))
            self._target_pan_x = 0.0
            self._target_pan_y = 0.0
        else:
            self._target_focus_progress = 0.0
            self._target_zoom_factor = 1.0
            self._target_pan_x = 0.0
            self._target_pan_y = 0.0
        self.update()

    def mouseDoubleClickEvent(self, event):
        try:
            if event.button() == QtCore.Qt.LeftButton and self._solar_scene_rect.contains(event.position()):
                item = self._hit_test_item(event.position())
                if item and item.get("kind") == "planet":
                    name = str(item.get("name", "") or "")
                    if name:
                        self._set_focus_planet(name)
                        event.accept()
                        return
                self._set_focus_planet(None)
                event.accept()
                return
        except Exception:
            pass
        super().mouseDoubleClickEvent(event)

    def _build_star_field(self) -> list[tuple[float, float, float, int]]:
        rng = random.Random(42)
        stars = []
        for _ in range(220):
            stars.append((
                rng.random(),
                rng.random(),
                rng.uniform(0.7, 2.1),
                rng.randint(35, 140),
            ))
        return stars

    def _start_skyfield_prepare(self):
        if not SKYFIELD_AVAILABLE or self._skyfield_loading or self._skyfield_ready:
            return
        self._skyfield_loading = True
        self._status = "Loading Skyfield ephemerides…"
        self.update()

        def _worker():
            try:
                # IMPORTANT v7.9.7:
                # Do NOT call SkyfieldLoader(...)(de421.bsp) directly here.
                # In PyInstaller builds Skyfield's internal downloader can crash with
                # "NoneType object has no attribute flush". Use SUMO's hybrid loader
                # instead: local/cache first, then controlled download into
                # APP_DIR/skyfield_data/de421.bsp.
                loader, eph, source_text = get_skyfield_loader_and_ephemeris()
                ts = loader.timescale()
                self._skyfield_loader = loader
                self._skyfield_ts = ts
                self._skyfield_eph = eph
                self._skyfield_eph_source = source_text
                self._skyfield_ready = True
                self._skyfield_error = ""
                self._status = f"Skyfield/JPL active • {source_text}"
                debug_log(f"SolarSystemDashboard Skyfield ready: {source_text}")
            except Exception as e:
                self._skyfield_ready = False
                self._skyfield_error = str(e)
                self._status = f"Skyfield unavailable • {e}"
                debug_log(f"SolarSystemDashboard Skyfield prepare failed: {e}")
            finally:
                self._skyfield_loading = False
                # Always retry ISS TLE once Skyfield state changes. If Skyfield is OK,
                # _prepare_iss_tle() will build the EarthSatellite. If not, it will expose
                # a clean ERROR instead of staying stuck in loading/pending.
                try:
                    QtCore.QTimer.singleShot(0, self._prepare_iss_tle)
                except Exception:
                    pass
                try:
                    self.skyfield_state_changed.emit()
                except Exception:
                    pass

        self._skyfield_thread = threading.Thread(target=_worker, daemon=True)
        self._skyfield_thread.start()

    def _object_tooltip_text(self, row: dict) -> str:
        try:
            body = row.get("body", {}) or {}
            name = str(row.get("name", body.get("name", "Object")))
            kind = str(row.get("kind", body.get("kind", "object"))).strip().lower()
            distance_au = float(row.get("distance_au", body.get("radius_au", 0.0)) or 0.0)
            lat_deg = float(row.get("lat_deg", 0.0) or 0.0)
            angle_deg = (math.degrees(float(row.get("angle_rad", 0.0) or 0.0)) + 360.0) % 360.0
            subtitle = str(row.get("subtitle", "") or "")
            source = str(row.get("source", "approximate") or "approximate")
            label_map = {"planet": "Planet", "spacecraft": "Spacecraft", "comet": "Comet", "station": "Station"}
            lines = [f"<b>{name}</b>", label_map.get(kind, "Object")]
            if subtitle:
                lines.append(subtitle)
            if kind in ("planet", "spacecraft", "comet"):
                lines.append(f"Distance from Sun: {distance_au:.3f} AU")
                lines.append(f"Ecliptic longitude: {angle_deg:.1f}°")
                lines.append(f"Ecliptic latitude: {lat_deg:+.2f}°")
            period_days = float(body.get("period_days", 0.0) or 0.0)
            if period_days > 0 and kind in ("planet", "comet"):
                lines.append(
                    f"Nominal period: {period_days / 365.25:.2f} years"
                    if period_days >= 365.25 else
                    f"Nominal period: {period_days:.1f} days"
                )
            target = str(row.get("target", body.get("target", "")) or "")
            if target:
                lines.append(f"Target: {target}")
            if kind in ("spacecraft", "comet"):
                a_au = float(row.get("a_au", body.get("a_au", 0.0)) or 0.0)
                ecc = float(row.get("ecc", body.get("ecc", 0.0)) or 0.0)
                lines.append(f"Stylized heliocentric orbit: a={a_au:.2f} AU, e={ecc:.3f}")
            if kind == "planet":
                moons = self._MOONS.get(name, [])
                if moons:
                    lines.append(f"Visible moons in zoom mode: {', '.join(m['name'] for m in moons)}")
                lines.append("Source: Skyfield/JPL" if source == "skyfield" else "Source: Approximate fallback")
            return "<br>".join(lines)
        except Exception:
            return "<b>Object</b>"

    def _simple_tooltip(self, title: str, lines: list[str]) -> str:
        return "<br>".join([f"<b>{title}</b>"] + [str(x) for x in lines if x])

    def _update_hover_tooltip(self, global_pos, local_pos) -> bool:
        try:
            point = QtCore.QPointF(local_pos)
            for item in reversed(self._hover_items):
                center = item.get("center")
                radius = float(item.get("radius", 0.0) or 0.0)
                if center is None or radius <= 0.0:
                    continue
                dx = point.x() - center.x()
                dy = point.y() - center.y()
                if (dx * dx) + (dy * dy) <= (radius * radius):
                    name = str(item.get("name", ""))
                    if self._hover_name != name:
                        self._hover_name = name
                        QtWidgets.QToolTip.showText(global_pos, item.get("tooltip", ""), self)
                    return True
        except Exception:
            pass

        if self._hover_name is not None:
            self._hover_name = None
            QtWidgets.QToolTip.hideText()
        return False

    def leaveEvent(self, event):
        self._hover_name = None
        self._dragging = False
        self._last_mouse_pos = None
        QtWidgets.QToolTip.hideText()
        super().leaveEvent(event)

    def _approx_planet_positions(self, now_utc: datetime) -> list[dict]:
        rows = []
        days = (now_utc - self._epoch).total_seconds() / 86400.0
        for body in self._PLANETS:
            period = max(1.0, float(body["period_days"]))
            angle_deg = float(body["phase_deg"]) + (days / period) * 360.0
            angle_rad = math.radians(angle_deg)
            dist = float(body["radius_au"])
            x_au = math.cos(angle_rad) * dist
            y_au = math.sin(angle_rad) * dist
            rows.append({
                "name": body["name"],
                "kind": "planet",
                "body": body,
                "distance_au": dist,
                "angle_rad": angle_rad,
                "lat_deg": 0.0,
                "x_au": x_au,
                "y_au": y_au,
                "z_au": 0.0,
                "source": "approximate",
                "subtitle": body.get("subtitle", ""),
            })
        return rows

    def _skyfield_planet_positions(self, now_utc: datetime) -> list[dict]:
        if not (self._skyfield_ready and self._skyfield_ts and self._skyfield_eph):
            return self._approx_planet_positions(now_utc)

        try:
            t = self._skyfield_ts.from_datetime(now_utc)
            sun = self._skyfield_eph["sun"]
            rows = []
            for body in self._PLANETS:
                key = body.get("skyfield_key")
                if not key or key not in self._skyfield_eph:
                    continue
                target = self._skyfield_eph[key]
                vec = sun.at(t).observe(target)
                pos = vec.frame_xyz(ecliptic_frame).au
                x_au = float(pos[0])
                y_au = float(pos[1])
                z_au = float(pos[2])
                distance_au = max(1e-9, math.sqrt(x_au*x_au + y_au*y_au + z_au*z_au))
                angle_rad = math.atan2(y_au, x_au)
                lat_deg = math.degrees(math.asin(max(-1.0, min(1.0, z_au / distance_au))))
                rows.append({
                    "name": body["name"],
                    "kind": "planet",
                    "body": body,
                    "distance_au": distance_au,
                    "angle_rad": angle_rad,
                    "lat_deg": lat_deg,
                    "x_au": x_au,
                    "y_au": y_au,
                    "z_au": z_au,
                    "source": "skyfield",
                    "subtitle": body.get("subtitle", ""),
                })
            if rows:
                return rows
        except Exception as e:
            self._status = f"Skyfield unavailable • {e}"

        return self._approx_planet_positions(now_utc)

    def _current_rows(self) -> list[dict]:
        now_utc = datetime.now(timezone.utc)
        key = (now_utc.minute // 1, round(self._zoom_factor, 2), self._skyfield_ready)
        if key == self._position_cache_key and self._position_cache:
            return self._position_cache

        rows = []
        rows.extend(self._skyfield_planet_positions(now_utc))

        self._position_cache_key = key
        self._position_cache = rows
        return rows

    def _draw_space_background(self, painter: QtGui.QPainter, rect: QtCore.QRectF):
        grad = QtGui.QRadialGradient(rect.center(), max(rect.width(), rect.height()) * 0.72)
        grad.setColorAt(0.0, QtGui.QColor("#101b2a"))
        grad.setColorAt(0.45, QtGui.QColor("#0a1119"))
        grad.setColorAt(1.0, QtGui.QColor("#05080d"))
        painter.fillRect(rect, grad)

        for sx, sy, size, alpha in self._star_field:
            x = rect.left() + sx * rect.width()
            y = rect.top() + sy * rect.height()
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(220, 235, 255, alpha))
            painter.drawEllipse(QtCore.QPointF(x, y), size, size)

        vignette = QtGui.QRadialGradient(rect.center(), max(rect.width(), rect.height()) * 0.85)
        vignette.setColorAt(0.0, QtGui.QColor(0, 0, 0, 0))
        vignette.setColorAt(0.72, QtGui.QColor(0, 0, 0, 0))
        vignette.setColorAt(1.0, QtGui.QColor(0, 0, 0, 115))
        painter.fillRect(rect, vignette)

    def _draw_solar_system_overview(self, painter: QtGui.QPainter, solar_rect: QtCore.QRectF, opacity: float = 1.0):
        if opacity <= 0.0:
            return
        painter.save()
        painter.setOpacity(max(0.0, min(1.0, opacity)))
        solar_scene = solar_rect.adjusted(12, 44, -12, -12)
        center = QtCore.QPointF(
            solar_scene.center().x() + self._pan_x,
            solar_scene.center().y() + self._pan_y,
        )
        max_radius_px = min(solar_scene.width(), solar_scene.height()) * 0.47 * self._zoom_factor

        self._draw_sun(painter, center)

        rows = self._current_rows()
        planet_rows = [r for r in rows if r["kind"] == "planet"]
        other_rows = [r for r in rows if r["kind"] != "planet"]

        for row in planet_rows:
            dist_au = float(row["distance_au"])
            orbit_r = self._orbital_scale(dist_au, max_radius_px)
            self._draw_orbit(painter, center, orbit_r, alpha=65 if dist_au < 3 else 45)

        for row in other_rows:
            a_au = float(row.get("a_au", row["distance_au"]))
            orbit_r = self._orbital_scale(a_au, max_radius_px)
            self._draw_orbit(painter, center, orbit_r, alpha=22)

        draw_rows = sorted(rows, key=lambda r: float(r["distance_au"]))
        for row in draw_rows:
            body = row["body"]
            name = row["name"]
            kind = row["kind"]
            angle = float(row["angle_rad"])
            dist_au = float(row["distance_au"])
            orbit_r = self._orbital_scale(dist_au, max_radius_px)
            x = center.x() + math.cos(angle) * orbit_r
            y = center.y() + math.sin(angle) * orbit_r * 0.94

            base_size = float(body.get("size", 6.0))
            radius = max(2.6, base_size * (0.82 + 0.10 * math.log1p(self._zoom_factor)))
            hovered = (self._hover_name == name)

            if kind == "planet":
                if name == "Saturn":
                    self._draw_saturn_rings(painter, QtCore.QPointF(x, y), radius)
                self._draw_body_sphere(painter, QtCore.QPointF(x, y), radius, body.get("body", "#cccccc"), hovered)
                if name == "Jupiter":
                    self._draw_jupiter_bands(painter, QtCore.QPointF(x, y), radius)

            if kind == "planet":
                self._draw_label(painter, name, x, y, hovered)

            hover_radius = max(8.0, radius + 4.0)
            self._hover_items.append({
                "name": name,
                "kind": kind,
                "center": QtCore.QPointF(x, y),
                "radius": hover_radius,
                "tooltip": self._object_tooltip_text(row),
            })
        painter.restore()

    def _draw_focused_planet_scene(self, painter: QtGui.QPainter, solar_rect: QtCore.QRectF, planet_name: str, opacity: float = 1.0):
        if opacity <= 0.0:
            return
        rows = self._current_rows()
        row = next((r for r in rows if r.get("kind") == "planet" and r.get("name") == planet_name), None)
        if row is None:
            return

        painter.save()
        painter.setOpacity(max(0.0, min(1.0, opacity)))

        solar_scene = solar_rect.adjusted(12, 44, -12, -12)
        center = solar_scene.center()
        body = row["body"]
        moons = self._MOONS.get(planet_name, [])
        max_moon_orbit = max([float(m.get("orbit_r", 0.0) or 0.0) for m in moons] + [0.0])

        usable_radius = min(solar_scene.width(), solar_scene.height()) * 0.42
        if max_moon_orbit > 0.0:
            planet_radius = min(usable_radius * 0.58, usable_radius * (0.82 - min(0.36, max_moon_orbit / 120.0)))
        else:
            planet_radius = usable_radius * 0.72
        planet_radius = max(32.0, planet_radius)

        halo_grad = QtGui.QRadialGradient(center, planet_radius * 1.55)
        halo_grad.setColorAt(0.0, QtGui.QColor(120, 170, 255, 38))
        halo_grad.setColorAt(0.68, QtGui.QColor(70, 110, 180, 16))
        halo_grad.setColorAt(1.0, QtGui.QColor(0, 0, 0, 0))
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(halo_grad)
        painter.drawEllipse(center, planet_radius * 1.55, planet_radius * 1.55)

        if planet_name == "Saturn":
            self._draw_saturn_rings(painter, center, planet_radius)
        self._draw_body_sphere(painter, center, planet_radius, body.get("body", "#cccccc"), True)
        if planet_name == "Jupiter":
            self._draw_jupiter_bands(painter, center, planet_radius)

        title_font = QtGui.QFont("Segoe UI", 12, QtGui.QFont.Bold)
        painter.setFont(title_font)
        painter.setPen(QtGui.QColor("#ecf3ff"))
        title_rect = QtCore.QRectF(solar_scene.left() + 6, solar_scene.top() + 4, solar_scene.width() - 12, 24)
        painter.drawText(title_rect, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, f"{planet_name} system")

        sub_font = QtGui.QFont("Segoe UI", 9)
        painter.setFont(sub_font)
        painter.setPen(QtGui.QColor("#9cb0c7"))
        sub_rect = QtCore.QRectF(solar_scene.left() + 6, solar_scene.top() + 28, solar_scene.width() - 12, 18)
        moon_count = len(moons)
        sub_text = "Double-click empty space to return • no wheel zoom"
        if moon_count > 0:
            sub_text = f"{moon_count} visible moon{'s' if moon_count > 1 else ''} • double-click empty space to return"
        painter.drawText(sub_rect, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, sub_text)

        now_utc = datetime.now(timezone.utc)
        days = (now_utc - self._epoch).total_seconds() / 86400.0
        orbit_scale = 1.0
        if max_moon_orbit > 0.0:
            orbit_scale = min(planet_radius * 1.45 / max_moon_orbit, min(solar_scene.width(), solar_scene.height()) * 0.44 / max_moon_orbit)

        # Pre-calculate moon positions with depth for 3D sorting
        moon_data = []
        for moon in moons:
            moon_orbit = float(moon["orbit_r"]) * orbit_scale
            moon_period = max(0.05, float(moon["period_days"]))
            moon_phase = math.radians(float(moon["phase_deg"]) + (days / moon_period) * 360.0)
            mx = center.x() + math.cos(moon_phase) * moon_orbit
            my = center.y() + math.sin(moon_phase) * moon_orbit * 0.65
            
            # Depth based on sin(angle): positive = behind, negative = in front
            depth = math.sin(moon_phase)
            
            moon_radius = max(4.0, planet_radius * 0.085 * float(moon.get("size", 2.2)) / 2.8)
            
            moon_data.append({
                "moon": moon,
                "depth": depth,
                "moon_orbit": moon_orbit,
                "moon_phase": moon_phase,
                "mx": mx,
                "my": my,
                "moon_radius": moon_radius,
            })
        
        # Draw orbits first
        for data in moon_data:
            painter.setPen(QtGui.QPen(QtGui.QColor(190, 205, 225, 58), 1.2))
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawEllipse(center, data["moon_orbit"], data["moon_orbit"] * 0.65)
        
        # Draw moons behind the planet (depth > 0) in reduced opacity
        for data in sorted(moon_data, key=lambda d: d["depth"], reverse=True):
            if data["depth"] > 0:
                moon = data["moon"]
                mx, my = data["mx"], data["my"]
                moon_radius = data["moon_radius"]
                
                # Draw with reduced opacity to indicate it's behind
                painter.save()
                painter.setOpacity(0.35)
                self._draw_body_sphere(painter, QtCore.QPointF(mx, my), moon_radius, moon["body"], False)
                painter.restore()
                
                # Add to hover items but mark as behind
                self._hover_items.append({
                    "name": moon["name"],
                    "kind": "moon",
                    "center": QtCore.QPointF(mx, my),
                    "radius": max(8.0, moon_radius + 4.0),
                    "tooltip": f"<b>{moon['name']}</b> (behind)<br> of {planet_name}<br>Orbital period: {float(moon['period_days']):.3f} days",
                })
        
        # Draw moons in front of the planet (depth <= 0) in full opacity
        for data in sorted(moon_data, key=lambda d: d["depth"]):
            if data["depth"] <= 0:
                moon = data["moon"]
                mx, my = data["mx"], data["my"]
                moon_radius = data["moon_radius"]
                
                self._draw_body_sphere(painter, QtCore.QPointF(mx, my), moon_radius, moon["body"], False)
                self._draw_label(painter, moon["name"], mx, my, False)
                
                self._hover_items.append({
                    "name": moon["name"],
                    "kind": "moon",
                    "center": QtCore.QPointF(mx, my),
                    "radius": max(8.0, moon_radius + 4.0),
                    "tooltip": f"<b>{moon['name']}</b><br> of {planet_name}<br>Orbital period: {float(moon['period_days']):.3f} days",
                })

        self._hover_items.append({
            "name": planet_name,
            "kind": "planet",
            "center": center,
            "radius": max(18.0, planet_radius + 8.0),
            "tooltip": self._object_tooltip_text(row),
        })
        painter.restore()

    def _draw_panel_card(self, painter: QtGui.QPainter, rect: QtCore.QRectF, title: str, subtitle: str = ""):
        painter.save()
        painter.setPen(QtGui.QPen(QtGui.QColor(70, 96, 125, 140), 1.2))
        panel_grad = QtGui.QLinearGradient(rect.topLeft(), rect.bottomLeft())
        panel_grad.setColorAt(0.0, QtGui.QColor(10, 18, 28, 170))
        panel_grad.setColorAt(1.0, QtGui.QColor(6, 11, 18, 205))
        painter.setBrush(panel_grad)
        painter.drawRoundedRect(rect, 14, 14)

        title_rect = QtCore.QRectF(rect.left() + 12, rect.top() + 10, rect.width() - 24, 18)
        sub_rect = QtCore.QRectF(rect.left() + 12, rect.top() + 28, rect.width() - 24, 14)
        painter.setFont(QtGui.QFont("Segoe UI", 10, QtGui.QFont.Bold))
        painter.setPen(QtGui.QColor("#ecf3ff"))
        painter.drawText(title_rect, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, title)
        if subtitle:
            painter.setFont(QtGui.QFont("Segoe UI", 8))
            painter.setPen(QtGui.QColor("#91a8c4"))
            painter.drawText(sub_rect, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, subtitle)
        painter.restore()

    def _draw_sun(self, painter: QtGui.QPainter, center: QtCore.QPointF):
        for radius, color in [
            (68, QtGui.QColor(255, 180, 70, 24)),
            (48, QtGui.QColor(255, 190, 80, 42)),
            (30, QtGui.QColor(255, 210, 110, 85)),
        ]:
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(center, radius, radius)

        core_grad = QtGui.QRadialGradient(center, 20)
        core_grad.setColorAt(0.0, QtGui.QColor("#fff7cf"))
        core_grad.setColorAt(0.35, QtGui.QColor("#ffd56a"))
        core_grad.setColorAt(0.8, QtGui.QColor("#ff9f2f"))
        core_grad.setColorAt(1.0, QtGui.QColor("#ff7b21"))
        painter.setBrush(core_grad)
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 220, 140, 110), 1.2))
        painter.drawEllipse(center, 18, 18)

    def _orbital_scale(self, au: float, max_radius_px: float) -> float:
        return math.log1p(max(0.0, au) * 1.85) * max_radius_px / math.log1p(30.07 * 1.85)

    def _draw_orbit(self, painter: QtGui.QPainter, center: QtCore.QPointF, radius_px: float, alpha: int = 90):
        pen = QtGui.QPen(QtGui.QColor(135, 160, 190, alpha), 1.0)
        pen.setStyle(QtCore.Qt.SolidLine)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawEllipse(center, radius_px, radius_px)

    def _draw_body_sphere(self, painter: QtGui.QPainter, center: QtCore.QPointF, radius: float, color_hex: str, hovered: bool = False):
        base = QtGui.QColor(color_hex)
        highlight = QtGui.QColor(255, 255, 255, 150 if hovered else 105)
        shadow = QtGui.QColor(0, 0, 0, 110)

        painter.setPen(QtCore.Qt.NoPen)

        shadow_center = QtCore.QPointF(center.x() + radius * 0.28, center.y() + radius * 0.28)
        painter.setBrush(shadow)
        painter.drawEllipse(shadow_center, radius * 1.02, radius * 1.02)

        grad = QtGui.QRadialGradient(
            QtCore.QPointF(center.x() - radius * 0.33, center.y() - radius * 0.36),
            radius * 1.25
        )
        grad.setColorAt(0.0, highlight)
        grad.setColorAt(0.18, base.lighter(150))
        grad.setColorAt(0.62, base)
        grad.setColorAt(1.0, base.darker(175))
        painter.setBrush(grad)
        painter.drawEllipse(center, radius, radius)

        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 55 if hovered else 28), 1.0))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawEllipse(center, radius, radius)

    def _draw_moon_sphere(self, painter: QtGui.QPainter, center: QtCore.QPointF, radius: float, hovered: bool = False):
        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        shadow_center = QtCore.QPointF(center.x() + radius * 0.34, center.y() + radius * 0.36)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(0, 0, 0, 105))
        painter.drawEllipse(shadow_center, radius * 1.02, radius * 1.02)

        moon_path = QtGui.QPainterPath()
        moon_path.addEllipse(center, radius, radius)

        grad = QtGui.QRadialGradient(
            QtCore.QPointF(center.x() - radius * 0.34, center.y() - radius * 0.38),
            radius * 1.24
        )
        grad.setColorAt(0.00, QtGui.QColor(255, 255, 255, 235 if hovered else 220))
        grad.setColorAt(0.22, QtGui.QColor(244, 246, 250, 255))
        grad.setColorAt(0.60, QtGui.QColor(214, 218, 225, 255))
        grad.setColorAt(0.84, QtGui.QColor(160, 167, 177, 255))
        grad.setColorAt(1.00, QtGui.QColor(108, 114, 123, 255))
        painter.fillPath(moon_path, grad)

        painter.save()
        painter.setClipPath(moon_path)

        maria_specs = [
            (-0.30, -0.18, 0.42, 0.26, -18, QtGui.QColor(116, 121, 130, 92)),   # Oceanus Procellarum
            (-0.08, -0.24, 0.22, 0.17, -8, QtGui.QColor(104, 110, 120, 82)),    # Mare Imbrium
            (0.13, -0.12, 0.18, 0.14, 8, QtGui.QColor(106, 111, 121, 78)),      # Serenitatis
            (0.23, -0.01, 0.16, 0.12, 4, QtGui.QColor(110, 115, 124, 75)),      # Tranquillitatis
            (0.34, -0.05, 0.13, 0.11, 12, QtGui.QColor(102, 108, 118, 82)),     # Crisium
            (0.10, 0.17, 0.18, 0.12, 0, QtGui.QColor(112, 117, 127, 68)),       # Nubium/Fecunditatis zone
            (-0.18, 0.10, 0.15, 0.10, -12, QtGui.QColor(118, 123, 132, 60)),    # Humorum area
        ]
        for ox, oy, wx, hy, rot, color in maria_specs:
            painter.save()
            painter.translate(center)
            painter.rotate(rot)
            rect = QtCore.QRectF((ox - wx / 2.0) * radius, (oy - hy / 2.0) * radius, wx * radius, hy * radius)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(rect)
            painter.restore()

        crater_specs = [
            (-0.44, -0.40, 0.16, 0.16, 85),
            (-0.04, -0.05, 0.07, 0.07, 54),
            (0.18, 0.22, 0.08, 0.08, 40),
            (0.33, -0.24, 0.06, 0.06, 46),
        ]
        for ox, oy, wr, hr, alpha in crater_specs:
            ring = QtGui.QColor(245, 247, 250, alpha)
            fill = QtGui.QColor(140, 145, 154, max(20, alpha - 32))
            rect = QtCore.QRectF(center.x() + (ox - wr / 2.0) * radius, center.y() + (oy - hr / 2.0) * radius, wr * radius, hr * radius)
            painter.setPen(QtGui.QPen(ring, max(0.8, radius * 0.028)))
            painter.setBrush(fill)
            painter.drawEllipse(rect)

        phase_grad = QtGui.QLinearGradient(
            QtCore.QPointF(center.x() - radius * 0.95, center.y() - radius * 0.10),
            QtCore.QPointF(center.x() + radius * 1.05, center.y() + radius * 0.14),
        )
        phase_grad.setColorAt(0.00, QtGui.QColor(14, 18, 28, 162))
        phase_grad.setColorAt(0.36, QtGui.QColor(18, 24, 36, 112))
        phase_grad.setColorAt(0.54, QtGui.QColor(28, 36, 52, 36))
        phase_grad.setColorAt(0.70, QtGui.QColor(0, 0, 0, 0))
        phase_grad.setColorAt(1.00, QtGui.QColor(0, 0, 0, 0))
        painter.fillPath(moon_path, phase_grad)

        rim_grad = QtGui.QRadialGradient(QtCore.QPointF(center.x() - radius * 0.82, center.y() - radius * 0.22), radius * 1.35)
        rim_grad.setColorAt(0.72, QtGui.QColor(255, 255, 255, 0))
        rim_grad.setColorAt(0.90, QtGui.QColor(210, 225, 255, 0 if hovered else 18))
        rim_grad.setColorAt(1.00, QtGui.QColor(220, 235, 255, 44 if hovered else 28))
        painter.fillPath(moon_path, rim_grad)
        painter.restore()

        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 64 if hovered else 34), 1.0))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawEllipse(center, radius, radius)
        painter.restore()

    def _draw_saturn_rings(self, painter: QtGui.QPainter, center: QtCore.QPointF, radius: float):
        painter.save()
        painter.translate(center)
        painter.rotate(-18)

        ring_pen_outer = QtGui.QPen(QtGui.QColor(235, 213, 164, 110), 3.0)
        ring_pen_mid = QtGui.QPen(QtGui.QColor(210, 185, 135, 90), 1.6)
        ring_pen_inner = QtGui.QPen(QtGui.QColor(255, 238, 188, 65), 1.0)

        painter.setBrush(QtCore.Qt.NoBrush)
        painter.setPen(ring_pen_outer)
        painter.drawEllipse(QtCore.QRectF(-radius*2.1, -radius*0.72, radius*4.2, radius*1.44))

        painter.setPen(ring_pen_mid)
        painter.drawEllipse(QtCore.QRectF(-radius*1.85, -radius*0.58, radius*3.7, radius*1.16))

        painter.setPen(ring_pen_inner)
        painter.drawEllipse(QtCore.QRectF(-radius*1.55, -radius*0.47, radius*3.1, radius*0.94))
        painter.restore()

    def _draw_jupiter_bands(self, painter: QtGui.QPainter, center: QtCore.QPointF, radius: float):
        painter.save()
        path = QtGui.QPainterPath()
        path.addEllipse(center, radius, radius)
        painter.setClipPath(path)

        for dy, alpha, width in [(-0.45, 48, 0.22), (-0.12, 36, 0.16), (0.18, 42, 0.18), (0.46, 30, 0.14)]:
            r = QtCore.QRectF(center.x() - radius, center.y() + dy * radius - width * radius / 2, radius * 2, width * radius)
            painter.fillRect(r, QtGui.QColor(125, 85, 50, alpha))
        painter.restore()

    def _draw_spacecraft(self, painter: QtGui.QPainter, center: QtCore.QPointF, radius: float, body_color: str, hovered: bool):
        painter.save()
        painter.translate(center)

        glow = QtGui.QColor(255, 255, 255, 45 if hovered else 25)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(glow)
        painter.drawEllipse(QtCore.QPointF(0, 0), radius * 1.65, radius * 1.65)

        painter.setBrush(QtGui.QColor(body_color))
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 120), 1.0))
        painter.drawRoundedRect(QtCore.QRectF(-radius*0.55, -radius*0.35, radius*1.1, radius*0.7), 2, 2)

        panel_color = QtGui.QColor("#7fb8ff")
        painter.setBrush(panel_color)
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawRect(QtCore.QRectF(-radius*1.5, -radius*0.16, radius*0.75, radius*0.32))
        painter.drawRect(QtCore.QRectF(radius*0.75, -radius*0.16, radius*0.75, radius*0.32))

        painter.setBrush(QtGui.QColor("#d7dde6"))
        painter.drawEllipse(QtCore.QPointF(radius*0.75, 0), radius*0.18, radius*0.18)
        painter.restore()

    def _draw_label(self, painter: QtGui.QPainter, text: str, x: float, y: float, hovered: bool = False):
        font = QtGui.QFont("Segoe UI", 8 if not hovered else 9)
        font.setBold(hovered)
        painter.setFont(font)

        metrics = QtGui.QFontMetrics(font)
        w = metrics.horizontalAdvance(text) + 10
        h = metrics.height() + 4
        rect = QtCore.QRectF(x + 8, y - h * 0.7, w, h)

        bg = QtGui.QColor(10, 18, 28, 160 if hovered else 120)
        border = QtGui.QColor(160, 185, 215, 85 if hovered else 45)

        painter.setPen(QtGui.QPen(border, 1.0))
        painter.setBrush(bg)
        painter.drawRoundedRect(rect, 7, 7)

        painter.setPen(QtGui.QColor("#e8eef7" if hovered else "#c9d5e5"))
        painter.drawText(rect, QtCore.Qt.AlignCenter, text)

    def _normalize_degrees(self, value: float) -> float:
        try:
            v = float(value)
        except Exception:
            return 0.0
        while v <= -180.0:
            v += 360.0
        while v > 180.0:
            v -= 360.0
        return v

    def _safe_xyz_km(self, position_obj) -> tuple[float, float, float] | None:
        try:
            xyz = None
            if hasattr(position_obj, "xyz"):
                xyz = position_obj.xyz.km
            elif hasattr(position_obj, "position"):
                xyz = position_obj.position.km
            if xyz is None:
                return None
            return float(xyz[0]), float(xyz[1]), float(xyz[2])
        except Exception:
            return None

    def _prepare_iss_tle(self):
        if not SKYFIELD_AVAILABLE or EarthSatellite is None:
            self._iss_tle_loading = False
            self._iss_tle_error = f"Skyfield unavailable: {SKYFIELD_IMPORT_ERROR}"
            debug_log(f"ISS TLE skipped: {self._iss_tle_error}")
            try:
                self.skyfield_state_changed.emit()
            except Exception:
                pass
            return
        if self._iss_tle_loading:
            debug_log("ISS TLE skipped: load already in progress")
            return
        if self._skyfield_ts is None:
            self._iss_tle_loading = False
            if getattr(self, "_skyfield_loading", False):
                self._iss_tle_error = "Skyfield loading"
                debug_log("ISS TLE waiting for Skyfield timescale")
                # Skyfield is prepared asynchronously. The first ISS TLE load can
                # happen too early during startup, so keep retrying until the
                # timescale is actually available instead of ending in ERROR.
                if not getattr(self, "_iss_waiting_for_skyfield_retry", False):
                    self._iss_waiting_for_skyfield_retry = True
                    try:
                        def _retry_after_skyfield():
                            try:
                                self._iss_waiting_for_skyfield_retry = False
                                self._prepare_iss_tle()
                            except Exception as retry_exc:
                                debug_log(f"ISS TLE retry-after-skyfield failed: {retry_exc}")
                        QtCore.QTimer.singleShot(2000, _retry_after_skyfield)
                        debug_log("ISS TLE retry scheduled after Skyfield loading")
                    except Exception as sched_exc:
                        debug_log(f"ISS TLE retry schedule failed: {sched_exc}")
            else:
                self._iss_tle_error = str(getattr(self, "_skyfield_error", "") or "Skyfield timescale unavailable")
                debug_log(f"ISS TLE skipped: {self._iss_tle_error}")
            try:
                self.skyfield_state_changed.emit()
            except Exception:
                pass
            return
        self._iss_tle_loading = True
        self._iss_refresh_attempts += 1
        debug_log(f"ISS TLE prepare start attempt={self._iss_refresh_attempts}")

        def _worker():
            sat = None
            epoch_text = ""
            err = ""
            loaded_from = ""
            previous_sat = self._iss_satellite
            previous_epoch_text = self._iss_tle_epoch_text
            previous_loaded_from = self._iss_tle_loaded_from

            def _parse_tle_blob(blob: str):
                """Parse ISS TLE robustly from Celestrak stations.txt or GP response.

                The previous parser accepted the first valid TLE triplet it found. That
                could fail silently or select the wrong satellite depending on the
                response format. We now explicitly prefer ISS (ZARYA) / CATNR 25544,
                then fall back to any 25544 TLE pair.
                """
                lines = [ln.strip() for ln in str(blob or "").splitlines() if ln.strip()]
                if len(lines) < 2:
                    return None

                # 1) Preferred Celestrak stations.txt format:
                #    ISS (ZARYA)
                #    1 25544U ...
                #    2 25544 ...
                for i in range(len(lines) - 2):
                    name = lines[i]
                    line1 = lines[i + 1]
                    line2 = lines[i + 2]
                    if (
                        "ISS" in name.upper()
                        and line1.startswith("1 25544")
                        and line2.startswith("2 25544")
                    ):
                        return name, line1, line2

                # 2) GP/TLE response or cache without a recognizable name.
                for i in range(len(lines) - 1):
                    line1 = lines[i]
                    line2 = lines[i + 1]
                    if line1.startswith("1 25544") and line2.startswith("2 25544"):
                        name = "ISS (ZARYA)"
                        if i > 0 and not lines[i - 1].startswith(("1 ", "2 ")):
                            name = lines[i - 1]
                        return name, line1, line2

                return None

            def _candidate_paths() -> list[Path]:
                seen = set()
                out = []
                for p in [
                    Path(self._iss_tle_cache_path),
                    resource_path('assets/iss_stations.tle'),
                    resource_path('assets/iss.tle'),
                ]:
                    key = str(p)
                    if key not in seen:
                        out.append(p)
                        seen.add(key)
                return out

            def _build_satellite(parsed, source_label: str):
                tle_name, line1, line2 = parsed
                satellite = EarthSatellite(line1, line2, tle_name, self._skyfield_ts)
                info_text = tle_name
                try:
                    epoch_dt = satellite.epoch.utc_datetime()
                    info_text = f"{tle_name} • TLE epoch {epoch_dt.strftime('%Y-%m-%d %H:%M UTC')}"
                except Exception:
                    pass
                return satellite, info_text, source_label

            def _load_best_local():
                best = None
                for candidate in _candidate_paths():
                    try:
                        if not candidate.exists():
                            continue
                        blob = candidate.read_text(encoding='utf-8', errors='ignore')
                        parsed = _parse_tle_blob(blob)
                        if parsed is None:
                            debug_log(f"ISS TLE local candidate invalid: {candidate}")
                            continue
                        mtime = float(candidate.stat().st_mtime)
                        age_s = max(0.0, time.time() - mtime)
                        item = {
                            'path': candidate,
                            'parsed': parsed,
                            'mtime': mtime,
                            'age_s': age_s,
                        }
                        if best is None or item['mtime'] > best['mtime']:
                            best = item
                    except Exception as e:
                        debug_log(f"ISS TLE local candidate read failed: {candidate} • {e}")
                return best

            try:
                local_item = _load_best_local()
                if local_item is not None:
                    debug_log(
                        f"ISS TLE local candidate selected: {local_item['path']} age_hours={local_item['age_s'] / 3600.0:.2f}"
                    )
                    try:
                        sat, epoch_text, loaded_from = _build_satellite(local_item['parsed'], f"local cache • {local_item['path'].name}")
                        debug_log(f"ISS TLE local load OK: {epoch_text}")
                    except Exception as e:
                        sat = None
                        epoch_text = ""
                        loaded_from = ""
                        err = f"ISS local TLE parse failed: {e}"
                        debug_log(err)
                else:
                    debug_log("ISS TLE local candidate not found")

                remote_error = ""
                for attempt in range(1, 4):
                    try:
                        debug_log(f"ISS TLE remote fetch attempt={attempt} url={self._iss_tle_url}")
                        remote_blob = safe_get_text(self._iss_tle_url, timeout=20)
                        debug_log(f"ISS TLE remote blob size={len(remote_blob)}")
                        remote_parsed = _parse_tle_blob(remote_blob)
                        if remote_parsed is None:
                            raise ValueError('remote ISS TLE not found in response')
                        try:
                            Path(self._iss_tle_cache_path).write_text(remote_blob, encoding='utf-8')
                            debug_log(f"ISS TLE cache updated: {self._iss_tle_cache_path}")
                        except Exception as cache_exc:
                            debug_log(f"ISS TLE cache write failed: {cache_exc}")
                        sat, epoch_text, loaded_from = _build_satellite(remote_parsed, 'remote refresh')
                        err = ""
                        remote_error = ""
                        debug_log(f"ISS TLE remote load OK: {epoch_text}")
                        break
                    except Exception as e:
                        remote_error = str(e)
                        debug_log(f"ISS TLE remote fetch failed attempt={attempt}: {remote_error}")
                        time.sleep(0.35 * attempt)

                if sat is None and local_item is not None:
                    try:
                        sat, epoch_text, loaded_from = _build_satellite(local_item['parsed'], f"local cache • {local_item['path'].name}")
                        err = f"Remote refresh failed, using cached TLE"
                        debug_log(err)
                    except Exception as e:
                        sat = None
                        epoch_text = ""
                        loaded_from = ""
                        err = f"ISS local fallback parse failed: {e}"
                        debug_log(err)

                if sat is None and previous_sat is not None:
                    sat = previous_sat
                    epoch_text = previous_epoch_text
                    loaded_from = previous_loaded_from or 'previous session cache'
                    err = err or (f"ISS refresh failed, keeping previous TLE • {remote_error}" if remote_error else "ISS refresh failed, keeping previous TLE")
                    debug_log(err)
                elif sat is not None and remote_error:
                    err = f"Remote refresh failed, using cached TLE • {remote_error}"
                    debug_log(err)
                elif sat is None and remote_error:
                    err = f"ISS TLE unavailable: {remote_error}"
                    debug_log(err)
                elif sat is None and not err:
                    err = "ISS TLE unavailable"
                    debug_log(err)
            except Exception as e:
                err = str(e)
                debug_log(f"ISS TLE worker fatal error: {err}")

            self._iss_waiting_for_skyfield_retry = False
            self._iss_satellite = sat
            self._iss_tle_epoch_text = epoch_text
            self._iss_tle_error = err
            self._iss_tle_loaded_from = loaded_from
            self._iss_tle_loading = False

            if sat is None:
                try:
                    QtCore.QTimer.singleShot(15000, self._prepare_iss_tle)
                    debug_log("ISS TLE retry scheduled in 15s")
                except Exception:
                    pass

            try:
                self.skyfield_state_changed.emit()
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True).start()

    def _earth_rotation_longitude_deg(self, now_utc: datetime) -> float:
        try:
            return self._normalize_degrees(((now_utc.hour * 3600 + now_utc.minute * 60 + now_utc.second) / 86164.0905) * 360.0 - 180.0)
        except Exception:
            return 0.0

    def _orthographic_project(self, lat_deg: float, lon_deg: float, center: QtCore.QPointF, radius: float, lon0_deg: float, lat0_deg: float = 12.0):
        lat = math.radians(float(lat_deg))
        lon = math.radians(float(lon_deg))
        lon0 = math.radians(float(lon0_deg))
        lat0 = math.radians(float(lat0_deg))
        cosc = math.sin(lat0) * math.sin(lat) + math.cos(lat0) * math.cos(lat) * math.cos(lon - lon0)
        x = radius * math.cos(lat) * math.sin(lon - lon0)
        y = -radius * (math.cos(lat0) * math.sin(lat) - math.sin(lat0) * math.cos(lat) * math.cos(lon - lon0))
        return QtCore.QPointF(center.x() + x, center.y() + y), cosc > 0.0, cosc

    def _earth_continent_polygons(self) -> list[tuple[str, list[tuple[float, float]]]]:
        return [
            ("North America", [(-168, 71), (-160, 66), (-150, 61), (-142, 58), (-134, 55), (-128, 50), (-124, 46), (-124, 40), (-120, 36), (-117, 33), (-114, 31), (-111, 29), (-107, 27), (-103, 25), (-99, 24), (-95, 25), (-91, 28), (-88, 30), (-85, 29), (-83, 26), (-82, 23), (-80, 20), (-79, 17), (-77, 12), (-79, 9), (-82, 10), (-84, 14), (-86, 18), (-90, 21), (-95, 23), (-101, 28), (-107, 31), (-114, 34), (-119, 39), (-124, 44), (-129, 50), (-138, 56), (-150, 61)]),
            ("South America", [(-81, 12), (-78, 8), (-75, 4), (-72, 0), (-70, -5), (-68, -10), (-66, -16), (-64, -22), (-62, -28), (-60, -34), (-60, -40), (-62, -46), (-66, -52), (-70, -55), (-73, -53), (-75, -48), (-76, -42), (-77, -36), (-78, -28), (-79, -20), (-80, -10), (-81, 0), (-81, 12)]),
            ("Greenland", [(-54, 60), (-50, 64), (-44, 69), (-38, 73), (-30, 75), (-24, 72), (-22, 68), (-28, 63), (-36, 60), (-46, 59), (-54, 60)]),
            ("Europe", [(-10, 36), (-8, 42), (-5, 46), (0, 49), (6, 51), (12, 54), (18, 57), (24, 59), (30, 60), (34, 58), (31, 54), (26, 50), (20, 47), (15, 45), (10, 43), (5, 41), (0, 40), (-4, 39), (-8, 37), (-10, 36)]),
            ("Africa", [(-17, 35), (-10, 33), (-4, 30), (4, 27), (12, 21), (18, 14), (23, 7), (28, -2), (31, -10), (34, -19), (31, -27), (27, -32), (20, -34), (12, -35), (5, -33), (0, -26), (-4, -16), (-8, -5), (-12, 8), (-15, 20), (-17, 35)]),
            ("Asia", [(30, 58), (40, 60), (52, 58), (66, 56), (82, 56), (96, 59), (112, 54), (126, 50), (140, 46), (150, 42), (156, 36), (151, 28), (140, 22), (128, 18), (118, 20), (106, 18), (96, 15), (86, 20), (76, 26), (66, 30), (56, 32), (48, 33), (42, 36), (36, 42), (32, 48), (30, 58)]),
            ("Arabia", [(35, 31), (42, 30), (49, 27), (54, 23), (56, 18), (52, 15), (46, 13), (42, 15), (39, 20), (36, 25), (35, 31)]),
            ("India", [(68, 23), (74, 25), (80, 23), (85, 19), (87, 14), (84, 9), (79, 7), (74, 9), (71, 13), (69, 18), (68, 23)]),
            ("Japan", [(130, 33), (134, 35), (138, 37), (141, 40), (143, 43)]),
            ("Australia", [(113, -11), (120, -16), (128, -19), (136, -21), (145, -25), (151, -31), (148, -37), (141, -39), (133, -35), (124, -31), (117, -24), (113, -18), (113, -11)]),
            ("Antarctica", [(-180, -72), (-150, -74), (-120, -76), (-90, -78), (-60, -79), (-30, -80), (0, -80), (30, -79), (60, -78), (90, -77), (120, -76), (150, -74), (180, -72)]),
        ]

    def _subsolar_point(self, when_utc: datetime) -> tuple[float, float]:
        try:
            dt = when_utc.astimezone(timezone.utc)
            doy = dt.timetuple().tm_yday
            hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
            gamma = 2.0 * math.pi / 365.0 * (doy - 1 + (hour - 12.0) / 24.0)
            decl = (
                0.006918
                - 0.399912 * math.cos(gamma)
                + 0.070257 * math.sin(gamma)
                - 0.006758 * math.cos(2 * gamma)
                + 0.000907 * math.sin(2 * gamma)
                - 0.002697 * math.cos(3 * gamma)
                + 0.00148 * math.sin(3 * gamma)
            )
            eqtime = 229.18 * (
                0.000075
                + 0.001868 * math.cos(gamma)
                - 0.032077 * math.sin(gamma)
                - 0.014615 * math.cos(2 * gamma)
                - 0.040849 * math.sin(2 * gamma)
            )
            minutes = hour * 60.0
            lon_deg = 180.0 - ((minutes + eqtime) / 4.0)
            lon_deg = self._normalize_degrees(lon_deg)
            return math.degrees(decl), lon_deg
        except Exception:
            return 0.0, 0.0

    
    
    def _earth_geojson_path(self) -> Path | None:
        candidates = [
            ASSETS_DIR / "world_countries.geojson",
            APP_DIR / "assets" / "world_countries.geojson",
            Path(__file__).resolve().parent / "assets" / "world_countries.geojson",
        ]
        for candidate in candidates:
            try:
                if candidate.exists():
                    return candidate
            except Exception:
                continue
        return None

    def _simplify_geo_ring(self, coords, target_points: int = 120) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        if not isinstance(coords, list) or len(coords) < 3:
            return points

        step = max(1, int(len(coords) / max(16, target_points)))
        for i in range(0, len(coords), step):
            item = coords[i]
            try:
                lon = float(item[0])
                lat = float(item[1])
            except Exception:
                continue
            lat = max(-89.5, min(89.5, lat))
            if points and abs(points[-1][0] - lon) < 1e-6 and abs(points[-1][1] - lat) < 1e-6:
                continue
            points.append((lon, lat))

        try:
            last = coords[-1]
            lon = float(last[0])
            lat = max(-89.5, min(89.5, float(last[1])))
            if not points or abs(points[-1][0] - lon) > 1e-6 or abs(points[-1][1] - lat) > 1e-6:
                points.append((lon, lat))
        except Exception:
            pass

        if len(points) >= 3 and (abs(points[0][0] - points[-1][0]) > 1e-6 or abs(points[0][1] - points[-1][1]) > 1e-6):
            points.append(points[0])
        return points

    def _earth_geojson_polygons(self) -> list[dict]:
        cached = getattr(self, "_earth_geojson_polygons_cache", None)
        if cached is not None:
            return cached

        polygons: list[dict] = []
        geo_path = self._earth_geojson_path()
        if geo_path is None:
            debug_log("Earth geojson not found, falling back to manual continents")
            self._earth_geojson_polygons_cache = []
            return []

        try:
            payload = json.loads(geo_path.read_text(encoding="utf-8"))
            features = payload.get("features", []) if isinstance(payload, dict) else []
            for feature in features:
                geometry = (feature or {}).get("geometry", {}) or {}
                props = (feature or {}).get("properties", {}) or {}
                gtype = geometry.get("type")
                coords = geometry.get("coordinates", [])
                if gtype == "Polygon":
                    poly_list = [coords]
                elif gtype == "MultiPolygon":
                    poly_list = coords
                else:
                    continue

                for poly in poly_list:
                    if not poly:
                        continue
                    exterior = self._simplify_geo_ring(poly[0], target_points=140)
                    holes = []
                    for hole in poly[1:]:
                        simplified_hole = self._simplify_geo_ring(hole, target_points=60)
                        if len(simplified_hole) >= 4:
                            holes.append(simplified_hole)
                    if len(exterior) >= 4:
                        polygons.append({"exterior": exterior, "holes": holes, "properties": props})
            debug_log(f"Earth geojson loaded: {geo_path} polygons={len(polygons)}")
        except Exception as e:
            debug_log(f"Earth geojson load failed: {e}")
            polygons = []

        self._earth_geojson_polygons_cache = polygons
        return polygons

    def _unwrap_geo_ring(self, ring: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if not ring:
            return []
        out: list[tuple[float, float]] = []
        prev_lon = None
        for lon, lat in ring:
            current_lon = float(lon)
            if prev_lon is not None:
                while current_lon - prev_lon > 180.0:
                    current_lon -= 360.0
                while current_lon - prev_lon < -180.0:
                    current_lon += 360.0
            out.append((current_lon, float(lat)))
            prev_lon = current_lon
        return out

    def _geo_ring_to_path(self, ring: list[tuple[float, float]], map_width: int, map_height: int, wrap_offset: float = 0.0) -> QtGui.QPainterPath:
        path = QtGui.QPainterPath()
        pts = self._unwrap_geo_ring(ring)
        if len(pts) < 3:
            return path
        for i, (lon, lat) in enumerate(pts):
            x = ((lon + 180.0) / 360.0) * float(map_width) + float(wrap_offset)
            y = ((90.0 - lat) / 180.0) * float(map_height)
            pt = QtCore.QPointF(x, y)
            if i == 0:
                path.moveTo(pt)
            else:
                path.lineTo(pt)
        path.closeSubpath()
        return path

    def _geo_polygon_to_path(self, exterior: list[tuple[float, float]], holes: list[list[tuple[float, float]]], map_width: int, map_height: int, wrap_offset: float = 0.0) -> QtGui.QPainterPath:
        path = QtGui.QPainterPath()
        path.setFillRule(QtCore.Qt.OddEvenFill)
        path.addPath(self._geo_ring_to_path(exterior, map_width, map_height, wrap_offset=wrap_offset))
        for hole in holes:
            if len(hole) >= 4:
                path.addPath(self._geo_ring_to_path(hole, map_width, map_height, wrap_offset=wrap_offset))
        return path

    def _build_landmask_texture_from_geojson(self, width: int = 1024, height: int = 512) -> QtGui.QImage:
        polygons = self._earth_geojson_polygons()
        if not polygons:
            debug_log("Earth geojson empty, using manual fallback texture")
            return self._generate_manual_fallback_texture(width, height)

        width = max(256, int(width))
        height = max(128, int(height))
        ext_width = width * 3
        ext = QtGui.QImage(ext_width, height, QtGui.QImage.Format_ARGB32)
        ext.fill(QtGui.QColor("#103055"))

        painter = QtGui.QPainter(ext)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)

        ocean_grad = QtGui.QLinearGradient(0, 0, 0, height)
        ocean_grad.setColorAt(0.00, QtGui.QColor("#112c4e"))
        ocean_grad.setColorAt(0.22, QtGui.QColor("#1d4d7d"))
        ocean_grad.setColorAt(0.50, QtGui.QColor("#4f93cf"))
        ocean_grad.setColorAt(0.76, QtGui.QColor("#255f99"))
        ocean_grad.setColorAt(1.00, QtGui.QColor("#102b49"))
        painter.fillRect(0, 0, ext_width, height, ocean_grad)

        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(255, 255, 255, 12))
        painter.drawRoundedRect(QtCore.QRectF(0, height * 0.42, ext_width, height * 0.16), height * 0.05, height * 0.05)

        land_path = QtGui.QPainterPath()
        land_path.setFillRule(QtCore.Qt.WindingFill)
        for poly in polygons:
            land_path.addPath(self._geo_polygon_to_path(poly["exterior"], poly["holes"], width, height, wrap_offset=width))

        painter.fillPath(land_path, QtGui.QColor("#6f9c63"))

        painter.save()
        painter.setClipPath(land_path)

        def lonlat_to_ext(lon: float, lat: float) -> QtCore.QPointF:
            x = ((float(lon) + 180.0) / 360.0) * float(width) + float(width)
            y = ((90.0 - float(lat)) / 180.0) * float(height)
            return QtCore.QPointF(x, y)

        desert_specs = [
            (-8.0, 22.0, 30.0, 18.0, QtGui.QColor(187, 159, 95, 165)),
            (47.0, 24.0, 18.0, 10.0, QtGui.QColor(193, 164, 103, 150)),
            (134.0, -25.0, 24.0, 13.0, QtGui.QColor(177, 150, 94, 145)),
            (-112.0, 33.0, 14.0, 8.0, QtGui.QColor(180, 154, 97, 105)),
            (78.0, 41.0, 24.0, 10.0, QtGui.QColor(192, 166, 110, 95)),
            (-70.0, -23.0, 10.0, 6.0, QtGui.QColor(188, 160, 108, 90)),
        ]
        for lon, lat, span_lon, span_lat, color in desert_specs:
            c = lonlat_to_ext(lon, lat)
            painter.setBrush(color)
            painter.setPen(QtCore.Qt.NoPen)
            painter.drawEllipse(QtCore.QRectF(c.x() - (span_lon / 360.0) * width * 0.5, c.y() - (span_lat / 180.0) * height * 0.5, (span_lon / 360.0) * width, (span_lat / 180.0) * height))

        forest_specs = [
            (-60.0, -6.0, 24.0, 11.0, QtGui.QColor(74, 126, 71, 86)),
            (22.0, 0.0, 18.0, 10.0, QtGui.QColor(68, 120, 67, 72)),
            (107.0, 8.0, 18.0, 9.0, QtGui.QColor(66, 118, 65, 68)),
        ]
        for lon, lat, span_lon, span_lat, color in forest_specs:
            c = lonlat_to_ext(lon, lat)
            painter.setBrush(color)
            painter.setPen(QtCore.Qt.NoPen)
            painter.drawEllipse(QtCore.QRectF(c.x() - (span_lon / 360.0) * width * 0.5, c.y() - (span_lat / 180.0) * height * 0.5, (span_lon / 360.0) * width, (span_lat / 180.0) * height))

        ice_grad_n = QtGui.QLinearGradient(0, 0, 0, height * 0.22)
        ice_grad_n.setColorAt(0.0, QtGui.QColor(245, 248, 252, 250))
        ice_grad_n.setColorAt(1.0, QtGui.QColor(245, 248, 252, 0))
        painter.fillRect(0, 0, ext_width, int(height * 0.24), ice_grad_n)

        ice_grad_s = QtGui.QLinearGradient(0, height, 0, height * 0.72)
        ice_grad_s.setColorAt(0.0, QtGui.QColor(245, 248, 252, 250))
        ice_grad_s.setColorAt(1.0, QtGui.QColor(245, 248, 252, 0))
        painter.fillRect(0, int(height * 0.72), ext_width, int(height * 0.28), ice_grad_s)

        painter.restore()

        painter.setPen(QtGui.QPen(QtGui.QColor(218, 236, 213, 54), 1.2))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawPath(land_path)

        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 16), 1.0))
        relief_lines = [
            [(-112, 58), (-108, 46), (-104, 35)],
            [(-77, -7), (-73, -18), (-70, -30), (-69, -42)],
            [(6, 46), (16, 44), (26, 42)],
            [(70, 34), (83, 31), (96, 30), (108, 28)],
            [(31, 9), (27, -2), (24, -12), (20, -24)],
        ]
        for line in relief_lines:
            path = QtGui.QPainterPath()
            for i, (lon, lat) in enumerate(line):
                pt = lonlat_to_ext(lon, lat)
                if i == 0:
                    path.moveTo(pt)
                else:
                    path.lineTo(pt)
            painter.drawPath(path)

        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(255, 255, 255, 10))
        painter.drawEllipse(QtCore.QRectF(width * 0.95, height * 0.18, width * 0.26, height * 0.07))
        painter.drawEllipse(QtCore.QRectF(width * 1.42, height * 0.25, width * 0.20, height * 0.05))
        painter.drawEllipse(QtCore.QRectF(width * 1.62, height * 0.56, width * 0.22, height * 0.05))
        painter.drawEllipse(QtCore.QRectF(width * 1.18, height * 0.63, width * 0.15, height * 0.04))

        painter.end()
        return ext.copy(width, 0, width, height)

    def _generate_manual_fallback_texture(self, width: int = 1024, height: int = 512):
        width = max(256, int(width))
        height = max(128, int(height))
        img = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32)
        img.fill(QtGui.QColor("#103055"))
        painter = QtGui.QPainter(img)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        ocean_grad = QtGui.QLinearGradient(0, 0, 0, height)
        ocean_grad.setColorAt(0.00, QtGui.QColor("#143a66"))
        ocean_grad.setColorAt(0.22, QtGui.QColor("#1f5a95"))
        ocean_grad.setColorAt(0.50, QtGui.QColor("#4b8fd1"))
        ocean_grad.setColorAt(0.76, QtGui.QColor("#2a68aa"))
        ocean_grad.setColorAt(1.00, QtGui.QColor("#12385f"))
        painter.fillRect(0, 0, width, height, ocean_grad)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor("#6f9c63"))
        for _name, poly in self._earth_continent_polygons():
            path = QtGui.QPainterPath()
            for i, (lon, lat) in enumerate(poly):
                x = ((lon + 180.0) / 360.0) * width
                y = ((90.0 - lat) / 180.0) * height
                if i == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            path.closeSubpath()
            painter.drawPath(path)
        painter.end()
        return img

    def _generate_earth_texture(self, width: int = 1024, height: int = 512):
        """Generate a lightweight stylized Earth texture from the GeoJSON world map."""
        return self._build_landmask_texture_from_geojson(width, height)

    def _earth_texture_image(self):

        try:
            if self._earth_texture is None:
                self._earth_texture = self._generate_earth_texture(1024, 512).convertToFormat(QtGui.QImage.Format_ARGB32)
                self._earth_texture_size = (self._earth_texture.width(), self._earth_texture.height())
                debug_log("Earth texture generated in code (stylized lightweight mode)")
        except Exception as e:
            debug_log(f"Earth texture generation failed: {e}")
            self._earth_texture = None
            self._earth_texture_size = (0, 0)
        return self._earth_texture

    def _earth_globe_cache_image(self, diameter: int, lon0_deg: float, lat0_deg: float = 0.0, when_utc: Optional[datetime] = None) -> Optional[QtGui.QImage]:
        texture = self._earth_texture_image()
        if when_utc is None:
            when_utc = datetime.now(timezone.utc)

        target_diameter = max(32, int(diameter))
        render_diameter = max(48, int(target_diameter * float(self._earth_globe_render_scale)))
        render_diameter = min(render_diameter, 420)

        cache_key = (
            render_diameter,
            round(float(lon0_deg) / 3.0) * 3.0,
            round(float(lat0_deg) / 3.0) * 3.0,
            when_utc.strftime("%Y-%m-%d %H:%M"),
        )
        cached = self._earth_globe_cache
        if cache_key == self._earth_globe_cache_key and isinstance(cached, QtGui.QImage) and not cached.isNull():
            return cached

        img = QtGui.QImage(render_diameter, render_diameter, QtGui.QImage.Format_ARGB32)
        img.fill(QtCore.Qt.transparent)

        sun_lat_deg, sun_lon_deg = self._subsolar_point(when_utc)
        sun_lat = math.radians(sun_lat_deg)
        sun_lon = math.radians(sun_lon_deg)
        l_x = math.cos(sun_lat) * math.cos(sun_lon)
        l_y = math.cos(sun_lat) * math.sin(sun_lon)
        l_z = math.sin(sun_lat)

        tex_w, tex_h = self._earth_texture_size if texture is not None else (0, 0)
        lon0 = math.radians(float(lon0_deg))
        lat0 = math.radians(float(lat0_deg))

        if texture is None or tex_w <= 0 or tex_h <= 0:
            texture = self._generate_earth_texture(512, 256).convertToFormat(QtGui.QImage.Format_ARGB32)
            tex_w, tex_h = texture.width(), texture.height()

        inv = 2.0 / max(1.0, render_diameter - 1)
        ocean_fallback = QtGui.QColor("#4a97d8")

        for py in range(render_diameter):
            ny = 1.0 - (py + 0.5) * inv
            for px in range(render_diameter):
                nx = (px + 0.5) * inv - 1.0
                rr = nx * nx + ny * ny
                if rr > 1.0:
                    continue

                rho = math.sqrt(rr)
                if rho <= 1e-9:
                    lat = lat0
                    lon = lon0
                else:
                    c = math.asin(min(1.0, rho))
                    sin_c = math.sin(c)
                    cos_c = math.cos(c)
                    lat = math.asin(cos_c * math.sin(lat0) + (ny * sin_c * math.cos(lat0) / rho))
                    lon = lon0 + math.atan2(nx * sin_c, rho * math.cos(lat0) * cos_c - ny * math.sin(lat0) * sin_c)

                lat_deg = math.degrees(lat)
                lon_deg = self._normalize_degrees(math.degrees(lon))

                tx = int(((lon_deg + 180.0) / 360.0) * tex_w) % tex_w
                ty = int(((90.0 - lat_deg) / 180.0) * tex_h)
                ty = max(0, min(tex_h - 1, ty))
                base = texture.pixelColor(tx, ty) if texture is not None else ocean_fallback

                n_x = math.cos(lat) * math.cos(lon)
                n_y = math.cos(lat) * math.sin(lon)
                n_z = math.sin(lat)
                illum = n_x * l_x + n_y * l_y + n_z * l_z

                limb = max(0.0, math.sqrt(max(0.0, 1.0 - rr)))
                daylight = max(0.0, illum)
                twilight = 0.22 + 0.78 * daylight
                r = int(min(255, base.red() * twilight + 12 * (1.0 - twilight)))
                g = int(min(255, base.green() * twilight + 18 * (1.0 - twilight)))
                b = int(min(255, base.blue() * twilight + 34 * (1.0 - twilight)))

                # subtle atmospheric rim
                rim = (1.0 - limb) ** 1.6
                r = min(255, int(r + 26 * rim))
                g = min(255, int(g + 32 * rim))
                b = min(255, int(b + 54 * rim))

                alpha = 255
                img.setPixelColor(px, py, QtGui.QColor(r, g, b, alpha))

        self._earth_globe_cache_key = cache_key
        self._earth_globe_cache = img
        return img

    def _draw_earth_globe_with_real_continents(self, painter: QtGui.QPainter, center: QtCore.QPointF, earth_radius: float, lon0_deg: float, lat0_deg: float = 0.0, when_utc: Optional[datetime] = None):
        diameter = max(32, int(earth_radius * 2.0))
        img = self._earth_globe_cache_image(diameter, lon0_deg, lat0_deg, when_utc=when_utc)

        painter.save()
        for r, alpha in [(earth_radius * 1.44, 16), (earth_radius * 1.24, 26), (earth_radius * 1.10, 42)]:
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(80, 165, 255, alpha))
            painter.drawEllipse(center, r, r)

        painter.setPen(QtGui.QPen(QtGui.QColor(170, 220, 255, 68), 1.2))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawEllipse(center, earth_radius * 1.01, earth_radius * 1.01)

        target = QtCore.QRectF(center.x() - earth_radius, center.y() - earth_radius, earth_radius * 2.0, earth_radius * 2.0)
        if isinstance(img, QtGui.QImage) and not img.isNull():
            painter.drawImage(target, img)
        else:
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor("#3f8de0"))
            painter.drawEllipse(target)
        painter.restore()

    def _iss_state_at(self, when_utc: datetime) -> dict:
        state = {'lat_deg': None, 'lon_deg': None, 'alt_km': None, 'source': 'fallback'}
        try:
            if self._iss_satellite is not None and self._skyfield_ts is not None and wgs84 is not None:
                t = self._skyfield_ts.from_datetime(when_utc)
                geocentric = self._iss_satellite.at(t)
                lat, lon = wgs84.latlon_of(geocentric)
                h = wgs84.height_of(geocentric)
                state['lat_deg'] = float(lat.degrees)
                state['lon_deg'] = float(lon.degrees)
                state['alt_km'] = float(h.km)
                state['source'] = 'skyfield'
                return state
        except Exception as e:
            if self._iss_satellite is not None:
                self._iss_tle_error = f"ISS propagation failed: {e}"
                debug_log(self._iss_tle_error)
        elapsed_s = (when_utc - self._epoch).total_seconds()
        state['lat_deg'] = 51.6 * math.sin(elapsed_s / 5400.0 * 2.0 * math.pi)
        state['lon_deg'] = self._normalize_degrees((elapsed_s / 5550.0) * 360.0 - 180.0)
        state['alt_km'] = 420.0
        return state

    def _greenwich_sidereal_angle_deg(self, when_utc: datetime) -> float:
        """Approximate Greenwich mean sidereal angle in degrees."""
        try:
            dt = when_utc.astimezone(timezone.utc) if when_utc.tzinfo is not None else when_utc.replace(tzinfo=timezone.utc)
            year = dt.year
            month = dt.month
            day = dt.day
            hour = dt.hour + (dt.minute / 60.0) + (dt.second / 3600.0) + (dt.microsecond / 3_600_000_000.0)
            if month <= 2:
                year -= 1
                month += 12
            a = year // 100
            b = 2 - a + (a // 4)
            jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b - 1524.5 + hour / 24.0
            t = (jd - 2451545.0) / 36525.0
            gmst = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * (t * t) - (t * t * t) / 38710000.0
            return self._normalize_degrees(gmst)
        except Exception:
            return 0.0

    def _moon_state_at(self, when_utc: datetime) -> dict:
        state = {'lat_deg': 0.0, 'lon_deg': 0.0, 'distance_km': 384400.0, 'source': 'fallback'}
        try:
            if self._skyfield_ready and self._skyfield_ts is not None and self._skyfield_eph is not None:
                t = self._skyfield_ts.from_datetime(when_utc)
                earth = self._skyfield_eph['earth']
                moon = self._skyfield_eph['moon']
                moon_obs = earth.at(t).observe(moon)
                xyz = self._safe_xyz_km(moon_obs)
                if xyz is not None:
                    mx, my, mz = xyz
                    state['distance_km'] = max(1.0, math.sqrt(mx * mx + my * my + mz * mz))
                    # Convert the geocentric inertial vector to an Earth-fixed longitude so the 
                    # is displayed in the same rotating frame as the Earth System globe.
                    gst_deg = self._greenwich_sidereal_angle_deg(when_utc)
                    gst_rad = math.radians(gst_deg)
                    cos_g = math.cos(gst_rad)
                    sin_g = math.sin(gst_rad)
                    ex = (mx * cos_g) + (my * sin_g)
                    ey = (-mx * sin_g) + (my * cos_g)
                    ez = mz
                    state['lat_deg'] = math.degrees(math.atan2(ez, math.hypot(ex, ey)))
                    state['lon_deg'] = self._normalize_degrees(math.degrees(math.atan2(ey, ex)))
                    state['source'] = 'skyfield'
                    return state
        except Exception:
            pass
        elapsed_s = (when_utc - self._epoch).total_seconds()
        state['lat_deg'] = 5.1 * math.sin(elapsed_s / (27.321 * 86400.0) * 2.0 * math.pi)
        state['lon_deg'] = self._normalize_degrees((elapsed_s / (27.321 * 86400.0)) * 360.0)
        return state

    def _earth_system_state(self) -> dict:
        now_utc = datetime.now(timezone.utc)
        key = now_utc.strftime('%Y-%m-%d %H:%M:%S')
        if key == self._earth_panel_cache_key and isinstance(self._earth_panel_cache, dict):
            return self._earth_panel_cache

        iss_now = self._iss_state_at(now_utc)
        moon_now = self._moon_state_at(now_utc)
        state = {
            'now_utc': now_utc,
            'iss_lat_deg': iss_now['lat_deg'],
            'iss_lon_deg': iss_now['lon_deg'],
            'iss_alt_km': iss_now['alt_km'],
            'iss_source': iss_now['source'],
            'moon_lat_deg': moon_now['lat_deg'],
            'moon_lon_deg': moon_now['lon_deg'],
            'moon_distance_km': moon_now['distance_km'],
            'moon_source': moon_now['source'],
            'view_lat0_deg': float(iss_now['lat_deg'] if iss_now['lat_deg'] is not None else 0.0),
            'view_lon0_deg': float(iss_now['lon_deg'] if iss_now['lon_deg'] is not None else 0.0),
            'track_samples': [],
        }

        samples = []
        step = int(max(30, self._earth_track_step_seconds))
        span_min = int(max(8, self._earth_track_minutes))
        for seconds in range(-span_min * 60, span_min * 60 + 1, step):
            dt = now_utc + timedelta(seconds=seconds)
            item = self._iss_state_at(dt)
            if item['lat_deg'] is None or item['lon_deg'] is None:
                continue
            samples.append({
                'offset_s': seconds,
                'lat_deg': float(item['lat_deg']),
                'lon_deg': float(item['lon_deg']),
                'alt_km': float(item['alt_km'] or 420.0),
                'source': item['source'],
            })
        state['track_samples'] = samples

        self._earth_panel_cache_key = key
        self._earth_panel_cache = state
        return state

    def _format_iss_hhmm(self, dt_obj) -> str:
        """Format ISS pass time in the computer/user local timezone."""
        try:
            if isinstance(dt_obj, datetime):
                if dt_obj.tzinfo is None:
                    dt_obj = dt_obj.replace(tzinfo=timezone.utc)
                return dt_obj.astimezone().strftime("%H:%M")
        except Exception:
            pass
        return "--:--"

    def _compute_next_iss_pass(self, minutes_ahead: int = 180) -> Optional[dict]:
        """Compute next visible ISS pass from observer coordinates using current TLE.
        Cached for about one minute because Skyfield find_events is relatively expensive.
        """
        try:
            if self._iss_satellite is None or self._skyfield_ts is None or wgs84 is None:
                return None
            now_utc = datetime.now(timezone.utc)
            cache_key = now_utc.strftime("%Y-%m-%d %H:%M")
            if cache_key == getattr(self, "_iss_pass_cache_key", ""):
                cached = getattr(self, "_iss_pass_cache", None)
                return cached if isinstance(cached, dict) else None

            observer = wgs84.latlon(
                float(getattr(self, "_iss_observer_lat", 48.1173)),
                float(getattr(self, "_iss_observer_lon", -1.6778)),
                elevation_m=float(getattr(self, "_iss_observer_alt_m", 60.0)),
            )
            t0 = self._skyfield_ts.from_datetime(now_utc)
            t1 = self._skyfield_ts.from_datetime(now_utc + timedelta(minutes=int(minutes_ahead)))
            min_el = float(getattr(self, "_iss_min_elevation_deg", 10.0))
            times, events = self._iss_satellite.find_events(observer, t0, t1, altitude_degrees=min_el)

            current = {}
            passes = []
            for t, event in zip(times, events):
                dt = t.utc_datetime().replace(tzinfo=timezone.utc)
                ev = int(event)
                if ev == 0:
                    current = {"aos": dt}
                elif ev == 1 and current:
                    current["max"] = dt
                    try:
                        topocentric = (self._iss_satellite - observer).at(t)
                        alt, az, distance = topocentric.altaz()
                        current["max_elev_deg"] = float(alt.degrees)
                    except Exception:
                        current["max_elev_deg"] = None
                elif ev == 2 and current:
                    current["los"] = dt
                    passes.append(current)
                    current = {}

            result = passes[0] if passes else None
            self._iss_pass_cache_key = cache_key
            self._iss_pass_cache = result
            return result
        except Exception as e:
            try:
                debug_log(f"ISS next pass compute failed: {e}")
            except Exception:
                pass
            return None

    def _iss_next_pass_text(self) -> str:
        try:
            if self._iss_satellite is None:
                if getattr(self, "_iss_tle_loading", False):
                    return "NEXT PASS\nTLE loading..."
                return "NEXT PASS\nTLE unavailable"

            p = self._compute_next_iss_pass(minutes_ahead=360)
            if not p:
                min_el = float(getattr(self, "_iss_min_elevation_deg", 10.0))
                return f"NEXT PASS\nNo pass > {min_el:.0f}° in next 6h"

            aos = p.get("aos")
            max_t = p.get("max")
            los = p.get("los")
            now_utc = datetime.now(timezone.utc)
            countdown = "--"
            duration_txt = ""
            if isinstance(aos, datetime) and isinstance(los, datetime):
                if aos <= now_utc <= los:
                    countdown = "IN PASS"
                else:
                    delta_s = int((aos - now_utc).total_seconds())
                    if delta_s >= 0:
                        if delta_s >= 3600:
                            countdown = f"T-{delta_s // 3600:d}h{(delta_s % 3600) // 60:02d}"
                        else:
                            countdown = f"T-{delta_s // 60:02d}m"
                duration_min = max(0, int((los - aos).total_seconds() // 60))
                duration_txt = f"  {duration_min} min" if duration_min else ""

            elev = p.get("max_elev_deg")
            elev_txt = ""
            try:
                if elev is not None:
                    elev_txt = f" ({float(elev):.0f}°)"
            except Exception:
                elev_txt = ""

            return (
                "NEXT PASS\n"
                f"AOS {self._format_iss_hhmm(aos)}  LOS {self._format_iss_hhmm(los)}{duration_txt}\n"
                f"MAX {self._format_iss_hhmm(max_t)}{elev_txt}  {countdown}"
            )
        except Exception:
            return "NEXT PASS\n--"

    def _draw_earth_panel(self, painter: QtGui.QPainter, rect: QtCore.QRectF):
        state = self._earth_system_state()
        self._draw_panel_card(painter, rect, "ISS Tracker", "Real-time ISS tracking • Earth-centered view")

        content = rect.adjusted(12, 44, -12, -12)
        center = QtCore.QPointF(content.center().x(), content.center().y() + 8)
        earth_radius = max(34.0, min(content.width(), content.height()) * 0.255)
        view_lat0_deg = float(state.get('view_lat0_deg', 0.0) or 0.0)
        view_lon0_deg = float(state.get('view_lon0_deg', 0.0) or 0.0)

        # Earth glow
        for r, alpha in [(earth_radius * 1.46, 18), (earth_radius * 1.28, 28), (earth_radius * 1.12, 44)]:
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(80, 165, 255, alpha))
            painter.drawEllipse(center, r, r)

        #  removed on purpose to enlarge the Earth-only tracking view.

        self._draw_earth_globe_with_real_continents(painter, center, earth_radius, view_lon0_deg, view_lat0_deg, when_utc=state.get('now_utc'))

        # ISS track: past solid, future dotted
        track_samples = list(state.get('track_samples') or [])
        past_points = []
        future_points = []
        current_sample = None
        for item in track_samples:
            pt, visible, cosc = self._orthographic_project(item['lat_deg'], item['lon_deg'], center, earth_radius * 0.985, view_lon0_deg, view_lat0_deg)
            if not visible:
                continue
            if int(item.get('offset_s', 0)) < 0:
                past_points.append(pt)
            elif int(item.get('offset_s', 0)) > 0:
                future_points.append(pt)
            else:
                current_sample = item

        if len(past_points) >= 2:
            path = QtGui.QPainterPath(past_points[0])
            for pt in past_points[1:]:
                path.lineTo(pt)
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 208, 120, 150), 2.2, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
            painter.drawPath(path)
        if len(future_points) >= 2:
            path = QtGui.QPainterPath(future_points[0])
            for pt in future_points[1:]:
                path.lineTo(pt)
            painter.setPen(QtGui.QPen(QtGui.QColor(140, 220, 255, 130), 1.8, QtCore.Qt.DashLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
            painter.drawPath(path)

        # ISS kept at a fixed reading position; globe rotates below it.
        iss_marker = QtCore.QPointF(center.x() - earth_radius * 1.05, center.y() - earth_radius * 0.26)
        nadir = QtCore.QPointF(center.x(), center.y())
        painter.setPen(QtGui.QPen(QtGui.QColor(140, 185, 235, 85), 1.2, QtCore.Qt.DashLine))
        painter.drawLine(iss_marker, nadir)
        self._draw_spacecraft(painter, iss_marker, earth_radius * 0.15, "#ffffff", hovered=(self._hover_name == "ISS"))
        self._draw_label(painter, "ISS", iss_marker.x(), iss_marker.y(), self._hover_name == "ISS")
        iss_lat = state.get('iss_lat_deg')
        iss_lon = state.get('iss_lon_deg')
        iss_alt = state.get('iss_alt_km')
        self._hover_items.append({
            "name": "ISS",
            "center": iss_marker,
            "radius": earth_radius * 0.34,
            "tooltip": self._simple_tooltip("ISS", [f"Source: {'Skyfield + TLE' if state.get('iss_source') == 'skyfield' else ('TLE loading' if getattr(self, '_iss_tle_loading', False) else 'Fallback')}", f"Latitude: {float(iss_lat or 0.0):+.2f}°", f"Longitude: {float(iss_lon or 0.0):+.2f}°", f"Altitude: {float(iss_alt or 0.0):.1f} km", self._iss_tle_epoch_text if self._iss_tle_epoch_text else (('TLE loading...' if getattr(self, '_iss_tle_loading', False) else None) or self._iss_tle_error or 'TLE pending'), self._iss_tle_loaded_from if self._iss_tle_loaded_from else ""]),
        })

        # Current sub-satellite point on globe
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(255, 245, 210, 220))
        painter.drawEllipse(center, 4.5, 4.5)

        self._hover_items.append({
            "name": "Earth",
            "center": center,
            "radius": earth_radius,
            "tooltip": self._simple_tooltip("Earth", ["ISS-centered tracking view", "Globe orientation follows the current ISS sub-satellite point", " removed to maximize Earth display size"]),
        })

        # NEXT PASS block: independent orbital information, not mixed with solar data.
        try:
            pass_text = self._iss_next_pass_text()
            pass_font = QtGui.QFont("Consolas", 9, QtGui.QFont.Bold)
            painter.setFont(pass_font)
            painter.setPen(QtGui.QColor("#d7dde6"))
            pass_rect = QtCore.QRectF(rect.left() + 16, rect.top() + 44, rect.width() - 32, 58)
            painter.drawText(pass_rect, QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop, pass_text)
        except Exception:
            pass

        info_font = QtGui.QFont("Segoe UI", 8)
        painter.setFont(info_font)
        painter.setPen(QtGui.QColor("#92a7c3"))
        info_rect = QtCore.QRectF(rect.left() + 12, rect.bottom() - 26, rect.width() - 24, 16)
        status_bits = []
        status_bits.append(": Skyfield" if state.get('moon_source') == 'skyfield' else ": fallback")
        if state.get('iss_source') == 'skyfield':
            status_bits.append("ISS: real TLE")
        elif self._iss_satellite is not None:
            status_bits.append("ISS: cached TLE")
        elif getattr(self, '_iss_tle_loading', False):
            status_bits.append("ISS: loading TLE...")
        else:
            status_bits.append("ISS: fallback")
        status_bits.append("Track: past solid / future dashed")
        painter.drawText(info_rect, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, " • ".join(status_bits))

    def paintEvent(self, event):
        self._hover_items = []

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing, True)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)

        outer = self.rect().adjusted(4, 4, -4, -4)

        panel_path = QtGui.QPainterPath()
        panel_path.addRoundedRect(QtCore.QRectF(outer), 16, 16)
        painter.setClipPath(panel_path)

        self._draw_space_background(painter, QtCore.QRectF(outer))

        header_h = 58
        footer_h = 24

        title_rect = QtCore.QRectF(outer.left() + 14, outer.top() + 8, outer.width() - 28, 24)
        subtitle_rect = QtCore.QRectF(outer.left() + 14, outer.top() + 30, outer.width() - 28, 18)
        status_rect = QtCore.QRectF(outer.left() + 14, outer.bottom() - footer_h, outer.width() - 28, 16)

        title_font = QtGui.QFont("Segoe UI", 16, QtGui.QFont.Bold)
        painter.setFont(title_font)
        painter.setPen(QtGui.QColor("#ecf3ff"))
        painter.drawText(title_rect, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, self._title)

        sub_font = QtGui.QFont("Segoe UI", 9)
        painter.setFont(sub_font)
        painter.setPen(QtGui.QColor("#9cb0c7"))
        painter.drawText(subtitle_rect, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, self._subtitle)

        painter.setPen(QtGui.QColor(120, 145, 175, 60))
        painter.drawLine(
            QtCore.QPointF(outer.left() + 14, outer.top() + 52),
            QtCore.QPointF(outer.right() - 14, outer.top() + 52)
        )

        body_rect = QtCore.QRectF(
            outer.left() + 12,
            outer.top() + header_h,
            outer.width() - 24,
            outer.height() - header_h - footer_h - 8,
        )

        gap = 12.0
        solar_w = body_rect.width() * 0.67
        earth_w = body_rect.width() - solar_w - gap
        solar_rect = QtCore.QRectF(body_rect.left(), body_rect.top(), solar_w, body_rect.height())
        earth_rect = QtCore.QRectF(solar_rect.right() + gap, body_rect.top(), earth_w, body_rect.height())
        self._solar_scene_rect = solar_rect
        self._earth_panel_rect = earth_rect

        panel_subtitle = "Double-click a planet to zoom • drag to pan • wheel zoom disabled"
        if self._focused_planet and self._focus_progress > 0.02:
            panel_subtitle = f"Focused view: {self._focused_planet} • double-click empty space to return"
        self._draw_panel_card(painter, solar_rect, "Solar System", panel_subtitle)
        self._draw_earth_panel(painter, earth_rect)

        overview_opacity = max(0.0, 1.0 - self._focus_progress)
        focused_opacity = max(0.0, self._focus_progress)
        self._draw_solar_system_overview(painter, solar_rect, overview_opacity)
        if self._focused_planet:
            self._draw_focused_planet_scene(painter, solar_rect, self._focused_planet, focused_opacity)

        badge_text = self._focused_planet if self._focused_planet and self._focus_progress > 0.5 else f"Zoom ×{self._zoom_factor:.2f}"
        badge_font = QtGui.QFont("Consolas", 9, QtGui.QFont.Bold)
        painter.setFont(badge_font)
        badge_rect = QtCore.QRectF(solar_rect.right() - 94, solar_rect.top() + 10, 84, 22)
        painter.setPen(QtGui.QPen(QtGui.QColor(145, 170, 200, 80), 1.0))
        painter.setBrush(QtGui.QColor(10, 18, 28, 125))
        painter.drawRoundedRect(badge_rect, 8, 8)
        painter.setPen(QtGui.QColor("#dce7f5"))
        painter.drawText(badge_rect, QtCore.Qt.AlignCenter, badge_text)

        pan_text = f"Pan {int(self._pan_x):+d},{int(self._pan_y):+d}"
        pan_rect = QtCore.QRectF(badge_rect.left() - 98, badge_rect.top(), 90, 22)
        painter.setPen(QtGui.QPen(QtGui.QColor(145, 170, 200, 55), 1.0))
        painter.setBrush(QtGui.QColor(10, 18, 28, 105))
        painter.drawRoundedRect(pan_rect, 8, 8)
        painter.setPen(QtGui.QColor("#bad0e8"))
        painter.drawText(pan_rect, QtCore.Qt.AlignCenter, pan_text)

        painter.setFont(QtGui.QFont("Segoe UI", 8))
        painter.setPen(QtGui.QColor("#7e93ab"))
        painter.drawText(status_rect, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, self._status)




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

        # Tooltip for the right-side value (palette explanation)
        self.lbl_big.setToolTip("")

        header.addWidget(self.lbl_title, 1)
        header.addWidget(self.lbl_big, 0)
        v.addLayout(header)

        axis = pg.graphicsItems.DateAxisItem.DateAxisItem(orientation="bottom")
        axis_items = {"bottom": axis}
        if getattr(self.cfg, "y_axis_kind", "linear") == "xray_class":
            axis_items["left"] = pg.AxisItem(orientation="left")
        self.plot = pg.PlotWidget(axisItems=axis_items)
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

        if getattr(self.cfg, "y_axis_kind", "linear") == "xray_class":
            self.plot.setLogMode(x=False, y=True)
            axL.setTicks([[
                (-8.0, "A"),
                (-7.0, "B"),
                (-6.0, "C"),
                (-5.0, "M"),
                (-4.0, "X"),
            ]])

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
            if getattr(self.cfg, "y_axis_kind", "linear") == "xray_class":
                series = np.array([1e-8, 1e-8], dtype=float)
            else:
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
            if getattr(self.cfg, "y_axis_kind", "linear") == "xray_class":
                yy = yy[yy > 0]
                if yy.size:
                    y_max = float(np.max(np.log10(yy)))
                    y_min = float(np.min(np.log10(yy)))
                    pad = (y_max - y_min) * 0.15 if y_max != y_min else 0.35
                    self.plot.setYRange(max(-8.4, y_min - pad), min(-3.6, y_max + pad), padding=0)
                else:
                    self.plot.setYRange(-8.2, -3.8, padding=0)
            elif yy.size:
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
# SUMO MUF MAP (fetcher + builder + embedded widget)
# =====================================================
USER_AGENT = "SUMO-KC2G-Fetcher/1.2"
KC2G_STATIONS_PAGE = "https://prop.kc2g.com/stations/"
CANDIDATE_JSON_ENDPOINTS = [
    "https://prop.kc2g.com/api/stations.json",
    "https://prop.kc2g.com/stations.json",
    "https://prop.kc2g.com/api/stations",
]
EUROPE_BOUNDS = {
    "lat_min": 30.0,
    "lat_max": 72.0,
    "lon_min": -15.0,
    "lon_max": 40.0,
}


def _kc2g_utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _kc2g_normalize_lon(lon):
    if lon is None:
        return None
    while lon > 180:
        lon -= 360
    while lon < -180:
        lon += 360
    return lon


def _kc2g_is_in_europe(lat, lon) -> bool:
    if lat is None or lon is None:
        return False
    lon = _kc2g_normalize_lon(lon)
    return (
        EUROPE_BOUNDS["lat_min"] <= lat <= EUROPE_BOUNDS["lat_max"]
        and EUROPE_BOUNDS["lon_min"] <= lon <= EUROPE_BOUNDS["lon_max"]
    )


def _kc2g_score_station(item) -> float:
    age = float(item.get("age_minutes") or 120.0)
    cs = float(item.get("cs") or 0.0)
    muf = float(item.get("muf") or 0.0)
    age_score = max(0.0, 120.0 - age)
    return round(age_score * 0.5 + cs * 0.3 + muf * 0.2, 2)


def _kc2g_parse_timestamp(ts):
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        try:
            return datetime.fromtimestamp(float(ts), tz=timezone.utc)
        except Exception:
            return None
    ts = str(ts).strip()
    if not ts:
        return None
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(ts, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _kc2g_age_minutes_from_timestamp(ts):
    dt = _kc2g_parse_timestamp(ts)
    if dt is None:
        return None
    return round((datetime.now(timezone.utc) - dt).total_seconds() / 60.0, 1)


class KC2GFetcher:
    def __init__(self, timeout: int = 20, debug_dir: str = "giro_debug", session=None) -> None:
        self.timeout = timeout
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def fetch_json_candidates(self):
        for url in CANDIDATE_JSON_ENDPOINTS:
            try:
                r = self.session.get(url, timeout=self.timeout)
                if not r.ok:
                    continue
                ctype = r.headers.get("content-type", "").lower()
                if "json" not in ctype and not r.text.strip().startswith(("[", "{")):
                    continue
                data = r.json()
                (self.debug_dir / "kc2g_raw_source.json").write_text(
                    json.dumps(data, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                return data
            except Exception:
                continue
        return None

    def fetch_html_page(self) -> str:
        r = self.session.get(KC2G_STATIONS_PAGE, timeout=self.timeout)
        r.raise_for_status()
        html = r.text
        (self.debug_dir / "kc2g_raw_source.html").write_text(html, encoding="utf-8")
        return html

    def normalize_json_station(self, item):
        station_obj = item.get("station") if isinstance(item, dict) else {}
        if not isinstance(station_obj, dict):
            station_obj = {}

        ursi = item.get("ursi") or item.get("code") or station_obj.get("code") or item.get("id")
        station_name = item.get("station_name") or item.get("name") or station_obj.get("name")
        lat = item.get("lat") if item.get("lat") is not None else item.get("latitude") if item.get("latitude") is not None else station_obj.get("latitude")
        lon = item.get("lon") if item.get("lon") is not None else item.get("longitude") if item.get("longitude") is not None else station_obj.get("longitude")
        muf = item.get("muf") if item.get("muf") is not None else item.get("mufd") if item.get("mufd") is not None else item.get("MUFD")
        fof2 = item.get("fof2") if item.get("fof2") is not None else item.get("foF2")
        md = item.get("md") if item.get("md") is not None else item.get("m_d")
        tec = item.get("tec")
        hmf2 = item.get("hmf2") if item.get("hmf2") is not None else item.get("hmF2")
        foe = item.get("foe") if item.get("foe") is not None else item.get("foE")
        cs = item.get("confidence") if item.get("confidence") is not None else item.get("cs") if item.get("cs") is not None else item.get("confidence_score")
        timestamp = (
            item.get("timestamp") or item.get("time") or item.get("datetime") or item.get("updated_at")
            or item.get("updated") or item.get("last_updated") or item.get("observed") or item.get("observed_at") or item.get("ts")
        )

        try:
            lat = float(lat) if lat is not None else None
            lon = _kc2g_normalize_lon(float(lon)) if lon is not None else None
            muf = float(muf) if muf is not None else None
            fof2 = float(fof2) if fof2 is not None else None
            md = float(md) if md is not None else None
            tec = float(tec) if tec is not None else None
            hmf2 = float(hmf2) if hmf2 is not None else None
            foe = float(foe) if foe is not None else None
            cs = float(cs) if cs is not None else None
        except Exception:
            return None

        if muf is None and fof2 is None:
            return None

        age = _kc2g_age_minutes_from_timestamp(timestamp)
        out = {
            "ursi": str(ursi) if ursi is not None else None,
            "station_name": str(station_name) if station_name is not None else None,
            "lat": lat,
            "lon": lon,
            "muf": muf,
            "fof2": fof2,
            "md": md,
            "tec": tec,
            "hmf2": hmf2,
            "foe": foe,
            "cs": cs,
            "timestamp": timestamp,
            "age_minutes": age,
            "is_europe": _kc2g_is_in_europe(lat, lon),
            "source": "kc2g_json",
        }
        out["score"] = _kc2g_score_station(out)
        return out

    def parse_json_payload(self, payload):
        items = []
        if isinstance(payload, list):
            for row in payload:
                if isinstance(row, dict):
                    norm = self.normalize_json_station(row)
                    if norm:
                        items.append(norm)
        elif isinstance(payload, dict):
            if isinstance(payload.get("stations"), list):
                for row in payload["stations"]:
                    if isinstance(row, dict):
                        norm = self.normalize_json_station(row)
                        if norm:
                            items.append(norm)
            else:
                for key, row in payload.items():
                    if isinstance(row, dict):
                        row = dict(row)
                        row.setdefault("ursi", key)
                        norm = self.normalize_json_station(row)
                        if norm:
                            items.append(norm)
        return items

    def parse_html_table(self, html: str):
        return []

    def fetch_active_stations(self, fresh_limit_minutes: int = 60):
        payload = self.fetch_json_candidates()
        if payload is not None:
            stations = self.parse_json_payload(payload)
            source_mode = "json"
        else:
            html = self.fetch_html_page()
            stations = self.parse_html_table(html)
            source_mode = "html"

        cleaned, seen = [], set()
        for s in stations:
            key = (s.get("ursi"), s.get("timestamp"), s.get("lat"), s.get("lon"))
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(s)

        active, stale, unknown_age = [], [], []
        for s in cleaned:
            age = s.get("age_minutes")
            if age is None:
                unknown_age.append(s)
            elif age <= fresh_limit_minutes:
                active.append(s)
            else:
                stale.append(s)

        active.sort(key=lambda x: (-x.get("score", 0.0), x.get("age_minutes", 999999)))
        stale.sort(key=lambda x: (x.get("age_minutes", 999999), -x.get("score", 0.0)))

        map_ready_world = [s for s in active if s.get("lat") is not None and s.get("lon") is not None]
        map_ready_europe = [s for s in map_ready_world if s.get("is_europe")]

        result = {
            "generated_at": _kc2g_utc_now_iso(),
            "kc2g_source_mode": source_mode,
            "source_page": KC2G_STATIONS_PAGE,
            "candidate_json_endpoints": CANDIDATE_JSON_ENDPOINTS,
            "fresh_limit_minutes": fresh_limit_minutes,
            "total_parsed": len(cleaned),
            "active_count": len(active),
            "stale_count": len(stale),
            "unknown_age_count": len(unknown_age),
            "map_ready_world_count": len(map_ready_world),
            "map_ready_europe_count": len(map_ready_europe),
            "best_global_station": active[0] if active else None,
            "best_europe_station": next((s for s in active if s.get("is_europe")), None),
            "active_pool": active,
            "stale_pool": stale,
            "unknown_age_pool": unknown_age,
            "map_ready_world_pool": map_ready_world,
            "map_ready_europe_pool": map_ready_europe,
        }
        (self.debug_dir / "kc2g_sumo_pool.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        (self.debug_dir / "kc2g_muf_map_points_world.json").write_text(json.dumps(map_ready_world, indent=2, ensure_ascii=False), encoding="utf-8")
        (self.debug_dir / "kc2g_muf_map_points_europe.json").write_text(json.dumps(map_ready_europe, indent=2, ensure_ascii=False), encoding="utf-8")
        return result




class KC2GMufWorldMapWidgetV3_7WithMap(QtWidgets.QWidget):
    def __init__(self, json_path: str, parent=None) -> None:
        super().__init__(parent)
        self.json_path = Path(json_path)
        self.show_stations = True
        self.show_grid_lines = False
        self.show_labels = False
        self.show_legend = True
        self.show_stats = False
        self.help_visible = False
        self.display_mode = "estimated"
        self.grid_layer_name = "muf_grid_filled"
        self.min_muf = MUF_FIXED_SCALE_MIN
        self.max_muf = MUF_FIXED_SCALE_MAX
        self.heatmap_alpha = 255
        self.gamma = 0.92
        self._payload = {}
        self._latitudes = []
        self._longitudes = []
        self._stations = []
        self._grid_raw = []
        self._grid_filled = []
        self._grid_smoothed = []
        self._grid = []
        self._grid_estimated = []
        self._estimated_mask = []
        self._current_stats = {}
        self._map_features = []
        self._geojson_loaded_path = ""
        self.show_map = True
        self._heatmap_cache_image = None
        self._heatmap_cache_rect = QtCore.QRect()
        self._heatmap_cache_key = None
        self._json_mtime_ns = None
        self.setObjectName("kc2gMufMap")
        self.setMinimumSize(720, 380)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.load_json()
        self.load_geojson()

    def _invalidate_render_cache(self) -> None:
        self._heatmap_cache_image = None
        self._heatmap_cache_rect = QtCore.QRect()
        self._heatmap_cache_key = None

    def load_json(self, force: bool = False) -> bool:
        try:
            if not self.json_path.exists():
                self._json_mtime_ns = None
                self._payload = {}
                self._latitudes = []
                self._longitudes = []
                self._stations = []
                self._grid_raw = []
                self._grid_filled = []
                self._grid_smoothed = []
                self._grid = []
                self._grid_estimated = []
                self._estimated_mask = []
                self._current_stats = {}
                self._invalidate_render_cache()
                self.update()
                return True
            stat = self.json_path.stat()
            mtime_ns = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9)))
            if (not force) and self._json_mtime_ns == mtime_ns and self._payload:
                return False
            data = json.loads(self.json_path.read_text(encoding="utf-8"))
            self._json_mtime_ns = mtime_ns
            self._payload = data
            self._latitudes = data.get("latitudes", [])
            self._longitudes = data.get("longitudes", [])
            self._stations = data.get("stations", [])
            self._grid_raw = data.get("muf_grid", []) or data.get("muf_grid_raw", []) or []
            self._grid_filled = data.get("muf_grid_filled", [])
            self._grid_smoothed = data.get("muf_grid_smoothed", [])
            self._grid_estimated, self._estimated_mask = self._build_estimated_grid()
            self._apply_display_mode()
            return True
        except Exception:
            self._json_mtime_ns = None
            self._payload = {}
            self._latitudes = []
            self._longitudes = []
            self._stations = []
            self._grid_raw = []
            self._grid_filled = []
            self._grid_smoothed = []
            self._grid = []
            self._grid_estimated = []
            self._estimated_mask = []
            self._current_stats = {}
            self._invalidate_render_cache()
            self.update()
            return True

    def load_geojson(self, path=None):
        candidates = []
        if path:
            candidates.append(Path(path))
        candidates.extend([
            APP_DIR / "assets" / "world_countries.geojson",
            Path("assets") / "world_countries.geojson",
            Path(__file__).resolve().parent / "assets" / "world_countries.geojson",
        ])
        self._map_features = []
        self._geojson_loaded_path = ""
        for p in candidates:
            try:
                if p.exists():
                    data = json.loads(p.read_text(encoding="utf-8"))
                    self._map_features = data.get("features", [])
                    self._geojson_loaded_path = str(p)
                    break
            except Exception:
                continue

    def reload(self, force: bool = False) -> None:
        changed = self.load_json(force=force)
        if not self._map_features:
            self.load_geojson()
        if changed:
            self._invalidate_render_cache()
        self.update()

    def has_valid_grid(self) -> bool:
        return bool(self._latitudes and self._longitudes and self._grid)

    def resizeEvent(self, event) -> None:
        self._invalidate_render_cache()
        super().resizeEvent(event)

    def status_text(self) -> str:
        point_count = int(self._payload.get("point_count") or len(self._stations) or 0)
        generated_at = str(self._payload.get("generated_at") or "")
        src = Path(self._geojson_loaded_path).name if self._geojson_loaded_path else "map MISSING"
        return f"SUMO MUF • points={point_count} • {generated_at} • {src}"

    def _apply_display_mode(self) -> None:
        if self.display_mode == "raw":
            self.grid_layer_name = "muf_grid_raw"
            self._grid = self._grid_raw
        elif self.display_mode == "filled":
            self.grid_layer_name = "muf_grid_filled"
            self._grid = self._grid_filled
        elif self.display_mode == "estimated":
            self.grid_layer_name = "muf_grid_estimated"
            if self._grid_smoothed:
                self._grid = self._grid_smoothed
            else:
                self._grid = self._grid_estimated if self._grid_estimated else self._grid_filled
        elif self.display_mode == "delta":
            self.grid_layer_name = "delta(filled-raw)"
            self._grid = self._build_delta_grid()
        elif self.display_mode == "classes":
            self.grid_layer_name = "muf_grid_classes"
            self._grid = self._grid_smoothed if self._grid_smoothed else (self._grid_filled if self._grid_filled else self._grid_raw)
        else:
            self.grid_layer_name = self.display_mode
            self._grid = self._grid_smoothed if self._grid_smoothed else self._grid_filled
        self._auto_adjust_range_and_stats()
        self._invalidate_render_cache()
        self.update()

    def _build_delta_grid(self):
        if not self._grid_raw or not self._grid_filled:
            return []
        out = []
        rows = min(len(self._grid_raw), len(self._grid_filled))
        for i in range(rows):
            raw_row = self._grid_raw[i] if i < len(self._grid_raw) else []
            filled_row = self._grid_filled[i] if i < len(self._grid_filled) else []
            cols = min(len(raw_row), len(filled_row))
            row = []
            for j in range(cols):
                rv = raw_row[j]
                fv = filled_row[j]
                if self._is_valid_value(rv) and self._is_valid_value(fv):
                    row.append(float(fv) - float(rv))
                else:
                    row.append(None)
            out.append(row)
        return out

    def _build_estimated_grid(self):
        source = self._grid_smoothed if self._grid_smoothed else (self._grid_filled if self._grid_filled else self._grid_raw)
        if not source:
            return [], []
        grid = [list(row) for row in source]
        mask = [[False for _ in row] for row in grid]
        rows = len(grid)
        valid_points = []
        for i, row in enumerate(grid):
            lat = self._latitudes[i] if i < len(self._latitudes) else float(i)
            for j, value in enumerate(row):
                if self._is_valid_value(value):
                    lon = self._longitudes[j] if j < len(self._longitudes) else float(j)
                    valid_points.append((i, j, lat, lon, float(value)))
        if not valid_points:
            return grid, mask
        max_radius_cells = 24
        max_neighbors = 28
        lat_scale = 10.0
        lon_scale = 20.0
        sigma = 1.35
        radius = 3.2
        for i in range(rows):
            row = grid[i]
            lat = self._latitudes[i] if i < len(self._latitudes) else float(i)
            for j in range(len(row)):
                if self._is_valid_value(row[j]):
                    continue
                lon = self._longitudes[j] if j < len(self._longitudes) else float(j)
                neighbors = []
                for ii, jj, plat, plon, value in valid_points:
                    di = abs(ii - i)
                    dj = abs(jj - j)
                    if di > max_radius_cells or dj > max_radius_cells:
                        continue
                    dx = (plon - lon) / lon_scale
                    dy = (plat - lat) / lat_scale
                    dist2 = dx * dx + dy * dy
                    dist = math.sqrt(dist2)
                    if dist <= 1e-9:
                        neighbors = [(1.0, value)]
                        break
                    if dist <= radius:
                        neighbors.append((dist2, value))
                if not neighbors:
                    continue
                neighbors.sort(key=lambda x: x[0])
                neighbors = neighbors[:max_neighbors]
                num = 0.0
                den = 0.0
                for dist2, value in neighbors:
                    w = math.exp(-dist2 / sigma)
                    num += w * value
                    den += w
                if den > 0.0:
                    row[j] = num / den
                    mask[i][j] = True
        smoothed = [list(row) for row in grid]
        kernel = ((1.0, 2.0, 1.0), (2.0, 4.0, 2.0), (1.0, 2.0, 1.0))
        for _ in range(3):
            next_grid = [list(row) for row in smoothed]
            for i, row in enumerate(smoothed):
                for j, value in enumerate(row):
                    if not mask[i][j] or not self._is_valid_value(value):
                        continue
                    num = 0.0
                    den = 0.0
                    for ki, di in enumerate((-1, 0, 1)):
                        ii = i + di
                        if ii < 0 or ii >= len(smoothed):
                            continue
                        nrow = smoothed[ii]
                        for kj, dj in enumerate((-1, 0, 1)):
                            jj = j + dj
                            if jj < 0 or jj >= len(nrow):
                                continue
                            neighbor = nrow[jj]
                            if not self._is_valid_value(neighbor):
                                continue
                            weight = kernel[ki][kj]
                            num += weight * float(neighbor)
                            den += weight
                    if den > 0.0:
                        next_grid[i][j] = num / den
            smoothed = next_grid
        return smoothed, mask

    def _is_valid_value(self, v) -> bool:
        if v is None:
            return False
        try:
            return not math.isnan(v)
        except TypeError:
            return True

    def _iter_values(self, grid=None):
        src = self._grid if grid is None else grid
        for row in src:
            for v in row:
                if self._is_valid_value(v):
                    yield float(v)

    def _compute_stats(self, values):
        if not values:
            return {}
        vals = sorted(values)
        n = len(vals)
        def pct(p):
            idx = min(n - 1, max(0, int(round((n - 1) * p))))
            return vals[idx]
        return {
            "count": n,
            "min": vals[0],
            "max": vals[-1],
            "mean": statistics.fmean(vals),
            "median": statistics.median(vals),
            "p10": pct(0.10),
            "p25": pct(0.25),
            "p75": pct(0.75),
            "p90": pct(0.90),
            "stdev": statistics.pstdev(vals) if n > 1 else 0.0,
        }

    def _auto_adjust_range_and_stats(self) -> None:
        values = list(self._iter_values())
        self._current_stats = self._compute_stats(values)

        if self.display_mode == "delta":
            if not values:
                self.min_muf = -1.0
                self.max_muf = 1.0
                return
            values.sort()
            absmax = max(abs(values[0]), abs(values[-1]), 0.5)
            self.min_muf = -absmax
            self.max_muf = absmax
            return

        # Fixed MUF scale for operational readability:
        # the legend stays stable and the colors move on the map with real data changes.
        self.min_muf = float(MUF_FIXED_SCALE_MIN)
        self.max_muf = float(MUF_FIXED_SCALE_MAX)

    def paintEvent(self, event) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        self._draw_background(painter, self.rect())
        plot_rect = self.rect().adjusted(10, 10, -10, -36)
        if self._latitudes and self._longitudes and self._grid:
            self._draw_heatmap_pixmap(painter, plot_rect)
        if self.show_map:
            self._draw_map(painter, plot_rect)
        if self.show_grid_lines:
            self._draw_graticule(painter, plot_rect)
        if self.show_stations and self._stations:
            self._draw_stations(painter, plot_rect)
        self._draw_border(painter, plot_rect)
        if self.show_legend:
            self._draw_legend(painter, self.rect().adjusted(14, self.height() - 24, -14, -8))
        if self.show_stats:
            self._draw_stats_overlay(painter, plot_rect)
        if self.help_visible:
            self._draw_help_overlay(painter, plot_rect)
        painter.end()

    def _draw_background(self, painter, rect):
        grad = QtGui.QLinearGradient(rect.topLeft(), rect.bottomLeft())
        grad.setColorAt(0.0, QtGui.QColor(5, 10, 20))
        grad.setColorAt(1.0, QtGui.QColor(10, 18, 32))
        painter.fillRect(rect, grad)

    def _draw_border(self, painter, rect):
        painter.setPen(QtGui.QPen(QtGui.QColor(180, 200, 220, 90), 1))
        painter.drawRect(rect)

    def _draw_graticule(self, painter, rect):
        pen = QtGui.QPen(QtGui.QColor(220, 220, 220, 30), 1)
        painter.setPen(pen)
        for lon in range(-180, 181, 30):
            x = self._lon_to_x(lon, rect)
            painter.drawLine(QtCore.QPointF(x, rect.top()), QtCore.QPointF(x, rect.bottom()))
        for lat in range(-60, 91, 15):
            y = self._lat_to_y(lat, rect)
            painter.drawLine(QtCore.QPointF(rect.left(), y), QtCore.QPointF(rect.right(), y))

    def _draw_heatmap_pixmap(self, painter, rect):
        lats = self._latitudes
        lons = self._longitudes
        grid = self._grid
        if len(lats) < 2 or len(lons) < 2:
            return

        rows = len(lats)
        cols = len(lons)
        rect_i = QtCore.QRect(rect.toRect()) if isinstance(rect, QtCore.QRectF) else QtCore.QRect(rect)
        cache_key = (
            cols,
            rows,
            int(rect_i.width()),
            int(rect_i.height()),
            round(float(self.min_muf), 3),
            round(float(self.max_muf), 3),
            int(self.heatmap_alpha),
            round(float(self.gamma), 3),
            self.display_mode,
            self.grid_layer_name,
            len(self._stations),
            str(self._payload.get("generated_at") or ""),
        )

        if (
            self._heatmap_cache_image is not None
            and self._heatmap_cache_key == cache_key
            and self._heatmap_cache_rect == rect_i
        ):
            painter.save()
            painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
            painter.drawImage(rect, self._heatmap_cache_image)
            painter.restore()
            return

        img = QtGui.QImage(cols, rows, QtGui.QImage.Format.Format_ARGB32)
        img.fill(QtGui.QColor(8, 14, 24, 255))
        lat_ascending = len(lats) >= 2 and lats[0] < lats[-1]
        for i in range(rows):
            if i >= len(grid):
                continue
            row = grid[i]
            yy = (rows - 1 - i) if lat_ascending else i
            for j in range(cols):
                if j >= len(row):
                    continue
                value = row[j]
                if not self._is_valid_value(value):
                    continue
                color = self._value_to_color(float(value))
                img.setPixelColor(j, yy, color)

        scaled = img.scaled(
            max(1, int(rect_i.width())),
            max(1, int(rect_i.height())),
            QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self._heatmap_cache_image = scaled
        self._heatmap_cache_rect = rect_i
        self._heatmap_cache_key = cache_key

        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawImage(rect, scaled)
        painter.restore()

    def _draw_map(self, painter, rect):
        if not self._map_features:
            return
        painter.save()
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 180), 1))
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        for feat in self._map_features:
            geom = feat.get("geometry", {})
            coords = geom.get("coordinates", [])
            gtype = geom.get("type")
            def draw_polygon(poly):
                for ring in poly:
                    points = []
                    for lon, lat in ring:
                        x = self._lon_to_x(lon, rect)
                        y = self._lat_to_y(lat, rect)
                        points.append(QtCore.QPointF(x, y))
                    if len(points) > 1:
                        painter.drawPolyline(points)
            if gtype == "Polygon":
                draw_polygon(coords)
            elif gtype == "MultiPolygon":
                for poly in coords:
                    draw_polygon(poly)
        painter.restore()

    def _draw_stations(self, painter, rect):
        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        for station in self._stations:
            lat = station.get("lat")
            lon = station.get("lon")
            if lat is None or lon is None:
                continue
            x = self._lon_to_x(float(lon), rect)
            y = self._lat_to_y(float(lat), rect)
            outer_pen = QtGui.QPen(QtGui.QColor(230, 240, 255, 145), 1.0)
            painter.setPen(outer_pen)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QtCore.QPointF(x, y), 2.6, 2.6)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QColor(10, 18, 28, 190))
            painter.drawEllipse(QtCore.QPointF(x, y), 0.95, 0.95)
            if self.show_labels:
                label = station.get("ursi") or ""
                painter.setPen(QtGui.QPen(QtGui.QColor(245, 245, 245, 215), 1))
                painter.drawText(QtCore.QPointF(x + 5, y - 5), label)
        painter.restore()

    def _draw_stats_overlay(self, painter, rect):
        stats = self._current_stats or {}
        raw_count = sum(1 for _ in self._iter_values(self._grid_raw))
        filled_count = sum(1 for _ in self._iter_values(self._grid_filled))
        estimated_count = sum(1 for _ in self._iter_values(self._grid_estimated))
        estimated_added = sum(1 for row in self._estimated_mask for v in row if v)
        delta_count = sum(1 for _ in self._iter_values(self._build_delta_grid())) if self.display_mode == "delta" else None
        lines = [
            f"MODE: {self.display_mode.upper()}    LAYER: {self.grid_layer_name}",
            f"RANGE: {self.min_muf:.2f} -> {self.max_muf:.2f} (fixed)" if self.display_mode != "delta" else f"RANGE: {self.min_muf:.2f} -> {self.max_muf:.2f}",
            f"RAW cells={raw_count}    FILLED cells={filled_count}    EST={estimated_count}",
            f"Estimated gap cells={estimated_added}",
        ]
        if delta_count is not None:
            lines.append(f"DELTA valid cells={delta_count}")
        if stats:
            lines.extend([
                f"min={stats['min']:.2f}   max={stats['max']:.2f}   mean={stats['mean']:.2f}   med={stats['median']:.2f}",
                f"p10={stats['p10']:.2f}   p25={stats['p25']:.2f}   p75={stats['p75']:.2f}   p90={stats['p90']:.2f}",
                f"stdev={stats['stdev']:.2f}   count={stats['count']}",
            ])
        else:
            lines.append("No valid data in current layer")
        box_w = 390
        box_h = 18 + len(lines) * 16
        box = QtCore.QRectF(rect.left() + 10, rect.top() + 10, box_w, box_h)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(7, 12, 18, 210))
        painter.drawRoundedRect(box, 8, 8)
        painter.setPen(QtGui.QPen(QtGui.QColor(220, 235, 245, 235), 1))
        painter.setFont(QtGui.QFont("Consolas", 9))
        y = box.top() + 16
        for line in lines:
            painter.drawText(QtCore.QPointF(box.left() + 10, y), line)
            y += 16

    def _draw_help_overlay(self, painter, rect):
        lines = [
            "R=RAW  F=FILLED  E=ESTIMATED  D=DELTA  C=CLASSES",
            "S=stats  G=grid  L=labels  T=stations  M=map",
            f"Fixed MUF scale: {MUF_FIXED_SCALE_MIN:.0f}-{MUF_FIXED_SCALE_MAX:.0f} MHz",
            "H=help  O=legend  Space=reload JSON",
        ]
        box_w = 350
        box_h = 18 + len(lines) * 16
        box = QtCore.QRectF(rect.right() - box_w - 10, rect.top() + 10, box_w, box_h)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(7, 12, 18, 205))
        painter.drawRoundedRect(box, 8, 8)
        painter.setPen(QtGui.QPen(QtGui.QColor(220, 235, 245, 235), 1))
        painter.setFont(QtGui.QFont("Consolas", 9))
        y = box.top() + 16
        for line in lines:
            painter.drawText(QtCore.QPointF(box.left() + 10, y), line)
            y += 16

    def _draw_legend(self, painter, rect):
        x = rect.left()
        y = rect.top()
        w = min(420, rect.width() - 120)
        h = 12
        grad = QtGui.QLinearGradient(x, y, x + w, y)
        if self.display_mode == "delta":
            delta_stops = [(0.00, QtGui.QColor(35, 85, 220, 255)), (0.50, QtGui.QColor(220, 220, 220, 255)), (1.00, QtGui.QColor(230, 60, 40, 255))]
            for t, c in delta_stops:
                grad.setColorAt(t, c)
        elif self.display_mode == "classes":
            class_stops = [(0.00, QtGui.QColor(35, 85, 220, 255)), (0.25, QtGui.QColor(35, 170, 90, 255)), (0.50, QtGui.QColor(235, 215, 40, 255)), (0.75, QtGui.QColor(255, 140, 20, 255)), (1.00, QtGui.QColor(230, 50, 30, 255))]
            for t, c in class_stops:
                grad.setColorAt(t, c)
        else:
            for t, c in self._continuous_color_stops():
                grad.setColorAt(t, c)
        painter.setPen(QtGui.QPen(QtGui.QColor(220, 220, 220, 120), 1))
        painter.setBrush(grad)
        painter.drawRoundedRect(QtCore.QRectF(x, y, w, h), 4, 4)
        painter.setPen(QtGui.QPen(QtGui.QColor(230, 230, 230, 200), 1))
        for t in [0.0, 0.25, 0.50, 0.75, 1.0]:
            tx = x + t * w
            value = self.min_muf + t * (self.max_muf - self.min_muf)
            painter.drawLine(QtCore.QPointF(tx, y + h), QtCore.QPointF(tx, y + h + 3))
            painter.drawText(QtCore.QPointF(tx - 10, y + h + 14), f"{value:.0f}")
        painter.drawText(QtCore.QPointF(x + w + 12, y + 10), self.grid_layer_name)

    def _lon_to_x(self, lon: float, rect) -> float:
        lon = max(-180.0, min(180.0, lon))
        return rect.left() + ((lon + 180.0) / 360.0) * rect.width()

    def _lat_to_y(self, lat: float, rect) -> float:
        lat = max(-60.0, min(80.0, lat))
        return rect.top() + ((80.0 - lat) / 140.0) * rect.height()

    def _continuous_color_stops(self):
        """Continuous MUF palette with softer transitions between zones."""
        return [
            (0.00, QtGui.QColor(8, 24, 125, self.heatmap_alpha)),
            (0.08, QtGui.QColor(16, 52, 178, self.heatmap_alpha)),
            (0.18, QtGui.QColor(25, 102, 225, self.heatmap_alpha)),
            (0.30, QtGui.QColor(36, 165, 232, self.heatmap_alpha)),
            (0.42, QtGui.QColor(26, 186, 138, self.heatmap_alpha)),
            (0.54, QtGui.QColor(112, 205, 82, self.heatmap_alpha)),
            (0.66, QtGui.QColor(232, 221, 62, self.heatmap_alpha)),
            (0.78, QtGui.QColor(246, 172, 34, self.heatmap_alpha)),
            (0.90, QtGui.QColor(240, 106, 28, self.heatmap_alpha)),
            (1.00, QtGui.QColor(210, 42, 38, self.heatmap_alpha)),
        ]

    def _interpolate_color(self, t: float, stops):
        t = max(0.0, min(1.0, t))
        if t <= stops[0][0]:
            return stops[0][1]
        if t >= stops[-1][0]:
            return stops[-1][1]
        for idx in range(len(stops) - 1):
            t1, c1 = stops[idx]
            t2, c2 = stops[idx + 1]
            if t1 <= t <= t2:
                k = (t - t1) / max(1e-6, (t2 - t1))
                return QtGui.QColor(int(c1.red() + (c2.red() - c1.red()) * k), int(c1.green() + (c2.green() - c1.green()) * k), int(c1.blue() + (c2.blue() - c1.blue()) * k), self.heatmap_alpha)
        return QtGui.QColor(255, 255, 255, self.heatmap_alpha)

    def _normalize(self, value: float) -> float:
        if self.max_muf <= self.min_muf:
            return 0.0
        t = (value - self.min_muf) / (self.max_muf - self.min_muf)
        t = max(0.0, min(1.0, t))
        gamma = float(getattr(self, "gamma", 1.0) or 1.0)
        if gamma <= 0.0:
            gamma = 1.0
        return max(0.0, min(1.0, t ** gamma))

    def _value_to_color(self, value: float):
        if self.display_mode == "delta":
            t = self._normalize(value)
            return self._interpolate_color(t, [(0.00, QtGui.QColor(35, 85, 220, 255)), (0.50, QtGui.QColor(220, 220, 220, 255)), (1.00, QtGui.QColor(230, 60, 40, 255))])
        if self.display_mode == "classes":
            stats = self._current_stats or {}
            p10 = stats.get("p10", self.min_muf)
            p25 = stats.get("p25", self.min_muf + (self.max_muf - self.min_muf) * 0.25)
            p75 = stats.get("p75", self.min_muf + (self.max_muf - self.min_muf) * 0.75)
            p90 = stats.get("p90", self.max_muf)
            if value <= p10:
                return QtGui.QColor(35, 85, 220, 255)
            if value <= p25:
                return QtGui.QColor(35, 170, 90, 255)
            if value <= p75:
                return QtGui.QColor(235, 215, 40, 255)
            if value <= p90:
                return QtGui.QColor(255, 140, 20, 255)
            return QtGui.QColor(230, 50, 30, 255)
        t = self._normalize(value)
        return self._interpolate_color(t, self._continuous_color_stops())

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == QtCore.Qt.Key.Key_R:
            self.display_mode = "raw"
            self._apply_display_mode(); return
        if key == QtCore.Qt.Key.Key_F:
            self.display_mode = "filled"
            self._apply_display_mode(); return
        if key == QtCore.Qt.Key.Key_E:
            self.display_mode = "estimated"
            self._apply_display_mode(); return
        if key == QtCore.Qt.Key.Key_D:
            self.display_mode = "delta"
            self._apply_display_mode(); return
        if key == QtCore.Qt.Key.Key_C:
            self.display_mode = "classes"
            self._apply_display_mode(); return
        if key == QtCore.Qt.Key.Key_S:
            self.show_stats = not self.show_stats; self.update(); return
        if key == QtCore.Qt.Key.Key_G:
            self.show_grid_lines = not self.show_grid_lines; self.update(); return
        if key == QtCore.Qt.Key.Key_L:
            self.show_labels = not self.show_labels; self.update(); return
        if key == QtCore.Qt.Key.Key_M:
            self.show_map = not self.show_map; self.update(); return
        if key == QtCore.Qt.Key.Key_T:
            self.show_stations = not self.show_stations; self.update(); return
        if key == QtCore.Qt.Key.Key_H:
            self.help_visible = not self.help_visible; self.update(); return
        if key == QtCore.Qt.Key.Key_O:
            self.show_legend = not self.show_legend; self.update(); return
        if key == QtCore.Qt.Key.Key_Space:
            self.reload(); return
        super().keyPressEvent(event)


class KC2GPipelineWorker(QtCore.QObject):
    finished = QtCore.Signal(dict)
    error = QtCore.Signal(str)

    def __init__(self, debug_dir: str, parent=None):
        super().__init__(parent)
        self.debug_dir = str(debug_dir)

    @QtCore.Slot()
    def run(self):
        try:
            debug_dir = Path(self.debug_dir)
            debug_dir.mkdir(parents=True, exist_ok=True)
            fetcher = KC2GFetcher(timeout=20, debug_dir=str(debug_dir))
            pool = fetcher.fetch_active_stations(fresh_limit_minutes=60)
            points_path = debug_dir / "kc2g_muf_map_points_world.json"
            grid_path = debug_dir / "kc2g_muf_world_grid_v2.json"
            builder = KC2GMufMapBuilderV2(str(points_path), str(grid_path))
            grid = builder.build_world_grid()
            result = {
                "points_path": str(points_path),
                "grid_path": str(grid_path),
                "active_count": int(pool.get("active_count", 0)),
                "point_count": int(grid.get("point_count", 0)),
                "generated_at": str(grid.get("generated_at", "")),
            }
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))
# =====================================================
# SETTINGS DIALOG
# =====================================================


class KC2GRemoteMapFetchWorker(QtCore.QObject):
    finished = QtCore.Signal(bytes, str, int)
    error = QtCore.Signal(str, int)

    def __init__(self, url: str, timeout: int = 20, generation: int = 0):
        super().__init__()
        self.url = str(url or KC2G_MUF_RENDER_URL).strip() or KC2G_MUF_RENDER_URL
        self.timeout = max(5, int(timeout))
        self.generation = int(generation)

    @QtCore.Slot()
    def run(self):
        try:
            r = requests.get(
                self.url,
                timeout=self.timeout,
                headers={
                    "User-Agent": "SUMO-SunMonitor/0.1",
                    "Accept": "image/svg+xml,image/*;q=0.9,*/*;q=0.8",
                },
            )
            r.raise_for_status()
            payload = r.content or b""
            if not payload:
                raise ValueError("Empty response from KC2G render server.")
            self.finished.emit(payload, self.url, self.generation)
        except Exception as e:
            self.error.emit(str(e), self.generation)


class KC2GRemoteMufMapWidget(QtWidgets.QWidget):
    def __init__(self, url: str = KC2G_MUF_RENDER_URL, parent=None):
        super().__init__(parent)
        self.setObjectName("kc2gRemoteMufMap")
        self.url = str(url or KC2G_MUF_RENDER_URL).strip() or KC2G_MUF_RENDER_URL
        self._svg_bytes = b""
        self._renderer = None
        self._pixmap_cache = None
        self._last_error = ""
        self._last_success_utc = ""
        self._fetch_thread = None
        self._fetch_worker = None
        self._fetch_in_progress = False
        self._loading = False
        self._fetch_generation = 0
        self._stopped = False
        self._alive = True
        self._status_text = "Loading KC2G MUF map…"
        self._site_text = "Source: KC2G • prop.kc2g.com"
        self.setMinimumSize(320, 220)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.reload)
        self._timer.start(KC2G_RENDER_REFRESH_SECONDS * 1000)

        QtCore.QTimer.singleShot(250, self.reload)

    def stop(self):
        self._stopped = True
        self._alive = False
        debug_log("KC2GRemoteMufMapWidget.stop()")
        try:
            if self._timer.isActive():
                self._timer.stop()
        except Exception:
            pass
        try:
            if self._fetch_worker is not None and hasattr(self._fetch_worker, "stop"):
                self._fetch_worker.stop()
        except Exception:
            pass
        try:
            if self._fetch_thread is not None:
                self._fetch_thread.quit()
                if not self._fetch_thread.wait(2500):
                    debug_log("KC2GRemoteMufMapWidget: fetch thread did not stop in time")
        except Exception as e:
            debug_log(f"KC2GRemoteMufMapWidget.stop error: {e}")
        self._fetch_thread = None
        self._fetch_worker = None
        self._fetch_in_progress = False
        self._loading = False

    def has_valid_map(self) -> bool:
        return bool(self._renderer is not None and self._renderer.isValid())

    def status_text(self) -> str:
        if self.has_valid_map():
            if self._last_success_utc:
                return f"KC2G render • {self._last_success_utc}"
            return "KC2G render ready."
        if self._last_error:
            return self._last_error
        return self._status_text

    def reload(self):
        try:
            # KC2GRemoteMufMapWidget also has a reload() method. Keep its original behavior.
            if not hasattr(self, "instrument"):
                self._gen = int(getattr(self, "_gen", 0)) + 1
                gen = self._gen
                debug_log(f"KC2GRemoteMufMapWidget.reload() gen={gen}")
                self._last_error = ""
                self._start_fetch_worker(gen)
                return

            if getattr(self, "_stopped", False):
                return
            if getattr(self, "_fetch_in_progress", False):
                debug_log(f"SohoLatestImageWidget reload skipped instrument={self.instrument} reason=in_progress")
                return
            self._gen = int(getattr(self, "_gen", 0)) + 1
            gen = self._gen
            self._fetch_in_progress = True
            self._loading = True
            self._last_error = ""
            debug_log(f"SohoLatestImageWidget begin fetch instrument={self.instrument} gen={gen}")

            self._worker_thread = QtCore.QThread(self)
            self._worker = SohoImageFetchWorker(self.instrument, timeout=20)
            self._worker.moveToThread(self._worker_thread)
            self._worker_thread.started.connect(self._worker.run)
            self._worker.finished.connect(lambda payload, image_url, site_url, g=gen: self._on_fetch_finished(payload, image_url, site_url, g))
            self._worker.error.connect(lambda msg, g=gen: self._on_fetch_error(msg, g))
            self._worker.finished.connect(self._worker_thread.quit)
            self._worker.error.connect(self._worker_thread.quit)
            self._worker_thread.finished.connect(self._worker.deleteLater)
            self._worker_thread.finished.connect(self._worker_thread.deleteLater)
            self._worker_thread.start()
        except Exception as e:
            if hasattr(self, "instrument"):
                self._fetch_in_progress = False
                self._loading = False
                try:
                    debug_log(f"SohoLatestImageWidget reload error instrument={self.instrument}: {e}")
                except Exception:
                    pass
            else:
                try:
                    debug_log(f"KC2GRemoteMufMapWidget.reload error: {e}")
                except Exception:
                    pass
            self._last_error = str(e)
            self.update()

    def _render_pixmap(self):
        if not getattr(self, '_alive', True) or not _qt_obj_alive(self):
            return
        self._pixmap_cache = None
        if not self.has_valid_map():
            self.update()
            return
        target = self.rect().adjusted(18, 18, -18, -42)
        if target.width() <= 10 or target.height() <= 10:
            self.update()
            return
        pm = QtGui.QPixmap(target.size())
        pm.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pm)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
        self._renderer.render(painter, QtCore.QRectF(0, 0, target.width(), target.height()))
        painter.end()
        self._pixmap_cache = pm
        self.update()

    @QtCore.Slot(bytes, str)
    def _on_fetch_finished(self, payload: bytes, source_url: str, generation: int | None = None):
        if self._stopped or not getattr(self, "_alive", True) or not _qt_obj_alive(self):
            debug_log(f"KC2GRemoteMufMapWidget._on_fetch_finished ignored after stop/deletion gen={generation}")
            return
        if generation is not None and int(generation) != int(self._fetch_generation):
            debug_log(f"KC2GRemoteMufMapWidget stale finished signal ignored gen={generation} current={self._fetch_generation}")
            return
        self._fetch_in_progress = False
        self._fetch_thread = None
        self._fetch_worker = None
        try:
            debug_log(f"KC2GRemoteMufMapWidget fetch finished gen={generation} url={source_url}")
            renderer = QtSvg.QSvgRenderer(payload)
            if not renderer.isValid():
                raise ValueError("Invalid SVG received from KC2G.")
            self._svg_bytes = bytes(payload)
            self._renderer = renderer
            self._last_error = ""
            self._last_success_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
            self._status_text = f"KC2G render • {self._last_success_utc}"
            self._site_text = f"Source: KC2G • {KC2G_SITE_URL}"
            self._render_pixmap()
        except Exception as e:
            self._last_error = str(e)
            self._status_text = "KC2G render unavailable."
            self.update()

    @QtCore.Slot(str)
    def _on_fetch_error(self, message: str, generation: int | None = None):
        if self._stopped or not getattr(self, "_alive", True) or not _qt_obj_alive(self):
            debug_log(f"KC2GRemoteMufMapWidget._on_fetch_error ignored after stop/deletion gen={generation}: {message}")
            return
        if generation is not None and int(generation) != int(self._fetch_generation):
            debug_log(f"KC2GRemoteMufMapWidget stale error signal ignored gen={generation} current={self._fetch_generation}: {message}")
            return
        self._fetch_in_progress = False
        self._loading = False
        self._fetch_thread = None
        self._fetch_worker = None
        self._last_error = str(message)
        self._status_text = "KC2G render unavailable."
        debug_log(f"KC2GRemoteMufMapWidget fetch error gen={generation}: {message}")
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.has_valid_map():
            self._render_pixmap()

    def mouseDoubleClickEvent(self, event):
        try:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(KC2G_SITE_URL))
        except Exception:
            pass
        super().mouseDoubleClickEvent(event)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QtGui.QColor("#0e141a"))

        border_rect = self.rect().adjusted(1, 1, -1, -1)
        painter.setPen(QtGui.QPen(QtGui.QColor("#2a3440"), 2))
        painter.setBrush(QtGui.QColor("#0e141a"))
        painter.drawRoundedRect(border_rect, 14, 14)

        title_rect = QtCore.QRectF(18, 10, max(40, self.width() - 36), 24)
        painter.setPen(QtGui.QColor("#44d16e"))
        painter.setFont(QtGui.QFont("Segoe UI", 14, QtGui.QFont.Bold))
        painter.drawText(title_rect, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, "KC2G MUF")

        map_rect = self.rect().adjusted(18, 18, -18, -42)
        if self._pixmap_cache is not None and not self._pixmap_cache.isNull():
            x = int(map_rect.x() + (map_rect.width() - self._pixmap_cache.width()) / 2)
            y = int(map_rect.y() + (map_rect.height() - self._pixmap_cache.height()) / 2)
            painter.drawPixmap(x, y, self._pixmap_cache)
        else:
            painter.setPen(QtGui.QColor("#d7dde6"))
            painter.setFont(QtGui.QFont("Segoe UI", 12, QtGui.QFont.Bold))
            msg = "Loading KC2G MUF map…"
            if self._last_error:
                msg = "KC2G map temporarily unavailable"
            painter.drawText(map_rect, QtCore.Qt.AlignCenter, msg)

        footer_rect = QtCore.QRectF(18, self.height() - 28, max(40, self.width() - 36), 18)
        painter.setFont(QtGui.QFont("Segoe UI", 9, QtGui.QFont.Bold))
        painter.setPen(QtGui.QColor("#aab6c5"))
        footer = self._site_text
        if self.has_valid_map() and self._last_success_utc:
            footer = f"{footer} • {self._last_success_utc}"
        elif self._last_error:
            footer = f"{footer} • {self._last_error}"
        painter.drawText(footer_rect, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, footer)


class SohoImageFetchWorker(QtCore.QObject):
    finished = QtCore.Signal(bytes, str, str, str)
    error = QtCore.Signal(str)

    def __init__(self, instrument: str = "eit_195", timeout: int = 20):
        super().__init__()
        inst = str(instrument or "eit_195").strip().lower()
        if inst not in SOHO_IMAGE_URLS:
            inst = "eit_195"
        self.instrument = inst
        self.timeout = max(5, int(timeout))

    @QtCore.Slot()
    def run(self):
        image_url = SOHO_IMAGE_URLS.get(self.instrument, SOHO_IMAGE_URLS["eit_195"])
        img_resp = None
        try:
            debug_log(f"SohoImageFetchWorker.run instrument={self.instrument} url={image_url}")
            img_resp = soho_safe_get(image_url, timeout=self.timeout)
            img_resp.raise_for_status()
            payload = img_resp.content or b""
            last_modified = str(img_resp.headers.get("Last-Modified") or "").strip()
            content_type = (img_resp.headers.get("Content-Type") or "").lower()
            debug_log(
                f"SohoImageFetchWorker response instrument={self.instrument} "
                f"status={img_resp.status_code} bytes={len(payload)} content_type={content_type or 'unknown'}"
            )
            if len(payload) < 512:
                raise ValueError(f"SOHO payload too small for {self.instrument}: {len(payload)} bytes")
            if not any(tag in content_type for tag in ("image/jpeg", "image/jpg", "image/png", "image/gif", "application/octet-stream", "binary/octet-stream")):
                debug_log(f"SohoImageFetchWorker unexpected content-type instrument={self.instrument}: {content_type or 'unknown'}")
            self.finished.emit(payload, image_url, SOHO_SITE_URL, last_modified)
        except Exception as e:
            debug_log(f"SohoImageFetchWorker error instrument={self.instrument}: {e}")
            self.error.emit(str(e))
        finally:
            try:
                if img_resp is not None:
                    img_resp.close()
            except Exception:
                pass


class SohoLatestImageWidget(QtWidgets.QWidget):
    def __init__(self, instrument: str = "eit_195", refresh_seconds: int = SOHO_REFRESH_SECONDS, parent=None):
        super().__init__(parent)
        self.setObjectName("sohoLatestImage")
        inst = str(instrument or "eit_195").strip().lower()
        if inst not in SOHO_IMAGE_URLS:
            inst = "eit_195"
        self.instrument = inst
        self.refresh_seconds = max(60, int(refresh_seconds))
        self._pixmap_original = QtGui.QPixmap()
        self._pixmap_cache = None
        self._last_error = ""
        self._loading = False
        self._fetch_in_progress = False
        self._last_success_utc = ""
        self._status_text = "Loading SOHO image…"
        self._source_page = SOHO_IMAGE_URLS.get(self.instrument, SOHO_IMAGE_URLS["eit_195"])
        self._source_image = ""
        self._fetch_thread = None
        self._fetch_worker = None
        self._fetch_in_progress = False
        self._loading = False
        self._fetch_generation = 0
        self._stopped = False
        self.setMinimumSize(320, 220)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.reload)
        self._timer.start(self.refresh_seconds * 1000)

        QtCore.QTimer.singleShot(300, self.reload)

    def stop(self):
        self._stopped = True
        debug_log(f"SohoLatestImageWidget.stop() instrument={self.instrument}")
        try:
            if self._timer.isActive():
                self._timer.stop()
        except Exception:
            pass
        try:
            if self._fetch_worker is not None and hasattr(self._fetch_worker, "stop"):
                self._fetch_worker.stop()
        except Exception:
            pass
        try:
            if self._fetch_thread is not None:
                self._fetch_thread.quit()
                if not self._fetch_thread.wait(2500):
                    debug_log(f"SohoLatestImageWidget thread did not stop in time instrument={self.instrument}")
        except Exception as e:
            debug_log(f"SohoLatestImageWidget.stop error instrument={self.instrument}: {e}")
        self._fetch_thread = None
        self._fetch_worker = None
        self._fetch_in_progress = False
        self._loading = False

    def set_instrument(self, instrument: str):
        inst = str(instrument or "eit_195").strip().lower()
        if inst not in SOHO_IMAGE_URLS:
            inst = "eit_195"
        if inst == self.instrument:
            return
        self.instrument = inst
        self._source_page = SOHO_IMAGE_URLS.get(self.instrument, SOHO_IMAGE_URLS["eit_195"])
        self._status_text = "Loading SOHO image…"
        self._last_error = ""
        self._last_success_utc = ""
        self._source_image = ""
        self._pixmap_original = QtGui.QPixmap()
        self._pixmap_cache = None
        self.update()
        self.reload()

    def set_refresh_seconds(self, seconds: int):
        self.refresh_seconds = max(60, int(seconds))
        try:
            self._timer.start(self.refresh_seconds * 1000)
        except Exception:
            pass

    def has_valid_image(self) -> bool:
        return bool(not self._pixmap_original.isNull())

    def status_text(self) -> str:
        if self.has_valid_image():
            if self._last_success_utc:
                return f"{SOHO_IMAGE_LABELS.get(self.instrument, 'SOHO')} • {self._last_success_utc}"
            return f"{SOHO_IMAGE_LABELS.get(self.instrument, 'SOHO')} ready."
        if self._last_error:
            return self._last_error
        return self._status_text

    def reload(self):
        if self._stopped or self._fetch_in_progress:
            return
        self._begin_fetch()

    def _begin_fetch(self):
        if self._stopped or self._fetch_in_progress:
            return
        if self._fetch_thread is not None:
            try:
                self._fetch_thread.quit()
                self._fetch_thread.wait(300)
            except Exception:
                pass
            self._fetch_thread = None
        if self._fetch_worker is not None:
            try:
                self._fetch_worker.deleteLater()
            except Exception:
                pass
            self._fetch_worker = None
        self._fetch_in_progress = True
        self._loading = True
        self._fetch_generation += 1
        generation = int(self._fetch_generation)
        debug_log(f"SohoLatestImageWidget begin fetch instrument={self.instrument} gen={generation}")
        self._status_text = f"Refreshing {SOHO_IMAGE_LABELS.get(self.instrument, 'SOHO')}…"
        self._fetch_thread = QtCore.QThread(self)
        self._fetch_worker = SohoImageFetchWorker(self.instrument, timeout=20)
        self._fetch_worker.moveToThread(self._fetch_thread)
        self._fetch_thread.started.connect(self._fetch_worker.run)
        self._fetch_worker.finished.connect(lambda payload, image_url, page_url, last_modified, gen=generation: self._on_fetch_finished(payload, image_url, page_url, last_modified, gen))
        self._fetch_worker.error.connect(lambda message, gen=generation: self._on_fetch_error(message, gen))
        self._fetch_worker.finished.connect(self._fetch_thread.quit)
        self._fetch_worker.error.connect(self._fetch_thread.quit)
        self._fetch_worker.finished.connect(self._fetch_worker.deleteLater)
        self._fetch_worker.error.connect(self._fetch_worker.deleteLater)
        self._fetch_thread.finished.connect(self._fetch_thread.deleteLater)
        self._fetch_thread.start()

    def _rerender_pixmap(self):
        if not getattr(self, '_alive', True) or not _qt_obj_alive(self):
            return
        self._pixmap_cache = None
        if not self.has_valid_image():
            self.update()
            return
        target = self.rect().adjusted(18, 38, -18, -42)
        if target.width() <= 10 or target.height() <= 10:
            self.update()
            return
        scaled = self._pixmap_original.scaled(
            target.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self._pixmap_cache = scaled
        self.update()

    @QtCore.Slot(bytes, str, str, str)
    def _on_fetch_finished(self, payload: bytes, image_url: str, page_url: str, last_modified: str, generation: int | None = None):
        if self._stopped or not getattr(self, "_alive", True) or not _qt_obj_alive(self):
            debug_log(f"SohoLatestImageWidget finished ignored after stop/deletion instrument={self.instrument} gen={generation}")
            return
        if generation is not None and int(generation) != int(self._fetch_generation):
            debug_log(f"SohoLatestImageWidget stale finished ignored instrument={self.instrument} gen={generation} current={self._fetch_generation}")
            return
        self._fetch_in_progress = False
        self._loading = False
        self._fetch_thread = None
        self._fetch_worker = None
        try:
            debug_log(f"SohoLatestImageWidget fetch finished instrument={self.instrument} gen={generation} url={image_url}")
            pm = QtGui.QPixmap()
            ok = pm.loadFromData(payload, "JPG")
            if not ok or pm.isNull():
                ok = pm.loadFromData(payload, "JPEG")
            if not ok or pm.isNull():
                ok = pm.loadFromData(payload, "PNG")
            if not ok or pm.isNull():
                ok = pm.loadFromData(payload)
            if not ok or pm.isNull():
                raise ValueError(f"Unable to decode SOHO image (instrument={self.instrument}, bytes={len(payload)}).")
            debug_log(
                f"SohoLatestImageWidget decode ok instrument={self.instrument} "
                f"bytes={len(payload)} size={pm.width()}x{pm.height()}"
            )
            self._pixmap_original = pm
            self._source_image = image_url
            self._source_page = page_url
            self._last_error = ""
            last_success_dt = parse_http_date(last_modified)
            if last_success_dt is None:
                last_success_dt = datetime.now(timezone.utc)
            self._last_success_utc = last_success_dt.strftime("%Y-%m-%d %H:%M:%SZ")
            self._status_text = f"{SOHO_IMAGE_LABELS.get(self.instrument, 'SOHO')} • {self._last_success_utc}"
            self._rerender_pixmap()
        except Exception as e:
            self._last_error = str(e)
            self._status_text = "SOHO image unavailable."
            debug_log(f"SohoLatestImageWidget decode error instrument={self.instrument} gen={generation}: {e}")
            self.update()

    @QtCore.Slot(str)
    def _on_fetch_error(self, message: str, generation: int | None = None):
        if self._stopped or not getattr(self, "_alive", True) or not _qt_obj_alive(self):
            debug_log(f"SohoLatestImageWidget error ignored after stop/deletion instrument={self.instrument} gen={generation}: {message}")
            return
        if generation is not None and int(generation) != int(self._fetch_generation):
            debug_log(f"SohoLatestImageWidget stale error ignored instrument={self.instrument} gen={generation} current={self._fetch_generation}: {message}")
            return
        self._fetch_in_progress = False
        self._loading = False
        self._fetch_thread = None
        self._fetch_worker = None
        self._last_error = str(message)
        self._status_text = "SOHO image unavailable."
        debug_log(f"SohoLatestImageWidget fetch error instrument={self.instrument} gen={generation}: {message}")
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.has_valid_image():
            self._rerender_pixmap()

    def mouseDoubleClickEvent(self, event):
        try:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(self._source_page or SOHO_SITE_URL))
        except Exception:
            pass
        super().mouseDoubleClickEvent(event)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QtGui.QColor("#0e141a"))

        border_rect = self.rect().adjusted(1, 1, -1, -1)
        painter.setPen(QtGui.QPen(QtGui.QColor("#2a3440"), 2))
        painter.setBrush(QtGui.QColor("#0e141a"))
        painter.drawRoundedRect(border_rect, 14, 14)

        title_rect = QtCore.QRectF(18, 10, max(40, self.width() - 36), 24)
        painter.setPen(QtGui.QColor("#44d16e"))
        painter.setFont(QtGui.QFont("Segoe UI", 14, QtGui.QFont.Bold))
        painter.drawText(title_rect, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, SOHO_IMAGE_LABELS.get(self.instrument, "SOHO"))

        image_rect = self.rect().adjusted(18, 38, -18, -42)
        if self._pixmap_cache is not None and not self._pixmap_cache.isNull():
            x = int(image_rect.x() + (image_rect.width() - self._pixmap_cache.width()) / 2)
            y = int(image_rect.y() + (image_rect.height() - self._pixmap_cache.height()) / 2)
            painter.drawPixmap(x, y, self._pixmap_cache)
        else:
            painter.setPen(QtGui.QColor("#d7dde6"))
            painter.setFont(QtGui.QFont("Segoe UI", 12, QtGui.QFont.Bold))
            msg = "Loading SOHO image…"
            if self._last_error:
                msg = "SOHO image temporarily unavailable"
            elif not getattr(self, "_loading", False):
                msg = "Waiting for next refresh…"
            painter.drawText(image_rect, QtCore.Qt.AlignCenter, msg)

        footer_rect = QtCore.QRectF(18, self.height() - 28, max(40, self.width() - 36), 18)
        painter.setFont(QtGui.QFont("Segoe UI", 9, QtGui.QFont.Bold))
        painter.setPen(QtGui.QColor("#aab6c5"))
        footer = ""
        painter.drawText(footer_rect, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, footer)


class SohoDashboardWidget(QtWidgets.QWidget):
    def __init__(self, refresh_seconds: int = SOHO_REFRESH_SECONDS, parent=None):
        super().__init__(parent)
        self.setObjectName("sohoDashboard")
        self.refresh_seconds = max(60, int(refresh_seconds))
        self._tiles = {}
        self.setStyleSheet("QWidget#sohoDashboard { background: #0e141a; border: 2px solid #2a3440; border-radius: 14px; }")

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        head = QtWidgets.QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(8)

        self.lbl_title = QtWidgets.QLabel("SOHO Dashboard")
        self.lbl_title.setStyleSheet("color:#44d16e; font-size: 28px; font-weight: 900;")
        head.addWidget(self.lbl_title, 0)
        head.addStretch(1)

        self.lbl_status = QtWidgets.QLabel("starting…")
        self.lbl_status.setStyleSheet("color:#aab6c5; font-size: 14px; font-weight: 700;")
        head.addWidget(self.lbl_status, 0)
        root.addLayout(head)

        self.lbl_sub = QtWidgets.QLabel("6 official SOHO realtime images shown together")
        self.lbl_sub.setStyleSheet("color:#d7dde6; font-size: 15px; font-family: Consolas; font-weight: 800;")
        root.addWidget(self.lbl_sub, 0)

        self.grid_frame = QtWidgets.QFrame()
        self.grid_frame.setStyleSheet("QFrame { background: transparent; border: 0px; }")
        grid = QtWidgets.QGridLayout(self.grid_frame)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        for idx, instrument in enumerate(SOHO_DASHBOARD_INSTRUMENTS):
            tile = SohoLatestImageWidget(instrument=instrument, refresh_seconds=self.refresh_seconds, parent=self.grid_frame)
            tile.setMinimumSize(180, 140)
            self._tiles[instrument] = tile
            grid.addWidget(tile, idx // 3, idx % 3)

        for col in range(3):
            grid.setColumnStretch(col, 1)
        for row in range(2):
            grid.setRowStretch(row, 1)

        root.addWidget(self.grid_frame, 1)
        
        # Initialize footer label and status timer
        self.lbl_footer = QtWidgets.QLabel("")
        self.lbl_footer.setStyleSheet("color:#aab6c5; font-size: 12px;")
        root.addWidget(self.lbl_footer, 0)
        
        # Start status refresh timer
        self._status_timer = QtCore.QTimer(self)
        self._status_timer.timeout.connect(self._refresh_status)
        self._status_timer.start(1500)
        QtCore.QTimer.singleShot(500, self._refresh_status)

    def set_refresh_seconds(self, seconds: int):
        self.refresh_seconds = max(60, int(seconds))
        for tile in self._tiles.values():
            try:
                tile.set_refresh_seconds(self.refresh_seconds)
            except Exception:
                pass
        self._refresh_status()

    def stop(self):
        try:
            if self._status_timer.isActive():
                self._status_timer.stop()
        except Exception:
            pass
        for tile in self._tiles.values():
            try:
                tile.stop()
            except Exception:
                pass

    def set_active(self, active: bool):
        active = bool(active)
        for tile in self._tiles.values():
            try:
                if active:
                    tile._stopped = False
                    tile.reload()
                else:
                    tile.stop()
            except Exception:
                pass

    def set_instrument(self, instrument: str):
        return

    def has_valid_image(self) -> bool:
        try:
            return any(tile.has_valid_image() for tile in self._tiles.values())
        except Exception:
            return False

    def status_text(self) -> str:
        ok = 0
        total = len(self._tiles)
        last_times = []
        for tile in self._tiles.values():
            try:
                if tile.has_valid_image():
                    ok += 1
                ts = str(getattr(tile, '_last_success_utc', '') or '').strip()
                if ts:
                    last_times.append(ts)
            except Exception:
                pass
        if ok <= 0:
            return 'Loading SOHO image…'
        if ok >= total and last_times:
            return f'All {total} SOHO panels ready • {max(last_times)}'
        return f'SOHO panels ready: {ok}/{total}'

    def _refresh_status(self):
        ok = 0
        errs = []
        for tile in self._tiles.values():
            try:
                if tile.has_valid_image():
                    ok += 1
                else:
                    err = str(getattr(tile, '_last_error', '') or '').strip()
                    if err:
                        errs.append(err)
            except Exception:
                pass
        total = len(self._tiles)
        if ok >= total:
            self.lbl_status.setText('ok')
        elif ok > 0:
            self.lbl_status.setText(f'partial {ok}/{total}')
        else:
            self.lbl_status.setText('updating…' if not errs else 'partial 0/6')
        self.lbl_footer.setText('')



class SettingsDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent=None,
        rss_url: str = "",
        nasa_api_key: str = "",
        time_mode: str = "utc",
        clock_grid_enabled: bool = False,
        clock_city_tl: str = "America/New_York",
        clock_city_tr: str = "Europe/London",
        clock_city_bl: str = "Asia/Tokyo",
        clock_city_br: str = "Australia/Sydney",
        clock_hour_chime_mode: str = "off",
        alert_volume: int = 90,
        hour_chime_volume: int = 90,
        display_mode: str = "solar",
        display_views: Optional[List[str]] = None,
        display_switch_seconds: int = 30,
        muf_source: str = "sumo",
        soho_instrument: str = "eit_195",
        soho_refresh_seconds: int = SOHO_REFRESH_SECONDS,
        rss_speed: int = RSS_SCROLL_PX_PER_TICK,
        dx_enabled: bool = False,
        dx_source: str = "dx",
        dx_host: str = "dxspider.co.uk",
        dx_port: int = 7300,
        dx_login: str = "",
        pota_zone: str = "worldwide",
        dapnet_config: Optional[dict] = None,
    ):
        super().__init__(parent)
        self._dapnet_cfg = normalize_dapnet_config(dapnet_config)
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
        link_row.addWidget(self.btn_get_key)
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
        form_time.addRow("Central clock:", self.cb_time_mode)

        self.cb_clock_grid_enabled = QtWidgets.QCheckBox("Replace the 9 solar tiles with the clock dashboard")
        self.cb_clock_grid_enabled.setChecked(bool(clock_grid_enabled))
        self.cb_clock_grid_enabled.setToolTip("Show one large center analog clock plus four smaller world clocks in the corners.")
        form_time.addRow("Dashboard:", self.cb_clock_grid_enabled)

        def _make_city_combo(selected_tz: str) -> QtWidgets.QComboBox:
            cb = QtWidgets.QComboBox()
            for city_name, tz_name in CLOCK_CITY_OPTIONS:
                cb.addItem(city_name, tz_name)
            target = (selected_tz or "").strip() or "Europe/London"
            idx = max(0, cb.findData(target))
            cb.setCurrentIndex(idx)
            cb.setMinimumWidth(180)
            return cb

        self.cb_clock_city_tl = _make_city_combo(clock_city_tl)
        self.cb_clock_city_tr = _make_city_combo(clock_city_tr)
        self.cb_clock_city_bl = _make_city_combo(clock_city_bl)
        self.cb_clock_city_br = _make_city_combo(clock_city_br)

        form_time.addRow("Top-left city:", self.cb_clock_city_tl)
        form_time.addRow("Top-right city:", self.cb_clock_city_tr)
        form_time.addRow("Bottom-left city:", self.cb_clock_city_bl)
        form_time.addRow("Bottom-right city:", self.cb_clock_city_br)

        self.cb_clock_hour_chime_mode = QtWidgets.QComboBox()
        self.cb_clock_hour_chime_mode.addItem("Disabled", "off")
        self.cb_clock_hour_chime_mode.addItem("Enabled", "on")
        chime_mode = (clock_hour_chime_mode or "off").strip().lower()
        if chime_mode in ("utc", "local"):
            chime_mode = "on"
        if chime_mode not in ("off", "on"):
            chime_mode = "on" if bool(clock_hour_chime_mode) else "off"
        idx_chime = max(0, self.cb_clock_hour_chime_mode.findData(chime_mode))
        self.cb_clock_hour_chime_mode.setCurrentIndex(idx_chime)
        self.cb_clock_hour_chime_mode.setToolTip("Plays assets/sounds/hour.wav 4 seconds before the top of the hour, following the current center clock mode (UTC or Local).")
        form_time.addRow("Hour chime:", self.cb_clock_hour_chime_mode)

        time_l.addLayout(form_time)

        hint_time = QtWidgets.QLabel(
            "• Central analog clock uses UTC or your local time.\n"
            "• The four corner clocks use the cities selected here.\n"
            "• Optional hour chime plays once at HH:59:56, 4 seconds before the top of the hour.\n"
            "• The chime follows the current center clock mode (UTC or Local).\n"
            "• When the dashboard is enabled, it replaces the 9 solar tiles area."
        )
        hint_time.setStyleSheet("color: #aab6c5; font-size: 11px;")
        hint_time.setWordWrap(True)
        time_l.addWidget(hint_time)
        time_l.addStretch(1)

        # ========== TAB 4: Sound ==========
        tab_sound = QtWidgets.QWidget()
        sound_l = QtWidgets.QVBoxLayout(tab_sound)
        sound_l.setContentsMargins(14, 14, 14, 14)
        sound_l.setSpacing(10)

        form_sound = QtWidgets.QFormLayout()
        form_sound.setHorizontalSpacing(10)
        form_sound.setVerticalSpacing(8)

        def _make_volume_slider(initial_value: int, tooltip: str):
            value = max(0, min(100, int(initial_value if initial_value is not None else 90)))

            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(10)

            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setSingleStep(1)
            slider.setPageStep(5)
            slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
            slider.setTickInterval(10)
            slider.setValue(value)
            slider.setToolTip(tooltip)
            slider.setMinimumWidth(260)

            label = QtWidgets.QLabel(f"{value} %")
            label.setMinimumWidth(48)
            label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            label.setStyleSheet("color: #d7dde6; font-weight: 800;")

            slider.valueChanged.connect(lambda v, lbl=label: lbl.setText(f"{int(v)} %"))

            row.addWidget(slider, 1)
            row.addWidget(label, 0)
            return slider, row, label

        self.sb_alert_volume, alert_volume_row, self.lbl_alert_volume_percent = _make_volume_slider(
            alert_volume,
            "Volume for solar alert sounds and the generic alert.wav feedback.",
        )
        form_sound.addRow("Solar alerts volume:", alert_volume_row)

        self.sb_hour_chime_volume, hour_chime_volume_row, self.lbl_hour_chime_volume_percent = _make_volume_slider(
            hour_chime_volume,
            "Volume for the hourly chime.",
        )
        form_sound.addRow("Hour chime volume:", hour_chime_volume_row)

        sound_l.addLayout(form_sound)

        hint_sound = QtWidgets.QLabel(
            "• These two volumes are independent.\n"
            "• The main Sound ON/OFF button still acts as a global mute.\n"
            "• 0% keeps the feature enabled but silent."
        )
        hint_sound.setStyleSheet("color: #aab6c5; font-size: 11px;")
        hint_sound.setWordWrap(True)
        sound_l.addWidget(hint_sound)
        sound_l.addStretch(1)

        # ========== TAB 5: Display ==========
        tab_display = QtWidgets.QWidget()
        display_l = QtWidgets.QVBoxLayout(tab_display)
        display_l.setContentsMargins(14, 14, 14, 14)
        display_l.setSpacing(10)

        form_display = QtWidgets.QFormLayout()
        form_display.setHorizontalSpacing(10)
        form_display.setVerticalSpacing(8)


        initial_views = normalize_display_views(display_views, display_mode)
        self.gb_display_views = QtWidgets.QGroupBox("Main views included in rotation")
        self.gb_display_views.setStyleSheet(
            "QGroupBox { color:#d7dde6; font-weight:800; border:1px solid #2a3440; border-radius:8px; margin-top:8px; padding-top:10px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding:0 4px; }"
        )
        display_views_layout = QtWidgets.QGridLayout(self.gb_display_views)
        display_views_layout.setContentsMargins(12, 14, 12, 10)
        display_views_layout.setHorizontalSpacing(18)
        display_views_layout.setVerticalSpacing(8)

        self.cb_view_solar = QtWidgets.QCheckBox("Solar tiles")
        self.cb_view_clock = QtWidgets.QCheckBox("Clock dashboard")
        self.cb_view_muf = QtWidgets.QCheckBox("SUMO MUF map")
        self.cb_view_soho = QtWidgets.QCheckBox("SOHO dashboard")
        self.cb_view_solarsystem = QtWidgets.QCheckBox("Solar System")
        self.cb_view_widgetdemo = QtWidgets.QCheckBox("Widget demo (safe)")

        self._display_view_checkboxes = {
            "solar": self.cb_view_solar,
            "clock": self.cb_view_clock,
            "muf": self.cb_view_muf,
            "soho": self.cb_view_soho,
            "solarsystem": self.cb_view_solarsystem,
            "widgetdemo": self.cb_view_widgetdemo,
        }
        for idx, key in enumerate(("solar", "clock", "muf", "soho", "solarsystem", "widgetdemo")):
            row = idx // 2
            col = idx % 2
            cb = self._display_view_checkboxes[key]
            cb.setChecked(key in initial_views)
            cb.setStyleSheet("color:#d7dde6; font-weight:700;")
            display_views_layout.addWidget(cb, row, col)

        form_display.addRow("Main display:", self.gb_display_views)

        self.sb_display_switch_seconds = QtWidgets.QSpinBox()
        self.sb_display_switch_seconds.setRange(5, 3600)
        self.sb_display_switch_seconds.setSuffix(" s")
        self.sb_display_switch_seconds.setValue(max(5, int(display_switch_seconds or 30)))
        self.sb_display_switch_seconds.setToolTip("Used only when alternating between solar tiles, clocks, MUF, SOHO, the Solar System dashboard and/or the safe widget demo.")
        form_display.addRow("Switch every:", self.sb_display_switch_seconds)

        self.cb_muf_source = QtWidgets.QComboBox()
        self.cb_muf_source.addItem("SUMO local render", "sumo")
        self.cb_muf_source.addItem("KC2G server render", "kc2g")
        ms = (muf_source or "sumo").strip().lower()
        if ms not in ("sumo", "kc2g"):
            ms = "sumo"
        idx_ms = max(0, self.cb_muf_source.findData(ms))
        self.cb_muf_source.setCurrentIndex(idx_ms)
        self.cb_muf_source.setToolTip("Choose whether SUMO shows its own MUF map or the render served by KC2G.")
        form_display.addRow("MUF source:", self.cb_muf_source)

        self.cb_soho_instrument = QtWidgets.QComboBox()
        self.cb_soho_instrument.addItem("6-image dashboard", "dashboard6")
        self.cb_soho_instrument.setCurrentIndex(0)
        self.cb_soho_instrument.setEnabled(False)
        self.cb_soho_instrument.setToolTip("SUMO now shows the 6 official SOHO realtime images together in one dashboard.")
        form_display.addRow("SOHO layout:", self.cb_soho_instrument)

        self.sb_soho_refresh_seconds = QtWidgets.QSpinBox()
        self.sb_soho_refresh_seconds.setRange(60, 3600)
        self.sb_soho_refresh_seconds.setSuffix(" s")
        self.sb_soho_refresh_seconds.setValue(max(60, int(soho_refresh_seconds or SOHO_REFRESH_SECONDS)))
        self.sb_soho_refresh_seconds.setToolTip("Refresh interval for the 6-image SOHO dashboard.")
        form_display.addRow("SOHO refresh:", self.sb_soho_refresh_seconds)

        display_l.addLayout(form_display)

        hint_display = QtWidgets.QLabel(
            "• Check one view only for a fixed main display.\n"
            "• Check two or more views to alternate automatically.\n"
            "• The switch delay below is used only when at least two views are checked.\n"
            "• MUF source: choose between the local SUMO render and the KC2G server render.\n"
            "• Widget demo: safe placeholder to validate the future widget architecture."
        )
        hint_display.setStyleSheet("color: #aab6c5; font-size: 11px;")
        hint_display.setWordWrap(True)
        display_l.addWidget(hint_display)
        display_l.addStretch(1)

        self.cb_clock_grid_enabled.setEnabled(False)
        self.cb_clock_grid_enabled.setToolTip("Legacy option kept for compatibility. Use the Display tab to choose the main display mode.")

        # ========== TAB 5: DX Cluster ==========
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

        # ========== TAB 6: DAPNET ==========
        tab_dapnet = QtWidgets.QWidget()
        dap_l = QtWidgets.QVBoxLayout(tab_dapnet)
        dap_l.setContentsMargins(14, 14, 14, 14)
        dap_l.setSpacing(10)

        form_dap = QtWidgets.QFormLayout()
        form_dap.setHorizontalSpacing(10)
        form_dap.setVerticalSpacing(8)

        self.cb_dapnet_enabled = QtWidgets.QCheckBox("Enable DAPNET alerts")
        self.cb_dapnet_enabled.setChecked(bool(self._dapnet_cfg.get("enabled", False)))
        form_dap.addRow("DAPNET:", self.cb_dapnet_enabled)

        self.ed_dapnet_user = QtWidgets.QLineEdit(str(self._dapnet_cfg.get("username") or ""))
        self.ed_dapnet_user.setPlaceholderText("DAPNET username")
        self.ed_dapnet_user.setClearButtonEnabled(True)
        form_dap.addRow("Username:", self.ed_dapnet_user)

        self.ed_dapnet_pass = QtWidgets.QLineEdit(str(self._dapnet_cfg.get("password") or ""))
        self.ed_dapnet_pass.setPlaceholderText("DAPNET password")
        self.ed_dapnet_pass.setEchoMode(QtWidgets.QLineEdit.Password)
        self.ed_dapnet_pass.setClearButtonEnabled(True)
        form_dap.addRow("Password:", self.ed_dapnet_pass)

        self.ed_dapnet_tx_group = QtWidgets.QLineEdit(str(self._dapnet_cfg.get("tx_group") or "f-53"))
        self.ed_dapnet_tx_group.setPlaceholderText("f-53")
        self.ed_dapnet_tx_group.setClearButtonEnabled(True)
        form_dap.addRow("TX group:", self.ed_dapnet_tx_group)

        self.ed_dapnet_callsigns = QtWidgets.QLineEdit(", ".join(normalize_callsigns(self._dapnet_cfg.get("callsigns"))))
        self.ed_dapnet_callsigns.setPlaceholderText("f4igv, f4fap")
        self.ed_dapnet_callsigns.setClearButtonEnabled(True)
        form_dap.addRow("Recipients:", self.ed_dapnet_callsigns)

        self.cb_dapnet_quick_ui_enabled = QtWidgets.QCheckBox("Show manual quick-send bar in main UI")
        self.cb_dapnet_quick_ui_enabled.setChecked(bool(self._dapnet_cfg.get("quick_ui_enabled", True)))
        form_dap.addRow("Manual UI:", self.cb_dapnet_quick_ui_enabled)

        dap_l.addLayout(form_dap)

        gb_xray = QtWidgets.QGroupBox("X-Ray alerts")
        gb_xray_l = QtWidgets.QFormLayout(gb_xray)
        xray_cfg = self._dapnet_cfg.get("xray", {})
        self.cb_dapnet_xray_enabled = QtWidgets.QCheckBox("Enable X-Ray alerts")
        self.cb_dapnet_xray_enabled.setChecked(bool(xray_cfg.get("enabled", False)))
        gb_xray_l.addRow("Module:", self.cb_dapnet_xray_enabled)
        self.cb_dapnet_xray_threshold = QtWidgets.QComboBox()
        for item in ("C5.0", "M1.0", "M5.0", "M9.0", "X1.0"):
            self.cb_dapnet_xray_threshold.addItem(item, item)
        idx = max(0, self.cb_dapnet_xray_threshold.findData(str(xray_cfg.get("threshold") or "M1.0").upper()))
        self.cb_dapnet_xray_threshold.setCurrentIndex(idx)
        gb_xray_l.addRow("Threshold:", self.cb_dapnet_xray_threshold)
        self.cb_dapnet_xray_send_start = QtWidgets.QCheckBox("Send start of storm")
        self.cb_dapnet_xray_send_start.setChecked(bool(xray_cfg.get("send_start", True)))
        gb_xray_l.addRow("Start message:", self.cb_dapnet_xray_send_start)
        self.cb_dapnet_xray_send_end = QtWidgets.QCheckBox("Send end of storm")
        self.cb_dapnet_xray_send_end.setChecked(bool(xray_cfg.get("send_end", True)))
        gb_xray_l.addRow("End message:", self.cb_dapnet_xray_send_end)
        self.cb_dapnet_xray_emergency = QtWidgets.QCheckBox("Emergency flag on start")
        self.cb_dapnet_xray_emergency.setChecked(bool(xray_cfg.get("emergency_on_start", True)))
        gb_xray_l.addRow("Emergency:", self.cb_dapnet_xray_emergency)
        dap_l.addWidget(gb_xray)

        gb_proton = QtWidgets.QGroupBox("Proton / Solar wind")
        gb_proton_l = QtWidgets.QFormLayout(gb_proton)
        proton_cfg = self._dapnet_cfg.get("proton", {})
        self.cb_dapnet_proton_enabled = QtWidgets.QCheckBox("Enable proton alerts")
        self.cb_dapnet_proton_enabled.setChecked(bool(proton_cfg.get("enabled", False)))
        gb_proton_l.addRow("Module:", self.cb_dapnet_proton_enabled)
        self.cb_dapnet_proton_threshold = QtWidgets.QComboBox()
        for item in ("S1", "S2", "S3", "S4", "S5"):
            self.cb_dapnet_proton_threshold.addItem(item, item)
        idx = max(0, self.cb_dapnet_proton_threshold.findData(str(proton_cfg.get("threshold") or "S1").upper()))
        self.cb_dapnet_proton_threshold.setCurrentIndex(idx)
        gb_proton_l.addRow("Threshold:", self.cb_dapnet_proton_threshold)
        self.sb_dapnet_proton_cooldown = QtWidgets.QSpinBox()
        self.sb_dapnet_proton_cooldown.setRange(5, 1440)
        self.sb_dapnet_proton_cooldown.setSuffix(" min")
        self.sb_dapnet_proton_cooldown.setValue(int(proton_cfg.get("cooldown_minutes", 30)))
        gb_proton_l.addRow("Cooldown:", self.sb_dapnet_proton_cooldown)
        self.cb_dapnet_proton_include_bzbt = QtWidgets.QCheckBox("Include Bz/Bt values")
        self.cb_dapnet_proton_include_bzbt.setChecked(bool(proton_cfg.get("include_bz_bt", True)))
        gb_proton_l.addRow("Message content:", self.cb_dapnet_proton_include_bzbt)
        dap_l.addWidget(gb_proton)

        gb_summary = QtWidgets.QGroupBox("Periodic solar summary")
        gb_summary_l = QtWidgets.QFormLayout(gb_summary)
        summary_cfg = self._dapnet_cfg.get("solar_summary", {})
        self.cb_dapnet_summary_enabled = QtWidgets.QCheckBox("Enable periodic solar summary")
        self.cb_dapnet_summary_enabled.setChecked(bool(summary_cfg.get("enabled", False)))
        gb_summary_l.addRow("Module:", self.cb_dapnet_summary_enabled)
        self.sb_dapnet_summary_interval = QtWidgets.QSpinBox()
        self.sb_dapnet_summary_interval.setRange(5, 1440)
        self.sb_dapnet_summary_interval.setSuffix(" min")
        self.sb_dapnet_summary_interval.setValue(int(summary_cfg.get("interval_minutes", 60)))
        gb_summary_l.addRow("Interval:", self.sb_dapnet_summary_interval)
        dap_l.addWidget(gb_summary)

        gb_iss = QtWidgets.QGroupBox("ISS alerts")
        gb_iss_l = QtWidgets.QFormLayout(gb_iss)
        iss_cfg = self._dapnet_cfg.get("iss", {})
        self.cb_dapnet_iss_enabled = QtWidgets.QCheckBox("Enable ISS alerts")
        self.cb_dapnet_iss_enabled.setChecked(bool(iss_cfg.get("enabled", False)))
        gb_iss_l.addRow("Module:", self.cb_dapnet_iss_enabled)
        self.cb_dapnet_iss_prealert = QtWidgets.QCheckBox("Prealert")
        self.cb_dapnet_iss_prealert.setChecked(bool(iss_cfg.get("prealert_enabled", True)))
        gb_iss_l.addRow("Prealert:", self.cb_dapnet_iss_prealert)
        self.cb_dapnet_iss_start = QtWidgets.QCheckBox("Pass start")
        self.cb_dapnet_iss_start.setChecked(bool(iss_cfg.get("start_enabled", True)))
        gb_iss_l.addRow("Start:", self.cb_dapnet_iss_start)
        self.cb_dapnet_iss_peak = QtWidgets.QCheckBox("Pass peak")
        self.cb_dapnet_iss_peak.setChecked(bool(iss_cfg.get("peak_enabled", True)))
        gb_iss_l.addRow("Peak:", self.cb_dapnet_iss_peak)
        self.cb_dapnet_iss_end = QtWidgets.QCheckBox("Pass end")
        self.cb_dapnet_iss_end.setChecked(bool(iss_cfg.get("end_enabled", True)))
        gb_iss_l.addRow("End:", self.cb_dapnet_iss_end)
        self.sb_dapnet_iss_prealert_minutes = QtWidgets.QSpinBox()
        self.sb_dapnet_iss_prealert_minutes.setRange(1, 120)
        self.sb_dapnet_iss_prealert_minutes.setSuffix(" min")
        self.sb_dapnet_iss_prealert_minutes.setValue(int(iss_cfg.get("prealert_minutes", 15)))
        gb_iss_l.addRow("Prealert delay:", self.sb_dapnet_iss_prealert_minutes)
        self.sb_dapnet_iss_min_elev = QtWidgets.QDoubleSpinBox()
        self.sb_dapnet_iss_min_elev.setRange(0.0, 90.0)
        self.sb_dapnet_iss_min_elev.setDecimals(1)
        self.sb_dapnet_iss_min_elev.setSuffix(" °")
        self.sb_dapnet_iss_min_elev.setValue(float(iss_cfg.get("min_elevation", 5.0)))
        gb_iss_l.addRow("Min elevation:", self.sb_dapnet_iss_min_elev)

        self.sb_dapnet_iss_observer_lat = QtWidgets.QDoubleSpinBox()
        self.sb_dapnet_iss_observer_lat.setRange(-90.0, 90.0)
        self.sb_dapnet_iss_observer_lat.setDecimals(4)
        self.sb_dapnet_iss_observer_lat.setSingleStep(0.0001)
        self.sb_dapnet_iss_observer_lat.setValue(float(iss_cfg.get("observer_lat", 48.1173)))
        gb_iss_l.addRow("Observer latitude:", self.sb_dapnet_iss_observer_lat)

        self.sb_dapnet_iss_observer_lon = QtWidgets.QDoubleSpinBox()
        self.sb_dapnet_iss_observer_lon.setRange(-180.0, 180.0)
        self.sb_dapnet_iss_observer_lon.setDecimals(4)
        self.sb_dapnet_iss_observer_lon.setSingleStep(0.0001)
        self.sb_dapnet_iss_observer_lon.setValue(float(iss_cfg.get("observer_lon", -1.6778)))
        gb_iss_l.addRow("Observer longitude:", self.sb_dapnet_iss_observer_lon)

        self.sb_dapnet_iss_observer_alt_m = QtWidgets.QSpinBox()
        self.sb_dapnet_iss_observer_alt_m.setRange(-500, 10000)
        self.sb_dapnet_iss_observer_alt_m.setSuffix(" m")
        self.sb_dapnet_iss_observer_alt_m.setValue(int(float(iss_cfg.get("observer_alt_m", 60))))
        gb_iss_l.addRow("Observer altitude:", self.sb_dapnet_iss_observer_alt_m)

        dap_l.addWidget(gb_iss)

        test_row = QtWidgets.QHBoxLayout()
        test_row.setContentsMargins(0, 0, 0, 0)
        test_row.setSpacing(8)
        self.btn_dapnet_test_xray = QtWidgets.QPushButton("Test X-Ray")
        self.btn_dapnet_test_xray.clicked.connect(lambda: self._send_dapnet_test_message("xray"))
        test_row.addWidget(self.btn_dapnet_test_xray, 0)
        self.btn_dapnet_test_proton = QtWidgets.QPushButton("Test Proton")
        self.btn_dapnet_test_proton.clicked.connect(lambda: self._send_dapnet_test_message("proton"))
        test_row.addWidget(self.btn_dapnet_test_proton, 0)
        self.btn_dapnet_test_solar = QtWidgets.QPushButton("Test Solar")
        self.btn_dapnet_test_solar.clicked.connect(lambda: self._send_dapnet_test_message("solar"))
        test_row.addWidget(self.btn_dapnet_test_solar, 0)
        self.btn_dapnet_test_iss = QtWidgets.QPushButton("Test ISS")
        self.btn_dapnet_test_iss.clicked.connect(lambda: self._send_dapnet_test_message("iss"))
        test_row.addWidget(self.btn_dapnet_test_iss, 0)
        test_row.addStretch(1)
        dap_l.addLayout(test_row)

        hint_dapnet = QtWidgets.QLabel(
            "• V7.5 adds the complete DAPNET alert engine and one dedicated test button per module.\n"
            "• X-Ray, Proton, Solar summary and ISS modules are live and use the current SUMO data/state.\n"
            "• Password is stored in the local SUMO config file for convenience."
        )
        hint_dapnet.setStyleSheet("color: #aab6c5; font-size: 11px;")
        hint_dapnet.setWordWrap(True)
        dap_l.addWidget(hint_dapnet)
        dap_l.addStretch(1)

        # Add tabs
        tabs.addTab(tab_rss, "RSS")
        tabs.addTab(tab_api, "API Keys")
        tabs.addTab(tab_time, "Clock")
        tabs.addTab(tab_sound, "Sound")
        tabs.addTab(tab_display, "Display")
        tabs.addTab(tab_dx, "DX Cluster")

        dap_scroll = QtWidgets.QScrollArea()
        dap_scroll.setWidgetResizable(True)
        dap_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        dap_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        dap_scroll.setWidget(tab_dapnet)
        tabs.addTab(dap_scroll, "DAPNET")

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

    def clock_grid_enabled(self) -> bool:
        mode = self.display_mode()
        if mode == "clock":
            return True
        if mode == "solar":
            return False
        return bool(getattr(self, "cb_clock_grid_enabled", None) and self.cb_clock_grid_enabled.isChecked())


    def display_views(self) -> list[str]:
        try:
            selected = [
                key for key in ("solar", "clock", "muf", "soho", "solarsystem", "widgetdemo")
                if getattr(self, "_display_view_checkboxes", {}).get(key) is not None
                and self._display_view_checkboxes[key].isChecked()
            ]
        except Exception:
            selected = []
        return normalize_display_views(selected, "solar")

    def display_mode(self) -> str:
        views = self.display_views()
        if len(views) == 1:
            return views[0]
        return "alternate_custom"

    def display_switch_seconds(self) -> int:
        try:
            return max(5, int(self.sb_display_switch_seconds.value()))
        except Exception:
            return 30

    def muf_source(self) -> str:
        try:
            data = self.cb_muf_source.currentData()
            val = (str(data) if data else "sumo").strip().lower()
        except Exception:
            val = "sumo"
        if val not in ("sumo", "kc2g"):
            val = "sumo"
        return val

    def soho_instrument(self) -> str:
        try:
            data = self.cb_soho_instrument.currentData()
            val = (str(data) if data else "eit_195").strip().lower()
        except Exception:
            val = "eit_195"
        if val not in ("dashboard6",):
            val = "dashboard6"
        return val

    def soho_refresh_seconds(self) -> int:
        try:
            return max(60, int(self.sb_soho_refresh_seconds.value()))
        except Exception:
            return SOHO_REFRESH_SECONDS

    def dapnet_config(self) -> dict:
        cfg = normalize_dapnet_config({
            "enabled": bool(self.cb_dapnet_enabled.isChecked()),
            "username": self.ed_dapnet_user.text().strip(),
            "password": self.ed_dapnet_pass.text(),
            "tx_group": self.ed_dapnet_tx_group.text().strip(),
            "callsigns": normalize_callsigns(self.ed_dapnet_callsigns.text()),
            "quick_ui_enabled": bool(self.cb_dapnet_quick_ui_enabled.isChecked()),
            "xray": {
                "enabled": bool(self.cb_dapnet_xray_enabled.isChecked()),
                "threshold": str(self.cb_dapnet_xray_threshold.currentData() or "M1.0"),
                "send_start": bool(self.cb_dapnet_xray_send_start.isChecked()),
                "send_end": bool(self.cb_dapnet_xray_send_end.isChecked()),
                "emergency_on_start": bool(self.cb_dapnet_xray_emergency.isChecked()),
            },
            "proton": {
                "enabled": bool(self.cb_dapnet_proton_enabled.isChecked()),
                "threshold": str(self.cb_dapnet_proton_threshold.currentData() or "S1"),
                "cooldown_minutes": int(self.sb_dapnet_proton_cooldown.value()),
                "include_bz_bt": bool(self.cb_dapnet_proton_include_bzbt.isChecked()),
            },
            "solar_summary": {
                "enabled": bool(self.cb_dapnet_summary_enabled.isChecked()),
                "interval_minutes": int(self.sb_dapnet_summary_interval.value()),
            },
            "iss": {
                "enabled": bool(self.cb_dapnet_iss_enabled.isChecked()),
                "prealert_enabled": bool(self.cb_dapnet_iss_prealert.isChecked()),
                "start_enabled": bool(self.cb_dapnet_iss_start.isChecked()),
                "peak_enabled": bool(self.cb_dapnet_iss_peak.isChecked()),
                "end_enabled": bool(self.cb_dapnet_iss_end.isChecked()),
                "prealert_minutes": int(self.sb_dapnet_iss_prealert_minutes.value()),
                "min_elevation": float(self.sb_dapnet_iss_min_elev.value()),
                "observer_lat": float(self.sb_dapnet_iss_observer_lat.value()),
                "observer_lon": float(self.sb_dapnet_iss_observer_lon.value()),
                "observer_alt_m": int(self.sb_dapnet_iss_observer_alt_m.value()),
            },
        })
        self._dapnet_cfg = cfg
        return cfg

    def _send_dapnet_test_message(self, module_name: str = "generic"):
        cfg = self.dapnet_config()
        if not cfg.get("username") or not cfg.get("password") or not normalize_callsigns(cfg.get("callsigns")):
            self.sb.showMessage("DAPNET test: missing username, password or recipients", 6000)
            return

        module_key = str(module_name or "generic").strip().lower()
        emergency = False
        if module_key == "xray":
            text = "[TEST] XRAY START M2.3"
            emergency = bool(cfg.get("xray", {}).get("emergency_on_start", True))
        elif module_key == "proton":
            text = "[TEST] PROTON S2 150PFU Bz -7 Bt 11"
        elif module_key == "solar":
            text = "[TEST] SOLAR SFI 180 SSN 90 Kp 3.0 X C1.2"
        elif module_key == "iss":
            text = "ISS TEST 21:45 MAX 62DEG"
        else:
            text = "SUMO V7.5 DAPNET test message"

        client = DapnetClient(cfg)
        ok, msg = client.send_message(text, emergency=emergency, force=True)
        prefix = f"DAPNET {module_key} test "
        self.sb.showMessage((prefix + "OK: " if ok else prefix + "FAILED: ") + str(msg), 8000)

    def _refresh_dapnet_quick_bar(self):
        try:
            if hasattr(self, "dapnet_quick_bar") and self.dapnet_quick_bar is not None:
                self.dapnet_quick_bar._update_state()
        except Exception:
            pass

    def _clock_city_value(self, attr_name: str, fallback: str) -> str:
        try:
            cb = getattr(self, attr_name)
            data = cb.currentData()
            val = (str(data) if data else fallback).strip()
            return val if val in CLOCK_CITY_MAP.values() else fallback
        except Exception:
            return fallback

    def clock_city_tl(self) -> str:
        return self._clock_city_value("cb_clock_city_tl", "America/New_York")

    def clock_city_tr(self) -> str:
        return self._clock_city_value("cb_clock_city_tr", "Europe/London")

    def clock_city_bl(self) -> str:
        return self._clock_city_value("cb_clock_city_bl", "Asia/Tokyo")

    def clock_city_br(self) -> str:
        return self._clock_city_value("cb_clock_city_br", "Australia/Sydney")

    def clock_hour_chime_mode(self) -> str:
        try:
            data = self.cb_clock_hour_chime_mode.currentData()
            mode = (str(data) if data else "off").strip().lower()
            return mode if mode in ("off", "on") else "off"
        except Exception:
            return "off"

    def alert_volume(self) -> int:
        try:
            return max(0, min(100, int(self.sb_alert_volume.value())))
        except Exception:
            return 90

    def hour_chime_volume(self) -> int:
        try:
            return max(0, min(100, int(self.sb_hour_chime_volume.value())))
        except Exception:
            return 90

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

            <p style="margin-top:10px;"><b>MUF map source</b></p>
            <p>
              When the <b>KC2G server render</b> source is selected in Settings, the MUF map displayed in SUMO
              is loaded from the KC2G servers and remains the property of its respective provider.
              Source website: <a href="https://prop.kc2g.com/">https://prop.kc2g.com/</a>
            </p>

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

        if not isinstance(data, list) or not data:
            return t, vals

        # NOAA current format: list of dicts like
        # {"time_tag": "...", "Kp": 1.33, ...}
        if isinstance(data[0], dict):
            for row in data:
                try:
                    t_epoch = parse_time_to_epoch(row.get("time_tag"))
                    v = float(row.get("Kp"))
                    if math.isnan(t_epoch):
                        continue
                    t.append(t_epoch)
                    vals.append(v)
                except Exception:
                    continue
            return t, vals

        # Legacy NOAA format: header row + indexed rows
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

    def _donki_cme_probability_strict(self) -> tuple[float, str, str]:
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
            return float("nan"), "N/A", ""


        def extract_impact_eta_text(ev: dict) -> str:
            """Best-effort ETA extraction from DONKI CME analysis impact lists."""
            try:
                analyses = ev.get("cmeAnalyses") or []
                for a in analyses:
                    if not isinstance(a, dict):
                        continue
                    impact_list = a.get("impactList") or []
                    if not isinstance(impact_list, list):
                        continue
                    for imp in impact_list:
                        if not isinstance(imp, dict):
                            continue
                        txt = " ".join(str(v).lower() for v in imp.values() if v is not None)
                        if "earth" not in txt:
                            continue
                        for key in ("arrivalTime", "impactTime", "time21_5", "time", "estimatedShockArrivalTime"):
                            val = imp.get(key)
                            if val:
                                return str(val).strip()
            except Exception:
                pass
            return ""


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
        best_eta = ""
        for ev in events[-40:]:
            if isinstance(ev, dict):
                score = score_event(ev)
                if score > best:
                    best = score
                    best_eta = extract_impact_eta_text(ev)

        if best <= 0:
            return 0.0, "NONE", ""
        if best < 40:
            return best, "LOW", best_eta
        if best < 70:
            return best, "MED", best_eta
        return best, "HIGH", best_eta

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

                    payload["xray_x_raw"] = np.array(t_list, dtype=float)
                    payload["xray_series_raw"] = np.array(xr_vals, dtype=float)
                    payload["xray_x"], payload["xray_series"] = smooth_xray_series_for_plot(
                        payload["xray_x_raw"],
                        payload["xray_series_raw"],
                    )
                    payload["xray_now"] = float(payload["xray_series"][-1]) if len(payload["xray_series"]) else float("nan")
                    cls, mag = xray_flare_class(payload["xray_now"])
                    payload["xray_class"] = cls
                    payload["xray_mag"] = mag
                    payload["xray_label"] = xray_class_label(cls, mag)
                    payload["xray_latest_peak_label"] = latest_xray_peak_label(
                        payload["xray_x"],
                        payload["xray_series"],
                    )

                    self._last_good["xray_x_raw"] = payload["xray_x_raw"]
                    self._last_good["xray_series_raw"] = payload["xray_series_raw"]
                    self._last_good["xray_x"] = payload["xray_x"]
                    self._last_good["xray_series"] = payload["xray_series"]
                    self._last_good["xray_now"] = payload["xray_now"]
                    self._last_good["xray_class"] = payload["xray_class"]
                    self._last_good["xray_mag"] = payload["xray_mag"]
                    self._last_good["xray_label"] = payload["xray_label"]
                    self._last_good["xray_latest_peak_label"] = payload["xray_latest_peak_label"]
                except Exception as e:
                    payload["xray_x_raw"] = self._last_good.get("xray_x_raw", np.array([], dtype=float))
                    payload["xray_series_raw"] = self._last_good.get("xray_series_raw", np.array([], dtype=float))
                    payload["xray_x"] = self._last_good.get("xray_x", np.array([], dtype=float))
                    payload["xray_series"] = self._last_good.get("xray_series", np.array([], dtype=float))
                    payload["xray_now"] = self._last_good.get("xray_now", float("nan"))
                    payload["xray_class"] = self._last_good.get("xray_class", "?")
                    payload["xray_mag"] = self._last_good.get("xray_mag", float("nan"))
                    payload["xray_label"] = self._last_good.get("xray_label", "?")
                    payload["xray_latest_peak_label"] = self._last_good.get("xray_latest_peak_label", "?")
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
                    payload["partial_error"] += f" SSN:{e} SFI:{e}"

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
                        prob, level, eta_text = self._donki_cme_probability_strict()
                        payload["cme_prob"] = prob
                        payload["cme_level"] = level
                        payload["cme_eta"] = eta_text

                        ts_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                        if not (isinstance(prob, float) and math.isnan(prob)):
                            db_insert_cme(DB_PATH, ts_utc, prob, level)
                        self._last_donki_fetch = now
                    except Exception as e:
                        payload["partial_error"] += f" CME:{e}"

                self.data_ready.emit(payload)

                # FIX v7.2.2: log RAM usage periodiquement pour detecter les fuites memoire
                try:
                    import psutil
                    rss_mb = psutil.Process().memory_info().rss // (1024 * 1024)
                    debug_log(f"[MEM] RAM usage: {rss_mb} MB")
                except Exception:
                    pass  # psutil optionnel - pas bloquant si absent

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
    - Soleil en haut, émet des CME (croissants) qui descendent et s'élargissent
    - Laser (traits) avec SPACE
    Controls: ←/→ (ou A/D), SPACE, ESC

    Upgrades:
    - Soucoupe volante tous les 200 points jusqu'à 1000 (inclus).
    - Détruire la soucoupe donne +1 laser (cumulable).
    - Les améliorations sont perdues si le vaisseau est touché par une CME
      OU si une CME atteint le bas sans être détruite.

    Endgame:
    - À 1000 points: déclenche "CARRINGTON MODE" pendant 60 secondes (beaucoup plus de CME).
      Un gros warning s'affiche au déclenchement.
    - Si le joueur survit 60 secondes: BIG BOSS.
      Le soleil se transforme en pieuvre géante (avec symbole illuminati) qui:
        * envoie des bébés pieuvres qui rebondissent,
        * tire aussi des projectiles vers le joueur.
      Victoire après 50 tirs sur la pieuvre géante.
    """
    # --- constants (kept local to this class) ---
    _CARRINGTON_DURATION_S = 60.0
    _CARRINGTON_BANNER_S = 3.2
    _BOSS_HP = 50

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SUMO Defense (B/W)")
        self.setModal(True)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        self.setFixedSize(900, 600)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        self._w = self.width()
        self._h = self.height()

        self._rng = np.random.default_rng()

        # Ship
        self._ship_speed = 7
        self._ship_x = self._w // 2
        self._ship_y = self._h - 70
        self._left = False
        self._right = False

        # Entities
        self._shots = []        # {"x","y"}
        self._cmes = []         # {"x","y","vy","w","h","grow"}
        self._score = 0
        self._lives = 3
        self._cooldown = 0

        self._frame = 0
        self._spawn_every = 110  # base

        # Upgrades / UFO
        self._laser_level = 1
        self._next_upgrade_score = 200
        self._ufo = None          # {"x","y","vx","vy","r"}
        self._ufo_pending = False

        # Carrington / Boss state
        self._phase = "normal"  # normal | carrington | boss | win
        self._carrington_started = False
        self._carrington_start_t = None
        self._banner_until_t = None

        # Boss
        self._boss_active = False
        self._boss_hp = self._BOSS_HP
        self._boss_hits = 0
        self._boss_x = self._w // 2
        self._boss_y = 70
        self._boss_r = 46.0
        self._boss_shots = []    # projectiles fired by boss: {"x","y","vx","vy","r"}
        self._boss_fire_cd = 0
        self._babies = []        # bouncing baby octos: {"x","y","vx","vy","r"}
        self._baby_spawn_cd = 0

        # Load logo and make it white silhouette
        pm = QtGui.QPixmap(str(ASSETS_DIR / "logo.png"))
        if pm.isNull():
            pm = QtGui.QPixmap(64, 64)
            pm.fill(QtGui.QColor("white"))
        pm = pm.scaled(64, 64, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self._ship_pix = self._to_white_pixmap(pm)

        # Loop (~60 fps)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # -------------------------
    # Input
    # -------------------------
    def keyPressEvent(self, e: QtGui.QKeyEvent):
        if self._phase == "win" or self._lives <= 0:
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
        elif e.key() == QtCore.Qt.Key_F9:
            # DEBUG: force Carrington Mode immediately
            if self._phase not in ("carrington", "boss", "win"):
                self._score = max(self._score, 1000)
                if not getattr(self, "_carrington_started", False):
                    self._start_carrington(time.monotonic())
        elif e.key() == QtCore.Qt.Key_F10:
            # DEBUG: jump directly to the boss
            if self._phase != "boss" and self._phase != "win":
                self._score = max(self._score, 1000)
                self._start_boss()
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e: QtGui.QKeyEvent):
        if e.key() in (QtCore.Qt.Key_Left, QtCore.Qt.Key_A):
            self._left = False
        elif e.key() in (QtCore.Qt.Key_Right, QtCore.Qt.Key_D):
            self._right = False
        super().keyReleaseEvent(e)

    # -------------------------
    # Helpers
    # -------------------------
    def _to_white_pixmap(self, pm: QtGui.QPixmap) -> QtGui.QPixmap:
        img = pm.toImage().convertToFormat(QtGui.QImage.Format_ARGB32)
        w, h = img.width(), img.height()
        for y in range(h):
            for x in range(w):
                c = QtGui.QColor(img.pixelColor(x, y))
                if c.alpha() > 0:
                    img.setPixelColor(x, y, QtGui.QColor(255, 255, 255, c.alpha()))
        return QtGui.QPixmap.fromImage(img)

    def _restart_game(self):
        self._score = 0
        self._lives = 3
        self._frame = 0
        self._cooldown = 0
        self._left = False
        self._right = False
        self._shots.clear()
        self._cmes.clear()
        self._ship_x = self._w // 2

        self._laser_level = 1
        self._next_upgrade_score = 200
        self._ufo = None
        self._ufo_pending = False

        self._phase = "normal"
        self._carrington_started = False
        self._carrington_start_t = None
        self._banner_until_t = None

        self._boss_active = False
        self._boss_hp = self._BOSS_HP
        self._boss_hits = 0
        self._boss_shots.clear()
        self._babies.clear()
        self._boss_fire_cd = 0
        self._baby_spawn_cd = 0

        if not self._timer.isActive():
            self._timer.start()
        self.update()

    # -------------------------
    # Weapons
    # -------------------------
    def _fire(self):
        if self._cooldown > 0:
            return
        self._cooldown = 10

        n = max(1, int(self._laser_level))
        spacing = 12.0
        center = float(self._ship_x)
        start = -((n - 1) / 2.0) * spacing
        y0 = float(self._ship_y - 18)
        for i in range(n):
            x = center + (start + i * spacing)
            self._shots.append({"x": float(x), "y": float(y0)})

    # -------------------------
    # CME (crescent)
    # -------------------------
    def _spawn_cme(self):
        x = float(self._rng.integers(60, self._w - 60))
        h = float(self._rng.integers(18, 26))   # thickness
        w0 = float(self._rng.integers(22, 34))  # initial width
        vy = float(self._rng.uniform(0.6, 1.15))
        grow = float(self._rng.uniform(0.10, 0.22))
        self._cmes.append({"x": x, "y": 95.0, "vy": vy, "w": w0, "h": h, "grow": grow})

    def _draw_cme_crescent(self, p: QtGui.QPainter, x: float, y: float, w: float, h: float):
        """Croissant horizontal, ouvert vers le haut."""
        outer = QtGui.QPainterPath()
        inner = QtGui.QPainterPath()

        rect = QtCore.QRectF(x - w / 2.0, y - h / 2.0, w, h)
        outer.addEllipse(rect)

        inset_w = w * 0.78
        inset_h = h * 0.78
        dy = -h * 0.22  # up => opening up
        inner_rect = QtCore.QRectF(
            x - inset_w / 2.0,
            (y - inset_h / 2.0) + dy,
            inset_w,
            inset_h
        )
        inner.addEllipse(inner_rect)
        crescent = outer.subtracted(inner)
        p.drawPath(crescent)

    # -------------------------
    # UFO (upgrade)
    # -------------------------
    def _spawn_ufo(self):
        x = float(self._rng.integers(80, self._w - 80))
        y = float(self._rng.uniform(105.0, 160.0))
        r = 12.0
        vx = float(self._rng.choice([-1, 1]) * self._rng.uniform(1.6, 2.4))
        vy = float(self._rng.uniform(0.25, 0.45))
        self._ufo = {"x": x, "y": y, "vx": vx, "vy": vy, "r": r}
        self._ufo_pending = True

    def _award_ufo_upgrade(self):
        self._laser_level = int(self._laser_level) + 1
        self._ufo = None
        self._ufo_pending = False
        # upgrades every 200 points, capped to 1000
        self._next_upgrade_score = min(self._next_upgrade_score + 200, 1001)

    def _miss_ufo_upgrade(self):
        self._ufo = None
        self._ufo_pending = False
        self._next_upgrade_score = min(self._next_upgrade_score + 200, 1001)

    def _draw_ufo(self, p: QtGui.QPainter, x: float, y: float, r: float):
        p.drawArc(QtCore.QRectF(x - r * 0.7, y - r * 0.9, r * 1.4, r * 1.0), 0, 180 * 16)
        p.drawRoundedRect(QtCore.QRectF(x - r, y - r * 0.2, r * 2.0, r * 0.8), 6, 6)
        for i in (-0.6, 0.0, 0.6):
            p.drawEllipse(QtCore.QPointF(x + i * r, y + r * 0.25), 1.6, 1.6)

    # -------------------------
    # Carrington / Boss
    # -------------------------
    def _start_carrington(self, now_t: float):
        self._phase = "carrington"
        self._carrington_started = True
        self._carrington_start_t = now_t
        self._banner_until_t = now_t + self._CARRINGTON_BANNER_S

        # optional: wipe UFO opportunity at this point
        self._ufo = None
        self._ufo_pending = False
        self._next_upgrade_score = 1001  # hard lock upgrades after 1000

    def _start_boss(self):
        self._phase = "boss"
        self._boss_active = True
        self._boss_hp = self._BOSS_HP
        self._boss_hits = 0
        self._boss_shots.clear()
        self._babies.clear()
        self._boss_fire_cd = 0
        self._baby_spawn_cd = 0

        # Clear remaining CMEs to make the boss phase readable
        self._cmes.clear()
        self._ufo = None
        self._ufo_pending = False

    def _boss_fire(self):
        # Fire a projectile roughly aimed at the ship
        sx, sy = float(self._ship_x), float(self._ship_y)
        bx, by = float(self._boss_x), float(self._boss_y + 18)
        dx = sx - bx
        dy = sy - by
        dist = math.hypot(dx, dy) or 1.0
        speed = 3.4
        vx = (dx / dist) * speed
        vy = (dy / dist) * speed
        self._boss_shots.append({"x": bx, "y": by, "vx": vx, "vy": vy, "r": 5.0})

    def _spawn_baby(self):
        # Spawn a bouncing baby octopus somewhere under the boss
        x = float(self._boss_x + self._rng.uniform(-60, 60))
        y = float(self._boss_y + self._rng.uniform(40, 70))
        r = float(self._rng.uniform(8, 12))
        vx = float(self._rng.choice([-1, 1]) * self._rng.uniform(2.0, 3.4))
        vy = float(self._rng.choice([-1, 1]) * self._rng.uniform(1.8, 3.2))
        self._babies.append({"x": x, "y": y, "vx": vx, "vy": vy, "r": r})

    def _draw_illuminati(self, p: QtGui.QPainter, cx: float, cy: float, size: float, t: float):
        """Triangle + œil stylisé, léger 'glow' vivant."""
        tri = QtGui.QPolygonF([
            QtCore.QPointF(cx, cy - size * 0.65),
            QtCore.QPointF(cx - size * 0.60, cy + size * 0.45),
            QtCore.QPointF(cx + size * 0.60, cy + size * 0.45),
        ])
        p.drawPolygon(tri)

        eye_w = size * 0.65
        eye_h = size * 0.22
        blink = 0.70 + 0.30 * math.sin(t * 2.2)
        p.drawEllipse(QtCore.QRectF(cx - eye_w / 2.0, cy - (eye_h * blink) / 2.0, eye_w, eye_h * blink))
        p.drawEllipse(QtCore.QPointF(cx, cy), size * 0.06, size * 0.06)

    def _draw_boss_octopus(self, p: QtGui.QPainter, x: float, y: float, r: float, t: float):
        """Pieuvre géante avec tentacules animées + Illuminati."""
        # Tentacules (derrière le corps)
        n = 8
        base_y = y + r * 0.35
        for i in range(n):
            a = (-0.95 + i * (1.90 / (n - 1)))  # éventail
            base_x = x + math.sin(a) * r * 0.65

            segs = 12
            length = r * 1.45
            amp = r * 0.22
            phase = t * 2.4 + i * 0.55

            path = QtGui.QPainterPath(QtCore.QPointF(base_x, base_y))
            for s in range(1, segs + 1):
                u = s / segs
                px = base_x + math.sin(a) * (u * r * 0.35)
                py = base_y + u * length

                wiggle = math.sin(phase + u * 3.4) * amp * (0.20 + 0.80 * u)
                px += wiggle

                path.lineTo(QtCore.QPointF(px, py))

            p.drawPath(path)

            # ventouses (petites)
            for s in (segs - 3, segs - 2, segs - 1, segs):
                u = s / segs
                px = base_x + math.sin(a) * (u * r * 0.35)
                py = base_y + u * length
                wiggle = math.sin(phase + u * 3.4) * amp * (0.20 + 0.80 * u)
                px += wiggle
                p.drawEllipse(QtCore.QPointF(px, py), 1.6, 1.6)

        # Corps principal
        p.drawEllipse(QtCore.QPointF(x, y), r, r * 0.92)
        # Dôme (tête)
        p.drawArc(QtCore.QRectF(x - r * 0.95, y - r * 1.10, r * 1.90, r * 1.25), 0, 180 * 16)
        # Jupe (bord ondulé léger)
        p.drawRoundedRect(QtCore.QRectF(x - r * 0.88, y + r * 0.15, r * 1.76, r * 0.58), 12, 12)

        # Visage
        eye_dx = r * 0.32
        eye_y = y - r * 0.06
        p.drawEllipse(QtCore.QPointF(x - eye_dx, eye_y), r * 0.11, r * 0.13)
        p.drawEllipse(QtCore.QPointF(x + eye_dx, eye_y), r * 0.11, r * 0.13)
        pup = r * 0.045
        p.drawEllipse(QtCore.QPointF(x - eye_dx + math.sin(t * 1.8) * r * 0.03, eye_y), pup, pup)
        p.drawEllipse(QtCore.QPointF(x + eye_dx + math.sin(t * 1.8 + 1.1) * r * 0.03, eye_y), pup, pup)
        p.drawArc(QtCore.QRectF(x - r * 0.22, y + r * 0.10, r * 0.44, r * 0.26), 200 * 16, 140 * 16)

        # Illuminati sur le front
        self._draw_illuminati(p, x, y - r * 0.56, r * 0.55, t)

    def _draw_boss(self, p: QtGui.QPainter):
        x = float(self._boss_x)
        y = float(self._boss_y)
        r = float(self._boss_r)
        t = float(self._frame) / 60.0
        self._draw_boss_octopus(p, x, y, r, t)



    def _tentacle_path(self, x0: float, y0: float, length: float, amp: float) -> QtGui.QPainterPath:
        path = QtGui.QPainterPath()
        path.moveTo(x0, y0)
        steps = 6
        for i in range(1, steps + 1):
            t = i / steps
            x = x0 + math.sin(t * math.pi * 2.0) * amp * (1.0 - t * 0.2)
            y = y0 + t * length
            path.lineTo(x, y)
        return path

    def _draw_baby(self, p: QtGui.QPainter, b):
        x, y, r = b["x"], b["y"], b["r"]
        p.drawEllipse(QtCore.QPointF(x, y), r, r * 0.9)
        # tiny tentacles
        for i in (-1, 0, 1):
            p.drawLine(int(x + i * r * 0.5), int(y + r * 0.6),
                       int(x + i * r * 0.5), int(y + r * 1.5))

    # -------------------------
    # Main tick
    # -------------------------
    def _tick(self):
        self._frame += 1
        now_t = time.monotonic()

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

        # Phase transitions
        if (not self._carrington_started) and (self._score >= 1000):
            self._start_carrington(now_t)

        if self._phase == "carrington":
            if (now_t - (self._carrington_start_t or now_t)) >= self._CARRINGTON_DURATION_S:
                self._start_boss()

        # Spawning logic (CME)
        if self._phase in ("normal", "carrington"):
            if self._phase == "carrington":
                # High spawn rate + occasional bursts
                if (self._frame % 22) == 0:
                    self._spawn_cme()
                if (self._frame % 90) == 0:
                    for _ in range(3):
                        self._spawn_cme()
            else:
                if (self._frame % self._spawn_every) == 0:
                    self._spawn_cme()

        # Move CME + widen
        for c in self._cmes:
            c["y"] += c["vy"] + min(0.25, self._score * 0.00025)
            c["w"] += c["grow"] + min(0.06, self._score * 0.00002)
            c["h"] += 0.01

        # UFO spawn (only in normal phase, and only <= 1000)
        if self._phase == "normal":
            if (
                self._score >= self._next_upgrade_score
                and self._next_upgrade_score <= 1000
                and self._ufo is None
                and not self._ufo_pending
            ):
                self._spawn_ufo()

        # Move UFO
        if self._ufo is not None:
            self._ufo["x"] += self._ufo["vx"]
            self._ufo["y"] += self._ufo["vy"]
            if self._ufo["x"] < 35:
                self._ufo["x"] = 35
                self._ufo["vx"] = abs(self._ufo["vx"])
            elif self._ufo["x"] > (self._w - 35):
                self._ufo["x"] = float(self._w - 35)
                self._ufo["vx"] = -abs(self._ufo["vx"])
            if self._ufo["y"] > (self._h - 140):
                self._miss_ufo_upgrade()

        # Boss phase updates
        if self._phase == "boss" and self._boss_active:
            # boss firing cadence
            if self._boss_fire_cd > 0:
                self._boss_fire_cd -= 1
            else:
                self._boss_fire_cd = 42  # roughly every 0.7s
                self._boss_fire()

            # spawn bouncing babies
            if self._baby_spawn_cd > 0:
                self._baby_spawn_cd -= 1
            else:
                self._baby_spawn_cd = 55
                self._spawn_baby()

            # move boss projectiles
            new_bs = []
            for pr in self._boss_shots:
                pr["x"] += pr["vx"]
                pr["y"] += pr["vy"]
                if -20 <= pr["x"] <= self._w + 20 and -20 <= pr["y"] <= self._h + 20:
                    new_bs.append(pr)
            self._boss_shots = new_bs

            # move babies with bounce
            for b in self._babies:
                b["x"] += b["vx"]
                b["y"] += b["vy"]
                r = b["r"]
                if b["x"] < r:
                    b["x"] = r
                    b["vx"] = abs(b["vx"])
                elif b["x"] > (self._w - r):
                    b["x"] = self._w - r
                    b["vx"] = -abs(b["vx"])
                if b["y"] < 90 + r:
                    b["y"] = 90 + r
                    b["vy"] = abs(b["vy"])
                elif b["y"] > (self._h - 40 - r):
                    b["y"] = self._h - 40 - r
                    b["vy"] = -abs(b["vy"])

        # -------------------------
        # Collisions: shots with UFO, CME, babies, boss
        # -------------------------
        # Shots <-> UFO
        if self._ufo is not None:
            ux, uy, ur = self._ufo["x"], self._ufo["y"], self._ufo["r"]
            ufo_hit = False
            for s in self._shots:
                dx = s["x"] - ux
                dy = s["y"] - uy
                if (dx * dx + dy * dy) <= (ur * ur):
                    s["y"] = -9999
                    ufo_hit = True
                    break
            if ufo_hit:
                self._award_ufo_upgrade()

        # Shots <-> babies
        if self._phase == "boss":
            survivors = []
            for b in self._babies:
                hit = False
                bx, by, br = b["x"], b["y"], b["r"]
                for s in self._shots:
                    dx = s["x"] - bx
                    dy = s["y"] - by
                    if (dx * dx + dy * dy) <= (br * br):
                        hit = True
                        s["y"] = -9999
                        self._score += 5
                        break
                if not hit:
                    survivors.append(b)
            self._babies = survivors

        # Shots <-> boss
        if self._phase == "boss" and self._boss_active and self._boss_hp > 0:
            bx, by, br = float(self._boss_x), float(self._boss_y), float(self._boss_r * 0.75)
            boss_hit = 0
            for s in self._shots:
                dx = s["x"] - bx
                dy = s["y"] - by
                if (dx * dx + dy * dy) <= (br * br):
                    s["y"] = -9999
                    boss_hit += 1
            if boss_hit:
                self._boss_hp = max(0, self._boss_hp - boss_hit)
                self._boss_hits += boss_hit
                if self._boss_hp <= 0:
                    self._phase = "win"
                    self._timer.stop()

        # Shots <-> CME
        new_cmes = []
        for c in self._cmes:
            hit = False
            cx, cy = c["x"], c["y"]
            cr = max(c["w"], c["h"]) * 0.50
            for s in self._shots:
                dx = s["x"] - cx
                dy = s["y"] - cy
                if (dx * dx + dy * dy) <= (cr * cr):
                    hit = True
                    s["y"] = -9999
                    self._score += 10
                    break
            if not hit:
                new_cmes.append(c)
        self._cmes = new_cmes
        self._shots = [s for s in self._shots if s["y"] > 0]

        # -------------------------
        # Collisions: ship with CME / babies / boss projectiles
        # -------------------------
        ship_r = max(18.0, self._ship_pix.width() * 0.32)
        ship_cx = float(self._ship_x)
        ship_cy = float(self._ship_y)

        # ship <-> CME => lose life + reset upgrades
        survived = []
        touched = False
        for c in self._cmes:
            dx = ship_cx - c["x"]
            dy = ship_cy - c["y"]
            cr = max(c["w"], c["h"]) * 0.50
            if (dx * dx + dy * dy) <= ((ship_r + cr) * (ship_r + cr)):
                touched = True
                continue
            survived.append(c)
        self._cmes = survived
        if touched:
            self._lives -= 1
            self._laser_level = 1

        # ship <-> babies
        if self._phase == "boss":
            kept = []
            touched_baby = False
            for b in self._babies:
                dx = ship_cx - b["x"]
                dy = ship_cy - b["y"]
                if (dx * dx + dy * dy) <= ((ship_r + b["r"]) ** 2):
                    touched_baby = True
                    continue
                kept.append(b)
            self._babies = kept
            if touched_baby:
                self._lives -= 1
                self._laser_level = 1

        # ship <-> boss projectiles
        if self._phase == "boss":
            kept = []
            hit_pr = False
            for pr in self._boss_shots:
                dx = ship_cx - pr["x"]
                dy = ship_cy - pr["y"]
                if (dx * dx + dy * dy) <= ((ship_r + pr["r"]) ** 2):
                    hit_pr = True
                    continue
                kept.append(pr)
            self._boss_shots = kept
            if hit_pr:
                self._lives -= 1
                self._laser_level = 1

        # CME reaches bottom => lose life AND lose upgrades
        still = []
        for c in self._cmes:
            if c["y"] > (self._h - 40):
                self._lives -= 1
                self._laser_level = 1
            else:
                still.append(c)
        self._cmes = still

        if self._lives <= 0:
            self._timer.stop()

        self.update()

    # -------------------------
    # Rendering
    # -------------------------
    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        p.fillRect(self.rect(), QtGui.QColor("black"))

        pen = QtGui.QPen(QtGui.QColor("white"))
        pen.setWidth(2)
        p.setPen(pen)
        p.setBrush(QtCore.Qt.NoBrush)

        # Top entity: Sun or Boss
        if self._phase == "boss":
            self._draw_boss(p)
        else:
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
        p.drawText(14, 52, f"LASERS x{int(self._laser_level)}")

        if self._phase == "carrington":
            p.drawText(self._w - 260, 52, "CARRINGTON")

        # CME
        for c in self._cmes:
            self._draw_cme_crescent(p, c["x"], c["y"], c["w"], c["h"])

        # UFO
        if self._ufo is not None:
            self._draw_ufo(p, self._ufo["x"], self._ufo["y"], self._ufo["r"])

        # Boss projectiles / babies
        if self._phase == "boss":
            for pr in self._boss_shots:
                p.drawEllipse(QtCore.QPointF(pr["x"], pr["y"]), pr["r"], pr["r"])
            for b in self._babies:
                self._draw_baby(p, b)

        # Shots
        for s in self._shots:
            p.drawLine(int(s["x"]), int(s["y"]), int(s["x"]), int(s["y"] - 14))

        # Ship
        ship_left = int(self._ship_x - self._ship_pix.width() / 2)
        ship_top = int(self._ship_y - self._ship_pix.height() / 2)
        p.drawPixmap(ship_left, ship_top, self._ship_pix)

        # Carrington banner
        if self._banner_until_t is not None and time.monotonic() <= self._banner_until_t:
            p.save()
            p.setFont(QtGui.QFont("Consolas", 38, QtGui.QFont.Bold))
            p.drawText(self.rect(), QtCore.Qt.AlignCenter, "⚠  CARRINGTON MODE  ⚠")
            p.setFont(QtGui.QFont("Consolas", 18, QtGui.QFont.Bold))
            p.drawText(0, self._h // 2 + 54, self._w, 40, QtCore.Qt.AlignCenter, "EXTREME SOLAR STORM")
            # Big warning triangle
            tri = QtGui.QPainterPath()
            cx = self._w / 2.0
            cy = self._h / 2.0 - 120
            tri.moveTo(cx, cy - 50)
            tri.lineTo(cx - 60, cy + 55)
            tri.lineTo(cx + 60, cy + 55)
            tri.closeSubpath()
            p.drawPath(tri)
            p.restore()

        # End screens
        if self._lives <= 0:
            p.setFont(QtGui.QFont("Consolas", 30, QtGui.QFont.Bold))
            p.drawText(self.rect(), QtCore.Qt.AlignCenter, "GAME OVER\n\nR to restart  |  ESC to quit")

        if self._phase == "win":
            p.setFont(QtGui.QFont("Consolas", 30, QtGui.QFont.Bold))
            p.drawText(self.rect(), QtCore.Qt.AlignCenter, "YOU WIN!\n\nR to restart  |  ESC to quit")

        p.end()
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
        # Tooltip will be updated on each refresh
        self.lbl_muf.setToolTip(
            "MUF estimate (MHz).\n"
            "Refresh: every 60 s (main data loop).\n"
            "Trend: EMA-smoothed Δ over the last 30 minutes."
        )
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

        # --- MUF trend over last 30 minutes (smoothed, robust to missing data) ---
        now = time.time()

        # Exponential moving average (EMA) for smoothing
        muf_s = None
        try:
            if isinstance(muf, (int, float)) and math.isfinite(float(muf)):
                m = float(muf)
                prev = getattr(self, "_muf_ema", float("nan"))
                if not (isinstance(prev, float) and math.isfinite(prev)):
                    ema = m
                else:
                    alpha = 0.20  # smoothing factor (higher = more reactive)
                    ema = prev + alpha * (m - prev)
                self._muf_ema = ema
                muf_s = ema
        except Exception:
            muf_s = None

        # Keep a 30-minute history of smoothed MUF values
        if muf_s is not None:
            self._muf_hist.append((now, muf_s))

        cutoff = now - 1800  # 30 minutes
        self._muf_hist = [
            (t, v) for (t, v) in self._muf_hist
            if t >= cutoff and math.isfinite(float(v))
        ]

        muf_delta = 0.0
        trend = "→"
        if len(self._muf_hist) >= 2:
            v0 = float(self._muf_hist[0][1])
            v1 = float(self._muf_hist[-1][1])
            muf_delta = v1 - v0
            if muf_delta > 0.5:
                trend = "▲"
            elif muf_delta < -0.5:
                trend = "▼"

        muf_color, muf_text = _muf_color(muf)
        delta_txt = f"{trend} {muf_delta:+.1f} MHz"
        self.lbl_muf.setText(f"MUF: {muf_text}  {delta_txt}")
        self.lbl_muf.setStyleSheet(self._badge_style(muf_color))

        # Update MUF tooltip with current trend details
        try:
            self.lbl_muf.setToolTip(
                "MUF estimate (MHz), derived from SFI and Kp.\n"
                "Refresh: every 60 s (main data loop).\n"
                "Trend: EMA smoothing (α=0.20) over a 30-minute window.\n"
                f"Current: {muf_text} MHz\n"
                f"Δ30m (smoothed): {muf_delta:+.1f} MHz {trend}"
            )
        except Exception:
            pass

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



class DapnetQuickSendBar(QtWidgets.QFrame):
    """Compact quick DAPNET sender embedded in the bottom radio bar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("dapnetQuickSendBar")

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(6)

        self.lbl_title = QtWidgets.QLabel("DAPNET")
        self.lbl_title.setStyleSheet("color:#4aa3ff; font-size:12px; font-weight:900;")
        self.lbl_title.setMinimumWidth(62)
        lay.addWidget(self.lbl_title, 0)

        self.lbl_group = QtWidgets.QLabel("Group:")
        self.lbl_group.setStyleSheet("color:#aab6c5; font-size:11px; font-weight:800;")
        self.lbl_group.setMinimumWidth(42)
        lay.addWidget(self.lbl_group, 0)

        self.cb_tx_group = QtWidgets.QComboBox()
        self.cb_tx_group.addItems(["f-53", "all"])
        self.cb_tx_group.setMinimumWidth(76)
        self.cb_tx_group.setMaximumWidth(84)
        self.cb_tx_group.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.cb_tx_group.setStyleSheet("""
            QComboBox {
                background:#0f1720;
                color:#d7dde6;
                border:1px solid #2a3440;
                border-radius:6px;
                padding:3px 6px;
                font-size:11px;
                font-weight:800;
            }
        """)
        lay.addWidget(self.cb_tx_group, 0)

        self.lbl_to = QtWidgets.QLabel("To:")
        self.lbl_to.setStyleSheet("color:#aab6c5; font-size:11px; font-weight:800;")
        self.lbl_to.setMinimumWidth(18)
        lay.addWidget(self.lbl_to, 0)

        self.ed_to = QtWidgets.QLineEdit()
        self.ed_to.setPlaceholderText("F4IGV")
        self.ed_to.setClearButtonEnabled(True)
        self.ed_to.setMinimumWidth(92)
        self.ed_to.setMaximumWidth(116)
        self.ed_to.setStyleSheet("""
            QLineEdit {
                background:#0f1720;
                color:#d7dde6;
                border:1px solid #2a3440;
                border-radius:6px;
                padding:4px 6px;
                font-size:12px;
                font-weight:800;
            }
        """)
        lay.addWidget(self.ed_to, 0)

        self.ed_msg = QtWidgets.QLineEdit()
        self.ed_msg.setPlaceholderText("message (max 80)")
        self.ed_msg.setMaxLength(80)
        self.ed_msg.setClearButtonEnabled(True)
        self.ed_msg.setMinimumWidth(180)
        self.ed_msg.setStyleSheet("""
            QLineEdit {
                background:#0f1720;
                color:#d7dde6;
                border:1px solid #2a3440;
                border-radius:6px;
                padding:4px 6px;
                font-size:12px;
                font-weight:700;
            }
        """)
        lay.addWidget(self.ed_msg, 1)

        self.cb_emergency = QtWidgets.QCheckBox("EMERGENCY")
        self.cb_emergency.setStyleSheet("""
            QCheckBox {
                color:#ff6b6b;
                font-size:11px;
                font-weight:900;
            }
            QCheckBox::indicator {
                width:14px;
                height:14px;
                border:2px solid #ff6b6b;
                border-radius:3px;
                background:#0f1720;
            }
            QCheckBox::indicator:checked {
                background:#ff6b6b;
            }
        """)
        lay.addWidget(self.cb_emergency, 0)

        self.lbl_count = QtWidgets.QLabel("0/80")
        self.lbl_count.setStyleSheet("color:#aab6c5; font-size:11px; font-weight:800;")
        self.lbl_count.setMinimumWidth(34)
        self.lbl_count.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        lay.addWidget(self.lbl_count, 0)

        self.btn_send = QtWidgets.QPushButton("SEND")
        self.btn_send.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btn_send.setMinimumWidth(68)
        self.btn_send.setStyleSheet("""
            QPushButton {
                background:#4aa3ff;
                color:#0b0f12;
                border:0px;
                border-radius:6px;
                padding:5px 10px;
                font-size:11px;
                font-weight:900;
            }
            QPushButton:disabled {
                background:#2a3440;
                color:#7f8c8d;
            }
        """)
        lay.addWidget(self.btn_send, 0)

        self.lbl_status = QtWidgets.QLabel("")
        self.lbl_status.setMinimumWidth(58)
        self.lbl_status.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.lbl_status.setStyleSheet("color:#aab6c5; font-size:11px; font-weight:800;")
        lay.addWidget(self.lbl_status, 0)

        self.setStyleSheet("""
            QFrame#dapnetQuickSendBar {
                background:#0e141a;
                border:1px solid #2a3440;
                border-radius:10px;
            }
        """)

        self.ed_to.textChanged.connect(self._update_state)
        self.ed_msg.textChanged.connect(self._update_state)
        self.cb_tx_group.currentIndexChanged.connect(self._update_state)
        self.btn_send.clicked.connect(self._on_send_clicked)
        self.ed_msg.returnPressed.connect(self._on_send_clicked)
        self.ed_to.returnPressed.connect(self._focus_message_or_send)

        self._update_state()

    def _focus_message_or_send(self):
        if str(self.ed_msg.text() or "").strip():
            self._on_send_clicked()
        else:
            self.ed_msg.setFocus()

    def _mainwindow(self):
        try:
            p = self.parent()
            while p is not None:
                if isinstance(p, QtWidgets.QMainWindow):
                    return p
                p = p.parent()
            w = self.window()
            return w if isinstance(w, QtWidgets.QMainWindow) else None
        except Exception:
            return None

    def _config_ready(self) -> bool:
        mw = self._mainwindow()
        try:
            cfg = normalize_dapnet_config(getattr(mw, "_dapnet_cfg", {}))
            return bool(cfg.get("enabled")) and bool(cfg.get("username")) and bool(cfg.get("password"))
        except Exception:
            return False

    def set_compact(self, compact: bool):
        try:
            if compact:
                self.ed_msg.setMaximumWidth(360)
            else:
                self.ed_msg.setMaximumWidth(16777215)
        except Exception:
            pass

    def set_status_text(self, text: str, ok: bool | None = None):
        self.lbl_status.setText(str(text or ""))
        if ok is True:
            self.lbl_status.setStyleSheet("color:#44d16e; font-size:11px; font-weight:900;")
        elif ok is False:
            self.lbl_status.setStyleSheet("color:#ff4d4d; font-size:11px; font-weight:900;")
        else:
            self.lbl_status.setStyleSheet("color:#aab6c5; font-size:11px; font-weight:800;")

    def _update_state(self):
        try:
            count = len(str(self.ed_msg.text() or ""))
            self.lbl_count.setText(f"{count}/80")
            ready = bool(str(self.ed_to.text() or "").strip()) and bool(str(self.ed_msg.text() or "").strip()) and self._config_ready()
            self.btn_send.setEnabled(ready)
            if not self._config_ready():
                self.set_status_text("OFF", ok=False)
            elif not str(self.ed_to.text() or "").strip() or not str(self.ed_msg.text() or "").strip():
                self.set_status_text("", ok=None)
            else:
                self.set_status_text("READY", ok=True)
        except Exception:
            pass

    def _on_send_clicked(self):
        mw = self._mainwindow()
        try:
            debug_log(f"QUICKBAR click send mw={type(mw).__name__ if mw is not None else 'None'}")
        except Exception:
            pass
        if mw is None:
            self.set_status_text("NO MW", ok=False)
            return
        callsign = str(self.ed_to.text() or "").strip().upper()
        message = str(self.ed_msg.text() or "").strip()[:80]
        tx_group = str(self.cb_tx_group.currentText() or "").strip() or "f-53"
        emergency = bool(self.cb_emergency.isChecked())
        try:
            debug_log(f"QUICKBAR payload to={callsign} len={len(message)} tx_group={tx_group} emergency={emergency}")
        except Exception:
            pass
        if not callsign or not message:
            self._update_state()
            return
        if not hasattr(mw, "_send_dapnet_quick_message"):
            try:
                debug_log("QUICKBAR missing _send_dapnet_quick_message on MainWindow")
            except Exception:
                pass
            self.set_status_text("ERR", ok=False)
            return
        ok, resp = mw._send_dapnet_quick_message(callsign, message, tx_group=tx_group, emergency=emergency)
        if ok:
            self.set_status_text("SENT", ok=True)
            self.ed_msg.clear()
            self.cb_emergency.setChecked(False)  # Reset emergency checkbox after successful send
        else:
            self.set_status_text("FAIL", ok=False)
        try:
            debug_log(f"QUICKBAR result ok={ok} resp={resp}")
        except Exception:
            pass
        self._update_state()

class RadioBottomBar(QtWidgets.QFrame):
    """Bottom radio bar: HF at left + optional DAPNET quick send."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("radioBottomBar")
        self._hf_visible = True
        self._quick_send_enabled = True

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        self.hf_bar = HfBandBar()
        self.quick_send = DapnetQuickSendBar()

        lay.addWidget(self.hf_bar, 3)
        lay.addWidget(self.quick_send, 2)

        self.setStyleSheet("QFrame#radioBottomBar { background: transparent; border: 0px; }")
        self._sync_layout()

    def _sync_layout(self):
        lay = self.layout()
        self.hf_bar.setVisible(bool(self._hf_visible))
        self.quick_send.setVisible(bool(self._quick_send_enabled))
        self.quick_send.set_compact(bool(self._hf_visible))
        if self._hf_visible and self._quick_send_enabled:
            lay.setStretch(0, 3)
            lay.setStretch(1, 2)
            self.setVisible(True)
        elif self._hf_visible and not self._quick_send_enabled:
            lay.setStretch(0, 1)
            lay.setStretch(1, 0)
            self.setVisible(True)
        elif (not self._hf_visible) and self._quick_send_enabled:
            lay.setStretch(0, 0)
            lay.setStretch(1, 1)
            self.setVisible(True)
        else:
            lay.setStretch(0, 0)
            lay.setStretch(1, 0)
            self.setVisible(False)

    def set_hf_visible(self, visible: bool):
        self._hf_visible = bool(visible)
        self._sync_layout()

    def set_quick_send_enabled(self, enabled: bool):
        self._quick_send_enabled = bool(enabled)
        self._sync_layout()


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
            s.setVolume(max(0.0, min(1.0, float(getattr(self, "_alert_volume", 0.9)))))
            self._startup_sfx = s  # keep reference
            QtCore.QTimer.singleShot(250, s.play)
        except Exception:
            pass


    def _dbg(self, message: str, hardcore_only: bool = False) -> None:
        try:
            if not getattr(self, "_debug_mode", SUMO_DEFAULT_DEBUG_MODE):
                return
            if hardcore_only and not getattr(self, "_debug_hardcore", SUMO_DEFAULT_DEBUG_HARDCORE):
                return
            debug_log(f"[MainWindow] {message}")
            runtime_log(f"[MainWindow] {message}")
        except Exception:
            pass

    def _register_thread(self, name: str, thread, worker=None) -> None:
        try:
            self._registered_threads[str(name)] = {
                "thread": thread,
                "worker": worker,
                "started_monotonic": time.monotonic(),
            }
            self._dbg(f"register_thread {name}", hardcore_only=True)
        except Exception as exc:
            debug_log(f"register_thread error for {name}: {exc}")

    def _unregister_thread(self, name: str) -> None:
        try:
            self._registered_threads.pop(str(name), None)
            self._dbg(f"unregister_thread {name}", hardcore_only=True)
        except Exception as exc:
            debug_log(f"unregister_thread error for {name}: {exc}")

    def _safe_stop_worker_thread(self, name: str, worker_attr: str, thread_attr: str, wait_ms: int = 2000) -> None:
        worker = getattr(self, worker_attr, None)
        thread = getattr(self, thread_attr, None)
        self._dbg(f"safe_stop_worker_thread {name}", hardcore_only=True)
        self._clear_worker_restart_schedule(name)

        try:
            if worker is not None and hasattr(worker, "stop"):
                worker.stop()
        except Exception as exc:
            self._dbg(f"{name}: worker.stop() error: {exc}")

        try:
            if thread is not None:
                thread.quit()
                if not thread.wait(int(wait_ms)):
                    self._dbg(f"{name}: thread did not stop within {wait_ms} ms; force terminate")
                    try:
                        thread.terminate()
                        thread.wait(1000)
                    except Exception as exc2:
                        self._dbg(f"{name}: thread terminate error: {exc2}")
        except Exception as exc:
            self._dbg(f"{name}: thread.quit()/wait() error: {exc}")

        try:
            setattr(self, worker_attr, None)
        except Exception:
            pass
        try:
            setattr(self, thread_attr, None)
        except Exception:
            pass
        try:
            self._unregister_thread(name)
        except Exception:
            pass


    def _worker_restart_guard_reset_if_needed(self, name: str) -> None:
        try:
            now = time.monotonic()
            guard = self._worker_restart_guard.setdefault(str(name), {
                "window_start": now,
                "count": 0,
                "scheduled": False,
                "last_reason": "",
            })
            if (now - float(guard.get("window_start", now))) > float(getattr(self, "_worker_restart_window_seconds", 300.0)):
                guard["window_start"] = now
                guard["count"] = 0
                guard["scheduled"] = False
                guard["last_reason"] = ""
        except Exception:
            pass

    def _clear_worker_restart_schedule(self, name: str) -> None:
        try:
            self._worker_restart_guard_reset_if_needed(name)
            guard = self._worker_restart_guard.setdefault(str(name), {})
            guard["scheduled"] = False
        except Exception:
            pass

    def _can_restart_worker(self, name: str, reason: str = "") -> bool:
        try:
            self._worker_restart_guard_reset_if_needed(name)
            guard = self._worker_restart_guard.setdefault(str(name), {})
            count = int(guard.get("count", 0))
            limit = int(getattr(self, "_worker_restart_limit", 5))
            if count >= limit:
                self._dbg(f"restart blocked for {name}: limit reached ({count}/{limit}) reason={reason}")
                return False
            guard["count"] = count + 1
            guard["last_reason"] = str(reason or "")
            self._dbg(f"restart allowed for {name}: {guard['count']}/{limit} reason={reason}", hardcore_only=True)
            return True
        except Exception as exc:
            self._dbg(f"_can_restart_worker error for {name}: {exc}")
            return False

    def _schedule_worker_restart(self, name: str, reason: str = "", delay_ms: int = 3000) -> None:
        try:
            if getattr(self, "_closing", False):
                return
            starters = {
                "data_worker": self._start_data_worker,
                "rss_worker": self._start_rss_worker,
                "dx_worker": self._start_dx_worker,
                "kc2g_pipeline": self._start_kc2g_pipeline,
            }
            starter = starters.get(str(name))
            if starter is None:
                return

            self._worker_restart_guard_reset_if_needed(name)
            guard = self._worker_restart_guard.setdefault(str(name), {})
            if bool(guard.get("scheduled", False)):
                self._dbg(f"restart already scheduled for {name}", hardcore_only=True)
                return
            if not self._can_restart_worker(name, reason):
                return

            guard["scheduled"] = True

            def _restart():
                self._clear_worker_restart_schedule(name)
                if getattr(self, "_closing", False):
                    return
                try:
                    if str(name) == "dx_worker" and not bool(getattr(self, "_dx_enabled", False)):
                        self._dbg("DX restart skipped: panel disabled", hardcore_only=True)
                        return
                    if str(name) == "kc2g_pipeline" and str(getattr(self, "_muf_source", "sumo")).strip().lower() != "sumo":
                        self._dbg("KC2G restart skipped: kc2g pipeline not active source", hardcore_only=True)
                        return
                    self._dbg(f"restarting {name} after delay; reason={reason}")
                    starter()
                except Exception as exc:
                    self._dbg(f"restart execution failed for {name}: {exc}")

            QtCore.QTimer.singleShot(int(delay_ms), _restart)
        except Exception as exc:
            self._dbg(f"_schedule_worker_restart error for {name}: {exc}")

    def _on_managed_thread_finished(self, name: str) -> None:
        try:
            self._unregister_thread(name)
        except Exception:
            pass

        if getattr(self, "_closing", False):
            return

        try:
            if name == "data_worker":
                worker = getattr(self, "_worker", None)
                thread = getattr(self, "_thread", None)
            elif name == "rss_worker":
                worker = getattr(self, "_rss_worker", None)
                thread = getattr(self, "_rss_thread", None)
            elif name == "dx_worker":
                worker = getattr(self, "_dx_worker", None)
                thread = getattr(self, "_dx_thread", None)
            elif name == "kc2g_pipeline":
                worker = getattr(self, "_kc2g_worker", None)
                thread = getattr(self, "_kc2g_thread", None)
            else:
                worker = None
                thread = None

            if worker is None and thread is None:
                self._dbg(f"{name} finished after clean stop", hardcore_only=True)
                return

            # stale dangling refs are cleared before any restart
            try:
                if name == "data_worker":
                    self._worker = None
                    self._thread = None
                elif name == "rss_worker":
                    self._rss_worker = None
                    self._rss_thread = None
                elif name == "dx_worker":
                    self._dx_worker = None
                    self._dx_thread = None
                elif name == "kc2g_pipeline":
                    self._kc2g_worker = None
                    self._kc2g_thread = None
            except Exception:
                pass

            self._dbg(f"{name} thread finished unexpectedly")
            if name in ("data_worker", "rss_worker", "dx_worker"):
                self._schedule_worker_restart(name, reason="thread_finished", delay_ms=3000)
        except Exception as exc:
            self._dbg(f"_on_managed_thread_finished error for {name}: {exc}")

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SUMO - Sun Monitor")
        self.resize(1300, 780)

        self._cfg = load_config(CONFIG_PATH)
        self._debug_mode = safe_debug_enabled(self._cfg)
        self._debug_hardcore = safe_debug_hardcore(self._cfg)
        self._closing = False
        self._registered_threads = {}
        self._worker_restart_guard = {}
        self._worker_restart_limit = 5
        self._worker_restart_window_seconds = 300
        self._last_ui_heartbeat_monotonic = time.monotonic()
        self._last_display_change_monotonic = 0.0
        self._muf_refresh_in_progress = False
        self._muf_last_refresh_started = 0.0
        self._muf_last_refresh_elapsed_ms = 0
        self._muf_last_refresh_ok = False
        self._dbg(f"MainWindow init debug_mode={self._debug_mode} debug_hardcore={self._debug_hardcore}")
        try:
            self._dbg("Network limiter active: max 2 concurrent fetches")
        except Exception:
            pass

        # --- One-shot 5-minute border blink when panels ENTER red state ---
        self._kp_last_is_red = None  # type: bool | None
        self._sw_last_is_red = None  # type: bool | None


        # --- Sound alerts ---
        # Persisted in sumo_config.json ("sound_enabled")
        self._sound_enabled = bool(self._cfg.get("sound_enabled", True))
        try:
            self._alert_volume = max(0.0, min(1.0, float(self._cfg.get("alert_volume", 90)) / 100.0))
        except Exception:
            self._alert_volume = 0.9
        try:
            self._hour_chime_volume = max(0.0, min(1.0, float(self._cfg.get("hour_chime_volume", 90)) / 100.0))
        except Exception:
            self._hour_chime_volume = 0.9
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
                s.setVolume(self._alert_volume)
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

        # --- Main display rotation (solar tiles / clock dashboard / SUMO MUF map / SOHO dashboard / Solar System dashboard) ---
        self._display_mode = str(self._cfg.get("display_mode") or "").strip().lower()
        self._display_views = normalize_display_views(self._cfg.get("display_views"), self._display_mode)
        if not self._display_views:
            self._display_views = ["clock"] if bool(self._cfg.get("clock_grid_enabled", False)) else ["solar"]
        if self._display_mode not in ("solar", "clock", "muf", "soho", "solarsystem", "widgetdemo", "alternate_custom"):
            self._display_mode = self._display_views[0] if len(self._display_views) == 1 else "alternate_custom"
        try:
            self._display_switch_seconds = int(self._cfg.get("display_switch_seconds", 30))
        except Exception:
            self._display_switch_seconds = 30
        self._display_switch_seconds = max(5, self._display_switch_seconds)
        self._display_current_view = str(self._cfg.get("display_current_view") or "").strip().lower()
        if self._display_current_view not in VALID_DISPLAY_VIEWS:
            self._display_current_view = self._display_views[0]
        self._display_last_switch_monotonic = time.monotonic()

        self._muf_source = str(self._cfg.get("muf_source") or "sumo").strip().lower()
        if self._muf_source not in ("sumo", "kc2g"):
            self._muf_source = "sumo"

        self._soho_instrument = str(self._cfg.get("soho_instrument") or "dashboard6").strip().lower()
        if self._soho_instrument not in ("dashboard6",):
            self._soho_instrument = "dashboard6"
        try:
            self._soho_refresh_seconds = int(self._cfg.get("soho_refresh_seconds", SOHO_REFRESH_SECONDS))
        except Exception:
            self._soho_refresh_seconds = SOHO_REFRESH_SECONDS
        self._soho_refresh_seconds = max(60, int(self._soho_refresh_seconds))

        self._clock_grid_enabled = bool(self._cfg.get("clock_grid_enabled", False))
        self._clock_city_tl = str(self._cfg.get("clock_city_tl") or "America/New_York").strip()
        self._clock_city_tr = str(self._cfg.get("clock_city_tr") or "Europe/London").strip()
        self._clock_city_bl = str(self._cfg.get("clock_city_bl") or "Asia/Tokyo").strip()
        self._clock_city_br = str(self._cfg.get("clock_city_br") or "Australia/Sydney").strip()
        _clock_hour_chime_mode_cfg = self._cfg.get("clock_hour_chime_mode", None)
        if _clock_hour_chime_mode_cfg is None:
            self._clock_hour_chime_mode = "on" if bool(self._cfg.get("clock_hour_chime_enabled", False)) else "off"
        else:
            self._clock_hour_chime_mode = str(_clock_hour_chime_mode_cfg or "off").strip().lower()
        if self._clock_hour_chime_mode in ("utc", "local"):
            self._clock_hour_chime_mode = "on"
        if self._clock_hour_chime_mode not in ("off", "on"):
            self._clock_hour_chime_mode = "off"
        self._last_hour_chime_key = ""
        for attr_name, fallback in (
            ("_clock_city_tl", "America/New_York"),
            ("_clock_city_tr", "Europe/London"),
            ("_clock_city_bl", "Asia/Tokyo"),
            ("_clock_city_br", "Australia/Sydney"),
        ):
            val = getattr(self, attr_name)
            if val not in CLOCK_CITY_MAP.values():
                setattr(self, attr_name, fallback)

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

        # --- DAPNET settings / client ---
        self._dapnet_cfg = normalize_dapnet_config(self._cfg.get("dapnet"))
        self._cfg["dapnet"] = self._dapnet_cfg
        self._dapnet_client = DapnetClient(self._dapnet_cfg, logger=self._dbg)
        # Thread pool for non-blocking DAPNET operations
        from concurrent.futures import ThreadPoolExecutor
        self._dapnet_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="dapnet")
        self._dapnet_state = load_dapnet_state(DAPNET_STATE_PATH)
        self._dapnet_last_payload = {}
        self._dapnet_local_tz = safe_zoneinfo("Europe/Paris")
        self._dapnet_timer = QtCore.QTimer(self)
        self._dapnet_timer.setInterval(30 * 1000)
        self._dapnet_timer.timeout.connect(self._dapnet_periodic_tick)
        self._dapnet_timer.start()

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
        self.lbl_app.setAlignment(QtCore.Qt.AlignHCenter)
        self.lbl_sub = QtWidgets.QLabel("Sun Monitor")
        self.lbl_sub.setObjectName("appSubTitle")
        self.lbl_sub.setAlignment(QtCore.Qt.AlignHCenter)
        self.lbl_version = QtWidgets.QLabel(APP_VERSION)
        self.lbl_version.setObjectName("appVersion")
        self.lbl_version.setAlignment(QtCore.Qt.AlignHCenter)
        title_box.addWidget(self.lbl_app)
        title_box.addWidget(self.lbl_sub)
        title_box.addWidget(self.lbl_version)
        title_box.setAlignment(QtCore.Qt.AlignHCenter)

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

        self.lbl_status = QtWidgets.QLabel("Solar data: connecting…")
        self.lbl_status.setObjectName("statusText")
        self._set_data_status("connecting")

        # Put the data status in the bottom status bar (saves space in the top-right)
        try:
            self.statusBar().addPermanentWidget(self.lbl_status, 1)
        except Exception:
            pass

        self.lbl_iss_status = QtWidgets.QLabel("ISS TLE: loading...")
        self.lbl_iss_status.setObjectName("issStatusText")
        self.lbl_iss_status.setToolTip("ISS orbital data status. Independent from Solar data.")
        self.lbl_iss_status.setStyleSheet(f"color: {STATUS_WARN}; font-size: 12px;")
        try:
            self.statusBar().addPermanentWidget(self.lbl_iss_status, 0)
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

        # --- Bottom radio bar: HF openings + DAPNET quick send ---
        self.radio_bar = RadioBottomBar(self)
        self.hf_bar = self.radio_bar.hf_bar
        self.dapnet_quick_bar = self.radio_bar.quick_send
        root.addWidget(self.radio_bar)
        self.radio_bar.set_hf_visible(bool(getattr(self, "_hf_bar_visible", True)))
        self.radio_bar.set_quick_send_enabled(bool(self._dapnet_cfg.get("quick_ui_enabled", True)))
        try:
            preset = normalize_callsigns(self._dapnet_cfg.get("callsigns"))
            self.dapnet_quick_bar.ed_to.setText(preset[0].upper() if preset else "")
            cfg_group = str(self._dapnet_cfg.get("tx_group") or "f-53").strip() or "f-53"
            idx = self.dapnet_quick_bar.cb_tx_group.findText(cfg_group)
            self.dapnet_quick_bar.cb_tx_group.setCurrentIndex(idx if idx >= 0 else 0)
            self.dapnet_quick_bar._update_state()
        except Exception:
            pass

        self.main_grid = QtWidgets.QGridLayout()
        self.main_grid.setContentsMargins(0, 0, 0, 0)
        self.main_grid.setSpacing(10)
        root.addLayout(self.main_grid, 1)

        self.panels: dict[str, KpLikePanel] = {}
        self._panel_severity: dict[str, str] = {}

        configs = [
            ("CME",  PanelConfig("CME ARRIVAL PROBABILITY [N/A]", unit="%", big_value_fmt="{:.0f}")),
            ("XRAY", PanelConfig("X-RAYS (GOES 0.1-0.8nm)", unit="", big_value_fmt="{:.2e}", y_axis_kind="xray_class")),
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
            # Tooltip on the right-side value shows the color scale (palette)
            try:
                if key in ("SSN", "SFI"):
                    p.lbl_big.setToolTip(PALETTE_QUALITY_TOOLTIP)
                else:
                    p.lbl_big.setToolTip(PALETTE_RISK_TOOLTIP)
            except Exception:
                pass
            # Palette tooltip on the right-side value (does not get overwritten)
            try:
                if key in ("SSN", "SFI"):
                    p.set_value_tooltip(PALETTE_TOOLTIP_QUALITY)
                else:
                    p.set_value_tooltip(PALETTE_TOOLTIP_RISK)
            except Exception:
                pass
            # Widgets are laid out later (to support optional DX column)
            pass

        # Build the grid layout (9 panels, optional DX column, or clock dashboard)
        self.dx_panel = DxClusterPanel()
        self.dx_panel.setMinimumWidth(320)
        self.dx_panel.setMaximumWidth(380)
        self.clock_dashboard = ClockDashboard()
        self.clock_dashboard.set_center_mode(self._time_mode)
        self.clock_dashboard.set_city("tl", self._clock_city_tl)
        self.clock_dashboard.set_city("tr", self._clock_city_tr)
        self.clock_dashboard.set_city("bl", self._clock_city_bl)
        self.clock_dashboard.set_city("br", self._clock_city_br)
        self.solar_system_dashboard = SolarSystemDashboard()
        try:
            iss_cfg = normalize_dapnet_config(getattr(self, "_dapnet_cfg", {})).get("iss", {})
            self.solar_system_dashboard.set_iss_observer(
                float(iss_cfg.get("observer_lat", 48.1173)),
                float(iss_cfg.get("observer_lon", -1.6778)),
                float(iss_cfg.get("observer_alt_m", 60)),
                float(iss_cfg.get("min_elevation", 10.0)),
            )
        except Exception as exc:
            debug_log(f"ISS observer initial config failed: {exc}")
        try:
            self.solar_system_dashboard.skyfield_state_changed.connect(self._update_iss_tle_status)
        except Exception:
            pass
        self.widget_demo_panel = WidgetDemoPanel()
        self.widget_demo_panel.apply_config(getattr(self, "_cfg", {}))
        self._kc2g_debug_dir = APP_DIR / "giro_debug"
        self._kc2g_debug_dir.mkdir(parents=True, exist_ok=True)
        self._kc2g_grid_path = self._kc2g_debug_dir / "kc2g_muf_world_grid_v2.json"
        self.muf_world_widget = KC2GMufWorldMapWidgetV3_7WithMap(str(self._kc2g_grid_path))
        self.muf_world_widget.setStyleSheet("QWidget#kc2gMufMap { background: #0e141a; border: 2px solid #2a3440; border-radius: 14px; }")
        self.muf_remote_widget = KC2GRemoteMufMapWidget(KC2G_MUF_RENDER_URL)
        self.soho_widget = SohoDashboardWidget(
            refresh_seconds=self._soho_refresh_seconds,
        )
        self.soho_placeholder = QtWidgets.QFrame()
        self.soho_placeholder.setObjectName("sohoPlaceholder")
        self.soho_placeholder.setStyleSheet("QFrame#sohoPlaceholder { background: #0e141a; border: 2px solid #2a3440; border-radius: 14px; }")
        _soho_placeholder_layout = QtWidgets.QVBoxLayout(self.soho_placeholder)
        _soho_placeholder_layout.setContentsMargins(20, 20, 20, 20)
        _soho_placeholder_layout.setSpacing(10)
        self.soho_placeholder_title = QtWidgets.QLabel("SOHO Dashboard")
        self.soho_placeholder_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.soho_placeholder_title.setStyleSheet("color:#44d16e; font-size:28px; font-weight:900;")
        _soho_placeholder_layout.addStretch(1)
        _soho_placeholder_layout.addWidget(self.soho_placeholder_title)
        self.soho_placeholder_label = QtWidgets.QLabel("Loading SOHO image…")
        self.soho_placeholder_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.soho_placeholder_label.setWordWrap(True)
        self.soho_placeholder_label.setStyleSheet("color:#d7dde6; font-size:16px; font-weight:700;")
        _soho_placeholder_layout.addWidget(self.soho_placeholder_label)
        self.soho_placeholder_sub = QtWidgets.QLabel("The display will stay safe until the SOHO dashboard is ready.")
        self.soho_placeholder_sub.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.soho_placeholder_sub.setWordWrap(True)
        self.soho_placeholder_sub.setStyleSheet("color:#aab6c5; font-size:12px;")
        _soho_placeholder_layout.addWidget(self.soho_placeholder_sub)
        _soho_placeholder_layout.addStretch(1)
        self._main_display_registry = {
            "solar": None,
            "clock": self.clock_dashboard,
            "muf": None,
            "soho": self.soho_widget,
            "solarsystem": self.solar_system_dashboard,
            "widgetdemo": self.widget_demo_panel,
        }

        self.muf_placeholder = QtWidgets.QFrame()
        self.muf_placeholder.setObjectName("mufPlaceholder")
        self.muf_placeholder.setStyleSheet("QFrame#mufPlaceholder { background: #0e141a; border: 2px solid #2a3440; border-radius: 14px; }")
        _muf_placeholder_layout = QtWidgets.QVBoxLayout(self.muf_placeholder)
        _muf_placeholder_layout.setContentsMargins(20, 20, 20, 20)
        _muf_placeholder_layout.setSpacing(10)
        self.muf_placeholder_title = QtWidgets.QLabel("SUMO MUF")
        self.muf_placeholder_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.muf_placeholder_title.setStyleSheet("color:#44d16e; font-size:28px; font-weight:900;")
        _muf_placeholder_layout.addStretch(1)
        _muf_placeholder_layout.addWidget(self.muf_placeholder_title)
        self.muf_placeholder_label = QtWidgets.QLabel("Loading MUF map…")
        self.muf_placeholder_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.muf_placeholder_label.setWordWrap(True)
        self.muf_placeholder_label.setStyleSheet("color:#d7dde6; font-size:16px; font-weight:700;")
        _muf_placeholder_layout.addWidget(self.muf_placeholder_label)
        self.muf_placeholder_sub = QtWidgets.QLabel("The display will stay safe until the MUF grid is ready.")
        self.muf_placeholder_sub.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.muf_placeholder_sub.setWordWrap(True)
        self.muf_placeholder_sub.setStyleSheet("color:#aab6c5; font-size:12px;")
        _muf_placeholder_layout.addWidget(self.muf_placeholder_sub)
        _muf_placeholder_layout.addStretch(1)
        self._update_display_rotation(force_reset=True)
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
        if self._muf_source == "sumo":
            self._init_kc2g_pipeline()
        else:
            try:
                self.muf_remote_widget.reload()
            except Exception:
                pass


    def _init_kc2g_pipeline(self):
        self._kc2g_refresh_seconds = 15 * 60
        self._kc2g_last_error = ""
        self._kc2g_last_success_utc = ""
        self._pending_muf_refresh = False
        self._kc2g_thread = None
        self._kc2g_worker = None
        self._kc2g_timer = QtCore.QTimer(self)
        self._kc2g_timer.timeout.connect(self._start_kc2g_pipeline)
        self._kc2g_timer.start(self._kc2g_refresh_seconds * 1000)
        QtCore.QTimer.singleShot(1000, self._start_kc2g_pipeline)

    def _stop_kc2g_pipeline(self):
        try:
            if hasattr(self, "_kc2g_timer") and self._kc2g_timer is not None:
                self._kc2g_timer.stop()
                self._kc2g_timer.deleteLater()
        except Exception:
            pass
        self._kc2g_timer = None
        self._safe_stop_worker_thread("kc2g_pipeline", "_kc2g_worker", "_kc2g_thread", wait_ms=3000)

    def _refresh_muf_widget_if_visible(self, force: bool = False):
        try:
            if str(getattr(self, "_muf_source", "sumo") or "sumo").strip().lower() != "sumo":
                self._pending_muf_refresh = False
                return
            if getattr(self, "_muf_refresh_in_progress", False):
                self._pending_muf_refresh = True
                self._dbg("MUF refresh skipped because one is already running", hardcore_only=True)
                return
            w = getattr(self, "muf_world_widget", None)
            if w is None:
                return
            if self._effective_main_display() != "muf":
                self._pending_muf_refresh = True
                return
            self._muf_refresh_in_progress = True
            self._muf_last_refresh_started = time.monotonic()
            w.reload(force=force)
            self._pending_muf_refresh = False
            self._muf_last_refresh_ok = True
        except Exception as exc:
            self._muf_last_refresh_ok = False
            self._pending_muf_refresh = True
            self._dbg(f"MUF widget refresh error: {exc}")
        finally:
            try:
                self._muf_last_refresh_elapsed_ms = int((time.monotonic() - float(getattr(self, "_muf_last_refresh_started", time.monotonic()))) * 1000)
            except Exception:
                self._muf_last_refresh_elapsed_ms = 0
            self._muf_refresh_in_progress = False

    def _start_kc2g_pipeline(self):
        if getattr(self, "_closing", False):
            return
        self._clear_worker_restart_schedule("kc2g_pipeline")
        if str(getattr(self, "_muf_source", "sumo") or "sumo").strip().lower() != "sumo":
            return
        try:
            if getattr(self, "_kc2g_thread", None) is not None and self._kc2g_thread.isRunning():
                self._dbg("KC2G pipeline start skipped: already running", hardcore_only=True)
                return
        except Exception:
            pass
        self._kc2g_thread = QtCore.QThread(self)
        self._kc2g_worker = KC2GPipelineWorker(str(self._kc2g_debug_dir))
        self._kc2g_worker.moveToThread(self._kc2g_thread)
        self._kc2g_thread.started.connect(self._kc2g_worker.run)
        self._kc2g_worker.finished.connect(self._on_kc2g_pipeline_ready)
        self._kc2g_worker.error.connect(self._on_kc2g_pipeline_error)
        self._kc2g_worker.finished.connect(self._kc2g_thread.quit)
        self._kc2g_worker.error.connect(self._kc2g_thread.quit)
        self._kc2g_worker.finished.connect(self._kc2g_worker.deleteLater)
        self._kc2g_worker.error.connect(self._kc2g_worker.deleteLater)
        self._kc2g_thread.finished.connect(self._kc2g_thread.deleteLater)
        self._kc2g_thread.finished.connect(lambda: self._on_managed_thread_finished("kc2g_pipeline"))
        self._register_thread("kc2g_pipeline", self._kc2g_thread, self._kc2g_worker)
        self._dbg("KC2G pipeline start", hardcore_only=True)
        self._kc2g_thread.start()

    def _on_kc2g_pipeline_ready(self, result):
        try:
            self._dbg("KC2G pipeline ready", hardcore_only=True)
            self._kc2g_last_error = ""
            generated_at = str(result.get("generated_at") or "").strip()
            self._kc2g_last_success_utc = generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
            self._pending_muf_refresh = True
            self._refresh_muf_widget_if_visible(force=False)
            self._update_muf_placeholder_text()
            try:
                count = int(result.get("active_count", 0))
                pts = int(result.get("point_count", 0))
                self.statusBar().showMessage(f"SUMO MUF updated • active={count} • points={pts} • {self._kc2g_last_success_utc}", 6000)
            except Exception:
                pass
        except Exception:
            pass
        finally:
            self._kc2g_thread = None
            self._kc2g_worker = None

    def _on_kc2g_pipeline_error(self, message: str):
        self._dbg(f"KC2G pipeline error: {message}")
        self._kc2g_last_error = str(message)
        self._pending_muf_refresh = False
        try:
            self.statusBar().showMessage(f"SUMO MUF error: {message}", 8000)
        except Exception:
            pass
        try:
            self._update_muf_placeholder_text()
            if self._effective_main_display() == "muf":
                self._layout_main_grid()
        except Exception:
            pass
        self._kc2g_thread = None
        self._kc2g_worker = None

    def _start_data_worker(self):
        try:
            if getattr(self, "_thread", None) is not None and self._thread.isRunning():
                self._dbg("DataWorker start skipped: already running", hardcore_only=True)
                return
        except Exception:
            pass
        self._clear_worker_restart_schedule("data_worker")
        self._thread = QtCore.QThread(self)
        self._worker = DataWorker(refresh_seconds=60, nasa_api_key=self._nasa_api_key)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.data_ready.connect(self._on_data)
        self._worker.error.connect(self._on_error)
        self._worker.error.connect(lambda msg: self._dbg(f"DataWorker error: {msg}"))
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(lambda: self._on_managed_thread_finished("data_worker"))
        self._register_thread("data_worker", self._thread, self._worker)
        self._dbg("DataWorker start", hardcore_only=True)
        self._thread.start()

    def _restart_data_worker(self):
        self._safe_stop_worker_thread("data_worker", "_worker", "_thread")
        self._start_data_worker()

    def _start_rss_worker(self):
        try:
            if getattr(self, "_rss_thread", None) is not None and self._rss_thread.isRunning():
                self._dbg("RSS worker start skipped: already running", hardcore_only=True)
                return
        except Exception:
            pass
        self._clear_worker_restart_schedule("rss_worker")
        self._rss_thread = QtCore.QThread(self)
        self._rss_worker = RssWorker(self._rss_url, RSS_REFRESH_SECONDS)
        self._rss_worker.moveToThread(self._rss_thread)
        self._rss_thread.started.connect(self._rss_worker.run)
        self._rss_worker.rss_ready.connect(self._on_rss_text)
        self._rss_worker.rss_error.connect(self._on_rss_error)
        self._rss_worker.rss_error.connect(lambda msg: self._dbg(f"RSS worker error: {msg}"))
        self._rss_worker.rss_error.connect(self._rss_thread.quit)
        self._rss_thread.finished.connect(self._rss_thread.deleteLater)
        self._rss_thread.finished.connect(lambda: self._on_managed_thread_finished("rss_worker"))
        self._register_thread("rss_worker", self._rss_thread, self._rss_worker)
        self._dbg("RSS worker start", hardcore_only=True)
        self._rss_thread.start()

    def _restart_rss_worker(self, new_url: str):
        self._rss_url = str(new_url or self._rss_url).strip() or DEFAULT_RSS_URL
        self._safe_stop_worker_thread("rss_worker", "_rss_worker", "_rss_thread")
        self._start_rss_worker()

    def _open_settings(self):
        dlg = SettingsDialog(
            self,
            rss_url=self._rss_url,
            nasa_api_key=self._nasa_api_key,
            time_mode=self._time_mode,
            clock_grid_enabled=self._clock_grid_enabled,
            clock_city_tl=self._clock_city_tl,
            clock_city_tr=self._clock_city_tr,
            clock_city_bl=self._clock_city_bl,
            clock_city_br=self._clock_city_br,
            clock_hour_chime_mode=getattr(self, "_clock_hour_chime_mode", "off"),
            alert_volume=int(round(float(getattr(self, "_alert_volume", 0.9)) * 100)),
            hour_chime_volume=int(round(float(getattr(self, "_hour_chime_volume", 0.9)) * 100)),
            display_mode=self._display_mode,
            display_views=list(getattr(self, "_display_views", normalize_display_views(None, self._display_mode))),
            display_switch_seconds=self._display_switch_seconds,
            muf_source=self._muf_source,
            soho_instrument=getattr(self, "_soho_instrument", "eit_195"),
            soho_refresh_seconds=getattr(self, "_soho_refresh_seconds", SOHO_REFRESH_SECONDS),
            rss_speed=self._rss_speed,
            dx_enabled=self._dx_enabled,
            dx_source=self._dx_source,
            dx_host=self._dx_host,
            dx_port=self._dx_port,
            dx_login=self._dx_login,
            pota_zone=getattr(self, "_pota_zone", "worldwide"),
            dapnet_config=getattr(self, "_dapnet_cfg", None),
        )
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return

        new_rss = dlg.rss_url().strip() or DEFAULT_RSS_URL
        new_key = dlg.nasa_api_key().strip()
        new_mode = dlg.time_mode().strip().lower() or "utc"
        new_clock_grid_enabled = dlg.clock_grid_enabled()
        new_clock_city_tl = dlg.clock_city_tl()
        new_clock_city_tr = dlg.clock_city_tr()
        new_clock_city_bl = dlg.clock_city_bl()
        new_clock_city_br = dlg.clock_city_br()
        new_clock_hour_chime_mode = dlg.clock_hour_chime_mode()
        new_alert_volume = dlg.alert_volume()
        new_hour_chime_volume = dlg.hour_chime_volume()
        new_display_mode = dlg.display_mode()
        new_display_views = dlg.display_views()
        new_display_switch_seconds = dlg.display_switch_seconds()
        new_muf_source = dlg.muf_source()
        new_soho_instrument = dlg.soho_instrument()
        new_soho_refresh_seconds = dlg.soho_refresh_seconds()
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
        new_dapnet_cfg = dlg.dapnet_config()

        changed_rss = (new_rss != self._rss_url)
        changed_key = (new_key != self._nasa_api_key)
        changed_mode = (new_mode != self._time_mode)
        changed_clock_grid = (bool(new_clock_grid_enabled) != bool(self._clock_grid_enabled))
        changed_clock_cities = (
            new_clock_city_tl != self._clock_city_tl or
            new_clock_city_tr != self._clock_city_tr or
            new_clock_city_bl != self._clock_city_bl or
            new_clock_city_br != self._clock_city_br
        )
        changed_clock_hour_chime = (str(new_clock_hour_chime_mode) != str(getattr(self, "_clock_hour_chime_mode", "off")))
        changed_sound_levels = (
            int(new_alert_volume) != int(round(float(getattr(self, "_alert_volume", 0.9)) * 100)) or
            int(new_hour_chime_volume) != int(round(float(getattr(self, "_hour_chime_volume", 0.9)) * 100))
        )
        changed_display = (
            list(new_display_views) != list(getattr(self, "_display_views", normalize_display_views(None, getattr(self, "_display_mode", "solar")))) or
            str(new_display_mode) != str(self._display_mode) or
            int(new_display_switch_seconds) != int(self._display_switch_seconds) or
            str(new_muf_source) != str(getattr(self, "_muf_source", "sumo")) or
            str(new_soho_instrument) != str(getattr(self, "_soho_instrument", "dashboard6")) or
            int(new_soho_refresh_seconds) != int(getattr(self, "_soho_refresh_seconds", SOHO_REFRESH_SECONDS))
        )
        changed_speed = (int(new_speed) != int(self._rss_speed))
        changed_dx = (bool(new_dx_enabled) != bool(self._dx_enabled)) or (str(new_dx_source) != str(self._dx_source)) or (new_dx_host != self._dx_host) or (int(new_dx_port) != int(self._dx_port)) or (new_dx_login != self._dx_login) or (str(new_pota_zone) != str(getattr(self, "_pota_zone", "worldwide")) )
        changed_dapnet = (normalize_dapnet_config(new_dapnet_cfg) != normalize_dapnet_config(getattr(self, "_dapnet_cfg", {})))

        self._rss_url = new_rss
        self._nasa_api_key = new_key
        self._time_mode = new_mode
        self._clock_grid_enabled = bool(new_clock_grid_enabled)
        self._clock_city_tl = new_clock_city_tl
        self._clock_city_tr = new_clock_city_tr
        self._clock_city_bl = new_clock_city_bl
        self._clock_city_br = new_clock_city_br
        self._clock_hour_chime_mode = str(new_clock_hour_chime_mode).strip().lower() or "off"
        if self._clock_hour_chime_mode in ("utc", "local"):
            self._clock_hour_chime_mode = "on"
        if self._clock_hour_chime_mode not in ("off", "on"):
            self._clock_hour_chime_mode = "off"
        self._alert_volume = max(0.0, min(1.0, int(new_alert_volume) / 100.0))
        self._hour_chime_volume = max(0.0, min(1.0, int(new_hour_chime_volume) / 100.0))
        self._display_views = normalize_display_views(new_display_views, new_display_mode)
        self._display_mode = str(new_display_mode).strip().lower() or "solar"
        if self._display_mode not in ("solar", "clock", "muf", "soho", "solarsystem", "widgetdemo", "alternate_custom"):
            self._display_mode = self._display_views[0] if len(self._display_views) == 1 else "alternate_custom"
        self._display_switch_seconds = max(5, int(new_display_switch_seconds))
        self._muf_source = str(new_muf_source).strip().lower() or "sumo"
        if self._muf_source not in ("sumo", "kc2g"):
            self._muf_source = "sumo"
        self._soho_instrument = str(new_soho_instrument).strip().lower() or "dashboard6"
        if self._soho_instrument not in ("dashboard6",):
            self._soho_instrument = "dashboard6"
        self._soho_refresh_seconds = max(60, int(new_soho_refresh_seconds))
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
        self._dapnet_cfg = normalize_dapnet_config(new_dapnet_cfg)
        try:
            self._dapnet_client.update_config(self._dapnet_cfg)
        except Exception:
            self._dapnet_client = DapnetClient(self._dapnet_cfg, logger=self._dbg)
        self._refresh_dapnet_quick_bar()
        try:
            iss_cfg = normalize_dapnet_config(getattr(self, "_dapnet_cfg", {})).get("iss", {})
            if hasattr(self, "solar_system_dashboard") and self.solar_system_dashboard is not None:
                self.solar_system_dashboard.set_iss_observer(
                    float(iss_cfg.get("observer_lat", 48.1173)),
                    float(iss_cfg.get("observer_lon", -1.6778)),
                    float(iss_cfg.get("observer_alt_m", 60)),
                    float(iss_cfg.get("min_elevation", 10.0)),
                )
        except Exception as exc:
            self._dbg(f"ISS observer update failed: {exc}")

        self._cfg["rss_url"] = self._rss_url
        self._cfg["nasa_api_key"] = self._nasa_api_key
        self._cfg["time_mode"] = self._time_mode
        self._cfg["clock_grid_enabled"] = bool(self._clock_grid_enabled)
        self._cfg["clock_city_tl"] = self._clock_city_tl
        self._cfg["clock_city_tr"] = self._clock_city_tr
        self._cfg["clock_city_bl"] = self._clock_city_bl
        self._cfg["clock_city_br"] = self._clock_city_br
        self._cfg["clock_hour_chime_mode"] = str(self._clock_hour_chime_mode)
        self._cfg["clock_hour_chime_enabled"] = bool(self._clock_hour_chime_mode == "on")
        self._cfg["alert_volume"] = int(round(self._alert_volume * 100))
        self._cfg["hour_chime_volume"] = int(round(self._hour_chime_volume * 100))
        self._cfg["display_mode"] = self._display_mode
        self._cfg["display_views"] = list(getattr(self, "_display_views", ["solar"]))
        try:
            self.widget_demo_panel.apply_config(self._cfg)
        except Exception:
            pass
        self._cfg["display_switch_seconds"] = int(self._display_switch_seconds)
        self._cfg["muf_source"] = str(self._muf_source)
        self._cfg["soho_instrument"] = str(self._soho_instrument)
        self._cfg["soho_refresh_seconds"] = int(self._soho_refresh_seconds)
        self._cfg["clock_grid_enabled"] = bool(self._display_mode == "clock")
        self._cfg["rss_speed"] = int(self._rss_speed)
        self._cfg["dx_enabled"] = bool(self._dx_enabled)
        self._cfg["dx_source"] = str(self._dx_source)
        self._cfg["dx_host"] = self._dx_host
        self._cfg["dx_port"] = int(self._dx_port)
        self._cfg["dx_login"] = self._dx_login
        self._cfg["pota_zone"] = str(getattr(self, "_pota_zone", "worldwide"))
        self._cfg["dapnet"] = normalize_dapnet_config(getattr(self, "_dapnet_cfg", {}))
        save_config(CONFIG_PATH, self._cfg)

        if str(self._muf_source) == "sumo":
            if not hasattr(self, "_kc2g_timer") or self._kc2g_timer is None:
                self._init_kc2g_pipeline()
            else:
                try:
                    self._kc2g_timer.start(self._kc2g_refresh_seconds * 1000)
                except Exception:
                    pass
                try:
                    self._start_kc2g_pipeline()
                except Exception:
                    pass
        else:
            try:
                self._stop_kc2g_pipeline()
            except Exception:
                pass
            try:
                if hasattr(self, "muf_remote_widget") and self.muf_remote_widget is not None:
                    self.muf_remote_widget.reload()
            except Exception:
                pass

        if changed_rss:
            self.rss_ticker.setText("NASA Solar System News   •   updating RSS…")
            self._restart_rss_worker(self._rss_url)

        if changed_key:
            self._restart_data_worker()

        if changed_sound_levels:
            self._apply_sound_volumes()

        if changed_mode or changed_clock_grid or changed_clock_cities or changed_display or changed_clock_hour_chime:
            if str(getattr(self, "_clock_hour_chime_mode", "off")) == "off":
                self._last_hour_chime_key = ""
            self.clock_dashboard.set_center_mode(self._time_mode)
            self.clock_dashboard.set_city("tl", self._clock_city_tl)
            self.clock_dashboard.set_city("tr", self._clock_city_tr)
            self.clock_dashboard.set_city("bl", self._clock_city_bl)
            self.clock_dashboard.set_city("br", self._clock_city_br)
            self._update_display_rotation(force_reset=True)
            self._layout_main_grid()
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
            bg = BTN_GREEN
        else:
            self.btn_dx.setText("DX OFF")
            bg = BTN_RED

        self.btn_dx.setStyleSheet(f"""
            QPushButton {{
                background: {bg};
                color: {BTN_TEXT};
                border: 0px;
                border-radius: 6px;
                padding: 4px 10px;
                font-weight: 900;
            }}
        """)

    def _update_hfbar_button(self):
        """Update the HF ribbon toggle button in the top bar."""
        if getattr(self, "_hf_bar_visible", True):
            self.btn_hfbar.setText("HF BAR ON")
            bg = BTN_GREEN
        else:
            self.btn_hfbar.setText("HF BAR OFF")
            bg = BTN_RED

        self.btn_hfbar.setStyleSheet(f"""
            QPushButton {{
                background: {bg};
                color: {BTN_TEXT};
                border: 0px;
                border-radius: 6px;
                padding: 4px 10px;
                font-weight: 900;
            }}
        """)

    def _toggle_hf_bar(self):
        """Show/hide the HF openings ribbon under the top bar."""
        self._hf_bar_visible = not getattr(self, "_hf_bar_visible", True)
        self._cfg["hf_bar_visible"] = bool(self._hf_bar_visible)
        save_config(CONFIG_PATH, self._cfg)

        try:
            if hasattr(self, "radio_bar") and self.radio_bar is not None:
                self.radio_bar.set_hf_visible(bool(self._hf_bar_visible))
            elif hasattr(self, "hf_bar") and self.hf_bar is not None:
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

    def _active_muf_widget(self):
        src = str(getattr(self, "_muf_source", "sumo") or "sumo").strip().lower()
        if src == "kc2g":
            return getattr(self, "muf_remote_widget", None)
        return getattr(self, "muf_world_widget", None)

    def _muf_can_be_shown(self) -> bool:
        try:
            src = str(getattr(self, "_muf_source", "sumo") or "sumo").strip().lower()
            if src == "kc2g":
                w = getattr(self, "muf_remote_widget", None)
                return bool(w is not None and w.has_valid_map())
            w = getattr(self, "muf_world_widget", None)
            return bool(w is not None and w.has_valid_grid())
        except Exception:
            return False

    def _update_muf_placeholder_text(self) -> None:
        try:
            src = str(getattr(self, "_muf_source", "sumo") or "sumo").strip().lower()
            if self._muf_can_be_shown():
                self.muf_placeholder_title.setText("KC2G MUF" if src == "kc2g" else "SUMO MUF")
                self.muf_placeholder_label.setText("MUF map ready.")
                active = self._active_muf_widget()
                if active is not None and hasattr(active, "status_text"):
                    self.muf_placeholder_sub.setText(active.status_text())
                else:
                    self.muf_placeholder_sub.setText("")
            else:
                self.muf_placeholder_title.setText("KC2G MUF" if src == "kc2g" else "SUMO MUF")
                err = ""
                if src == "kc2g":
                    w = getattr(self, "muf_remote_widget", None)
                    if w is not None:
                        err = str(getattr(w, "_last_error", "") or "").strip()
                else:
                    err = str(getattr(self, "_kc2g_last_error", "") or "").strip()
                if err:
                    self.muf_placeholder_label.setText("MUF temporarily unavailable")
                    self.muf_placeholder_sub.setText(err)
                else:
                    self.muf_placeholder_label.setText("Loading MUF map…")
                    if src == "kc2g":
                        self.muf_placeholder_sub.setText("Waiting for the latest map from the KC2G servers.")
                    else:
                        self.muf_placeholder_sub.setText("The display will stay safe until the MUF grid is ready.")
        except Exception:
            pass

    def _soho_can_be_shown(self) -> bool:
        try:
            return getattr(self, "soho_widget", None) is not None
        except Exception:
            return False

    def _update_soho_placeholder_text(self) -> None:
        try:
            self.soho_placeholder_title.setText("SOHO Dashboard")
            if self._soho_can_be_shown():
                self.soho_placeholder_label.setText("SOHO dashboard ready.")
                w = getattr(self, "soho_widget", None)
                if w is not None and hasattr(w, "status_text"):
                    self.soho_placeholder_sub.setText(w.status_text())
                else:
                    self.soho_placeholder_sub.setText("")
            else:
                err = ""
                w = getattr(self, "soho_widget", None)
                if w is not None:
                    err = str(getattr(w, "_last_error", "") or "").strip()
                if err:
                    self.soho_placeholder_label.setText("SOHO image temporarily unavailable")
                    self.soho_placeholder_sub.setText(err)
                else:
                    self.soho_placeholder_label.setText("Loading SOHO image…")
                    self.soho_placeholder_sub.setText("Waiting for the latest official SOHO realtime images.")
        except Exception:
            pass

    def _display_sequence(self):
        seq = normalize_display_views(getattr(self, "_display_views", None), getattr(self, "_display_mode", "solar"))
        return [x for x in seq if x in VALID_DISPLAY_VIEWS] or ["solar"]

    def _effective_main_display(self) -> str:
        current = str(getattr(self, "_display_current_view", "solar") or "solar").strip().lower()
        seq = self._display_sequence()
        if current in seq:
            return current
        return seq[0]

    def _update_display_rotation(self, force_reset: bool = False):
        try:
            interval = max(5, int(getattr(self, "_display_switch_seconds", 30)))
        except Exception:
            interval = 30
        now_mono = time.monotonic()
        previous = self._effective_main_display()
        seq = self._display_sequence()

        if force_reset:
            self._display_last_switch_monotonic = now_mono
            preferred = seq[0]
            if preferred == "muf" and not self._muf_can_be_shown() and len(seq) > 1:
                preferred = next((x for x in seq if x != "muf"), seq[0])
            if preferred == "soho" and not self._soho_can_be_shown() and len(seq) > 1:
                preferred = next((x for x in seq if x != "soho"), seq[0])
            self._display_current_view = preferred
            return previous != self._effective_main_display()

        if len(seq) <= 1:
            preferred = seq[0]
            if preferred == "muf" and not self._muf_can_be_shown():
                preferred = previous if previous in ("solar", "clock", "soho", "solarsystem", "widgetdemo") else "solar"
            if preferred == "soho" and not self._soho_can_be_shown():
                preferred = previous if previous in ("solar", "clock", "muf", "solarsystem", "widgetdemo") else "solar"
            self._display_current_view = preferred
            self._display_last_switch_monotonic = now_mono
            return previous != self._effective_main_display()

        if (now_mono - float(getattr(self, "_display_last_switch_monotonic", now_mono))) >= interval:
            cur = str(getattr(self, "_display_current_view", seq[0]) or seq[0]).strip().lower()
            try:
                idx = seq.index(cur)
            except ValueError:
                idx = 0

            next_view = cur
            for step in range(1, len(seq) + 1):
                candidate = seq[(idx + step) % len(seq)]
                if candidate == "muf" and not self._muf_can_be_shown():
                    continue
                if candidate == "soho" and not self._soho_can_be_shown():
                    continue
                next_view = candidate
                break

            self._display_current_view = next_view
            self._display_last_switch_monotonic = now_mono
            if previous != next_view:
                self._last_display_change_monotonic = now_mono
                self._dbg(f"Display switched: {previous} -> {next_view}", hardcore_only=True)

        return previous != self._effective_main_display()

    def _layout_main_grid(self):
        """Lay out the 9 solar panels, the clock dashboard, the Solar System dashboard, the SUMO MUF widget, or the 6-image SOHO dashboard, optionally with a left DX column."""
        g = self.main_grid
        self._clear_layout(g)

        if getattr(self, "_dx_enabled", False):
            g.addWidget(self.dx_panel, 0, 0, 3, 1)
            positions = [(0, 1), (0, 2), (0, 3),
                         (1, 1), (1, 2), (1, 3),
                         (2, 1), (2, 2), (2, 3)]
            g.setColumnMinimumWidth(0, 320)
            g.setColumnStretch(0, 0)
            for cc in (1, 2, 3):
                g.setColumnStretch(cc, 1)
        else:
            positions = [(0, 0), (0, 1), (0, 2),
                         (1, 0), (1, 1), (1, 2),
                         (2, 0), (2, 1), (2, 2)]
            g.setColumnMinimumWidth(0, 0)
            g.setColumnMinimumWidth(3, 0)
            g.setColumnStretch(3, 0)
            for cc in (0, 1, 2):
                g.setColumnStretch(cc, 1)

        for rr in range(3):
            g.setRowStretch(rr, 1)

        current = self._effective_main_display()
        debug_log(f"_layout_main_grid current={current} dx={bool(getattr(self, '_dx_enabled', False))} muf_source={getattr(self, '_muf_source', 'sumo')}")
        if current == "clock":
            if getattr(self, "_dx_enabled", False):
                g.addWidget(self.clock_dashboard, 0, 1, 3, 3)
            else:
                g.addWidget(self.clock_dashboard, 0, 0, 3, 3)
            return

        if current == "solarsystem":
            if getattr(self, "_dx_enabled", False):
                g.addWidget(self.solar_system_dashboard, 0, 1, 3, 3)
            else:
                g.addWidget(self.solar_system_dashboard, 0, 0, 3, 3)
            return

        if current == "widgetdemo":
            try:
                self.widget_demo_panel.start()
                self.widget_demo_panel.apply_config(getattr(self, "_cfg", {}))
            except Exception:
                pass
            if getattr(self, "_dx_enabled", False):
                g.addWidget(self.widget_demo_panel, 0, 1, 3, 3)
            else:
                g.addWidget(self.widget_demo_panel, 0, 0, 3, 3)
            return

        if current == "muf":
            if getattr(self, "_pending_muf_refresh", False):
                self._refresh_muf_widget_if_visible(force=False)
            self._update_muf_placeholder_text()
            muf_target = self._active_muf_widget() if self._muf_can_be_shown() else self.muf_placeholder
            if getattr(self, "_dx_enabled", False):
                g.addWidget(muf_target, 0, 1, 3, 3)
            else:
                g.addWidget(muf_target, 0, 0, 3, 3)
            return

        if current == "soho":
            self._update_soho_placeholder_text()
            soho_target = self.soho_widget
            if getattr(self, "_dx_enabled", False):
                g.addWidget(soho_target, 0, 1, 3, 3)
            else:
                g.addWidget(soho_target, 0, 0, 3, 3)
            return

        order = ["CME", "XRAY", "KP", "SSN", "SFI", "BZBT", "P10", "AUR", "SW"]
        for key, (r, c) in zip(order, positions):
            w = self.panels.get(key)
            if w is not None:
                g.addWidget(w, r, c)

    def _start_dx_worker(self):
        self._dbg(f"DX worker start source={getattr(self, '_dx_source', 'dx')}", hardcore_only=True)
        """Start the left DX column worker (DX Cluster telnet OR POTA spots)."""
        try:
            if getattr(self, "_dx_thread", None) is not None and self._dx_thread.isRunning():
                self._dbg("DX worker start skipped: already running", hardcore_only=True)
                return
        except Exception:
            pass
        self._stop_dx_worker()
        self._clear_worker_restart_schedule("dx_worker")

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
        self._dx_thread.finished.connect(lambda: self._on_managed_thread_finished("dx_worker"))
        self._register_thread("dx_worker", self._dx_thread, self._dx_worker)
        self._dx_thread.start()

    def _stop_dx_worker(self):
        self._safe_stop_worker_thread("dx_worker", "_dx_worker", "_dx_thread", wait_ms=2000)
        try:
            self._pota_spot_buffer = []
            self._pota_flush_scheduled = False
        except Exception:
            pass

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
            self._dbg(f"DX error handler: {msg}")
            self.dx_panel.set_status(f"DX: offline ({msg})")
        except Exception:
            pass
        if not getattr(self, "_closing", False) and bool(getattr(self, "_dx_enabled", False)):
            self._schedule_worker_restart("dx_worker", reason=f"dx_error:{msg}", delay_ms=5000)

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
                s.setVolume(self._alert_volume)
                # QSoundEffect expects a file URL
                s.setSource(QtCore.QUrl.fromLocalFile(path))
                self._sfx[key] = s
        except Exception:
            # If QtMultimedia backend is missing, we will fall back to QApplication.beep().
            self._sfx = {}

    def _apply_sound_volumes(self):
        try:
            for s in getattr(self, "_sfx", {}).values():
                try:
                    s.setVolume(float(getattr(self, "_alert_volume", 0.9)))
                except Exception:
                    pass
            alert_sfx = getattr(self, "_alert_sfx", None)
            if alert_sfx is not None:
                try:
                    alert_sfx.setVolume(float(getattr(self, "_alert_volume", 0.9)))
                except Exception:
                    pass
            hour_sfx = getattr(self, "_hour_chime_sfx", None)
            if hour_sfx is not None:
                try:
                    hour_sfx.setVolume(float(getattr(self, "_hour_chime_volume", 0.9)))
                except Exception:
                    pass
        except Exception:
            pass

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
        # quick feedback: play alert.wav when turning sound ON
        if self._sound_enabled:
            play_alert_sound(self)

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

    def _iss_tle_status_text(self) -> tuple[str, str]:
        """Return (text, color) for ISS TLE status. Kept separate from Solar data.

        Loading is only shown while an actual TLE/Skyfield load is running.
        Otherwise SUMO exposes OK/ERROR so the label never stays stuck silently.
        """
        try:
            dash = getattr(self, "solar_system_dashboard", None)
            if dash is None:
                return "ISS TLE: ERROR", STATUS_ERR

            if getattr(dash, "_iss_satellite", None) is not None:
                return "ISS TLE: OK", STATUS_OK

            if getattr(dash, "_iss_tle_loading", False) or getattr(dash, "_skyfield_loading", False):
                return "ISS TLE: loading...", STATUS_WARN

            err = str(getattr(dash, "_iss_tle_error", "") or getattr(dash, "_skyfield_error", "") or "").strip()
            if err:
                return "ISS TLE: ERROR", STATUS_ERR

            return "ISS TLE: ERROR", STATUS_ERR
        except Exception:
            return "ISS TLE: ERROR", STATUS_ERR

    def _update_iss_tle_status(self):
        try:
            text, color = self._iss_tle_status_text()
            if hasattr(self, "lbl_iss_status") and self.lbl_iss_status is not None:
                self.lbl_iss_status.setText(text)
                self.lbl_iss_status.setStyleSheet(f"color: {color}; font-size: 12px; font-weight: 700;")
                try:
                    dash = getattr(self, "solar_system_dashboard", None)
                    detail = ""
                    if dash is not None:
                        detail = str(getattr(dash, "_iss_tle_epoch_text", "") or getattr(dash, "_iss_tle_error", "") or getattr(dash, "_skyfield_error", "") or "")
                    self.lbl_iss_status.setToolTip(detail)
                except Exception:
                    pass
        except Exception:
            pass

    def _set_data_status(self, state: str, details: str | None = None):
        label = "Solar data"
        max_len = 140
        if details:
            short = details[:max_len] + "..." if len(details) > max_len else details
        else:
            short = ""

        if state == "ok":
            text = f"{label}: OK"
            if short:
                text += f" ({short})"
            color = STATUS_OK
        elif state == "partial":
            text = f"{label}: partial"
            if short:
                text += f" ({short})"
            color = STATUS_WARN
        elif state == "error":
            if short:
                text = f"{label}: ERROR ({short})"
            else:
                text = f"{label}: ERROR"
            color = STATUS_ERR
        else:
            if short:
                text = f"{label}: connecting… ({short})"
            else:
                text = f"{label}: connecting…"
            color = STATUS_WARN

        self.lbl_status.setText(text)
        self.lbl_status.setStyleSheet(f"color: {color}; font-size: 12px;")

    def closeEvent(self, event):
        self._closing = True
        debug_log("MainWindow.closeEvent start")
        try:
            if hasattr(self, "_utc_timer") and self._utc_timer is not None:
                self._utc_timer.stop()
        except Exception:
            pass
        try:
            self._stop_dx_worker()
        except Exception:
            pass

        try:
            self._stop_kc2g_pipeline()
        except Exception:
            pass
        try:
            if hasattr(self, "muf_remote_widget") and self.muf_remote_widget is not None:
                self.muf_remote_widget.stop()
        except Exception:
            pass
        try:
            if hasattr(self, "soho_widget") and self.soho_widget is not None:
                self.soho_widget.stop()
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

        debug_log("MainWindow.closeEvent end")
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
        now_utc = datetime.now(timezone.utc)
        now_local = now_utc.astimezone()

        if getattr(self, "_time_mode", "utc") == "local":
            now = now_local
            label = "LOCAL"
        else:
            now = now_utc
            label = "UTC"

        t = now.strftime("%H:%M:%S")
        d = now.strftime("%Y-%m-%d")
        self.lbl_utc_badge.setText(f"{label} {t}\n{d}")
        try:
            self._update_iss_tle_status()
        except Exception:
            pass

        try:
            chime_mode = str(getattr(self, "_clock_hour_chime_mode", "off") or "off").strip().lower()
            chime_dt = now_local if getattr(self, "_time_mode", "utc") == "local" else now_utc

            if chime_mode == "on":
                if chime_dt.minute == 59 and chime_dt.second == 56:
                    target_dt = chime_dt + timedelta(seconds=4)
                    chime_prefix = "local" if getattr(self, "_time_mode", "utc") == "local" else "utc"
                    chime_key = f"{chime_prefix}:{target_dt.strftime('%Y-%m-%d %H')}"
                    if chime_key != getattr(self, "_last_hour_chime_key", ""):
                        self._last_hour_chime_key = chime_key
                        play_hour_chime(self)
            elif chime_mode == "off":
                self._last_hour_chime_key = ""
        except Exception:
            pass

        try:
            if hasattr(self, "clock_dashboard") and self.clock_dashboard is not None:
                self.clock_dashboard.set_center_mode(self._time_mode)
                self.clock_dashboard.update_times(now)
        except Exception:
            pass

        try:
            if self._update_display_rotation(force_reset=False):
                self._layout_main_grid()
        except Exception:
            pass

    def _style_panel_title_default(self, panel: KpLikePanel):
        panel.lbl_title.setStyleSheet("")

    def _style_panel_title_color(self, panel: KpLikePanel, color: str):
        panel.lbl_title.setStyleSheet(
            f"color: {color}; font-size: 20px; font-weight: 700; letter-spacing: 0.5px;"
        )


    # =====================================================
    # Coherent panel coloring (severity grammar)
    # =====================================================
    def _set_panel_severity(self, key: str, family: str, sev: str, blink_on_danger: bool = True):
        """Apply a coherent accent color based on severity.
        Play the generic alert sound only when a RISK panel enters DANGER/EXTREME.
        Blink only when entering DANGER/EXTREME for RISK family.
        """
        try:
            key = str(key)
            family = (str(family) or "risk").strip().lower()
            sev = str(sev or SEV_UNKNOWN).strip().upper()
        except Exception:
            family = "risk"
            sev = SEV_UNKNOWN

        if family not in ("risk", "quality"):
            family = "risk"

        order = RISK_ORDER if family == "risk" else QUAL_ORDER
        colors = RISK_COLORS if family == "risk" else QUAL_COLORS
        sev = _clamp_sev(order, sev)
        color = colors.get(sev, ACCENT_GREY)

        prev = self._panel_severity.get(key, SEV_UNKNOWN)
        self._panel_severity[key] = sev

        p = self.panels.get(key)
        if not p:
            return

        p.set_accent(color, blink=False)

        if family == "risk":
            try:
                prev_i = _sev_index(RISK_ORDER, str(prev))
                now_i = _sev_index(RISK_ORDER, sev)
                danger_i = _sev_index(RISK_ORDER, SEV_DANGER)

                entered_danger = (now_i >= danger_i) and (prev_i < danger_i)
                if entered_danger:
                    play_alert_sound(self)

                if blink_on_danger and entered_danger:
                    p.start_blink(300, border_only=True)
            except Exception:
                pass

    def _risk_kp_sev(self, kp: float) -> str:
        if _is_nan(kp):
            return SEV_UNKNOWN
        # Hysteresis around thresholds to avoid yo-yo.
        prev = self._panel_severity.get("KP", SEV_UNKNOWN)
        # thresholds up
        if kp >= 8.0:
            sev = SEV_EXTREME
        elif kp >= 7.0:
            sev = SEV_DANGER
        elif kp >= 5.0:
            sev = SEV_ALERT
        elif kp >= 4.0:
            sev = SEV_WATCH
        else:
            sev = SEV_OK

        # thresholds down (slightly lower to come back down)
        if prev == SEV_EXTREME and kp <= 7.7:
            sev = SEV_DANGER
        if prev == SEV_DANGER and kp <= 6.7:
            sev = SEV_ALERT
        if prev == SEV_ALERT and kp <= 4.7:
            sev = SEV_WATCH if kp >= 3.8 else SEV_OK
        if prev == SEV_WATCH and kp <= 3.8:
            sev = SEV_OK
        return sev

    def _risk_sw_sev(self, v: float) -> str:
        if _is_nan(v):
            return SEV_UNKNOWN
        prev = self._panel_severity.get("SW", SEV_UNKNOWN)
        if v >= 800:
            sev = SEV_EXTREME
        elif v >= 700:
            sev = SEV_DANGER
        elif v >= 600:
            sev = SEV_ALERT
        elif v >= 500:
            sev = SEV_WATCH
        else:
            sev = SEV_OK

        # Hysteresis (km/s)
        if prev == SEV_EXTREME and v <= 780:
            sev = SEV_DANGER
        if prev == SEV_DANGER and v <= 680:
            sev = SEV_ALERT
        if prev == SEV_ALERT and v <= 580:
            sev = SEV_WATCH if v >= 480 else SEV_OK
        if prev == SEV_WATCH and v <= 480:
            sev = SEV_OK
        return sev

    def _risk_xray_sev(self, flare_cls: str) -> str:
        c = (str(flare_cls or "?").strip().upper() or "?")
        if c == "X":
            return SEV_DANGER
        if c == "M":
            return SEV_ALERT
        if c == "C":
            return SEV_WATCH
        if c in ("A", "B"):
            return SEV_OK
        return SEV_UNKNOWN

    def _risk_cme_sev(self, level: str) -> str:
        s = (str(level or "N/A").strip().upper())
        if s in ("N/A", "NA", "NONE", ""):
            return SEV_UNKNOWN
        if s == "HIGH":
            return SEV_DANGER
        if s == "MEDIUM":
            return SEV_ALERT
        if s == "WATCH":
            return SEV_WATCH
        if s == "LOW":
            return SEV_OK
        # fallback: try to infer
        if "HIGH" in s:
            return SEV_DANGER
        if "MED" in s:
            return SEV_ALERT
        if "WAT" in s:
            return SEV_WATCH
        if "LOW" in s:
            return SEV_OK
        return SEV_UNKNOWN

    def _risk_p10_sev(self, s_level: str) -> str:
        s = (str(s_level or "S0").strip().upper())
        if s.startswith("S5"):
            return SEV_EXTREME
        if s.startswith("S4"):
            return SEV_DANGER
        if s.startswith("S3") or s.startswith("S2"):
            return SEV_ALERT
        if s.startswith("S1"):
            return SEV_WATCH
        if s.startswith("S0"):
            return SEV_OK
        return SEV_UNKNOWN

    def _risk_aur_sev(self, aur_max_pct: float) -> str:
        if _is_nan(aur_max_pct):
            return SEV_UNKNOWN
        v = float(aur_max_pct)
        if v >= 80:
            return SEV_EXTREME
        if v >= 60:
            return SEV_DANGER
        if v >= 40:
            return SEV_ALERT
        if v >= 20:
            return SEV_WATCH
        return SEV_OK

    def _risk_bzbt_sev(self, bz: float, bt: float) -> str:
        if _is_nan(bz):
            return SEV_UNKNOWN
        v = float(bz)
        # Only negative Bz is geoeffective.
        if v <= -15:
            sev = SEV_DANGER
        elif v <= -10:
            sev = SEV_ALERT
        elif v <= -5:
            sev = SEV_WATCH
        else:
            sev = SEV_OK

        # If Bt is weak, downgrade one step (less coupling potential).
        try:
            if not _is_nan(bt) and float(bt) < 5.0 and sev not in (SEV_UNKNOWN, SEV_OK):
                sev = _downgrade_sev(RISK_ORDER, sev, 1)
        except Exception:
            pass
        return sev

    def _qual_sfi_sev(self, sfi: float) -> str:
        if _is_nan(sfi):
            return SEV_UNKNOWN
        v = float(sfi)
        if v >= 160:
            return SEV_EXCELLENT
        if v >= 120:
            return SEV_GOOD
        if v >= 90:
            return SEV_FAIR
        return SEV_POOR

    def _qual_ssn_sev(self, ssn: float) -> str:
        if _is_nan(ssn):
            return SEV_UNKNOWN
        v = float(ssn)
        if v >= 150:
            return SEV_EXCELLENT
        if v >= 80:
            return SEV_GOOD
        if v >= 40:
            return SEV_FAIR
        return SEV_POOR

    
    def _refresh_cme_from_db(self):
        series = db_load_cme_series(DB_PATH, limit=60)
        last = db_load_cme_last(DB_PATH)

        if last is None:
            level = "N/A"
            self.panels["CME"].lbl_title.setText("CME ARRIVAL PROBABILITY [N/A]")
            self._style_panel_title_default(self.panels["CME"])
            self.panels["CME"].lbl_title.setToolTip(
                "NASA DONKI: no cached CME data yet.\n"
                "Display: X axis is a rolling 6-hour window (evenly spaced points)."
            )
            # Coherent severity coloring
            self._set_panel_severity("CME", "risk", SEV_UNKNOWN, blink_on_danger=False)

            now = time.time()
            x = np.linspace(now - 60 * 60 * 6, now, num=max(2, len(series) if series.size else 2))
            y = series if series.size else np.array([np.nan, np.nan], dtype=float)
            self.panels["CME"].set_data(y, float("nan"), y_range=(0, 100), color=RISK_COLORS[SEV_UNKNOWN], x=x)
            return

        prob, level, ts_utc = last
        eta_txt = ""
        try:
            eta_raw = str(getattr(self, "_last_cme_eta", "") or "").strip()
            eta_epoch = parse_time_to_epoch(eta_raw) if eta_raw else float("nan")
            if not (isinstance(eta_epoch, float) and math.isnan(eta_epoch)):
                eta_txt = datetime.fromtimestamp(float(eta_epoch), timezone.utc).strftime(" ETA %H:%M UTC")
        except Exception:
            eta_txt = ""
        self.panels["CME"].lbl_title.setText(f"CME ARRIVAL PROBABILITY [{level}]{eta_txt}")
        self._style_panel_title_default(self.panels["CME"])
        self.panels["CME"].lbl_title.setToolTip(
            f"""DONKI last update: {ts_utc}
ETA: {str(getattr(self, "_last_cme_eta", "") or "--")}
Heuristic: Earth mention + analyses + ENLIL/impactList
Optimized: age-weighted (~5 days) + 'likely/may' nuance
Display: X axis is a rolling 6-hour window (evenly spaced points; DONKI event times are in the tooltip header)
Note: this is an index, not a physical probability."""
        )

        sev = self._risk_cme_sev(level)
        self._set_panel_severity("CME", "risk", sev, blink_on_danger=True)
        accent = RISK_COLORS.get(sev, ACCENT_GREY)

        now = time.time()
        x = np.linspace(now - 60 * 60 * 6, now, num=max(2, len(series) if series.size else 2))
        y = series if series.size else np.array([np.nan, np.nan], dtype=float)
        self.panels["CME"].set_data(y, prob, y_range=(0, 100), color=accent, x=x)


    @QtCore.Slot(str)
    def _on_rss_text(self, text: str):
        self.rss_ticker.setText(text)

    @QtCore.Slot(str)
    def _on_rss_error(self, msg: str):
        self._dbg(f"RSS error handler: {msg}")
        self.rss_ticker.setText(f"RSS error: {msg}")
        if not getattr(self, "_closing", False):
            self._schedule_worker_restart("rss_worker", reason=f"rss_error:{msg}", delay_ms=5000)


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
            f"Latest activity peak: {d.get('xray_latest_peak_label', '?')}\n"
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
        if "cme_eta" in d:
            try:
                self._last_cme_eta = str(d.get("cme_eta") or "").strip()
            except Exception:
                self._last_cme_eta = ""

        state, details = self._solar_data_status_text(d)
        self._set_data_status(state, details)

        kp_now = d.get("kp_now", float("nan"))
        kp_g = kp_g_scale(kp_now)

        # Coherent severity for Kp (RISK family)
        kp_sev = self._risk_kp_sev(kp_now)
        self._set_panel_severity("KP", "risk", kp_sev, blink_on_danger=True)
        kp_accent = RISK_COLORS.get(kp_sev, ACCENT_GREY)

        self.panels["KP"].lbl_title.setText("K-INDEX (Planetary)")
        self._style_panel_title_default(self.panels["KP"])
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

        xray_sev = self._risk_xray_sev(flare_cls)
        self._set_panel_severity("XRAY", "risk", xray_sev, blink_on_danger=True)
        xray_accent = RISK_COLORS.get(xray_sev, ACCENT_GREY)

        self.panels["XRAY"].lbl_title.setText("X-RAYS (GOES 0.1-0.8nm)")
        self._style_panel_title_default(self.panels["XRAY"])
        self.panels["XRAY"].set_data(
            d.get("xray_series", np.array([])),
            xray_now,
            y_range=(-8.2, -3.8),
            color=xray_accent,
            x=d.get("xray_x", None)
        )

        if isinstance(xray_now, float) and math.isnan(xray_now):
            self.panels["XRAY"].set_big_text("--")
        else:
            # V7.8: radio-friendly display. The raw flux remains in the tooltip/logs;
            # the panel shows the useful GOES flare class only.
            self.panels["XRAY"].lbl_big.setStyleSheet("")
            self.panels["XRAY"].lbl_big.setText(
                f'<span style="color:{xray_accent}; font-weight:900;">{flare_label}</span>'
            )

        self.panels["SW"].lbl_title.setText("SOLAR WIND SPEED")
        self._style_panel_title_default(self.panels["SW"])
        sw_now = d.get("sw_speed_now", float("nan"))

        sw_sev = self._risk_sw_sev(sw_now)
        self._set_panel_severity("SW", "risk", sw_sev, blink_on_danger=True)
        sw_accent = RISK_COLORS.get(sw_sev, ACCENT_GREY)

        self.panels["SW"].set_data(
            d.get("sw_speed_series", np.array([])),
            sw_now,
            y_range=(0, 1200),
            color=sw_accent,
            x=d.get("sw_speed_x", None),
        )

        ssn_now = d.get("ssn_now", float("nan"))
        ssn_sev = self._qual_ssn_sev(ssn_now)
        self._set_panel_severity("SSN", "quality", ssn_sev, blink_on_danger=False)
        ssn_accent = QUAL_COLORS.get(ssn_sev, ACCENT_GREY)

        self.panels["SSN"].lbl_title.setText("SSN (Sunspot Number) daily index")
        self._style_panel_title_default(self.panels["SSN"])
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

        sfi_sev = self._qual_sfi_sev(sfi_now)
        self._set_panel_severity("SFI", "quality", sfi_sev, blink_on_danger=False)
        sfi_accent = QUAL_COLORS.get(sfi_sev, ACCENT_GREY)

        self.panels["SFI"].lbl_title.setText("SFI (F10.7 cm Flux) daily index")
        self._style_panel_title_default(self.panels["SFI"])
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
        bzbt_sev = self._risk_bzbt_sev(bz_now, bt_now)
        self._set_panel_severity("BZBT", "risk", bzbt_sev, blink_on_danger=True)

        if (isinstance(bz_now, float) and math.isnan(bz_now)) and (isinstance(bt_now, float) and math.isnan(bt_now)):
            p.set_big_text("--")
        else:
            bz_txt = "--" if (isinstance(bz_now, float) and math.isnan(bz_now)) else f"{bz_now:.1f}"
            bt_txt = "--" if (isinstance(bt_now, float) and math.isnan(bt_now)) else f"{bt_now:.1f}"
            # V7.8: Bz drives the visual priority, because southward Bz is the key HF/aurora risk signal.
            if isinstance(bz_now, float) and math.isnan(bz_now):
                bz_color = ACCENT_GREY
            elif bz_now <= -10:
                bz_color = STATUS_ERR
            elif bz_now <= -5:
                bz_color = "#ff9f43"
            elif bz_now <= 0:
                bz_color = STATUS_WARN
            else:
                bz_color = ACCENT_GREEN
            p.lbl_big.setStyleSheet("")
            p.lbl_big.setText(
                f'<span style="color:{bz_color}; font-weight:900;">Bz {bz_txt} nT</span>'
                f'&nbsp;&nbsp;<span style="color:#aab6c5; font-weight:700;">|</span>&nbsp;&nbsp;'
                f'<span style="color:{ACCENT_BLUE}; font-weight:800;">Bt {bt_txt}</span>'
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
        p10_sev = self._risk_p10_sev(s_level)
        self._set_panel_severity("P10", "risk", p10_sev, blink_on_danger=True)
        p_accent = RISK_COLORS.get(p10_sev, ACCENT_GREY)

        self.panels["P10"].lbl_title.setText("PROTONS P10 (>=10MeV)")
        self._style_panel_title_default(self.panels["P10"])
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
        aur_sev = self._risk_aur_sev(aur_now)
        aur_accent = RISK_COLORS.get(aur_sev, ACCENT_GREY)
        self._set_panel_severity("AUR", "risk", aur_sev, blink_on_danger=True)

        # Tooltip: show the 3 metrics (Max / Mean(active>0) / Area>10)
        tip_parts = []
        if not (isinstance(aur_now, float) and math.isnan(aur_now)):
            tip_parts.append(f"Max: {aur_now:.1f}%")
        if not (isinstance(aur_mean, float) and math.isnan(aur_mean)):
            tip_parts.append(f"Mean(active>0): {aur_mean:.1f}%")
        if not (isinstance(aur_area, float) and math.isnan(aur_area)):
            tip_parts.append(f"Area(cells>10): {aur_area:.1f}%")
        p_aur.lbl_title.setToolTip("Aurora" + (" • " + " • ".join(tip_parts) if tip_parts else ""))

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
        # V7.8: keep the tile readable. Max is the decision value; Mean/Area stay visible in the tooltip.
        p_aur.lbl_big.setText(
            f'<span style="color:{aur_accent}; font-size:24px; font-weight:900;">Max {_fmt_pct(aur_now)}</span>'
        )
        p_aur.lbl_big.setToolTip(
            f"Aurora details\nMean(active>0): {_fmt_pct(aur_mean)}\nArea(cells>10): {_fmt_pct(aur_area)}"
        )

        if "cme_prob" in d and "cme_level" in d:
            self._refresh_cme_from_db()

        self._dapnet_last_payload = dict(d or {})
        self._process_dapnet_from_payload(self._dapnet_last_payload)

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

    def _solar_data_status_text(self, payload: dict) -> tuple[str, str]:
        """Solar data status only. ISS TLE is displayed in its own status label."""
        pe = str(payload.get("partial_error") or "").strip()
        categories = [
            ("KP", "KP"),
            ("MAG", "MAG"),
            ("SW", "SW"),
            ("XRAY", "X-ray"),
            ("P10", "P10"),
            ("SSN", "SSN"),
            ("SFI", "SFI"),
            ("AUR", "AUR"),
            ("CME", "CME"),
        ]
        errors = {key: (f"{key}:" in pe) for key, _ in categories}
        statuses = [
            f"{label} ERR" if errors[key] else f"{label} OK"
            for key, label in categories
        ]
        if any(errors.values()):
            return "partial", " • ".join(statuses)
        return "ok", " • ".join(statuses)


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

# =====================================================
# DAPNET runtime glue for MainWindow
# =====================================================
def _dapnet_parse_iso8601(value: str | None) -> datetime | None:
    try:
        if not value:
            return None
        s = str(value).strip()
        if not s:
            return None
        if s.endswith('Z'):
            s = s[:-1] + '+00:00'
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _dapnet_now_utc(self) -> datetime:
    return datetime.now(timezone.utc)


def _dapnet_format_local(self, dt_utc: datetime | None) -> str:
    try:
        if dt_utc is None:
            return '--:--'
        tz = getattr(self, '_dapnet_local_tz', timezone.utc) or timezone.utc
        return dt_utc.astimezone(tz).strftime('%H:%M')
    except Exception:
        try:
            return dt_utc.astimezone(timezone.utc).strftime('%H:%M UTC') if dt_utc else '--:--'
        except Exception:
            return '--:--'


def _dapnet_xray_rank(label: str | None) -> float:
    try:
        s = str(label or '').strip().upper()
        m = re.match(r'^([ABCMX])\s*([0-9]+(?:\.[0-9]+)?)$', s)
        if not m:
            return -1.0
        letter = m.group(1)
        mag = float(m.group(2))
        base = {'A':0,'B':10,'C':20,'M':30,'X':40}.get(letter, -10)
        return base + mag
    except Exception:
        return -1.0


def _dapnet_s_rank(level: str | None) -> int:
    try:
        s = str(level or '').strip().upper()
        m = re.match(r'^S([0-5])$', s)
        return int(m.group(1)) if m else -1
    except Exception:
        return -1


def _dapnet_send_message(self, module: str, text: str, *, emergency: bool = False, state_key: str | None = None) -> tuple[bool, str]:
    try:
        ok, resp = self._dapnet_client.send_message(text, emergency=emergency)
    except Exception as exc:
        ok, resp = False, str(exc)
    try:
        self._dbg(f'DAPNET {module}: {"OK" if ok else "FAIL"} • {text} • {resp}')
    except Exception:
        pass
    if ok:
        try:
            self._dapnet_state['last_success_utc'] = _dapnet_now_utc(self).isoformat()
            if state_key:
                self._dapnet_state[state_key] = self._dapnet_state['last_success_utc']
            save_dapnet_state(DAPNET_STATE_PATH, self._dapnet_state)
        except Exception:
            pass
    return ok, resp


def _dapnet_compute_next_iss_pass(self, now_utc: datetime | None = None) -> dict | None:
    now_utc = now_utc or _dapnet_now_utc(self)
    try:
        cfg = normalize_dapnet_config(getattr(self, '_dapnet_cfg', {})).get('iss', {})
        min_elev = float(cfg.get('min_elevation', 5.0))
        sat = getattr(self, '_iss_satellite', None)
        if sat is None or self._skyfield_ts is None or wgs84 is None:
            return None
        observer = wgs84.latlon(float(cfg.get('observer_lat', 48.1173)), float(cfg.get('observer_lon', -1.6778)), elevation_m=float(cfg.get('observer_alt_m', 60)))
        step_seconds = 60
        total_seconds = 12 * 3600
        started = False
        start_dt = None
        end_dt = None
        peak_dt = None
        peak_el = -999.0
        tdt = now_utc
        prev_above = False
        for offset in range(0, total_seconds + step_seconds, step_seconds):
            if offset == 0:
                tdt = now_utc
            else:
                tdt = now_utc + timedelta(seconds=offset)
            t = self._skyfield_ts.from_datetime(tdt)
            difference = sat - observer
            topocentric = difference.at(t)
            alt, az, distance = topocentric.altaz()
            elev = float(alt.degrees)
            above = elev >= min_elev
            if above and not started:
                started = True
                start_dt = tdt
                peak_dt = tdt
                peak_el = elev
            if started:
                if elev > peak_el:
                    peak_el = elev
                    peak_dt = tdt
                if (not above) and prev_above:
                    end_dt = tdt
                    break
            prev_above = above
        if not started or start_dt is None:
            return None
        if end_dt is None:
            end_dt = (peak_dt or start_dt) + timedelta(minutes=8)
        prealert_dt = start_dt - timedelta(minutes=max(1, int(cfg.get('prealert_minutes', 15))))
        return {
            'start_utc': start_dt.isoformat(),
            'peak_utc': (peak_dt or start_dt).isoformat(),
            'end_utc': end_dt.isoformat(),
            'prealert_utc': prealert_dt.isoformat(),
            'peak_elev_deg': round(float(peak_el), 1),
        }
    except Exception as exc:
        try:
            self._dbg(f'DAPNET ISS compute failed: {exc}')
        except Exception:
            pass
        return None


def _dapnet_check_iss(self, now_utc: datetime) -> None:
    cfg = normalize_dapnet_config(getattr(self, '_dapnet_cfg', {})).get('iss', {})
    if not cfg.get('enabled', False):
        return
    state = getattr(self, '_dapnet_state', {})
    pass_info = state.get('iss_next_pass') if isinstance(state.get('iss_next_pass'), dict) else None
    end_dt = _dapnet_parse_iso8601((pass_info or {}).get('end_utc'))
    if pass_info is None or end_dt is None or now_utc >= end_dt + timedelta(minutes=1):
        pass_info = _dapnet_compute_next_iss_pass(self, now_utc)
        state['iss_next_pass'] = pass_info or {}
        save_dapnet_state(DAPNET_STATE_PATH, state)
    if not pass_info:
        return

    start_dt = _dapnet_parse_iso8601(pass_info.get('start_utc'))
    peak_dt = _dapnet_parse_iso8601(pass_info.get('peak_utc'))
    end_dt = _dapnet_parse_iso8601(pass_info.get('end_utc'))
    prealert_dt = _dapnet_parse_iso8601(pass_info.get('prealert_utc'))
    pass_id = pass_info.get('start_utc') or 'unknown'
    sent = state.setdefault('iss_sent', {})
    peak_el = float(pass_info.get('peak_elev_deg', 0.0) or 0.0)

    def _maybe(flag_name: str, enabled: bool, when_dt: datetime | None, message: str):
        if not enabled or when_dt is None:
            return
        key = f'{pass_id}|{flag_name}'
        if sent.get(key):
            return
        if now_utc >= when_dt:
            ok, _ = _dapnet_send_message(self, f'ISS/{flag_name}', message)
            if ok:
                sent[key] = True
                save_dapnet_state(DAPNET_STATE_PATH, state)

    _maybe('prealert', bool(cfg.get('prealert_enabled', True)), prealert_dt,
           f'ISS soon AOS {_dapnet_format_local(self, start_dt)} max {peak_el:.0f}d')
    _maybe('start', bool(cfg.get('start_enabled', True)), start_dt,
           f'ISS AOS now peak {peak_el:.0f}d LOS {_dapnet_format_local(self, end_dt)}')
    _maybe('peak', bool(cfg.get('peak_enabled', True)), peak_dt,
           f'ISS peak now {peak_el:.0f}d over horizon')
    _maybe('end', bool(cfg.get('end_enabled', True)), end_dt,
           'ISS LOS now')


def _process_dapnet_from_payload(self, payload: dict | None) -> None:
    payload = dict(payload or {})
    cfg = normalize_dapnet_config(getattr(self, '_dapnet_cfg', {}))
    if not cfg.get('enabled', False):
        return
    now_utc = _dapnet_now_utc(self)
    state = getattr(self, '_dapnet_state', {})

    xcfg = cfg.get('xray', {})
    if xcfg.get('enabled', False):
        current_label = str(payload.get('xray_label') or payload.get('xray_class') or '').strip().upper()
        current_rank = _dapnet_xray_rank(current_label)
        threshold_label = str(xcfg.get('threshold') or 'M1.0').strip().upper()
        threshold_rank = _dapnet_xray_rank(threshold_label)
        active = bool(current_rank >= threshold_rank >= 0)
        prev_active = bool(state.get('xray_active', False))
        state['xray_active'] = active
        state['xray_last_label'] = current_label
        if active and not prev_active and bool(xcfg.get('send_start', True)):
            emergency = bool(xcfg.get('emergency_on_start', True))
            _dapnet_send_message(self, 'XRAY/start', f'XRAY {current_label} started thr {threshold_label}', emergency=emergency)
        elif (not active) and prev_active and bool(xcfg.get('send_end', True)):
            _dapnet_send_message(self, 'XRAY/end', f'XRAY ended last {state.get("xray_last_sent_label") or state.get("xray_last_label") or threshold_label}')
        if active:
            state['xray_last_sent_label'] = current_label

    pcfg = cfg.get('proton', {})
    if pcfg.get('enabled', False):
        current_s = str(payload.get('s_level') or proton_s_scale(float(payload.get('p10_now', float('nan')))) ).strip().upper()
        current_rank = _dapnet_s_rank(current_s)
        threshold_s = str(pcfg.get('threshold') or 'S1').strip().upper()
        threshold_rank = _dapnet_s_rank(threshold_s)
        if current_rank >= threshold_rank >= 0:
            last_dt = _dapnet_parse_iso8601(state.get('proton_last_alert_utc'))
            cooldown = max(5, int(pcfg.get('cooldown_minutes', 30)))
            if last_dt is None or (now_utc - last_dt) >= timedelta(minutes=cooldown):
                p10 = payload.get('p10_now', float('nan'))
                msg = f'PROTON {current_s} p10 {p10:.1f} PFU' if isinstance(p10, (int, float)) and not math.isnan(float(p10)) else f'PROTON {current_s}'
                if bool(pcfg.get('include_bz_bt', True)):
                    bz = payload.get('bz_now', float('nan'))
                    bt = payload.get('bt_now', float('nan'))
                    if isinstance(bz, (int, float)) and not math.isnan(float(bz)):
                        msg += f' Bz {float(bz):+.1f}'
                    if isinstance(bt, (int, float)) and not math.isnan(float(bt)):
                        msg += f' Bt {float(bt):.1f}'
                ok, _ = _dapnet_send_message(self, 'PROTON', msg[:80])
                if ok:
                    state['proton_last_alert_utc'] = now_utc.isoformat()

    scfg = cfg.get('solar_summary', {})
    if scfg.get('enabled', False):
        last_dt = _dapnet_parse_iso8601(state.get('solar_summary_last_utc'))
        interval = max(5, int(scfg.get('interval_minutes', 60)))
        if last_dt is None or (now_utc - last_dt) >= timedelta(minutes=interval):
            kp = payload.get('kp_now', float('nan'))
            xray = str(payload.get('xray_label') or payload.get('xray_class') or '?')
            sfi = payload.get('sfi_now', float('nan'))
            ssn = payload.get('ssn_now', float('nan'))
            sw = payload.get('sw_speed_now', float('nan'))
            aind = payload.get('dsi_x', '')
            parts = [f'SOLAR X {xray}']
            if isinstance(kp, (int, float)) and not math.isnan(float(kp)):
                parts.append(f'Kp {float(kp):.1f}')
            if isinstance(sw, (int, float)) and not math.isnan(float(sw)):
                parts.append(f'SW {float(sw):.0f}')
            if isinstance(sfi, (int, float)) and not math.isnan(float(sfi)):
                parts.append(f'SFI {float(sfi):.0f}')
            if isinstance(ssn, (int, float)) and not math.isnan(float(ssn)):
                parts.append(f'SSN {float(ssn):.0f}')
            a_txt = _dapnet_sanitize_a_index(aind)
            if a_txt is not None:
                parts.append(f'A {a_txt}')
            ok, _ = _dapnet_send_message(self, 'SOLAR', _dapnet_compact_text(parts))
            if ok:
                state['solar_summary_last_utc'] = now_utc.isoformat()

    _dapnet_check_iss(self, now_utc)
    save_dapnet_state(DAPNET_STATE_PATH, state)


def _dapnet_periodic_tick(self) -> None:
    try:
        payload = getattr(self, '_dapnet_last_payload', {}) or {}
        # Run DAPNET processing in background thread to avoid blocking UI
        self._dapnet_executor.submit(_process_dapnet_from_payload, self, payload)
    except Exception as exc:
        try:
            self._dbg(f'DAPNET periodic tick failed: {exc}')
        except Exception:
            pass


MainWindow._dapnet_periodic_tick = _dapnet_periodic_tick
MainWindow._process_dapnet_from_payload = _process_dapnet_from_payload
MainWindow._dapnet_compute_next_iss_pass = _dapnet_compute_next_iss_pass
MainWindow._dapnet_check_iss = _dapnet_check_iss
MainWindow._dapnet_send_message = _dapnet_send_message


def _mainwin_send_dapnet_quick_message(self, callsign: str, text: str, tx_group: str | None = None, emergency: bool = False) -> tuple[bool, str]:
    try:
        self._dbg(f"DAPNET QUICK CALLED raw_to={callsign} raw_text={text} raw_group={tx_group} emergency={emergency}")
    except Exception:
        pass

    callsign = str(callsign or "").strip().upper()
    text = str(text or "").strip()[:80]
    tx_group = str(tx_group or self._dapnet_cfg.get("tx_group") or "f-53").strip() or "f-53"

    if not callsign:
        return False, "Missing callsign"
    if not text:
        return False, "Empty message"

    cfg = normalize_dapnet_config(getattr(self, "_dapnet_cfg", {}))

    try:
        self._dapnet_client.update_config(cfg)
        client = self._dapnet_client
    except Exception:
        client = DapnetClient(cfg, logger=self._dbg)
        self._dapnet_client = client

    ok, msg = client.send_message(text, emergency=emergency, force=True, recipients=[callsign], tx_group=tx_group)

    try:
        self._dbg(f"DAPNET QUICK RESULT {'OK' if ok else 'FAIL'} • {callsign} • {text} • TX={tx_group} • EMERGENCY={emergency} • {msg}")
    except Exception:
        pass

    return ok, str(msg)


def _mainwin_refresh_dapnet_quick_bar(self):
    try:
        if hasattr(self, "dapnet_quick_bar") and self.dapnet_quick_bar is not None:
            self.dapnet_quick_bar._update_state()
    except Exception:
        pass


MainWindow._send_dapnet_quick_message = _mainwin_send_dapnet_quick_message
MainWindow._refresh_dapnet_quick_bar = _mainwin_refresh_dapnet_quick_bar



def _mainwin_init_skyfield_resources(self):
    try:
        if not SKYFIELD_AVAILABLE:
            self._skyfield_loader = None
            self._skyfield_eph = None
            self._skyfield_ts = None
            self._skyfield_eph_source = ""
            self._dbg(f"Skyfield unavailable at import: {SKYFIELD_IMPORT_ERROR}")
            return False
        self._skyfield_loader, self._skyfield_eph, self._skyfield_eph_source = get_skyfield_loader_and_ephemeris()
        try:
            self._skyfield_ts = self._skyfield_loader.timescale()
        except Exception as exc:
            self._dbg(f"Skyfield timescale init failed: {exc}")
            self._skyfield_ts = None
        self._dbg(f"Skyfield ephemeris ready: {self._skyfield_eph_source}")
        return True
    except Exception as exc:
        self._skyfield_loader = None
        self._skyfield_eph = None
        self._skyfield_ts = None
        self._skyfield_eph_source = ""
        self._dbg(f"Skyfield init failed: {exc}")
        return False


MainWindow._init_skyfield_resources = _mainwin_init_skyfield_resources


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

