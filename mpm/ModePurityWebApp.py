"""
ModePurityWebApp.py

Browser-based GUI for thermogram preprocessing, beam-tilt analysis,
phase retrieval, and LP mode-purity evaluation.

Run:
    python3 ModePurityWebApp.py
"""
from __future__ import annotations

import base64
import io
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.path import Path as MplPath

try:
    import pandas as pd
except ModuleNotFoundError as exc:
    raise SystemExit(
        "pandas is required. Install with: python3 -m pip install pandas"
    ) from exc

try:
    from scipy.special import jn, jn_zeros
except ModuleNotFoundError as exc:
    raise SystemExit(
        "scipy is required. Install with: python3 -m pip install scipy"
    ) from exc

from AngularSpectrumFFT import propagate_angular_spectrum
from ThermogramProcessor import initial_cleanup, read_ppf_corners

try:
    import cv2
except ModuleNotFoundError as exc:
    raise SystemExit(
        "opencv-python is required. Install with: python3 -m pip install opencv-python"
    ) from exc

try:
    from nicegui import run, ui
except ModuleNotFoundError as exc:
    raise SystemExit(
        "nicegui is required. Install with: python3 -m pip install nicegui"
    ) from exc


@dataclass
class CsvMeta:
    path: Path
    z_mm: float
    frame: int | None
    stamp12: str | None
    is_bg: bool


@dataclass
class ThermogramEntry:
    beam_path: Path
    bg_path: Path
    ppf_path: Path
    z_mm: float
    frame: int | None
    stamp12: str | None
    delta_t_pt: np.ndarray
    delta_t_pt_raw: np.ndarray
    centroid_x_mm: float
    centroid_y_mm: float
    sigma_x_mm: float
    sigma_y_mm: float
    centroid_x_mm_raw: float
    centroid_y_mm_raw: float
    sigma_x_mm_raw: float
    sigma_y_mm_raw: float
    preview_url: str


@dataclass
class PhaseResult:
    u_pr: np.ndarray
    u_eval: np.ndarray
    z_eval_mm: float
    err_history: np.ndarray
    iterations_done: int
    best_iteration: int
    best_error: float
    stop_reason: str
    axis_purity: dict[str, float]
    axis_center_mm: tuple[float, float]
    output_dir: Path


Z_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*mm", re.IGNORECASE)
FRAME_RE = re.compile(r"_(\d+)f(?:_BG)?$", re.IGNORECASE)
STAMP12_RE = re.compile(r"_(\d{12})(?:_|$)")
DATE8_RE = re.compile(r"(20\d{6})")
PPF_Z_STAMP_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)mm_(\d{12})", re.IGNORECASE)
PPF_DATE_ONLY_RE = re.compile(r"^(20\d{6})$", re.IGNORECASE)

MAJOR_MODES = {
    "LP0,1",
    "LP0,2",
    "LP0,3",
    "LP1,1even",
    "LP1,1odd",
    "LP2,1even",
    "LP2,1odd",
    "LP1,2even",
    "LP1,2odd",
}


class AppState:
    entries: list[ThermogramEntry] = []
    selected_entry_keys: set[str] = set()
    load_notes: list[str] = []
    tilt_info: dict[str, float] | None = None
    tilt_plot_url: str | None = None
    progress: float = 0.0
    status: str = "Idle"
    running: bool = False
    last_error: float | None = None
    phase_result: PhaseResult | None = None
    mode_table_rows: list[dict[str, Any]] = []
    axis_sum_percent: float = 0.0
    tilt_comp_enabled: bool = False


STATE = AppState()


def entry_key(entry: ThermogramEntry) -> str:
    return str(entry.beam_path.resolve())


def parse_csv_meta(path: Path) -> CsvMeta | None:
    name = path.stem
    m_z = Z_RE.search(name)
    if not m_z:
        return None
    z_mm = float(m_z.group(1))

    m_frame = FRAME_RE.search(name)
    frame = int(m_frame.group(1)) if m_frame else None

    m_stamp = STAMP12_RE.search(name)
    stamp12 = m_stamp.group(1) if m_stamp else None

    is_bg = name.upper().endswith("_BG")
    return CsvMeta(path=path, z_mm=z_mm, frame=frame, stamp12=stamp12, is_bg=is_bg)


def score_bg(pattern: CsvMeta, bg: CsvMeta) -> tuple[int, int, int]:
    ts_mismatch = 0
    if pattern.stamp12 is not None and bg.stamp12 is not None:
        ts_mismatch = 0 if pattern.stamp12 == bg.stamp12 else 1

    if pattern.frame is not None and bg.frame is not None:
        frame_delta = abs(pattern.frame - bg.frame)
        after_penalty = 1 if bg.frame > pattern.frame else 0
    else:
        frame_delta = 10**9
        after_penalty = 1
    return (ts_mismatch, frame_delta, after_penalty)


def choose_ppf_for(pattern: CsvMeta, ppf_files: list[Path]) -> Path | None:
    if not ppf_files:
        return None

    date8 = pattern.stamp12[:8] if pattern.stamp12 else None
    best: tuple[int, Path] | None = None

    for ppf in ppf_files:
        stem = ppf.stem
        score = 10

        m = PPF_Z_STAMP_RE.search(stem)
        if m:
            z2 = float(m.group(1))
            stamp2 = m.group(2)
            if math.isclose(z2, pattern.z_mm, abs_tol=1e-9) and pattern.stamp12 == stamp2:
                score = 0
            elif math.isclose(z2, pattern.z_mm, abs_tol=1e-9):
                score = 2
            elif date8 and stamp2.startswith(date8):
                score = 4
        elif PPF_DATE_ONLY_RE.search(stem):
            if date8 and stem == date8:
                score = 1
            else:
                score = 5
        else:
            m_date = DATE8_RE.search(stem)
            if m_date and date8 and m_date.group(1) == date8:
                score = 3
            elif m_date:
                score = 6

        if best is None or score < best[0]:
            best = (score, ppf)

    return best[1] if best else ppf_files[0]


def format_stamp(stamp12: str | None) -> str:
    if stamp12 is None or len(stamp12) != 12:
        return "-"
    return f"{stamp12[0:4]}-{stamp12[4:6]}-{stamp12[6:8]} {stamp12[8:10]}:{stamp12[10:12]}"


def order_corners(corners: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = corners.sum(axis=1)
    diff = np.diff(corners, axis=1)
    rect[0] = corners[np.argmin(s)]      # UL
    rect[2] = corners[np.argmax(s)]      # LR
    rect[1] = corners[np.argmin(diff)]   # UR
    rect[3] = corners[np.argmax(diff)]   # LL
    return rect


def moment_stats(img: np.ndarray, size_mm: float) -> tuple[float, float, float, float]:
    n, m = img.shape
    px = size_mm / n  # vertical pixel size (rows)
    py = size_mm / m  # horizontal pixel size (cols)
    # x = horizontal (axis 1, columns, right = positive)
    # y = vertical   (axis 0, rows,    up   = positive → row 0 が上端なので符号反転)
    x = (np.arange(m) - (m - 1) / 2.0) * py
    y = ((n - 1) / 2.0 - np.arange(n)) * px   # 上端 → 正、下端 → 負
    Y, X = np.meshgrid(y, x, indexing="ij")  # Y along axis 0 (rows), X along axis 1 (cols)

    w = np.clip(img.astype(np.float64), 0.0, None)
    s = w.sum()
    if s <= 0:
        nan = float("nan")
        return nan, nan, nan, nan

    cx = float((w * X).sum() / s)
    cy = float((w * Y).sum() / s)
    sig_x = float(np.sqrt(np.clip((w * (X - cx) ** 2).sum() / s, 0.0, None)))
    sig_y = float(np.sqrt(np.clip((w * (Y - cy) ** 2).sum() / s, 0.0, None)))
    return cx, cy, sig_x, sig_y


def array_to_data_url(data: np.ndarray) -> str:
    arr = np.clip(data.astype(np.float64), 0.0, None)
    vmax = np.percentile(arr, 99.5)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    norm = np.clip(arr / vmax, 0.0, 1.0)
    rgb = (colormaps["inferno"](norm)[..., :3] * 255).astype(np.uint8)
    bgr = rgb[..., ::-1]
    # 中心に白い点線の十字線を描画
    h, w = bgr.shape[:2]
    cx, cy = w // 2, h // 2
    dash = 4  # 点線のピッチ (px)
    color = (255, 255, 255)
    for x in range(0, w):
        if (x // dash) % 2 == 0:
            bgr[cy, x] = color
    for y in range(0, h):
        if (y // dash) % 2 == 0:
            bgr[y, cx] = color
    ok, encoded = cv2.imencode(".png", bgr)
    if not ok:
        return ""
    return f"data:image/png;base64,{base64.b64encode(encoded.tobytes()).decode('ascii')}"


def preprocess_pair(
    beam_path: Path,
    bg_path: Path,
    ppf_path: Path,
    image_size_mm: float,
    pt_pixels: int,
    tilt_comp_enabled: bool = False,
    target_tilt_x_deg: float = 0.0,
    target_tilt_y_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    df_beam, _ = initial_cleanup(beam_path)
    df_bg, _ = initial_cleanup(bg_path)

    a = df_beam.to_numpy(float)
    b = df_bg.to_numpy(float)
    if a.shape != b.shape:
        raise ValueError(f"size mismatch: {beam_path.name} vs {bg_path.name}")

    delta_t = a - b
    h, w = delta_t.shape

    corners = read_ppf_corners(ppf_path).copy()
    # corners[:, 1] = (h - 1) - corners[:, 1]  # removed: flipud も廃止したため不要
    corners = order_corners(corners)

    yy, xx = np.mgrid[0:h, 0:w]
    mask = MplPath(corners).contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(h, w)
    delta_t_trim = np.where(mask, delta_t, 0.0)

    dst = np.array(
        [[0, 0], [pt_pixels - 1, 0], [pt_pixels - 1, pt_pixels - 1], [0, pt_pixels - 1]],
        dtype=np.float32,
    )
    mat = cv2.getPerspectiveTransform(corners, dst)
    delta_t_pt = cv2.warpPerspective(
        delta_t_trim.astype(np.float32),
        mat,
        (pt_pixels, pt_pixels),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    delta_t_pt_raw = delta_t_pt
    if tilt_comp_enabled:
        delta_t_pt = compensate_target_tilt(
            delta_t_pt_raw,
            image_size_mm=image_size_mm,
            tilt_x_deg=target_tilt_x_deg,
            tilt_y_deg=target_tilt_y_deg,
        )
    return delta_t_pt_raw, delta_t_pt


def compensate_target_tilt(
    img: np.ndarray,
    image_size_mm: float,
    tilt_x_deg: float,
    tilt_y_deg: float,
) -> np.ndarray:
    """
    Approximate small target-plane tilt compensation by resampling the
    perspective-transformed target image onto a beam-normal virtual plane.
    tilt_x_deg: rotation about x-axis, affecting y-scale.
    tilt_y_deg: rotation about y-axis, affecting x-scale.
    """
    tilt_x_rad = math.radians(float(tilt_x_deg))
    tilt_y_rad = math.radians(float(tilt_y_deg))
    cos_x = math.cos(tilt_x_rad)
    cos_y = math.cos(tilt_y_rad)
    if abs(cos_x) < 1e-6 or abs(cos_y) < 1e-6:
        raise ValueError("target tilt is too close to 90 deg")
    if math.isclose(cos_x, 1.0, rel_tol=0.0, abs_tol=1e-12) and math.isclose(
        cos_y, 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        return img

    n, m = img.shape
    px_x = image_size_mm / n
    px_y = image_size_mm / m

    x_perp = (np.arange(n, dtype=np.float32) - (n - 1) / 2.0) * px_x
    y_perp = (np.arange(m, dtype=np.float32) - (m - 1) / 2.0) * px_y
    X_perp, Y_perp = np.meshgrid(x_perp, y_perp, indexing="ij")

    X_src_mm = X_perp / cos_y
    Y_src_mm = Y_perp / cos_x

    map_x = (Y_src_mm / px_y) + (m - 1) / 2.0
    map_y = (X_src_mm / px_x) + (n - 1) / 2.0

    corrected = cv2.remap(
        img.astype(np.float32),
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return corrected


def collect_dataset(
    folder: Path,
    image_size_mm: float,
    pt_pixels: int,
    tilt_comp_enabled: bool = False,
    target_tilt_x_deg: float = 0.0,
    target_tilt_y_deg: float = 0.0,
) -> tuple[list[ThermogramEntry], list[str]]:
    csv_files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".csv"]
    ppf_files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".ppf"]

    metas = [m for p in csv_files if (m := parse_csv_meta(p)) is not None]
    patterns = [m for m in metas if not m.is_bg]
    bgs = [m for m in metas if m.is_bg]
    notes: list[str] = []

    if not patterns:
        raise ValueError("pattern CSV not found in the directory")
    if not bgs:
        raise ValueError("background CSV (_BG) not found in the directory")
    if not ppf_files:
        raise ValueError("PPF file not found in the directory")

    entries: list[ThermogramEntry] = []
    for p in patterns:
        bg_candidates = [b for b in bgs if math.isclose(b.z_mm, p.z_mm, abs_tol=1e-9)]
        if not bg_candidates:
            notes.append(f"BG not found for {p.path.name}")
            continue
        bg = sorted(bg_candidates, key=lambda b: score_bg(p, b))[0]
        ppf = choose_ppf_for(p, ppf_files)
        if ppf is None:
            notes.append(f"PPF not found for {p.path.name}")
            continue

        try:
            delta_t_pt_raw, delta_t_pt = preprocess_pair(
                p.path,
                bg.path,
                ppf,
                image_size_mm=image_size_mm,
                pt_pixels=pt_pixels,
                tilt_comp_enabled=tilt_comp_enabled,
                target_tilt_x_deg=target_tilt_x_deg,
                target_tilt_y_deg=target_tilt_y_deg,
            )
        except Exception as exc:
            notes.append(f"preprocess failed for {p.path.name}: {exc}")
            continue

        cx_raw, cy_raw, sig_x_raw, sig_y_raw = moment_stats(delta_t_pt_raw, image_size_mm)
        cx, cy, sig_x, sig_y = moment_stats(delta_t_pt, image_size_mm)

        entries.append(
            ThermogramEntry(
                beam_path=p.path,
                bg_path=bg.path,
                ppf_path=ppf,
                z_mm=p.z_mm,
                frame=p.frame,
                stamp12=p.stamp12,
                delta_t_pt=delta_t_pt,
                delta_t_pt_raw=delta_t_pt_raw,
                centroid_x_mm=cx,
                centroid_y_mm=cy,
                sigma_x_mm=sig_x,
                sigma_y_mm=sig_y,
                centroid_x_mm_raw=cx_raw,
                centroid_y_mm_raw=cy_raw,
                sigma_x_mm_raw=sig_x_raw,
                sigma_y_mm_raw=sig_y_raw,
                preview_url=array_to_data_url(delta_t_pt),
            )
        )

    entries.sort(key=lambda e: e.z_mm)
    return entries, notes


def compute_tilt(entries: list[ThermogramEntry]) -> dict[str, float] | None:
    if len(entries) < 2:
        return None
    z = np.array([e.z_mm for e in entries], dtype=np.float64)
    cx = np.array([e.centroid_x_mm for e in entries], dtype=np.float64)
    cy = np.array([e.centroid_y_mm for e in entries], dtype=np.float64)

    return compute_tilt_from_arrays(z, cx, cy)


def compute_tilt_from_arrays(
    z_mm: np.ndarray,
    cx_mm: np.ndarray,
    cy_mm: np.ndarray,
) -> dict[str, float] | None:
    if len(z_mm) < 2:
        return None

    fit_x = np.polyfit(z_mm, cx_mm, 1)
    fit_y = np.polyfit(z_mm, cy_mm, 1)
    slope_x, intercept_x = fit_x[0], fit_x[1]
    slope_y, intercept_y = fit_y[0], fit_y[1]

    theta_x = math.degrees(math.atan(slope_x))
    theta_y = math.degrees(math.atan(slope_y))
    theta_tot = math.degrees(math.atan(math.hypot(slope_x, slope_y)))
    theta_dir = math.degrees(math.atan2(slope_y, slope_x))

    return {
        "slope_x": float(slope_x),
        "slope_y": float(slope_y),
        "cx0": float(intercept_x),
        "cy0": float(intercept_y),
        "theta_x": float(theta_x),
        "theta_y": float(theta_y),
        "theta_tot": float(theta_tot),
        "theta_dir": float(theta_dir),
    }


def tilt_plot(entries: list[ThermogramEntry], tilt: dict[str, float]) -> str:
    z = np.array([e.z_mm for e in entries], dtype=np.float64)
    cx = np.array([e.centroid_x_mm for e in entries], dtype=np.float64)
    cy = np.array([e.centroid_y_mm for e in entries], dtype=np.float64)
    z_line = np.linspace(min(z.min(), 0.0), max(z.max(), 0.0), 64)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.0), dpi=120, sharex=True)
    ax1.scatter(z, cx, s=20)
    ax1.plot(z_line, tilt["cx0"] + tilt["slope_x"] * z_line)
    ax1.scatter([0], [tilt["cx0"]], marker="s", c="red", s=24)
    ax1.set_title("x-centroid vs z")
    ax1.set_xlabel("z [mm]")
    ax1.set_ylabel("cx [mm]")
    ax1.grid(alpha=0.2)

    ax2.scatter(z, cy, s=20)
    ax2.plot(z_line, tilt["cy0"] + tilt["slope_y"] * z_line)
    ax2.scatter([0], [tilt["cy0"]], marker="s", c="red", s=24)
    ax2.set_title("y-centroid vs z")
    ax2.set_xlabel("z [mm]")
    ax2.set_ylabel("cy [mm]")
    ax2.grid(alpha=0.2)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def amp_to_db(amplitude: np.ndarray, floor_db: float = -40.0) -> np.ndarray:
    amax = float(np.max(amplitude))
    if amax <= 0.0:
        return np.full_like(amplitude, floor_db, dtype=np.float64)
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(np.clip(amplitude / amax, 1e-12, None))
    return np.maximum(db, floor_db)


def normalize_thermograms_by_power(
    thermograms: list[np.ndarray],
    ref_index: int = 0,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Normalize plane-by-plane thermograms so total positive power matches reference plane.
    """
    if not thermograms:
        return [], np.array([], dtype=np.float64)

    powers = np.array([float(np.sum(np.clip(t, 0.0, None))) for t in thermograms], dtype=np.float64)
    if not np.isfinite(powers).all():
        powers = np.nan_to_num(powers, nan=0.0, posinf=0.0, neginf=0.0)

    ref_index = int(np.clip(ref_index, 0, len(thermograms) - 1))
    p_ref = float(powers[ref_index])
    if p_ref <= 0.0:
        nonzero = powers[powers > 0.0]
        p_ref = float(nonzero[0]) if nonzero.size else 1.0

    scales = np.ones(len(thermograms), dtype=np.float64)
    out: list[np.ndarray] = []
    for i, t in enumerate(thermograms):
        p = float(powers[i])
        s = p_ref / p if p > 0.0 else 1.0
        scales[i] = s
        out.append(t * s)
    return out, scales


def amplitude_misfit(
    u0: np.ndarray,
    zs_m: list[float],
    amps: list[np.ndarray],
    wavelength: float,
    dx_m: float,
    tilt_x: float,
    tilt_y: float,
) -> float:
    mis2 = 0.0
    ref2 = 0.0
    u = u0.copy()
    for k in range(len(zs_m)):
        if k > 0:
            dz = zs_m[k] - zs_m[k - 1]
            u = propagate_angular_spectrum(u, float(dz), wavelength, dx_m, -tilt_y, tilt_x)
        mis2 += float(np.sum((np.abs(u) - amps[k]) ** 2))
        ref2 += float(np.sum(amps[k] ** 2))
    return math.sqrt(mis2 / max(ref2, 1e-30))


def run_phase_retrieval(
    thermograms: list[np.ndarray],
    zs_m: list[float],
    wavelength: float,
    dx_m: float,
    tilt_x: float,
    tilt_y: float,
    n_iter: int,
    hio_beta: float,
    hio_cycles: int,
    er_cycles: int,
    hio_percent: float,
    auto_stop: bool = False,
    min_iter: int = 20,
    patience: int = 20,
    rel_improve_tol: float = 1e-4,
    abs_improve_tol: float = 1e-8,
    progress_cb: Callable[[int, int, float], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, int, int, float, str]:
    amplitudes = [np.sqrt(np.clip(i, 0.0, None)).astype(np.float64) for i in thermograms]
    u = amplitudes[0].astype(np.complex128)

    cycle = max(1, hio_cycles + er_cycles)
    min_iter = max(1, int(min_iter))
    patience = max(1, int(patience))
    rel_improve_tol = max(0.0, float(rel_improve_tol))
    abs_improve_tol = max(0.0, float(abs_improve_tol))

    err_history: list[float] = []
    n_planes = len(zs_m)
    best_u = u.copy()
    best_err = float("inf")
    best_iter = 0
    last_improve_iter = 0
    stop_reason = "max_iter"

    for i in range(n_iter):
        use_hio = (i % cycle) < hio_cycles

        for k in range(n_planes - 1):
            dz = zs_m[k + 1] - zs_m[k]
            u = propagate_angular_spectrum(u, float(dz), wavelength, dx_m, -tilt_y, tilt_x)
            a = amplitudes[k + 1]
            u_prop = u
            u_new = a * u_prop / np.maximum(np.abs(u_prop), 1e-20)
            if use_hio:
                mask = a >= np.percentile(a, hio_percent)
                u = np.where(mask, u_new, u_prop - hio_beta * u_new)
            else:
                u = u_new

        for k in range(n_planes - 1, 0, -1):
            dz = zs_m[k] - zs_m[k - 1]
            u = propagate_angular_spectrum(u, -float(dz), wavelength, dx_m, -tilt_y, tilt_x)
            a = amplitudes[k - 1]
            u_prop = u
            u_new = a * u_prop / np.maximum(np.abs(u_prop), 1e-20)
            if use_hio:
                mask = a >= np.percentile(a, hio_percent)
                u = np.where(mask, u_new, u_prop - hio_beta * u_new)
            else:
                u = u_new

        err = amplitude_misfit(u, zs_m, amplitudes, wavelength, dx_m, tilt_x, tilt_y)
        err_history.append(err)

        # Keep best-so-far solution and its iteration.
        if not np.isfinite(best_err):
            improved = True
        else:
            improve_th = max(abs_improve_tol, rel_improve_tol * max(best_err, 1e-12))
            improved = (best_err - err) > improve_th
        if improved:
            best_err = float(err)
            best_u = u.copy()
            best_iter = i + 1
            last_improve_iter = i + 1

        if progress_cb is not None:
            progress_cb(i + 1, n_iter, err)

        if auto_stop and (i + 1) >= min_iter and ((i + 1) - last_improve_iter) >= patience:
            stop_reason = f"plateau (patience={patience})"
            break

    iterations_done = len(err_history)
    if iterations_done < n_iter and stop_reason == "max_iter":
        stop_reason = "stopped"
    if best_iter == 0:
        # Fallback for pathological numerical cases.
        best_u = u.copy()
        best_err = float(err_history[-1]) if err_history else float("nan")
        best_iter = iterations_done

    return best_u, np.array(err_history, dtype=np.float64), iterations_done, best_iter, best_err, stop_reason


def pixel_coords(shape: tuple[int, int], dx_m: float) -> tuple[np.ndarray, np.ndarray]:
    n, m = shape
    x = (np.arange(n) - n / 2 + 0.5) * dx_m
    y = (np.arange(m) - m / 2 + 0.5) * dx_m
    return np.meshgrid(x, y, indexing="ij")


def centroid_of_field(u: np.ndarray, dx_m: float) -> tuple[float, float]:
    x, y = pixel_coords(u.shape, dx_m)
    w = np.abs(u) ** 2
    s = w.sum()
    if s <= 0:
        return 0.0, 0.0
    return float((w * x).sum() / s), float((w * y).sum() / s)


def generate_mode_specs(max_m: int, max_n: int) -> list[tuple[str, int, int, str | None]]:
    specs: list[tuple[str, int, int, str | None]] = []
    for n in range(1, max_n + 1):
        specs.append((f"LP0,{n}", 0, n, None))
    for m in range(1, max_m + 1):
        for n in range(1, max_n + 1):
            specs.append((f"LP{m},{n}even", m, n, "cos"))
            specs.append((f"LP{m},{n}odd", m, n, "sin"))
    return specs


def mode_name_sort_key(name: str) -> tuple[int, int, int, int]:
    """
    Natural sort key for LP mode names:
    LPm,n < LPm,nodd < LPm,neven by numeric m,n.
    """
    m = re.fullmatch(r"LP(\d+),(\d+)(odd|even)?", name)
    if not m:
        return (10**9, 10**9, 10**9, 10**9)
    mm = int(m.group(1))
    nn = int(m.group(2))
    suffix = m.group(3)
    if suffix is None:
        p = 0
    elif suffix == "odd":
        p = 1
    else:
        p = 2
    return (mm, nn, p, 0)


def mode_name_sort_token(name: str) -> str:
    mm, nn, pp, _ = mode_name_sort_key(name)
    return f"{mm:03d}-{nn:03d}-{pp}"


def calc_lp_purity(
    u: np.ndarray,
    dx_m: float,
    r_wg_m: float,
    center_m: tuple[float, float],
    max_m: int,
    max_n: int,
) -> dict[str, float]:
    n, m = u.shape
    cx, cy = center_m  # cx = horizontal (right+), cy = vertical (up+)
    # x = horizontal (axis 1, cols), y = vertical (axis 0, rows, up = positive)
    x = (np.arange(m) - m / 2) * dx_m - cx
    y = (n / 2 - np.arange(n)) * dx_m - cy   # 上端 → 正、下端 → 負
    Y, X = np.meshgrid(y, x, indexing="ij")  # Y along axis 0 (rows), X along axis 1 (cols)
    R = np.sqrt(X**2 + Y**2)
    Phi = np.arctan2(Y, X)
    mask = (R <= r_wg_m)

    dA = dx_m * dx_m
    p_tot = float(np.sum(np.abs(u) ** 2 * mask) * dA)
    if p_tot <= 0:
        return {}

    out: dict[str, float] = {}
    for name, mm, nn, parity in generate_mode_specs(max_m=max_m, max_n=max_n):
        chi = jn_zeros(mm, nn)[-1]
        v = jn(mm, chi * R / r_wg_m)
        if mm >= 1:
            if parity == "cos":
                v = v * np.cos(mm * Phi)
            else:
                v = v * np.sin(mm * Phi)
        v = v * mask
        denom = float(np.sum(np.abs(v) ** 2 * mask) * dA) * p_tot
        if denom <= 0:
            out[name] = 0.0
            continue
        c = np.sum(u.conj() * v * mask) * dA
        out[name] = 100.0 * float((np.abs(c) ** 2) / denom)
    return out


def save_reconstruction_plot(u: np.ndarray, frame_size_mm: float, out_png: Path) -> None:
    extent = (-frame_size_mm / 2, frame_size_mm / 2, frame_size_mm / 2, -frame_size_mm / 2)
    amp_lin = np.abs(u)
    amp_db = amp_to_db(amp_lin)
    phase = np.angle(u)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11.5, 3.8), sharey=True)
    ax1.imshow(amp_lin, cmap="inferno", extent=extent, origin="upper")
    ax1.set_title("|E| linear")
    ax1.set_xlabel("x [mm]")
    ax1.set_ylabel("y [mm]")

    im2 = ax2.imshow(amp_db, cmap="inferno", extent=extent, origin="upper", vmin=-40, vmax=0)
    ax2.set_title("|E| dB")
    ax2.set_xlabel("x [mm]")

    im3 = ax3.imshow(phase, cmap="hsv", extent=extent, origin="upper")
    ax3.set_title("phase")
    ax3.set_xlabel("x [mm]")
    fig.colorbar(im2, ax=ax2, fraction=0.046)
    fig.colorbar(im3, ax=ax3, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def field_view_urls(u: np.ndarray, frame_size_mm: float) -> tuple[str, str, str]:
    extent = (-frame_size_mm / 2, frame_size_mm / 2, frame_size_mm / 2, -frame_size_mm / 2)

    intensity = np.abs(u) ** 2
    amp_db = amp_to_db(np.abs(u))
    phase = np.angle(u)

    def _to_url(fig: plt.Figure) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"

    fig1, ax1 = plt.subplots(figsize=(4.8, 4.0), dpi=140)
    im1 = ax1.imshow(intensity, cmap="inferno", extent=extent, origin="upper")
    ax1.set_title("Intensity |E|²")
    ax1.set_xlabel("x [mm]")
    ax1.set_ylabel("y [mm]")
    fig1.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(4.8, 4.0), dpi=140)
    im2 = ax2.imshow(amp_db, cmap="inferno", extent=extent, origin="upper", vmin=-40, vmax=0)
    ax2.set_title("Intensity [dB]")
    ax2.set_xlabel("x [mm]")
    ax2.set_ylabel("y [mm]")
    fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="dB")
    fig2.tight_layout()

    fig3, ax3 = plt.subplots(figsize=(4.8, 4.0), dpi=140)
    im3 = ax3.imshow(phase, cmap="hsv", extent=extent, origin="upper", vmin=-np.pi, vmax=np.pi)
    ax3.set_title("Phase arg(E)")
    ax3.set_xlabel("x [mm]")
    ax3.set_ylabel("y [mm]")
    fig3.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    fig3.tight_layout()

    return _to_url(fig1), _to_url(fig2), _to_url(fig3)


def save_mode_txt(
    out_path: Path,
    axis_purity: dict[str, float],
    axis_center_mm: tuple[float, float],
) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        f.write("== Axis-centred LP purity [%] ==\n")
        f.write(
            f"axis center at evaluation plane = ({axis_center_mm[0]:.4f}, {axis_center_mm[1]:.4f}) mm\n"
        )
        for k in sorted(axis_purity.keys(), key=mode_name_sort_key):
            v = axis_purity[k]
            f.write(f"{k:<14}: {v:10.6f}\n")
        f.write(f"\nAxis sum [%]: {sum(axis_purity.values()):10.6f}\n")

        f.write("\n== Major modes (axis) [%] ==\n")
        for k in sorted(MAJOR_MODES, key=mode_name_sort_key):
            if k in axis_purity:
                f.write(f"{k:<14}: {axis_purity[k]:10.6f}\n")


def execute_phase_pipeline(
    entries: list[ThermogramEntry],
    freq_ghz: float,
    image_size_mm: float,
    wg_diameter_mm: float,
    axis_center_x_mm: float,
    axis_center_y_mm: float,
    n_iter: int,
    hio_beta: float,
    hio_cycles: int,
    er_cycles: int,
    hio_percent: float,
    auto_stop: bool,
    min_iter: int,
    patience: int,
    rel_improve_tol: float,
    abs_improve_tol: float,
    z_eval_mm: float,
    max_m: int,
    max_n: int,
    output_dir: Path,
    tilt_comp_enabled: bool = False,
    progress_cb: Callable[[int, int, float], None] | None = None,
) -> PhaseResult:
    def run_single_pipeline(
        thermograms_in: list[np.ndarray],
        centroid_x_mm_in: np.ndarray,
        centroid_y_mm_in: np.ndarray,
        progress_cb_in: Callable[[int, int, float], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, float, str, dict[str, float]]:
        thermograms, _scales = normalize_thermograms_by_power(thermograms_in, ref_index=0)
        dx_m = (image_size_mm * 1e-3) / thermograms[0].shape[0]

        tilt = compute_tilt_from_arrays(z_mm, centroid_x_mm_in, centroid_y_mm_in)
        if tilt is None:
            raise ValueError("at least 2 thermograms are required for tilt estimation")
        tilt_x = tilt["slope_x"]
        tilt_y = tilt["slope_y"]

        u_pr, err_history, iterations_done, best_iteration, best_error, stop_reason = run_phase_retrieval(
            thermograms=thermograms,
            zs_m=zs_m,
            wavelength=wavelength,
            dx_m=dx_m,
            tilt_x=tilt_x,
            tilt_y=tilt_y,
            n_iter=n_iter,
            hio_beta=hio_beta,
            hio_cycles=hio_cycles,
            er_cycles=er_cycles,
            hio_percent=hio_percent,
            auto_stop=auto_stop,
            min_iter=min_iter,
            patience=patience,
            rel_improve_tol=rel_improve_tol,
            abs_improve_tol=abs_improve_tol,
            progress_cb=progress_cb_in,
        )

        z_eval_m = z_eval_mm * 1e-3
        dz_ref = z_eval_m - zs_m[0]
        u_eval = propagate_angular_spectrum(u_pr, float(dz_ref), wavelength, dx_m, -tilt_y, tilt_x)
        axis_center_m = (axis_center_x_mm * 1e-3, axis_center_y_mm * 1e-3)
        axis_purity = calc_lp_purity(
            u=u_eval,
            dx_m=dx_m,
            r_wg_m=(wg_diameter_mm * 0.5) * 1e-3,
            center_m=axis_center_m,
            max_m=max_m,
            max_n=max_n,
        )
        return (
            u_pr,
            u_eval,
            err_history,
            iterations_done,
            best_iteration,
            best_error,
            stop_reason,
            axis_purity,
        )

    c_light = 299792458.0
    wavelength = c_light / (freq_ghz * 1e9)
    z_mm = np.array([e.z_mm for e in entries], dtype=np.float64)
    zs_m = [e.z_mm * 1e-3 for e in entries]
    thermograms = [e.delta_t_pt for e in entries] if tilt_comp_enabled else [e.delta_t_pt_raw for e in entries]
    centroid_x = (
        np.array([e.centroid_x_mm for e in entries], dtype=np.float64)
        if tilt_comp_enabled
        else np.array([e.centroid_x_mm_raw for e in entries], dtype=np.float64)
    )
    centroid_y = (
        np.array([e.centroid_y_mm for e in entries], dtype=np.float64)
        if tilt_comp_enabled
        else np.array([e.centroid_y_mm_raw for e in entries], dtype=np.float64)
    )

    u_pr, u_eval, err_history, iterations_done, best_iteration, best_error, stop_reason, axis_purity = (
        run_single_pipeline(
            thermograms_in=thermograms,
            centroid_x_mm_in=centroid_x,
            centroid_y_mm_in=centroid_y,
            progress_cb_in=progress_cb,
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_dir / "error_history.txt", err_history)
    plt.figure(figsize=(6.0, 3.2), dpi=130)
    plt.semilogy(err_history)
    plt.xlabel("iteration")
    plt.ylabel("amplitude error")
    plt.tight_layout()
    plt.savefig(output_dir / "error_curve.png", dpi=180)
    plt.close()

    save_reconstruction_plot(u_eval, frame_size_mm=image_size_mm, out_png=output_dir / "reconstruction.png")
    save_mode_txt(
        output_dir / "mode_purity.txt",
        axis_purity=axis_purity,
        axis_center_mm=(axis_center_x_mm, axis_center_y_mm),
    )
    return PhaseResult(
        u_pr=u_pr,
        u_eval=u_eval,
        z_eval_mm=float(z_eval_mm),
        err_history=err_history,
        iterations_done=iterations_done,
        best_iteration=best_iteration,
        best_error=best_error,
        stop_reason=stop_reason,
        axis_purity=axis_purity,
        axis_center_mm=(axis_center_x_mm, axis_center_y_mm),
        output_dir=output_dir,
    )


def prepare_mode_rows(result: PhaseResult) -> list[dict[str, Any]]:
    keys = sorted(result.axis_purity.keys())
    rows = []
    for k in keys:
        rows.append(
            {
                "mode": k,
                "mode_sort": mode_name_sort_token(k),
                "axis_percent": round(result.axis_purity.get(k, 0.0), 6),
                "major": "yes" if k in MAJOR_MODES else "",
            }
        )
    rows.sort(key=lambda r: (r["major"] != "yes", mode_name_sort_key(r["mode"])))
    return rows


def nice_float(v: float) -> str:
    if not np.isfinite(v):
        return "-"
    return f"{v:.3f}"


def build_ui() -> None:
    ui.dark_mode(False)
    ui.query("body").style("font-family: 'Noto Sans', 'Segoe UI', sans-serif;")

    ui.label("Mode Purity Measurement Web App").classes("text-h5")
    ui.label(
        "Load thermogram folder, preprocess automatically, inspect tilt, and run phase retrieval."
    ).classes("text-body2 text-grey-8")

    with ui.card().classes("w-full"):
        with ui.row().classes("items-end w-full"):
            folder_input = ui.input("Data folder path", value=str(Path.cwd())).classes("w-96")
            image_size_input = ui.number("Image size [mm]", value=240.0, step=1.0, format="%.1f")
            pt_pixels_input = ui.number("PT pixels", value=256, step=1, min=64, max=1024, format="%d")
            load_button = ui.button("Load + Preprocess", color="primary")
        with ui.row().classes("items-end w-full"):
            tilt_comp_input = ui.switch("Target tilt compensation", value=False)
            target_tilt_x_input = ui.number("Target tilt X [deg]", value=0.0, format="%.3f")
            target_tilt_y_input = ui.number("Target tilt Y [deg]", value=0.0, format="%.3f")
        notes_area = ui.log(max_lines=200).classes("w-full").style("height:120px;")

    gallery_card = ui.card().classes("w-full")
    tilt_card = ui.card().classes("w-full")
    with tilt_card:
        tilt_text = ui.label("Tilt: -")
        tilt_image = ui.image().classes("w-full").style("max-width: 820px;")

    with ui.card().classes("w-full"):
        ui.label("Phase Retrieval Settings").classes("text-subtitle1")
        with ui.row().classes("items-end"):
            freq_input = ui.number("Frequency [GHz]", value=170.0, format="%.3f")
            wg_diameter_input = ui.number("Waveguide diameter [mm]", value=63.5, format="%.3f")
            z_eval_input = ui.number("Evaluation z [mm]", value=0.0, format="%.1f")
        with ui.row().classes("items-end"):
            axis_center_x_input = ui.number("Axis center X [mm]", value=0.0, format="%.3f")
            axis_center_y_input = ui.number("Axis center Y [mm]", value=0.0, format="%.3f")
        with ui.row().classes("items-end"):
            n_iter_input = ui.number("Iterations", value=100, step=1, min=1, max=5000, format="%d")
            hio_beta_input = ui.number("HIO beta", value=0.75, format="%.3f")
            hio_cycles_input = ui.number("HIO cycles", value=4, step=1, min=0, max=20, format="%d")
            er_cycles_input = ui.number("ER cycles", value=1, step=1, min=1, max=20, format="%d")
            hio_percent_input = ui.number("HIO mask percentile", value=5.0, format="%.2f")
        with ui.row().classes("items-end"):
            auto_stop_input = ui.switch("Auto stop", value=True)
            min_iter_input = ui.number("Min iter", value=20, step=1, min=1, max=5000, format="%d")
            patience_input = ui.number("Patience", value=20, step=1, min=1, max=5000, format="%d")
            rel_tol_input = ui.number("Rel improve tol", value=1.0e-4, format="%.1e")
            abs_tol_input = ui.number("Abs improve tol", value=1.0e-8, format="%.1e")
        with ui.row().classes("items-end"):
            max_m_input = ui.number("LP m max", value=10, step=1, min=0, max=20, format="%d")
            max_n_input = ui.number("LP n max", value=10, step=1, min=1, max=20, format="%d")
            output_dir_input = ui.input("Output dir", value=str(Path.cwd() / "output_web")).classes("w-96")
            run_button = ui.button("Phase Retrieval", color="secondary")

    with ui.card().classes("w-full"):
        status_label = ui.label("Status: Idle")
        progress_bar = ui.linear_progress(value=0.0).classes("w-full")
        result_label = ui.label("Result: -").classes("text-body2")

    with ui.card().classes("w-full"):
        eval_field_label = ui.label("Retrieved field at evaluation plane: -").classes("text-subtitle1")
        with ui.row().classes("w-full no-wrap").style("overflow-x:auto; gap:12px;"):
            eval_intensity_image = ui.image().style("width:480px; height:400px; object-fit:contain;")
            eval_db_image = ui.image().style("width:480px; height:400px; object-fit:contain;")
            eval_phase_image = ui.image().style("width:480px; height:400px; object-fit:contain;")

    with ui.card().classes("w-full"):
        ui.label("LP Mode Purity [%]").classes("text-subtitle1")
        mode_sum_label = ui.label("Sum: Axis=-").classes("text-body2")
        mode_table = ui.table(
            columns=[
                {"name": "mode", "label": "Mode", "field": "mode_sort", "sortable": True},
                {"name": "axis_percent", "label": "Axis-centred [%]", "field": "axis_percent", "sortable": True},
                {"name": "major", "label": "Major", "field": "major", "sortable": True},
            ],
            rows=[],
            row_key="mode",
            pagination=20,
        ).classes("w-full")
        mode_table.add_slot(
            "body-cell-mode",
            r"""
            <q-td :props="props">
              {{ props.row.mode }}
            </q-td>
            """,
        )

    def get_selected_entries() -> list[ThermogramEntry]:
        return [e for e in STATE.entries if entry_key(e) in STATE.selected_entry_keys]

    def update_tilt_for_selected() -> None:
        selected = get_selected_entries()
        STATE.tilt_info = compute_tilt(selected)
        STATE.tilt_plot_url = tilt_plot(selected, STATE.tilt_info) if STATE.tilt_info else None

    def set_entry_selected(key: str, checked: bool) -> None:
        if checked:
            STATE.selected_entry_keys.add(key)
        else:
            STATE.selected_entry_keys.discard(key)
        update_tilt_for_selected()
        redraw_gallery()
        redraw_tilt()

    def select_all_entries() -> None:
        STATE.selected_entry_keys = {entry_key(e) for e in STATE.entries}
        update_tilt_for_selected()
        redraw_gallery()
        redraw_tilt()

    def clear_all_entries() -> None:
        STATE.selected_entry_keys = set()
        update_tilt_for_selected()
        redraw_gallery()
        redraw_tilt()

    def invert_selection() -> None:
        all_keys = {entry_key(e) for e in STATE.entries}
        STATE.selected_entry_keys = all_keys - STATE.selected_entry_keys
        update_tilt_for_selected()
        redraw_gallery()
        redraw_tilt()

    def redraw_gallery() -> None:
        gallery_card.clear()
        with gallery_card:
            selected = get_selected_entries()
            with ui.row().classes("items-center justify-between w-full"):
                ui.label("Preprocessed Thermograms (sorted by distance)").classes("text-subtitle1")
                ui.label(f"Selected for phase retrieval: {len(selected)} / {len(STATE.entries)}").classes("text-body2")
            with ui.row().classes("items-center"):
                ui.button("Select all", on_click=lambda: select_all_entries()).props("flat dense")
                ui.button("Clear", on_click=lambda: clear_all_entries()).props("flat dense")
                ui.button("Invert", on_click=lambda: invert_selection()).props("flat dense")
            with ui.row().classes("no-wrap").style("overflow-x:auto; width:100%; gap:12px; padding-bottom:6px;"):
                for e in STATE.entries:
                    k = entry_key(e)
                    is_checked = k in STATE.selected_entry_keys
                    with ui.card().style("min-width:280px;"):
                        ui.checkbox(
                            "Use for phase retrieval",
                            value=is_checked,
                            on_change=lambda ev, kk=k: set_entry_selected(kk, bool(ev.value)),
                        )
                        ui.image(e.preview_url).style("width:260px; height:240px; object-fit:contain;")
                        ui.label(f"z = {e.z_mm:.1f} mm")
                        ui.label(f"time = {format_stamp(e.stamp12)}")
                        ui.label(f"frame = {e.frame if e.frame is not None else '-'}")
                        ui.label(f"file = {e.beam_path.name}")
                        ui.label(f"centroid = ({nice_float(e.centroid_x_mm)}, {nice_float(e.centroid_y_mm)}) mm")
                        ui.label(f"width(sigma) = ({nice_float(e.sigma_x_mm)}, {nice_float(e.sigma_y_mm)}) mm")

    def redraw_tilt() -> None:
        n_sel = len(get_selected_entries())
        if STATE.tilt_info is None:
            tilt_text.set_text(f"Tilt: - (need at least 2 selected frames, selected={n_sel})")
            tilt_image.set_source("")
            return
        t = STATE.tilt_info
        tilt_text.set_text(
            "Tilt "
            f"[selected={n_sel}] "
            f"X={t['theta_x']:.3f} deg, Y={t['theta_y']:.3f} deg, "
            f"Total={t['theta_tot']:.3f} deg, phi={t['theta_dir']:.3f} deg, "
            f"Off-center(z=0)=({t['cx0']:.3f}, {t['cy0']:.3f}) mm"
        )
        tilt_image.set_source(STATE.tilt_plot_url or "")

    def redraw_table() -> None:
        mode_table.rows = STATE.mode_table_rows
        mode_table.update()
        mode_sum_label.set_text(f"Sum: Axis={STATE.axis_sum_percent:.6f} %")

    async def on_load_clicked() -> None:
        try:
            folder = Path(folder_input.value).expanduser().resolve()
            if not folder.is_dir():
                raise ValueError("directory not found")
            STATE.status = "Loading dataset..."
            status_label.set_text(f"Status: {STATE.status}")

            entries, notes = await run.io_bound(
                collect_dataset,
                folder,
                float(image_size_input.value),
                int(pt_pixels_input.value),
                bool(tilt_comp_input.value),
                float(target_tilt_x_input.value),
                float(target_tilt_y_input.value),
            )
            STATE.entries = entries
            STATE.selected_entry_keys = {entry_key(e) for e in entries}
            STATE.load_notes = notes
            STATE.tilt_comp_enabled = bool(tilt_comp_input.value)
            update_tilt_for_selected()
            STATE.phase_result = None
            STATE.mode_table_rows = []
            STATE.axis_sum_percent = 0.0
            eval_field_label.set_text("Retrieved field at evaluation plane: -")
            eval_intensity_image.set_source("")
            eval_phase_image.set_source("")
            redraw_gallery()
            redraw_tilt()
            redraw_table()

            notes_area.clear()
            notes_area.push(f"Loaded {len(entries)} entries from {folder}")
            if bool(tilt_comp_input.value):
                notes_area.push(
                    "[INFO] target tilt compensation enabled "
                    f"(X={float(target_tilt_x_input.value):.3f} deg, "
                    f"Y={float(target_tilt_y_input.value):.3f} deg)"
                )
            else:
                notes_area.push("[INFO] target tilt compensation disabled")
            for n in notes:
                notes_area.push(f"[WARN] {n}")
            STATE.status = f"Loaded {len(entries)} entries"
            status_label.set_text(f"Status: {STATE.status}")
        except Exception as exc:
            notes_area.push(f"[ERROR] {exc}")
            STATE.status = "Load failed"
            status_label.set_text(f"Status: {STATE.status}")

    async def on_phase_clicked() -> None:
        selected_entries = get_selected_entries()
        if len(selected_entries) < 2:
            ui.notify("Need at least 2 selected thermograms", color="negative")
            return

        STATE.running = True
        STATE.progress = 0.0
        STATE.last_error = None
        STATE.status = "Phase retrieval running..."
        status_label.set_text(f"Status: {STATE.status}")
        run_button.disable()

        def progress_cb(i: int, n: int, err: float) -> None:
            STATE.progress = float(i / max(n, 1))
            STATE.last_error = float(err)

        try:
            result = await run.io_bound(
                execute_phase_pipeline,
                selected_entries,
                float(freq_input.value),
                float(image_size_input.value),
                float(wg_diameter_input.value),
                float(axis_center_x_input.value),
                float(axis_center_y_input.value),
                int(n_iter_input.value),
                float(hio_beta_input.value),
                int(hio_cycles_input.value),
                int(er_cycles_input.value),
                float(hio_percent_input.value),
                bool(auto_stop_input.value),
                int(min_iter_input.value),
                int(patience_input.value),
                float(rel_tol_input.value),
                float(abs_tol_input.value),
                float(z_eval_input.value),
                int(max_m_input.value),
                int(max_n_input.value),
                Path(output_dir_input.value).expanduser().resolve(),
                bool(STATE.tilt_comp_enabled),
                progress_cb,
            )
            STATE.phase_result = result
            STATE.mode_table_rows = prepare_mode_rows(result)
            STATE.axis_sum_percent = float(sum(result.axis_purity.values()))
            redraw_table()
            STATE.status = "Phase retrieval completed"
            STATE.progress = 1.0
            i_url, db_url, p_url = field_view_urls(result.u_eval, float(image_size_input.value))
            eval_field_label.set_text(f"Retrieved field at evaluation plane: z = {result.z_eval_mm:.3f} mm")
            eval_intensity_image.set_source(i_url)
            eval_db_image.set_source(db_url)
            eval_phase_image.set_source(p_url)
            result_label.set_text(
                f"Result: output={result.output_dir} | axis-center=({result.axis_center_mm[0]:.3f}, {result.axis_center_mm[1]:.3f}) mm "
                f"| used={len(selected_entries)} frames | "
                f"iters={result.iterations_done} (best={result.best_iteration}, err={result.best_error:.4e}, stop={result.stop_reason})"
            )
            ui.notify("Phase retrieval completed", color="positive")
        except Exception as exc:
            STATE.status = f"Phase retrieval failed: {exc}"
            ui.notify(f"Phase retrieval failed: {exc}", color="negative")
        finally:
            STATE.running = False
            run_button.enable()
            status_label.set_text(f"Status: {STATE.status}")

    def timer_update() -> None:
        progress_bar.set_value(STATE.progress)
        if STATE.running and STATE.last_error is not None:
            status_label.set_text(f"Status: {STATE.status} (err={STATE.last_error:.4e})")
        elif not STATE.running:
            status_label.set_text(f"Status: {STATE.status}")

    load_button.on("click", on_load_clicked)
    run_button.on("click", on_phase_clicked)
    ui.timer(0.2, timer_update)
    redraw_gallery()
    redraw_tilt()
    redraw_table()


build_ui()


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="Mode Purity Measurement", reload=False, port=8081)
