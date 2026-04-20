"""
    PhaseRetrieval.py
    author:   Konan Yagasaki (kyoto Fusioneering Ltd.)
    version:  1.0 (13 June 2025)
    
    execute phase retrieval algorithm (ER and HIO) and reconstruct beam phase
    with FFT-based angular spectrum calculation in Python

    ---
    usage: python3 PhaseRetrieval.py [CSV Data Folder]/ --freq [f in GHz]  
    package: (python) numpy, matplotlib, scipy, tqdm
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn, jn_zeros 
from AngularSpectrumFFT import propagate_angular_spectrum as propagate_AS
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(it, **kwargs):
        return it


### Parameters -----------------------------------------------------------------------
C_LIGHT    = 299792458.0    # speed of light [m/s]

CSV_PIXELS = 143            # pixel size of thermograms
#CSV_PIXELS = 256            # pixel size of thermograms
FRAME_SIZE = 0.12           # thremogram size [m] - QST setup
#FRAME_SIZE = 0.15          # KF setup
DX         = FRAME_SIZE / CSV_PIXELS    # pixel size
R_WG       = 0.03015        # waveguide radius   

N_ITER     = 100            # iteration number
HIO_BETA   = 0.75           # beta for HIO

Z_REF      = 0.00           # reference plane (MOU out)
REGEX_Z    = r"([0-9.]+)mm"  # RegEx for picking z up
UNIT_MM    = True            # z is written in millimeter

OUT_DIR = Path("./output")   # output directory
OUT_DIR.mkdir(exist_ok=True)
EXTENT_MM = (-FRAME_SIZE/2*1e3, FRAME_SIZE/2*1e3,
             FRAME_SIZE/2*1e3, -FRAME_SIZE/2*1e3) # UL->UR->LR->LL

### 1. load CSVs (thermograms) ----------------------------------------------------------------------------------
def load_csv_stack(dir_path: Path, regex: str = REGEX_Z, unit_mm: bool = UNIT_MM
                   ) -> Tuple[List[np.ndarray], List[float]]:
    """
    load 256x256 CSVs (thermograms) from dir_path and return z [m]
    """
    pat = re.compile(regex, re.IGNORECASE)  # compile regex
    files = sorted(dir_path.glob("*.csv"))  # get CSVs
    if not files:
        sys.exit("[ERROR] no CSV found.")

    records = []
    for f in files:
        m = pat.search(f.name)
        if not m:
            sys.exit(f"[ERROR] cannot parse z from '{f.name}'... rename the CSV")
        
        z_val = float(m.group(1))
        z_m   = z_val*1.0e-3 if unit_mm else z_val
        thermogram = np.loadtxt(f, delimiter="\t")
        thermogram = np.flipud(thermogram)
        
        if thermogram.shape != (CSV_PIXELS, CSV_PIXELS):
            sys.exit(f"[ERROR] the size of {f} is {thermogram.shape}, expected {CSV_PIXELS} x {CSV_PIXELS}.")
            
        records.append((z_m, thermogram))
        
    records.sort(key=lambda t: t[0])
    zs, thermograms = zip(*records)
    
    return list(thermograms), list(zs)


### 2. evaluate beam tilt from CSVs -------------------------------------------------------------
def calc_beam_tilt(thermograms:List[np.ndarray], zs: List[float]) -> Tuple[float,float]:
    """
    evaluate θx, θy [rad] by linear-fitting moment centers
    """
    N = CSV_PIXELS
    
    x_grid = (np.arange(N) - N/2) * DX   # m
    y_grid = x_grid.copy()
    X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")

    cx, cy = [], []
    for I in thermograms:
        s = I.sum()
        cx.append((X*I).sum()/s)
        cy.append((Y*I).sum()/s)

    θx, _ = np.polyfit(zs, cx, 1)
    θy, _ = np.polyfit(zs, cy, 1)
    return θx, θy          # small angle ⇒ tanθ ≈ θ


def normalize_thermograms_by_power(thermograms: List[np.ndarray], ref_index: int = 0
                                   ) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    normalize each thermogram so total positive power matches the reference plane.
    """
    if not thermograms:
        return [], np.array([], dtype=np.float64)

    powers = np.array([float(np.sum(np.clip(t, 0.0, None))) for t in thermograms], dtype=np.float64)
    if not np.isfinite(powers).all():
        powers = np.nan_to_num(powers, nan=0.0, posinf=0.0, neginf=0.0)

    ref_index = int(np.clip(ref_index, 0, len(thermograms)-1))
    p_ref = float(powers[ref_index])
    if p_ref <= 0.0:
        nonzero = powers[powers > 0.0]
        p_ref = float(nonzero[0]) if nonzero.size else 1.0

    scales = np.ones(len(thermograms), dtype=np.float64)
    out: List[np.ndarray] = []
    for i, t in enumerate(thermograms):
        p = float(powers[i])
        s = p_ref / p if p > 0.0 else 1.0
        scales[i] = s
        out.append(t * s)
    return out, scales


### 3. save thermogram preview -------------------------------------------------------------------------
def save_preview(thermos: List[np.ndarray], zs: List[float],
                 fname: Path = OUT_DIR / "input_thermograms.png"):
    """
    display input thermograms
    """
    n = len(thermos)
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    axes = axes.reshape(n, 2)

    for i, (I, z) in enumerate(zip(thermos, zs)):
        amp = np.sqrt(np.clip(I, 0, None))
        amp_db = amp_to_db(amp)

        im0 = axes[i, 0].imshow(amp, cmap="inferno", extent=EXTENT_MM, origin="upper")
        axes[i, 0].set_title(f"z={z*1e3:.0f} mm (linear)")
        axes[i, 0].axis("off")

        im1 = axes[i, 1].imshow(amp_db, cmap="inferno", extent=EXTENT_MM, origin="upper",
                                vmin=-40, vmax=0)
        axes[i, 1].set_title("dB")
        axes[i, 1].axis("off")

    # colorbar for the second column
    fig.colorbar(im1, ax=axes[:, 1].ravel().tolist(), fraction=0.015)
    plt.tight_layout(); plt.savefig(fname, dpi=200)
    print(f"[INFO] preview saved → {fname}")


### 4. phase retrieval by ER and HIO -----------------------------------------------------------------------------
def phase_retieval_ER_HIO(thermograms, zs, wavelength, tilt_x, tilt_y):
    """
    execute phase retrieval by Error Reduction and Fienup's Hybrid Input-Output algorithms
    """
    n_planes = len(zs)
    amplitudes = [np.sqrt(np.clip(I, 0, None)).astype(np.float64) for I in thermograms]   # A ≈ sqrt(ΔT)
    #u = amplitudes[0] * np.exp(1j * 2 * np.pi * np.random.rand(*amplitudes[0].shape))    # inital u with random initial φ
    u = amplitudes[0].astype(np.complex128) # flat phase is enough
    
    ERR_EPS = 1.0e-4
    err_history: List[float] = []       # amplitude error on each iteration
    
    HIO_CYCLES = 4
    ER_CYCLES  = 1
    cycle = HIO_CYCLES + ER_CYCLES      # 4 HIOs -> 1 ER
    
    HIO_PERCENT = 5.0
    iterator = tqdm(range(N_ITER), desc="ER + HIO", ncols = 80, leave=True)

    for i in iterator:
        use_HIO = (i % cycle) < HIO_CYCLES          # regulation to use HIO

        # ---- forward propagation ---------------------------------
        for k in range(n_planes-1):
            dz = zs[k+1] - zs[k] # propagation distance
            u = propagate_AS(u, float(dz), wavelength, DX, tilt_x, tilt_y)
            a = amplitudes[k+1]
            u_prop = u
            u_new = a * u_prop / np.maximum(np.abs(u_prop), 1e-20)
            
            if use_HIO:
                mask = a >= np.percentile(a, HIO_PERCENT)  # mask the region of < 0.01*Max
                u = np.where(mask, u_new, u_prop - HIO_BETA * u_new)
            else:
                u = u_new

        # ---- backward propagation --------------------------------
        for k in range(n_planes-1, 0, -1):
            dz = zs[k] - zs[k-1]
            u = propagate_AS(u, -float(dz), wavelength, DX, tilt_x, tilt_y)
            a = amplitudes[k-1]
            u_prop = u
            u_new = a * u_prop / np.maximum(np.abs(u_prop), 1e-20)
            
            if use_HIO:
                mask = a >= np.percentile(a, HIO_PERCENT)  # mask the region of < 0.01*Max
                u = np.where(mask, u_new, u_prop - HIO_BETA * u_new)
            else:
                u = u_new

        # ---- error evaluation ----------------------------------------
        err = amplitude_misfit(u, zs, amplitudes, wavelength, DX, tilt_x, tilt_y)
        err_history.append(err)
        
        #iterator.set_postfix({"err": f"{err:.4e}", "mode": "HIO" if use_HIO else "ER"})
              
        ###if err < ERR_EPS:
           # print("[INFO] converged.")
            #break
        
    return u, np.array(err_history)



### 5. evaluate LPmn purity --------------------------------------------------------------
def calc_LPmn_purity(u: np.ndarray, center: Tuple[float, float] = (0.0, 0.0)) -> Dict[str, float]:
    """
    calculate LPmn mode purity of the field
    """
    N, M = u.shape
    cx, cy = center
    
    x = (np.arange(N) - N/2)*DX - cx
    y = (np.arange(M) - M/2)*DX - cy
    X, Y = np.meshgrid(x, y, indexing="ij")
    R     = np.sqrt(X**2 + Y**2)
    Phi   = np.arctan2(Y, X)
    mask  = (R <= R_WG)

    dA  = DX*DX
    w   = 1   # Jacobian (need not to be considered)

    P_tot = np.sum(np.abs(u)**2 * w * mask) * dA

    # LP modes is aligned with GA's evaluation
    modes = [
        ("LP01", 0, 1, None),
        ("LP02", 0, 2, None),
        ("LP03", 0, 3, None),
        ("LP04", 0, 4, None),
        ("LP05", 0, 5, None),
        ("LP06", 0, 6, None),
        ("LP07", 0, 7, None),
        ("LP11even", 1, 1, "cos"),
        ("LP11odd",  1, 1, "sin"),
        ("LP12even", 1, 2, "cos"),
        ("LP12odd",  1, 2, "sin"),
        ("LP13even", 1, 3, "cos"),
        ("LP13odd",  1, 3, "sin"),
        ("LP21even", 2, 1, "cos"),
        ("LP21odd",  2, 1, "sin"),
        ("LP22even", 2, 2, "cos"),
        ("LP22odd",  2, 2, "sin"),
        ("LP31even", 3, 1, "cos"),
        ("LP31odd",  3, 1, "sin"),
    ]

    result = {}
    for name, m, n, parity in modes:
        χmn = jn_zeros(m, n)[-1]
        V   = jn(m, χmn*R/R_WG)
        if m >= 1:
            V *= np.cos(m*Phi) if parity=="cos" else np.sin(m*Phi)
        V   *= mask                     

        c_mn = np.sum(u.conj()*V * w * mask) * dA
        P_mn = np.abs(c_mn)**2 / (                       # 
                (np.sum(np.abs(V)**2 * w * mask)*dA)     # ∫|V|²
                * P_tot)                                 # ∫|u|²
        result[name] = 100.0*P_mn # [%]

    return result


### 6. calculate amplitude error ------------------------------------------------------
def amplitude_misfit(u0: np.ndarray, zs: List[float], amps: List[np.ndarray],
                     λ: float, dx: float, θx: float, θy: float) -> float:
    """
    return amplitude RMS relative error on all measurement points
    """
    mis2, ref2 = 0.0, 0.0
    u = u0.copy()
    for k in range(len(zs)):
        if k > 0:
            dz = zs[k] - zs[k-1]
            u = propagate_AS(u, float(dz), λ, dx, θx, θy)
        mis2 += np.sum((np.abs(u) - amps[k])**2)
        ref2 += np.sum(amps[k]**2)
    return np.sqrt(mis2 / ref2)

def amp_to_db(A: np.ndarray, floor_db: float = -40.0) -> np.ndarray:
    """
    convert data into dB, and clip values below floor_db.
    """
    Amax = A.max()
    with np.errstate(divide="ignore"):
        Adb = 20.0 * np.log10(np.clip(A / Amax, 1e-12, None))
    return np.maximum(Adb, floor_db)

def pixel_coords(shape: tuple[int,int]) -> tuple[np.ndarray,np.ndarray]:
    """
    return center coords (X, Y)
    """
    N, M = shape
    x = (np.arange(N) - N/2 + 0.5) * DX   # cell center
    y = (np.arange(M) - M/2 + 0.5) * DX
    return np.meshgrid(x, y, indexing="ij")

def centroid(u: np.ndarray) -> tuple[float,float]:
    """
    return beam center
    """
    X, Y = pixel_coords(u.shape)
    w = np.abs(u)**2
    s = w.sum()
    return float((w*X).sum()/s), float((w*Y).sum()/s)


### 7. visualization --------------------------------------------------------------
def save_amp_phase(u: np.ndarray, fname: Path = OUT_DIR / "reconstruction.png"):
    amp_lin  = np.abs(u)
    amp_db   = amp_to_db(amp_lin)
    phase    = np.angle(u)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    ax1.imshow(amp_lin, cmap="inferno", extent=EXTENT_MM, origin="upper")
    ax1.set_title("|E| linear"); ax1.set_xlabel("x [mm]"); ax1.set_ylabel("y [mm]")

    im2 = ax2.imshow(amp_db, cmap="inferno", extent=EXTENT_MM, origin="upper",
                     vmin=-40, vmax=0)
    ax2.set_title("|E| (dB)"); ax2.set_xlabel("x [mm]")

    im3 = ax3.imshow(phase, cmap="hsv", extent=EXTENT_MM, origin="upper")
    ax3.set_title("Phase");  ax3.set_xlabel("x [mm]")

    fig.colorbar(im2, ax=ax2, fraction=0.046)
    fig.colorbar(im3, ax=ax3, fraction=0.046)
    
    fig.tight_layout(); fig.savefig(fname, dpi=300); plt.close(fig)
    print(f"[INFO] {fname} saved.")


### 8. command line interface --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="ER+HIO phase retrieval")
    ap.add_argument("csv_dir", help="directory where 256x256 ΔT thermograms are uploaded")
    ap.add_argument("--freq",  required=True, type=float, help="frequency [GHz]")
    args = ap.parse_args()

    csv_path = Path(args.csv_dir)
    if not csv_path.is_dir():
        sys.exit("[ERROR] specified path is not a directory.")

    # load CSVs
    thermograms, zs = load_csv_stack(csv_path)
    
    # save CSV preview
    save_preview(thermograms, zs)

    # calculate wavelength from frequency 
    wavelength = C_LIGHT / (args.freq * 1.0e9)
    print(f"[INFO] f = {args.freq:.3f} GHz → λ = {wavelength*1e3:.3f} mm")
    print(f"[INFO] planes loaded: {len(zs)}, first z = {zs[0]*1e3:.1f} mm")
    
    # calculate beam tilt
    tilt_x, tilt_y = calc_beam_tilt(thermograms, zs)
    print(f"[INFO] beam tilt: {tilt_x*180/np.pi:.2f}° horizontal, {tilt_y*180/np.pi:.2f}° vertical")

    # normalize plane power for robust phase retrieval
    thermograms_pr, power_scales = normalize_thermograms_by_power(thermograms, ref_index=0)
    print("[INFO] power normalization scales:", ", ".join(f"{s:.4f}" for s in power_scales))
    
    # phase retrieval
    u_pr, err_hist = phase_retieval_ER_HIO(thermograms_pr, zs, wavelength, tilt_x, tilt_y)
    np.savetxt(OUT_DIR/"error_history.txt", err_hist)
    plt.figure(); plt.semilogy(err_hist)
    plt.xlabel("iteration"); plt.ylabel("amplitude error")
    plt.tight_layout(); plt.savefig(OUT_DIR/"error_curve.png", dpi=200)
    print("[INFO] error curve saved → error_curve.png")
    
    # propagate to z = 0
    dz_ref = Z_REF - zs[0]
    u_ref  = propagate_AS(u_pr, dz_ref, wavelength, DX, tilt_x, tilt_y)
    print(f"[INFO] propagated to z = {Z_REF*1e3:.1f} mm (Δz = {dz_ref*1e3:.1f} mm)")
    
    ## evaluate mode purity -----------------
    pur_axis = calc_LPmn_purity(u_ref, center=(0.0, 0.0))
    print("\n== Axis-centred LPmn purity ==")
    for k, v in pur_axis.items():
        print(f"{k:<9}: {v:.4f} %")
    
    # evaluate again but ideal LP modes are defined on the beam centroid
    cx, cy = centroid(u_ref)
    pur_cen = calc_LPmn_purity(u_ref, center=(cx, cy))
    print(f"\nbeam centroid = ({cx*1e3:.2f} mm, {cy*1e3:.2f} mm)")
    print("== Centroid-centred LPmn purity ==")
    for k, v in pur_cen.items():
        print(f"{k:<9}: {v:.4f} %")
    
    # save purity indormation on txt
    purity_path = OUT_DIR / "mode_purity.txt"
    try:
        with purity_path.open("w", encoding="utf-8") as f:
            # --- axis-centered
            f.write("== Axis-centred LPmn purity ==\n")
            for k, v in pur_axis.items():
                f.write(f"{k:<9}: {(v*0.01):7.6f}\n")

            f.write("\n")

            # --- beam-centered
            f.write(f"beam centroid = ({cx*1e3:.2f} mm, {cy*1e3:.2f} mm)\n")
            f.write("== Centroid-centred LPmn purity ==\n")
            for k, v in pur_cen.items():
                f.write(f"{k:<9}: {(v*0.01):7.6f}\n")
    except OSError as e:
        print(f"[ERROR] cannot write mode_purity.txt → {e}")
    else:
        print(f"[INFO] mode purity saved → {purity_path.resolve()}")

    save_amp_phase(u_ref)
    
    comp_dir = OUT_DIR / "comparison"
    comp_dir.mkdir(exist_ok=True)

    # save A and phase
    u_tmp = u_pr.copy()
    for k, (z_plane, thermo) in enumerate(zip(zs, thermograms_pr)):
        if k > 0:
            dz = zs[k] - zs[k-1]
            u_tmp = propagate_AS(u_tmp, float(dz), wavelength,
                                 DX, tilt_x, tilt_y)

        A_input_lin = np.sqrt(np.clip(thermo, 0, None))
        A_input_db  = amp_to_db(A_input_lin)

        A_rec_lin   = np.abs(u_tmp)
        A_rec_db    = amp_to_db(A_rec_lin)

        fig, axes = plt.subplots(2, 3, figsize=(9, 6), sharex=True, sharey=True)
        ax1, ax2, ax0 = axes[0]
        ax3, ax4, ax5 = axes[1]

        im1 = ax1.imshow(A_input_lin, cmap="inferno",
                         extent=EXTENT_MM, origin="upper")
        ax1.set_title("input lin")
        ax1.set_xlabel("x [mm]");  ax1.set_ylabel("y [mm]")
        
        im2 = ax2.imshow(A_input_db, cmap="inferno", vmin=-40, vmax=0,
                         extent=EXTENT_MM, origin="upper")
        ax2.set_title("input dB")
        ax2.set_xlabel("x [mm]")
        
        ax0.axis("off")

        im3 = ax3.imshow(A_rec_lin, cmap="inferno",
                         extent=EXTENT_MM, origin="upper")
        ax3.set_title("retrieved lin")
        ax3.set_xlabel("x [mm]");  ax3.set_ylabel("y [mm]")

        im4 = ax4.imshow(A_rec_db, cmap="inferno", vmin=-40, vmax=0,
                         extent=EXTENT_MM, origin="upper")
        ax4.set_title("retrieved dB")
        ax4.set_xlabel("x [mm]")

        im5 = ax5.imshow(np.angle(u_tmp), cmap="hsv",
                         extent=EXTENT_MM, origin="upper")
        ax5.set_title("phase")
        ax5.set_xlabel("x [mm]")

        # colorbars
        # dB for input/retrieved data
        fig.colorbar(im4, ax=ax2, fraction=0.046, pad=0.04,
                     label="|E| [dB]")
        # for phase
        fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04,
                     label="Phase [rad]")

        out_png = comp_dir / f"compare_{k:02d}.png"
        #fig.tight_layout()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        print(f"[INFO] comparison saved → {out_png}")

if __name__ == "__main__":
    main()
