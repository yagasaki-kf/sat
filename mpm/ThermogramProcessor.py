"""
    ThermogramProcessor.py
    author:   Konan Yagasaki (kyoto Fusioneering Ltd.)
    version:  1.0 (1 May 2025)
    ---
    preprocess thermograms (**.CSV) by following procedures:
    ex) 100mm_202405131202_1082f.CSV ... [z position]_[date]_[frame].CSV

    0) export thermogram data from InfRec Analyzer
    1) preprocess header row/column of CSV file
    2) get ΔT by subtracting background from measured data
    3) do perspective transformation using cv2 based on the corner coordinates (**.PPF) ... [z_position]_[date].PPF
    ---
    usage:   python3 ThermogramProcessor.py --beam [measured].CSV --bg [background].CSV --ppf [corners].PPF
    package: pandas, numpy, matplotlib, scikit-image, opencv-python

    this is integrated version of previous PerspectiveTransformation.py and other scripts
"""
import argparse, os, tempfile
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
from matplotlib.path import Path as MplPath


### A. preprocess the input data -----------------------------------------------------------------------------------------
##  A-1. CSVs ------------------------------------------------
def initial_cleanup(path: Path, *, encoding="utf-8"):
    """
    clean up unnecessary header columns and rows on the CSV output from InfRec analyzer
    -------
    df :       pandas.DataFrame
    modified : bool ... True -> CSV has been cleaned up, and the boolean works as the sign of overwriting
    """
    # open CSV file on the path and read with UTF-8 encoding
    with open(path, encoding=encoding, errors="ignore") as f:   # errors="ignore" ignores undecodable string
        lines = f.readlines()                                   # storage lines

    # find a column including "y/x", return None if not
    yx_idx = next((i for i, ln in enumerate(lines) if "y/x" in ln), None)

    # if y/x was found, eliminate "y/x" coliumn and all above
    if yx_idx is not None:
        df = (
            pd.read_csv(
                StringIO("".join(lines[yx_idx + 1 :])), # reconstruct columns below "y/x"
                sep="\t",               # delimiter should be TAB
                header=None,
                engine="python",
            )
            .iloc[:, 1:]  # eliminate header index row
        )
        print("eliminate header column/row")
        return df, True

    # if t/x was not found, read without any cleaning up
    df = pd.read_csv(path, sep="\t", header=None, engine="python")
    return df, False


def inplace_save_csv(df: pd.DataFrame, path: Path):
    """
    overwrite DataFrame with delimiter TAB safely = not destructive
    - generate tmp file on the same directrory
    """
    with tempfile.NamedTemporaryFile(   # generate OS-assured safe tmp file
        "w", delete=False, dir=path.parent, encoding="utf-8", newline=""    # newline = "" ... invalidate new line and keep \n
    ) as tmp:
        df.to_csv(tmp, sep="\t", index=False, header=False)     # write DataFrame to CSV with delimiter TAB
        tmp_path = tmp.name
    os.replace(tmp_path, path)          # atomic replacement! yeah


##  A-2. PPF (corner file) -------------------------------------------
def read_ppf_corners(ppf_path: Path):
    """
    read PPF file including corner points

    4
    28,11,0.95,24,16777215    # upper-left  <---- first 2 rows are corner points (28, 11)
    255,12,0.95,24,16777215   # upper-right
    251,208,0.95,24,16777215  # lower-right
    27,205,0.95,24,16777215   # lower-left
    """
    with open(ppf_path, encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()[1:5]  # pick row 2-5 up by slicing

    pts = []
    for ln in lines:
        x, y, *_ = ln.replace(",", " ").split() # convert "," -> " " and split token, and then unpack 1 -> x, 2 -> y (after 3, ignore)
        pts.append([float(x), float(y)])        # append (x, y) pair

    if len(pts) != 4:   # if PPF doesn't have 4 corner data
        raise ValueError("corner data is missing... can't read PPF file correctly")
    
    return np.array(pts, dtype=np.float32)  # (4, 2) array of Float32



### B. Utility for Visualization -----------------------------------------------------------------------------
_cmap = mpl.colormaps.get_cmap("jet").copy()   # specify jet colormap (nandemo iikedone)

##  B-1. save contour (normal)
def save_contour(data, png_path, title=""):
    """
    save contour
    """
    plt.figure(figsize=(5, 4))
    plt.contourf(data, levels=200, cmap=_cmap)
    plt.title(title)
    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()

##  B-1. save contour (with overlaying a frame rectangle)
def save_contour_overlay(data, corners, png_path, title=""):
    """
    save contourwith overlaying a frame rectangle
    """
    plt.figure(figsize=(5, 4))
    plt.contourf(data, levels=200, cmap=_cmap)
    xs = list(corners[:, 0]) + [corners[0, 0]]
    ys = list(corners[:, 1]) + [corners[0, 1]]
    plt.plot(xs, ys, lw=0.5, color="white")
    plt.scatter(corners[:, 0], corners[:, 1], s=10, color="white", edgecolors="white")
    plt.title(title)
    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()



### ----------------------------------------------------------------------------------------------
### main function --------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="thermogram processor with OpenCV")
    ap.add_argument("--beam", required=True, help="measured data (.CSV)")   # these three are essential argument
    ap.add_argument("--bg",   required=True, help="background data (.CSV)")
    ap.add_argument("--ppf",  required=True, help="corner data (.ppf)")
    args = ap.parse_args()  # parse argument and storage 

    # set path as absolute path
    beam_path = Path(args.beam).resolve()
    bg_path   = Path(args.bg).resolve()
    ppf_path  = Path(args.ppf).resolve()

    base    = beam_path.stem        # hoge.CSV -> "hoge"
    out_dir = beam_path.parent      # output to the same directory


    ##  1. preprocessing -----------------------------------------------------------------------------
    df_beam, mod_beam = initial_cleanup(beam_path)
    if mod_beam:    # if True, save in-place
        inplace_save_csv(df_beam, beam_path)

    df_bg, mod_bg = initial_cleanup(bg_path)
    if mod_bg:      # if True, save in-place
        inplace_save_csv(df_bg, bg_path)


    ##  2. convert DataFrame -> NumPy array -------------------------
    A = df_beam.to_numpy(float)
    B = df_bg.to_numpy(float)
    if A.shape != B.shape:
        raise ValueError("the size of the beam data is not the same with that of background... recheck data")


    ##  3. evaluate ΔT by subtract B (BG) from A (beam data) -----------------------------------------
    deltaT = A - B
    deltaT_path = out_dir / f"{base}_deltaT.csv"
    pd.DataFrame(deltaT).to_csv(deltaT_path, sep="\t", index=False, header=False)   # wrap dataFrame and save with delimiter TAB


    ##  4. set region of interest (ROI) mask out-of-bounds data --------------------------------------
    h, w = deltaT.shape                             # get (rows, cols)
    corners = read_ppf_corners(ppf_path).copy()
    # corners[:, 1] = (h - 1) - corners[:, 1]  # removed: flipud も廃止したため不要

    # reorder corner points to UL > UR > LR > LL
    def order_corners(pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype=np.float32)

        s    = pts.sum(axis=1)         # y + x
        diff = np.diff(pts, axis=1)    # y - x

        rect[0] = pts[np.argmin(s)]    # upper-left  (minimum y+x)
        rect[2] = pts[np.argmax(s)]    # lower-right (maximum y+x)
        rect[1] = pts[np.argmin(diff)] # upper-right (minimum y-x)
        rect[3] = pts[np.argmax(diff)] # lower-left  (maximum y-x)
        return rect

    corners = order_corners(corners)

    # generate mask
    yy, xx = np.mgrid[0:h, 0:w]     # (h x w) 2D grid
    mask = MplPath(corners).contains_points(    # check whether the point is contained in the 
        np.c_[xx.ravel(), yy.ravel()]
    ).reshape(h, w)

    deltaT_trim = np.where(mask, deltaT, 0.0)           # convert out-of-bounds data to zero
    trim_path = out_dir / f"{base}_deltaT_trim.csv"
    pd.DataFrame(deltaT_trim).to_csv(trim_path, sep="\t", index=False, header=False)


    ##  5. execute perspective transformation (OpenCV, 256 x 256) ------------------------------------
    dst = np.array([[0, 0], [255, 0], [255, 255], [0, 255]], dtype=np.float32)  # corresponding point on output image

    M = cv2.getPerspectiveTransform(corners, dst)   # calc 3x3 homography matrix, M
    deltaT_pt = cv2.warpPerspective(
        deltaT_trim.astype(np.float32),     # input image (must be Float32)
        M,                                  # homography
        (256, 256),                         # output image siza (width x height)
        flags=cv2.INTER_CUBIC,              # interpolation ... using INTER_CUBIC, but idk how much difference btw INTER_LINEAR
        borderMode=cv2.BORDER_CONSTANT,     # all constant (0) on the out of bounds
        borderValue=0,
    )
    pt_path = out_dir / f"{base}_deltaT_PTed.csv"   # PerspectiveTransform-ed
    pd.DataFrame(deltaT_pt).to_csv(pt_path, sep="\t", index=False, header=False)


    ##  6. output contour as PNG
    save_contour(A, out_dir / f"{base}.png",    "beam pattern")     # input thermogram
    save_contour(B, out_dir / f"{base}_BG.png", "background data")  # input BG thermogram

    save_contour(deltaT,   out_dir / f"{base}_deltaT.png",      "ΔT profile")  # ΔT profile
    save_contour_overlay(deltaT,                                               # ΔT profile with frame
                         corners,
                         out_dir / f"{base}_deltaT_trim.png",
                         "ΔT profile with a frame rectangle",
    )
    save_contour(deltaT_pt, out_dir / f"{base}_deltaT_PTed.png", "ΔT profile (perspective transformed)")

    print("process finished -", beam_path)

if __name__ == "__main__":
    main()
