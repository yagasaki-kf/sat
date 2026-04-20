"""
CircularTargetPT.py

Web GUI for perspective transformation of thermograms projected on a circular target.
The target is defined by four marks: Top, Right, Bottom, Left.

Run:
    python3 CircularTargetPT.py
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import cv2
except ModuleNotFoundError as exc:
    raise SystemExit(
        "opencv-python is required. Install with: python3 -m pip install opencv-python"
    ) from exc

try:
    from nicegui import events, ui
except ModuleNotFoundError as exc:
    raise SystemExit(
        "nicegui is required. Install with: python3 -m pip install nicegui"
    ) from exc

try:
    from ThermogramProcessor import initial_cleanup
except ModuleNotFoundError as exc:
    raise SystemExit(
        "ThermogramProcessor.py was not found in this directory."
    ) from exc


ROOT_DIR = Path(__file__).resolve().parent
POINT_NAMES = ["Top", "Right", "Bottom", "Left"]


class State:
    source_data: np.ndarray | None = None
    transformed_data: np.ndarray | None = None
    source_path: Path | None = None
    output_dir: Path | None = None
    points: list[list[float]] = []


S = State()


def order_circle_points(pts: np.ndarray) -> np.ndarray:
    top = pts[np.argmin(pts[:, 1])]
    bottom = pts[np.argmax(pts[:, 1])]
    left = pts[np.argmin(pts[:, 0])]
    right = pts[np.argmax(pts[:, 0])]
    return np.array([top, right, bottom, left], dtype=np.float32)


def default_points(shape: tuple[int, int]) -> list[list[float]]:
    h, w = shape
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    rx = 0.33 * w
    ry = 0.33 * h
    return [
        [cx, cy - ry],  # Top
        [cx + rx, cy],  # Right
        [cx, cy + ry],  # Bottom
        [cx - rx, cy],  # Left
    ]


def ellipse_curve_from_four_points(points_trbl: np.ndarray, n: int = 360) -> np.ndarray | None:
    if points_trbl.shape != (4, 2):
        return None

    x = points_trbl[:, 0]
    y = points_trbl[:, 1]
    m = np.column_stack([x * x, x * y, y * y, x, y, np.ones_like(x)])
    _, _, vh = np.linalg.svd(m, full_matrices=True)
    null = vh[-2:, :].T

    mean_xy = points_trbl.mean(axis=0)
    best_curve: np.ndarray | None = None
    best_score = np.inf

    for th in np.linspace(0.0, np.pi, 721):
        co = np.cos(th)
        si = np.sin(th)
        coeff = null @ np.array([co, si], dtype=float)
        a, b, c, d, e, f = [float(v) for v in coeff]

        q = np.array([[a, 0.5 * b], [0.5 * b, c]], dtype=float)
        try:
            q_inv = np.linalg.inv(q)
        except np.linalg.LinAlgError:
            continue

        qv = np.array([0.5 * d, 0.5 * e], dtype=float)
        center = -q_inv @ qv
        k = float(qv @ q_inv @ qv - f)

        eigvals, eigvecs = np.linalg.eigh(q)
        if np.any(eigvals <= 1e-12):
            continue
        if k <= 1e-12:
            continue

        radii = np.sqrt(k / eigvals)
        t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=True)
        unit = np.vstack([radii[0] * np.cos(t), radii[1] * np.sin(t)])
        curve = (eigvecs @ unit).T + center[None, :]

        ratio = float(max(radii) / max(min(radii), 1e-12))
        score = float(np.sum((center - mean_xy) ** 2) + 1e-3 * ratio)
        if score < best_score:
            best_score = score
            best_curve = curve

    return best_curve


def draw_preview(
    data: np.ndarray,
    points: np.ndarray | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    draw_reference_circle: bool = False,
) -> str:
    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=120)
    ax.imshow(data, cmap="plasma", origin="upper", vmin=vmin, vmax=vmax)
    ax.set_xlabel("x [pixel]")
    ax.set_ylabel("y [pixel]")
    ax.grid(True, which="major", color="white", alpha=0.2, linewidth=0.6)

    if points is not None and len(points) == 4:
        ax.scatter(points[:, 0], points[:, 1], color="cyan", s=36)
        for i, (x, y) in enumerate(points):
            ax.text(x + 2, y + 2, str(i + 1), color="white", fontsize=9)
        p = order_circle_points(points)
        top, right, bottom, left = p
        ax.plot([left[0], right[0]], [left[1], right[1]], color="cyan", lw=1.0)
        ax.plot([top[0], bottom[0]], [top[1], bottom[1]], color="cyan", lw=1.0)

    if draw_reference_circle:
        h, w = data.shape
        m = float(min(w, h) - 1)
        c = m * 0.5
        margin = m * 0.08
        r = max(c - margin, 1.0)
        circ = plt.Circle((c, c), r, color="cyan", fill=False, lw=1.6)
        ax.add_patch(circ)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def get_plot_range(data: np.ndarray, tmin_value: float, tmax_value: float) -> tuple[float, float]:
    lo = float(tmin_value)
    hi = float(tmax_value)
    if not np.isfinite(lo):
        lo = float(np.nanmin(data))
    if not np.isfinite(hi):
        hi = float(np.nanmax(data))
    if hi <= lo:
        hi = lo + 1e-9
    return lo, hi


def get_transform(src_data: np.ndarray, points: np.ndarray, output_px: int) -> tuple[np.ndarray, np.ndarray]:
    src = order_circle_points(points.astype(np.float32))
    m = float(output_px - 1)
    c = m * 0.5
    margin = m * 0.08
    dst = np.array(
        [[c, margin], [m - margin, c], [c, m - margin], [margin, c]],
        dtype=np.float32,
    )

    mat = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        src_data.astype(np.float32),
        mat,
        (output_px, output_px),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped, mat


def csv_folders_under_root() -> list[str]:
    dirs: set[str] = set()
    for p in ROOT_DIR.rglob("*"):
        if not p.is_file() or p.suffix.lower() != ".csv":
            continue
        try:
            rel = p.parent.relative_to(ROOT_DIR)
            dirs.add(str(rel))
        except ValueError:
            continue
    return sorted(dirs)


def read_ppf_clockwise_top(ppf_path: Path) -> list[list[float]]:
    """Read first 4 points from PPF in order: Top -> Right -> Bottom -> Left."""
    with open(ppf_path, encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if len(lines) < 5:
        raise ValueError(f"PPF must include 4 point lines: {ppf_path}")

    points: list[list[float]] = []
    for ln in lines[1:5]:
        parts = ln.replace(",", " ").split()
        if len(parts) < 2:
            raise ValueError(f"Invalid PPF line: {ln}")
        x = float(parts[0])
        y = float(parts[1])
        points.append([x, y])

    return points


def main() -> None:
    ui.label("Circular Target Perspective Transformation").classes("text-h5")
    ui.label("Top/Right/Bottom/Left marks define the circular target for frontal reconstruction.").classes(
        "text-body2 text-grey-8"
    )

    with ui.card().classes("w-full"):
        with ui.row().classes("items-end"):
            csv_path = ui.input("Input CSV path").classes("w-80")
            folder_select = ui.select(options=[], label="Folder from ModePurityMeasurement").classes("w-80")
            csv_select = ui.select(options=[], label="CSV file in folder").classes("w-96")
            ppf_select = ui.select(options=[], label="PPF file in folder (optional)").classes("w-96")
            refresh_btn = ui.button("Refresh list")
            out_dir = ui.input("Output directory", value=str(Path.cwd() / "output_circle_pt")).classes("w-96")
        with ui.row().classes("items-end"):
            output_px = ui.number("Output size [pixel]", value=256, min=64, max=2048, step=1, format="%d")
            tmin = ui.number("T min", value=0.0, step=0.1, format="%.3f").classes("w-40")
            tmax = ui.number("T max", value=1.0, step=0.1, format="%.3f").classes("w-40")
            auto_range_btn = ui.button("Auto range")
            load_btn = ui.button("Load CSV", color="primary")
            reset_btn = ui.button("Reset points")
            transform_btn = ui.button("Transform", color="secondary")
            save_btn = ui.button("Save", color="positive")

    point_labels: list = []
    point_x_inputs: list = []
    point_y_inputs: list = []

    with ui.row().classes("w-full items-start"):
        with ui.card().classes("w-1/2"):
            ui.label("Original").classes("text-subtitle1")
            with ui.row().classes("w-full items-start no-wrap"):
                with ui.column().classes("w-2/3"):
                    source_img = ui.interactive_image(source="")
                    source_img.style("width:100%; max-width:760px;")
                    point_info = ui.label("Points: 0 / 4")
                    mouse_pos = ui.label("Mouse: x=-, y=-")
                    active_point = ui.select(
                        options={0: "Point1 Top", 1: "Point2 Right", 2: "Point3 Bottom", 3: "Point4 Left"},
                        value=0,
                        label="Point to set by click",
                    ).classes("w-64")
                with ui.column().classes("w-1/3"):
                    ui.label("Point Coordinates").classes("text-subtitle2")
                    for _ in range(4):
                        with ui.row().classes("items-end"):
                            point_labels.append(ui.label("-").classes("w-24"))
                            point_x_inputs.append(ui.number("x [px]", value=0.0, step=1.0, format="%.2f").classes("w-32"))
                            point_y_inputs.append(ui.number("y [px]", value=0.0, step=1.0, format="%.2f").classes("w-32"))
        with ui.card().classes("w-1/2"):
            ui.label("Transformed (frontal)").classes("text-subtitle1")
            transformed_img = ui.image(source="")
            transformed_img.style("width:100%; max-width:760px;")
            save_info = ui.label("Output: -")

    log = ui.log(max_lines=300).classes("w-full").style("height:180px;")
    syncing_point_inputs = False

    def clamp_points() -> None:
        if S.source_data is None:
            return
        h, w = S.source_data.shape
        for i in range(min(4, len(S.points))):
            x, y = S.points[i]
            S.points[i] = [float(np.clip(x, 0, w - 1)), float(np.clip(y, 0, h - 1))]

    def update_point_inputs_from_state() -> None:
        nonlocal syncing_point_inputs
        syncing_point_inputs = True
        for i in range(4):
            point_labels[i].set_text(f"{i+1} ({POINT_NAMES[i]})")
            if i < len(S.points):
                point_x_inputs[i].set_value(float(S.points[i][0]))
                point_y_inputs[i].set_value(float(S.points[i][1]))
            else:
                point_x_inputs[i].set_value(0.0)
                point_y_inputs[i].set_value(0.0)
        syncing_point_inputs = False

    def update_points_from_inputs() -> None:
        if syncing_point_inputs:
            return
        if S.source_data is None:
            return
        S.points = [
            [float(point_x_inputs[i].value or 0.0), float(point_y_inputs[i].value or 0.0)]
            for i in range(4)
        ]
        clamp_points()
        update_source_view()
        S.transformed_data = None
        update_transformed_view()

    def update_source_view() -> None:
        if S.source_data is None:
            source_img.set_source("")
            point_info.set_text("Points: 0 / 4")
            return
        lo, hi = get_plot_range(S.source_data, float(tmin.value), float(tmax.value))
        pts = np.array(S.points, dtype=np.float32) if len(S.points) == 4 else None
        source_img.set_source(draw_preview(S.source_data, points=pts, vmin=lo, vmax=hi))
        point_info.set_text(f"Points: {len(S.points)} / 4")

    def update_transformed_view() -> None:
        if S.transformed_data is None:
            transformed_img.set_source("")
            return
        lo, hi = get_plot_range(S.transformed_data, float(tmin.value), float(tmax.value))
        transformed_img.set_source(
            draw_preview(
                S.transformed_data,
                vmin=lo,
                vmax=hi,
                draw_reference_circle=True,
            )
        )

    def set_auto_range() -> None:
        if S.source_data is None:
            return
        lo = float(np.nanmin(S.source_data))
        hi = float(np.nanmax(S.source_data))
        if hi <= lo:
            hi = lo + 1e-9
        tmin.set_value(lo)
        tmax.set_value(hi)
        update_source_view()
        update_transformed_view()

    def on_mouse(e: events.MouseEventArguments) -> None:
        if S.source_data is None or len(S.points) != 4:
            mouse_pos.set_text("Mouse: x=-, y=-")
            return
        x = float(e.image_x)
        y = float(e.image_y)
        if np.isfinite(x) and np.isfinite(y):
            mouse_pos.set_text(f"Mouse: x={x:.1f}, y={y:.1f}")
            if str(e.type).lower() == "click":
                idx = int(active_point.value)
                h, w = S.source_data.shape
                S.points[idx] = [float(np.clip(x, 0, w - 1)), float(np.clip(y, 0, h - 1))]
                update_point_inputs_from_state()
                update_source_view()
        else:
            mouse_pos.set_text("Mouse: x=-, y=-")

    source_img.on_mouse(on_mouse)

    def refresh_csv_menu() -> None:
        if not folder_select.value:
            csv_select.options = []
            ppf_select.options = []
            csv_select.update()
            ppf_select.update()
            return
        folder = ROOT_DIR / str(folder_select.value)
        csv_files = sorted([p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".csv"])
        ppf_files = sorted([p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".ppf"])
        csv_select.options = csv_files
        ppf_select.options = ppf_files
        if csv_files and csv_select.value not in csv_files:
            csv_select.set_value(csv_files[0])
        if not csv_files:
            csv_select.set_value(None)
        if ppf_files and ppf_select.value not in ppf_files:
            ppf_select.set_value(ppf_files[0])
        if not ppf_files:
            ppf_select.set_value(None)
        csv_select.update()
        ppf_select.update()

    def refresh_folder_menu() -> None:
        folders = csv_folders_under_root()
        folder_select.options = folders
        if folders and folder_select.value not in folders:
            folder_select.set_value("reference" if "reference" in folders else folders[0])
        folder_select.update()
        refresh_csv_menu()

    def apply_selected_csv_path() -> None:
        if folder_select.value and csv_select.value:
            p = ROOT_DIR / str(folder_select.value) / str(csv_select.value)
            csv_path.set_value(str(p.resolve()))

    def on_load() -> None:
        try:
            apply_selected_csv_path()
            p = Path(csv_path.value).expanduser().resolve()
            if not p.is_file():
                raise FileNotFoundError(f"CSV not found: {p}")
            df, _ = initial_cleanup(p)
            S.source_data = df.to_numpy(float)
            S.source_path = p
            S.output_dir = Path(out_dir.value).expanduser().resolve()
            S.points = default_points(S.source_data.shape)
            if folder_select.value and ppf_select.value:
                ppf_path = ROOT_DIR / str(folder_select.value) / str(ppf_select.value)
                if ppf_path.is_file():
                    S.points = read_ppf_clockwise_top(ppf_path)
                    log.push(f"[INFO] loaded ppf: {ppf_path}")
            S.transformed_data = None
            clamp_points()
            update_point_inputs_from_state()
            set_auto_range()
            update_source_view()
            update_transformed_view()
            log.push(f"[INFO] loaded: {p} shape={S.source_data.shape}")
        except Exception as ex:
            log.push(f"[ERROR] load failed: {ex}")
            ui.notify(f"Load failed: {ex}", color="negative")

    def on_reset() -> None:
        if S.source_data is None:
            return
        S.points = default_points(S.source_data.shape)
        S.transformed_data = None
        update_point_inputs_from_state()
        update_source_view()
        update_transformed_view()
        log.push("[INFO] points reset")

    def on_transform() -> None:
        try:
            if S.source_data is None:
                raise RuntimeError("load CSV first")
            if len(S.points) != 4:
                raise RuntimeError("4 points are required")
            pts = np.array(S.points, dtype=np.float32)
            warped, _ = get_transform(S.source_data, pts, int(output_px.value))
            S.transformed_data = warped
            update_transformed_view()
            log.push(f"[INFO] transformed: out={int(output_px.value)}x{int(output_px.value)}")
        except Exception as ex:
            log.push(f"[ERROR] transform failed: {ex}")
            ui.notify(f"Transform failed: {ex}", color="negative")

    def on_save() -> None:
        try:
            if S.transformed_data is None:
                raise RuntimeError("run transform first")
            if S.source_path is None:
                raise RuntimeError("source path missing")

            outp = S.output_dir or (Path.cwd() / "output_circle_pt")
            outp.mkdir(parents=True, exist_ok=True)

            stem = S.source_path.stem
            csv_out = outp / f"{stem}_PT_CIRCLE.csv"
            png_out = outp / f"{stem}_PT_CIRCLE.png"
            src_png_out = outp / f"{stem}.png"
            pts_out = outp / f"{stem}_PT_CIRCLE_points.txt"

            pd.DataFrame(S.transformed_data).to_csv(csv_out, sep="\t", index=False, header=False)

            lo, hi = get_plot_range(S.transformed_data, float(tmin.value), float(tmax.value))
            fig, ax = plt.subplots(figsize=(5.4, 4.4), dpi=160)
            ax.imshow(S.transformed_data, cmap="plasma", origin="upper", vmin=lo, vmax=hi)
            h, w = S.transformed_data.shape
            m = float(min(w, h) - 1)
            c = m * 0.5
            margin = m * 0.08
            r = max(c - margin, 1.0)
            ax.add_patch(plt.Circle((c, c), r, color="cyan", fill=False, lw=1.6))
            ax.set_title("Circular Target Perspective Transform")
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(png_out)
            plt.close(fig)

            if S.source_data is not None and len(S.points) == 4:
                lo_src, hi_src = get_plot_range(S.source_data, float(tmin.value), float(tmax.value))
                src_pts = np.array(S.points, dtype=np.float32)
                p = order_circle_points(src_pts)
                top, right, bottom, left = p

                fig0, ax0 = plt.subplots(figsize=(5.4, 4.4), dpi=160)
                ax0.imshow(S.source_data, cmap="plasma", origin="upper", vmin=lo_src, vmax=hi_src)
                ax0.scatter(src_pts[:, 0], src_pts[:, 1], color="cyan", s=36)
                for i, (x, y) in enumerate(src_pts):
                    ax0.text(float(x) + 2, float(y) + 2, str(i + 1), color="white", fontsize=9)
                ax0.plot([left[0], right[0]], [left[1], right[1]], color="cyan", lw=1.0)
                ax0.plot([top[0], bottom[0]], [top[1], bottom[1]], color="cyan", lw=1.0)
                ax0.set_title("Original Thermogram")
                ax0.axis("off")
                fig0.tight_layout()
                fig0.savefig(src_png_out)
                plt.close(fig0)

            with open(pts_out, "w", encoding="utf-8") as f:
                for i, (x, y) in enumerate(S.points, start=1):
                    f.write(f"p{i}_{POINT_NAMES[i-1]}={x:.3f},{y:.3f}\n")

            save_info.set_text(f"Output: {csv_out.name}, {png_out.name}, {src_png_out.name}, {pts_out.name}")
            log.push(f"[INFO] saved: {csv_out}")
            log.push(f"[INFO] saved: {png_out}")
            if S.source_data is not None and len(S.points) == 4:
                log.push(f"[INFO] saved: {src_png_out}")
            log.push(f"[INFO] saved: {pts_out}")
            ui.notify("Saved transformed CSV/PNG + original PNG + points", color="positive")
        except Exception as ex:
            log.push(f"[ERROR] save failed: {ex}")
            ui.notify(f"Save failed: {ex}", color="negative")

    load_btn.on("click", lambda: on_load())
    reset_btn.on("click", lambda: on_reset())
    transform_btn.on("click", lambda: on_transform())
    save_btn.on("click", lambda: on_save())
    auto_range_btn.on("click", lambda: set_auto_range())
    folder_select.on("update:model-value", lambda _: refresh_csv_menu())
    csv_select.on("update:model-value", lambda _: apply_selected_csv_path())
    refresh_btn.on("click", lambda: refresh_folder_menu())
    tmin.on("change", lambda _: (update_source_view(), update_transformed_view()))
    tmax.on("change", lambda _: (update_source_view(), update_transformed_view()))
    for i in range(4):
        point_x_inputs[i].on("change", lambda _: update_points_from_inputs())
        point_y_inputs[i].on("change", lambda _: update_points_from_inputs())

    update_point_inputs_from_state()
    refresh_folder_menu()
    ui.run(title="Circular Target PT", port=8084, reload=False)


if __name__ in {"__main__", "__mp_main__"}:
    main()
