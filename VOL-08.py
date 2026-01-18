# -*- coding: utf-8 -*-
"""
plot_results_from_volcanic_report_spyder_friendly.py

Versión tolerante a ejecución SIN argumentos (ideal para Spyder / runfile).
Si no se proporcionan --input y --outdir:
  1) intenta abrir un selector de archivo para elegir .zip o .xlsx,
  2) si no hay GUI, pide la ruta por consola,
  3) genera outdir automáticamente junto al input (RESULTS_PAPER).

Uso recomendado en terminal (opcional):
  python plot_results_from_volcanic_report_spyder_friendly.py --input RUTA --outdir RUTA

Salidas:
  - <outdir>/figures/*.png
  - <outdir>/tables/paper_results_tables.xlsx
  - <outdir>/figure_index.csv
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Utilidades generales
# -----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) + 1e-12
    return (x - med) / (1.4826 * mad)

def read_report_xlsx(report_path: Path) -> Dict[str, pd.DataFrame]:
    xl = pd.ExcelFile(report_path)
    sheets = {}
    for s in xl.sheet_names:
        sheets[s] = xl.parse(s)
    return sheets

def locate_report_from_zip(zip_path: Path, extract_dir: Path) -> Path:
    ensure_dir(extract_dir)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    candidates = list(extract_dir.rglob("volcanic_tremor_topo_deep_report.xlsx"))
    if not candidates:
        raise FileNotFoundError(
            "No se encontró 'volcanic_tremor_topo_deep_report.xlsx' dentro del zip extraído."
        )
    return candidates[0]


# -----------------------------
# Curación de tablas para Results
# -----------------------------
def build_tables_for_paper(
    sheets: Dict[str, pd.DataFrame],
    data_meta: pd.DataFrame,
    config: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    tables = {}

    wm = sheets.get("window_metrics", pd.DataFrame()).copy()
    if wm.empty:
        raise ValueError("La hoja 'window_metrics' está vacía o no existe.")

    cols_51 = [
        "window_idx", "center_sec", "valid",
        "rmse_AE1D", "rmse_AE2D",
        "spec_centroid", "spec_entropy"
    ]
    cols_52 = [
        "window_idx", "center_sec", "valid", "tda_computed",
        "wd_h0", "wd_h1",
        "betti_l2_h0", "betti_l2_h1",
        "topo_n_h0", "topo_total_persistence_h0", "topo_max_lifetime_h0",
        "topo_n_h1", "topo_total_persistence_h1", "topo_max_lifetime_h1"
    ]
    cols_53 = [
        "window_idx", "center_sec", "valid",
        "score_raw", "score_smooth",
        "alert", "cusum_gp", "cusum_gn",
        "rmse_AE2D", "wd_h0", "wd_h1", "betti_l2_h1"
    ]

    cols_51 = [c for c in cols_51 if c in wm.columns]
    cols_52 = [c for c in cols_52 if c in wm.columns]
    cols_53 = [c for c in cols_53 if c in wm.columns]

    tables["5_1_reconstruction_time_series"] = wm[cols_51].copy()
    tables["5_2_topology_time_series"] = wm[cols_52].copy()
    tables["5_3_early_warning_time_series"] = wm[cols_53].copy()

    for k in [
        "phase_stats", "segment_summary", "threshold_sweep", "alarm_cards",
        "early_warning_metrics", "lead_time", "lead_lag"
    ]:
        if k in sheets and not sheets[k].empty:
            tables[k] = sheets[k].copy()

    if data_meta is not None and not data_meta.empty:
        tables["data_meta"] = data_meta.copy()

    if config is not None and not config.empty:
        keep_cfg = [c for c in config.columns if c in (
            "RUN_ID", "WIN_SEC", "HOP_SEC", "BASELINE_END_SEC",
            "EMBED_DIM", "TDA_MAXDIM", "TDA_SUBSAMPLE", "TDA_STRIDE",
            "ALERT_Z", "SCORE_SMOOTH_WIN", "ZSCORE_WIN",
            "USE_CUSUM", "CUSUM_K", "CUSUM_H"
        )]
        tables["config_key"] = config[keep_cfg].copy() if keep_cfg else config.copy()

    return tables

def write_tables_to_excel(tables: Dict[str, pd.DataFrame], out_xlsx: Path) -> None:
    ensure_dir(out_xlsx.parent)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        for name, df in tables.items():
            safe_name = name[:31]
            df.to_excel(w, sheet_name=safe_name, index=False)


# -----------------------------
# Figuras para 5.1–5.3
# -----------------------------
def plot_5_1_reconstruction(wm: pd.DataFrame, data_meta: pd.DataFrame, config: pd.DataFrame, outdir: Path) -> None:
    ensure_dir(outdir)

    t = wm["center_sec"].to_numpy(dtype=float)
    rmse1 = wm["rmse_AE1D"].to_numpy(dtype=float) if "rmse_AE1D" in wm.columns else None
    rmse2 = wm["rmse_AE2D"].to_numpy(dtype=float) if "rmse_AE2D" in wm.columns else None

    transition = None
    baseline_end = None
    if data_meta is not None and not data_meta.empty and "transition_start_sec" in data_meta.columns:
        transition = float(data_meta.loc[0, "transition_start_sec"])
    if config is not None and not config.empty and "BASELINE_END_SEC" in config.columns:
        baseline_end = float(config.loc[0, "BASELINE_END_SEC"])

    plt.figure(figsize=(12, 4.8))
    if rmse1 is not None:
        plt.plot(t, rmse1, linewidth=1.2, label="RMSE AE-1D (waveform)")
    if rmse2 is not None:
        plt.plot(t, rmse2, linewidth=1.2, label="RMSE AE-2D (embedding)")
    if baseline_end is not None:
        plt.axvline(baseline_end, linestyle="--", linewidth=1.0, label="Baseline end")
    if transition is not None:
        plt.axvline(transition, linestyle="--", linewidth=1.0, label="Transition start")
    plt.xlabel("Time (s)")
    plt.ylabel("Reconstruction error (RMSE)")
    plt.title("5.1 Reconstruction error dynamics across time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "5_1_rmse_over_time.png", dpi=300)
    plt.close()

    if baseline_end is not None and rmse2 is not None:
        pre = rmse2[t <= baseline_end]
        post = rmse2[t > baseline_end]
        plt.figure(figsize=(8.5, 4.8))
        plt.boxplot([pre[~np.isnan(pre)], post[~np.isnan(post)]], labels=["Baseline", "Post-baseline"])
        plt.ylabel("RMSE AE-2D")
        plt.title("5.1 Baseline stability as low-distortion proxy")
        plt.tight_layout()
        plt.savefig(outdir / "5_1_baseline_vs_post_rmse_boxplot.png", dpi=300)
        plt.close()

def plot_5_2_topology(wm: pd.DataFrame, data_meta: pd.DataFrame, outdir: Path) -> None:
    ensure_dir(outdir)

    t = wm["center_sec"].to_numpy(dtype=float)
    transition = None
    if data_meta is not None and not data_meta.empty and "transition_start_sec" in data_meta.columns:
        transition = float(data_meta.loc[0, "transition_start_sec"])

    plt.figure(figsize=(12, 4.8))
    if "wd_h0" in wm.columns:
        plt.plot(t, wm["wd_h0"].to_numpy(dtype=float), linewidth=1.2, label="Wasserstein distance H0")
    if "wd_h1" in wm.columns:
        plt.plot(t, wm["wd_h1"].to_numpy(dtype=float), linewidth=1.2, label="Wasserstein distance H1")
    if transition is not None:
        plt.axvline(transition, linestyle="--", linewidth=1.0, label="Transition start")
    plt.xlabel("Time (s)")
    plt.ylabel("Persistence-diagram distance")
    plt.title("5.2 Temporal trajectories of persistence-diagram distances")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "5_2_wasserstein_h0_h1_over_time.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 4.8))
    if "betti_l2_h0" in wm.columns:
        plt.plot(t, wm["betti_l2_h0"].to_numpy(dtype=float), linewidth=1.2, label="Betti L2 H0")
    if "betti_l2_h1" in wm.columns:
        plt.plot(t, wm["betti_l2_h1"].to_numpy(dtype=float), linewidth=1.2, label="Betti L2 H1")
    if transition is not None:
        plt.axvline(transition, linestyle="--", linewidth=1.0, label="Transition start")
    plt.xlabel("Time (s)")
    plt.ylabel("Betti-curve L2 distance")
    plt.title("5.2 Betti-curve distance profiles")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "5_2_betti_l2_over_time.png", dpi=300)
    plt.close()

    metrics = [
        ("topo_total_persistence_h0", "Total persistence H0"),
        ("topo_total_persistence_h1", "Total persistence H1"),
        ("topo_max_lifetime_h1", "Max lifetime H1"),
    ]
    plt.figure(figsize=(12, 4.8))
    plotted = False
    for col, lab in metrics:
        if col in wm.columns:
            plt.plot(t, wm[col].to_numpy(dtype=float), linewidth=1.2, label=lab)
            plotted = True
    if transition is not None:
        plt.axvline(transition, linestyle="--", linewidth=1.0, label="Transition start")
    if plotted:
        plt.xlabel("Time (s)")
        plt.ylabel("Topological summary metric")
        plt.title("5.2 Topological summary metrics across time")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "5_2_topology_summary_metrics.png", dpi=300)
    plt.close()

def plot_5_3_early_warning(wm: pd.DataFrame, sheets: Dict[str, pd.DataFrame], data_meta: pd.DataFrame, outdir: Path) -> None:
    ensure_dir(outdir)

    t = wm["center_sec"].to_numpy(dtype=float)
    transition = None
    if data_meta is not None and not data_meta.empty and "transition_start_sec" in data_meta.columns:
        transition = float(data_meta.loc[0, "transition_start_sec"])

    s = wm["score_smooth"].to_numpy(dtype=float) if "score_smooth" in wm.columns else wm["score_raw"].to_numpy(dtype=float)
    alerts = wm["alert"].to_numpy(dtype=int) if "alert" in wm.columns else np.zeros_like(t, dtype=int)

    plt.figure(figsize=(12, 4.8))
    plt.plot(t, s, linewidth=1.3, label="Stability score (smooth)")
    if np.any(alerts == 1):
        plt.scatter(t[alerts == 1], s[alerts == 1], s=16, label="Alarms")
    if transition is not None:
        plt.axvline(transition, linestyle="--", linewidth=1.0, label="Transition start")
    plt.xlabel("Time (s)")
    plt.ylabel("Score")
    plt.title("5.3 Stability curve with alarms")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "5_3_stability_curve_with_alarms.png", dpi=300)
    plt.close()

    recon = wm["rmse_AE2D"].to_numpy(dtype=float) if "rmse_AE2D" in wm.columns else np.zeros_like(t)
    topo = np.zeros_like(t, dtype=float)
    for c in ["wd_h0", "wd_h1", "betti_l2_h1"]:
        if c in wm.columns:
            topo = topo + np.nan_to_num(wm[c].to_numpy(dtype=float), nan=0.0)

    z_recon = np.abs(robust_z(recon))
    z_topo = np.abs(robust_z(topo))
    denom = z_recon + z_topo + 1e-12
    frac_recon = z_recon / denom
    frac_topo = z_topo / denom

    plt.figure(figsize=(12, 4.8))
    plt.plot(t, frac_recon, linewidth=1.2, label="Reconstruction contribution (proxy)")
    plt.plot(t, frac_topo, linewidth=1.2, label="Topology contribution (proxy)")
    if transition is not None:
        plt.axvline(transition, linestyle="--", linewidth=1.0, label="Transition start")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized contribution")
    plt.title("5.3 Score decomposition (proxy): reconstruction vs topology")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "5_3_score_decomposition_proxy.png", dpi=300)
    plt.close()

    if "threshold_sweep" in sheets and not sheets["threshold_sweep"].empty:
        ts = sheets["threshold_sweep"].copy()
        if {"threshold", "lead_time_sec", "false_alarm_rate_per_hr"}.issubset(ts.columns):
            plt.figure(figsize=(10.5, 4.8))
            plt.plot(ts["threshold"].to_numpy(), ts["lead_time_sec"].to_numpy(), linewidth=1.3, label="Lead time (s)")
            plt.plot(ts["threshold"].to_numpy(), ts["false_alarm_rate_per_hr"].to_numpy(), linewidth=1.3, label="False alarm rate (per hr)")
            plt.xlabel("Threshold (z)")
            plt.title("5.3 Early-warning trade-offs (threshold sweep)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(outdir / "5_3_threshold_sweep_tradeoff.png", dpi=300)
            plt.close()

    ll = sheets.get("lead_lag", pd.DataFrame())
    if ll is not None and not ll.empty and {"lag_sec", "corr"}.issubset(ll.columns):
        plt.figure(figsize=(10.5, 4.8))
        plt.plot(ll["lag_sec"].to_numpy(dtype=float), ll["corr"].to_numpy(dtype=float), linewidth=1.3)
        plt.axhline(0.0, linewidth=1.0)
        plt.xlabel("Lag (s)")
        plt.ylabel("Correlation")
        plt.title("Lead–lag correlation (report sheet lead_lag)")
        plt.tight_layout()
        plt.savefig(outdir / "5_3_lead_lag_correlation.png", dpi=300)
        plt.close()


# -----------------------------
# Entrada tolerante a Spyder (sin argumentos)
# -----------------------------
def try_gui_pick_file() -> Path | None:
    """
    Intenta abrir un selector de archivo. Si Tkinter no está disponible,
    regresa None.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Seleccione el ZIP o XLSX del reporte",
            filetypes=[
                ("ZIP files", "*.zip"),
                ("Excel files", "*.xlsx *.xls"),
                ("All files", "*.*")
            ]
        )
        root.destroy()
        if file_path:
            return Path(file_path).expanduser().resolve()
        return None
    except Exception:
        return None

def pick_input_path_fallback() -> Path:
    p = try_gui_pick_file()
    if p is not None:
        return p
    # fallback consola
    raw = input("Ruta al .zip o .xlsx del reporte: ").strip().strip('"').strip("'")
    return Path(raw).expanduser().resolve()

def resolve_outdir_default(input_path: Path) -> Path:
    # Carpeta RESULTS_PAPER junto al archivo de entrada
    base = input_path.parent
    return (base / "RESULTS_PAPER").resolve()


# -----------------------------
# Core pipeline
# -----------------------------
def run_pipeline(input_path: Path, outdir: Path) -> None:
    outdir = outdir.expanduser().resolve()
    figdir = outdir / "figures"
    tabdir = outdir / "tables"
    ensure_dir(figdir)
    ensure_dir(tabdir)

    if input_path.suffix.lower() == ".zip":
        extract_dir = outdir / "_extracted_zip"
        report_path = locate_report_from_zip(input_path, extract_dir)
    elif input_path.suffix.lower() in [".xlsx", ".xls"]:
        report_path = input_path
    else:
        raise ValueError("El input debe ser un .zip o un .xlsx/.xls")

    sheets = read_report_xlsx(report_path)
    wm = sheets.get("window_metrics", pd.DataFrame()).copy()
    if wm.empty:
        raise ValueError("No se encontró contenido en la hoja 'window_metrics'.")

    data_meta = sheets.get("data_meta", pd.DataFrame())
    config = sheets.get("config", pd.DataFrame())

    # Tablas curadas para paper
    paper_tables = build_tables_for_paper(sheets, data_meta, config)
    out_tables = tabdir / "paper_results_tables.xlsx"
    write_tables_to_excel(paper_tables, out_tables)

    # Figuras
    plot_5_1_reconstruction(wm, data_meta, config, figdir)
    plot_5_2_topology(wm, data_meta, figdir)
    plot_5_3_early_warning(wm, sheets, data_meta, figdir)

    # Índice de figuras
    index_rows = [{"figure_file": f.name, "path": str(f)} for f in sorted(figdir.glob("*.png"))]
    pd.DataFrame(index_rows).to_csv(outdir / "figure_index.csv", index=False, encoding="utf-8")

    print("OK")
    print(f"Input: {input_path}")
    print(f"Reporte leído: {report_path}")
    print(f"Figuras: {figdir}")
    print(f"Tablas curadas: {out_tables}")
    print(f"Índice: {outdir / 'figure_index.csv'}")


def parse_args_lenient() -> argparse.Namespace:
    """
    Parser 'suave': si faltan args, no falla; devuelve None y se usa modo interactivo.
    """
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--input", required=False, help="Ruta al .zip o al .xlsx del reporte")
    ap.add_argument("--outdir", required=False, help="Directorio de salida para figuras y tablas")
    return ap.parse_args()


def main() -> None:
    args = parse_args_lenient()

    if args.input is None or str(args.input).strip() == "":
        input_path = pick_input_path_fallback()
    else:
        input_path = Path(str(args.input)).expanduser().resolve()

    if args.outdir is None or str(args.outdir).strip() == "":
        outdir = resolve_outdir_default(input_path)
    else:
        outdir = Path(str(args.outdir)).expanduser().resolve()

    run_pipeline(input_path, outdir)


if __name__ == "__main__":
    main()
