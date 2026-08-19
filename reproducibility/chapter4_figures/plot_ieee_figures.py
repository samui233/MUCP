#!/usr/bin/env python3
"""Render IEEE-ready Chapter IV figures from filtered per-sample results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parent
VARIANTS = tuple("ABCDEFG")

ARIAL_FILES = (
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Italic.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Bold_Italic.ttf"),
)

LINE_STYLES = {
    "A": {"color": "#4D4D4D", "marker": "o", "linestyle": "-"},
    "B": {"color": "#0072B2", "marker": "s", "linestyle": "--"},
    "C": {"color": "#E69F00", "marker": "^", "linestyle": "-."},
    "D": {"color": "#009E73", "marker": "D", "linestyle": ":"},
    "E": {"color": "#D55E00", "marker": "v", "linestyle": (0, (5, 1))},
    "F": {"color": "#CC79A7", "marker": "P", "linestyle": (0, (3, 1, 1, 1))},
    "G": {"color": "#56B4E9", "marker": "X", "linestyle": (0, (1, 1))},
}
MODALITY_LABELS = {
    "A": "A (W)",
    "B": "B (W+P)",
    "C": "C (W+P+M)",
    "D": "D (W+P+I)",
    "E": "E (W+I)",
    "F": "F (W+P+M+I)",
    "G": "G (P+M+I)",
}



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "figures" / "chapter4_ieee")
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_ROOT / "results" / "chapter4_ieee_plot_data"
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_paths(results_root: Path) -> dict[str, Any]:
    samples = {"csi": {}, "beam": {}}
    for variant in VARIANTS:
        for task in ("csi", "beam"):
            samples[task][variant] = (
                results_root
                / "chapter4_sample_metrics"
                / f"{task}_{variant}_test_rollout_samples.npz"
            )
    filters = {
        task: results_root / "chapter4_connected_filter" / f"{task}_test_connected_mask.npz"
        for task in ("csi", "beam")
    }
    return {"samples": samples, "filters": filters}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing numeric result: {path}")
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def validate_alignment(samples: dict[str, np.ndarray], mask: dict[str, np.ndarray], path: Path) -> None:
    for key in ("route_id", "bs_id", "start_frame"):
        if not np.array_equal(samples[key], mask[key]):
            raise RuntimeError(f"Sample/filter key mismatch for {key}: {path}")


def read_numeric_data(results_root: Path) -> tuple[dict[str, Any], list[Path]]:
    paths = source_paths(results_root)
    used_paths: list[Path] = []
    direct: dict[str, dict[str, dict[str, float]]] = {"csi": {}, "beam": {}}
    rollout: dict[str, dict[str, dict[str, list[float]]]] = {"csi": {}, "beam": {}}

    filters = {task: load_npz(path) for task, path in paths["filters"].items()}
    used_paths.extend(paths["filters"].values())
    filter_metadata: dict[str, Any] = {}
    for task, values in filters.items():
        keep = values["keep"].astype(bool)
        filter_metadata[task] = {
            "threshold_db": float(values["threshold_db"]),
            "rule": "minimum optimal-beam gain across all 16 sequence frames",
            "frame_stride": int(values["frame_stride"]),
            "total_samples": int(len(keep)),
            "retained_samples": int(keep.sum()),
            "removed_samples": int((~keep).sum()),
        }

    for variant in VARIANTS:
        csi_path = paths["samples"]["csi"][variant]
        beam_path = paths["samples"]["beam"][variant]
        csi = load_npz(csi_path)
        beam = load_npz(beam_path)
        used_paths.extend((csi_path, beam_path))
        validate_alignment(csi, filters["csi"], csi_path)
        validate_alignment(beam, filters["beam"], beam_path)

        csi_keep = filters["csi"]["keep"].astype(bool)
        beam_keep = filters["beam"]["keep"].astype(bool)
        csi_error = csi["error"][csi_keep].astype(np.float64)
        csi_power = csi["power"][csi_keep].astype(np.float64)
        csi_nmse = csi_error.sum(axis=0) / np.maximum(csi_power.sum(axis=0), 1e-30)
        csi_direct_nmse = csi_error[:, :2].sum() / max(csi_power[:, :2].sum(), 1e-30)
        direct["csi"][variant] = {
            "nmse_db": float(10.0 * np.log10(max(csi_direct_nmse, 1e-30))),
        }
        direct["beam"][variant] = {
            "top1_pct": 100.0 * float(beam["correct1"][beam_keep, :2].mean()),
            "top3_pct": 100.0 * float(beam["correct3"][beam_keep, :2].mean()),
            "normalized_gain_pct": 100.0
            * float(beam["normalized_gain"][beam_keep, :2].mean()),
        }
        rollout["csi"][variant] = {
            "nmse_db": [
                float(value) for value in 10.0 * np.log10(np.maximum(csi_nmse, 1e-30))
            ],
        }
        rollout["beam"][variant] = {
            "top3_pct": [
                100.0 * float(value)
                for value in beam["correct3"][beam_keep].mean(axis=0)
            ],
        }

    return {
        "evaluation_filter": filter_metadata,
        "direct": direct,
        "rollout": rollout,
    }, used_paths


def configure_ieee_style() -> None:
    missing = [path for path in ARIAL_FILES if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Arial font files are missing: {missing}")
    for path in ARIAL_FILES:
        font_manager.fontManager.addfont(path)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "font.size": 8.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.15,
            "lines.markersize": 4.2,
            "hatch.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def style_axis(axis: plt.Axes, vertical_grid: bool = False) -> None:
    axis.set_axisbelow(True)
    axis.yaxis.grid(True, color="#D9D9D9", linewidth=0.45, linestyle="-")
    if vertical_grid:
        axis.xaxis.grid(True, color="#E6E6E6", linewidth=0.4, linestyle="-")
    else:
        axis.xaxis.grid(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(width=0.7, length=3.0)


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        0.5,
        -0.265,
        label,
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        fontweight="bold",
    )


def truncated_axis_mark(axis: plt.Axes) -> None:
    kwargs = {"transform": axis.transAxes, "color": "black", "clip_on": False, "linewidth": 0.75}
    axis.plot((-0.012, 0.012), (-0.012, 0.018), **kwargs)
    axis.plot((-0.012, 0.012), (0.018, 0.048), **kwargs)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int = 600) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(pdf_path, facecolor="white", transparent=False)
    fig.savefig(png_path, dpi=dpi, facecolor="white", transparent=False)
    plt.close(fig)
    with Image.open(png_path) as image:
        gray = ImageOps.grayscale(image).convert("RGB")
        gray.save(output_dir / f"{stem}_gray_preview.png", dpi=(dpi, dpi))


def figure1(data: dict[str, Any], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))
    fig.subplots_adjust(left=0.080, right=0.992, bottom=0.225, top=0.785, wspace=0.285)
    x = np.arange(len(VARIANTS), dtype=np.float64)

    csi = np.asarray([data["direct"]["csi"][variant]["nmse_db"] for variant in VARIANTS])
    colors = [LINE_STYLES[variant]["color"] for variant in VARIANTS]
    bars = axes[0].bar(x, csi, width=0.66, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_xlim(-0.6, 6.6)
    axes[0].set_ylim(-11.0, 0.0)
    axes[0].set_yticks(np.arange(-10, 1, 2))
    axes[0].set_xticks(x, VARIANTS)
    axes[0].set_xlabel("Modality combination")
    axes[0].set_ylabel("NMSE (dB)")
    axes[0].axhline(0.0, color="black", linewidth=0.7)
    for bar, value in zip(bars, csi):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            value - 0.18,
            f"{value:.2f}",
            ha="center",
            va="top",
            fontsize=8,
        )
    style_axis(axes[0])
    panel_label(axes[0], "(a)")

    width = 0.235
    beam = data["direct"]["beam"]
    series = (
        ("Top-1", np.asarray([beam[v]["top1_pct"] for v in VARIANTS]), "#0072B2", None),
        ("Top-3", np.asarray([beam[v]["top3_pct"] for v in VARIANTS]), "#E69F00", "///"),
        (
            "Normalized beam gain",
            np.asarray([beam[v]["normalized_gain_pct"] for v in VARIANTS]),
            "#009E73",
            "xx",
        ),
    )
    for offset, (label, values, color, hatch) in enumerate(series):
        axes[1].bar(
            x + (offset - 1) * width,
            values,
            width=width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            hatch=hatch,
        )
    axes[1].set_xlim(-0.65, 6.65)
    axes[1].set_ylim(65.0, 100.0)
    axes[1].set_yticks(np.arange(65, 101, 5))
    axes[1].set_xticks(x, VARIANTS)
    axes[1].set_xlabel("Modality combination")
    axes[1].set_ylabel("Performance (%)")
    axes[1].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.045),
        ncol=3,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.1,
        handletextpad=0.45,
    )
    style_axis(axes[1])
    truncated_axis_mark(axes[1])
    panel_label(axes[1], "(b)")
    save_figure(fig, output_dir, "fig1_task_modality_comparison")


def figure2(data: dict[str, Any], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.2))
    fig.subplots_adjust(left=0.080, right=0.992, bottom=0.215, top=0.725, wspace=0.275)
    steps = np.arange(1, 9)

    for variant in VARIANTS:
        style = LINE_STYLES[variant]
        axes[0].plot(
            steps,
            data["rollout"]["csi"][variant]["nmse_db"],
            label=MODALITY_LABELS[variant],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.15,
            markersize=4.2,
            markeredgewidth=0.65,
            markeredgecolor=style["color"],
        )
    axes[0].set_xlim(0.75, 8.25)
    axes[0].set_xticks(steps)
    axes[0].set_ylim(-10.5, -3.0)
    axes[0].set_yticks(np.arange(-10.0, -2.9, 1.0))
    axes[0].set_xlabel("Prediction step")
    axes[0].set_ylabel("NMSE (dB)")
    style_axis(axes[0], vertical_grid=True)
    panel_label(axes[0], "(a)")

    beam_values: list[float] = []
    for variant in VARIANTS:
        values = data["rollout"]["beam"][variant]["top3_pct"]
        beam_values.extend(values)
        style = LINE_STYLES[variant]
        axes[1].plot(
            steps,
            values,
            label=MODALITY_LABELS[variant],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.15,
            markersize=4.2,
            markeredgewidth=0.65,
            markeredgecolor=style["color"],
        )
    lower = max(0.0, 5.0 * math.floor((min(beam_values) - 1.0) / 5.0))
    axes[1].set_xlim(0.75, 8.25)
    axes[1].set_xticks(steps)
    axes[1].set_ylim(lower, 100.0)
    axes[1].set_yticks(np.arange(lower, 100.1, 5.0))
    axes[1].set_xlabel("Prediction step")
    axes[1].set_ylabel("Top-3 accuracy (%)")
    style_axis(axes[1], vertical_grid=True)
    panel_label(axes[1], "(b)")

    handles, labels = axes[0].get_legend_handles_labels()
    # Matplotlib fills multi-row legends column-wise. Reorder the entries so
    # that the visual reading order is A--D on the first row and E--G below.
    legend_order = [0, 4, 1, 5, 2, 6, 3]
    handles = [handles[index] for index in legend_order]
    labels = [labels[index] for index in legend_order]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=4,
        frameon=False,
        handlelength=1.8,
        columnspacing=0.9,
        handletextpad=0.35,
    )
    save_figure(fig, output_dir, "fig2_recursive_comparison")


def write_plot_data(data: dict[str, Any], paths: list[Path], data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    fig1_rows = []
    for variant in VARIANTS:
        fig1_rows.append(
            {
                "variant": variant,
                "csi_nmse_db": data["direct"]["csi"][variant]["nmse_db"],
                **data["direct"]["beam"][variant],
            }
        )
    fig2_rows = []
    for variant in VARIANTS:
        for index in range(8):
            fig2_rows.append(
                {
                    "variant": variant,
                    "prediction_step": index + 1,
                    "csi_nmse_db": data["rollout"]["csi"][variant]["nmse_db"][index],
                    "beam_top3_pct": data["rollout"]["beam"][variant]["top3_pct"][index],
                }
            )

    for filename, rows in (("fig1_plot_data.csv", fig1_rows), ("fig2_plot_data.csv", fig2_rows)):
        with (data_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    (data_dir / "figure_plot_data.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "policy": (
            "All plotted values were aggregated directly from saved per-sample NPZ results after "
            "applying the same data-defined connected-link mask to every A-G variant; no image "
            "digitization or manual value editing."
        ),
        "evaluation_filter": data["evaluation_filter"],
        "sources": [
            {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": sha256(path)}
            for path in sorted(set(paths))
        ],
    }
    (data_dir / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with (data_dir / "fig1_plot_data.csv").open("r", encoding="utf-8", newline="") as handle:
        reloaded = list(csv.DictReader(handle))
    for original, current in zip(fig1_rows, reloaded):
        if current["variant"] != original["variant"]:
            raise RuntimeError("Fig. 1 CSV variant mismatch")
        for key in original.keys() - {"variant"}:
            if float(current[key]) != float(original[key]):
                raise RuntimeError(f"Fig. 1 CSV numeric mismatch: {original['variant']} {key}")


def pdf_font_report(output_dir: Path) -> str:
    reports = []
    for path in (
        output_dir / "fig1_task_modality_comparison.pdf",
        output_dir / "fig2_recursive_comparison.pdf",
    ):
        result = subprocess.run(["pdffonts", str(path)], check=True, text=True, capture_output=True)
        reports.append(f"## {path.name}\n\n```text\n{result.stdout.rstrip()}\n```")
        lines = [line.split() for line in result.stdout.splitlines()[2:] if line.strip()]
        if not lines or any("yes" not in fields for fields in lines):
            raise RuntimeError(f"PDF font embedding check failed: {path}")
    return "\n\n".join(reports) + "\n"


def validate_outputs(output_dir: Path, data_dir: Path) -> None:
    expected = {
        "fig1_task_modality_comparison.png": (4296, 1800),
        "fig2_recursive_comparison.png": (4296, 1920),
    }
    validation: dict[str, Any] = {"png_dimensions": {}, "font": "Arial", "font_sizes_pt": {}}
    for filename, dimensions in expected.items():
        with Image.open(output_dir / filename) as image:
            actual = image.size
            validation["png_dimensions"][filename] = list(actual)
        if actual != dimensions:
            raise RuntimeError(f"Unexpected PNG dimensions for {filename}: {actual}, expected {dimensions}")
    validation["font_sizes_pt"] = {
        "axis_labels": 9,
        "ticks": 8.5,
        "legend": 8,
        "panel_labels": 9,
    }
    validation["gray_previews"] = [
        "fig1_task_modality_comparison_gray_preview.png",
        "fig2_recursive_comparison_gray_preview.png",
    ]
    validation["numeric_sources"] = "source_manifest.json"
    (data_dir / "validation_report.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (data_dir / "pdf_fonts.txt").write_text(pdf_font_report(output_dir), encoding="utf-8")


def main() -> int:
    args = parse_args()
    configure_ieee_style()
    data, paths = read_numeric_data(args.results_root)
    write_plot_data(data, paths, args.data_dir)
    figure1(data, args.output_dir)
    figure2(data, args.output_dir)
    validate_outputs(args.output_dir, args.data_dir)
    print(f"Saved IEEE figures to: {args.output_dir}")
    print(f"Saved exact plotting data to: {args.data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
