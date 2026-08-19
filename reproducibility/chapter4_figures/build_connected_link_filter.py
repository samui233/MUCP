#!/usr/bin/env python3
"""Build reproducible connected-link masks from prepared SimART HDF5 shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


HERE = Path(__file__).resolve().parent
TASK_SETTINGS = {
    "csi": {"frame_stride": 1},
    "beam": {"frame_stride": 4},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--threshold-db", type=float, default=-100.0)
    parser.add_argument(
        "--results-root", type=Path, default=HERE / "results" / "chapter4_sample_metrics"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=HERE / "results" / "chapter4_connected_filter"
    )
    return parser.parse_args()


def build_task_mask(
    task: str, reference_path: Path, dataset_root: Path, threshold_db: float
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(reference_path) as archive:
        route_ids = archive["route_id"].astype(np.int64)
        bs_ids = archive["bs_id"].astype(np.int64)
        start_frames = archive["start_frame"].astype(np.int64)

    frame_stride = int(TASK_SETTINGS[task]["frame_stride"])
    offsets = np.arange(16, dtype=np.int64) * frame_stride
    minimum_gain_db = np.empty(len(route_ids), dtype=np.float32)
    for route_id in np.unique(route_ids):
        indices = np.flatnonzero(route_ids == route_id)
        shard = dataset_root / "shards" / f"route_{int(route_id):04d}.h5"
        with h5py.File(shard, "r", swmr=True, libver="latest") as handle:
            gains = handle["optimal_beam_gain_db"]
            for index in indices:
                selected = int(start_frames[index]) + offsets
                minimum_gain_db[index] = float(gains[selected, int(bs_ids[index])].min())

    keep = minimum_gain_db >= threshold_db
    arrays = {
        "route_id": route_ids,
        "bs_id": bs_ids,
        "start_frame": start_frames,
        "minimum_gain_db": minimum_gain_db,
        "keep": keep,
        "threshold_db": np.asarray(threshold_db, dtype=np.float64),
        "frame_stride": np.asarray(frame_stride, dtype=np.int64),
        "sequence_frames": np.asarray(16, dtype=np.int64),
    }
    summary = {
        "task": task,
        "rule": "minimum optimal-beam gain across all 16 sequence frames",
        "threshold_db": threshold_db,
        "frame_stride": frame_stride,
        "total_samples": int(len(keep)),
        "retained_samples": int(keep.sum()),
        "removed_samples": int((~keep).sum()),
        "minimum_gain_db_quantiles": {
            str(q): float(np.quantile(minimum_gain_db, q))
            for q in (0.0, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0)
        },
    }
    return arrays, summary


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for task in TASK_SETTINGS:
        reference = args.results_root / f"{task}_A_test_rollout_samples.npz"
        arrays, summary = build_task_mask(task, reference, args.dataset_root, args.threshold_db)
        output = args.output_dir / f"{task}_test_connected_mask.npz"
        np.savez_compressed(output, **arrays)
        summaries[task] = summary
        print(f"{task}: kept {summary['retained_samples']}/{summary['total_samples']} -> {output}")
    (args.output_dir / "filter_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
