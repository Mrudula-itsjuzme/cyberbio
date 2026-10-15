"""Prepare a deduplicated, source-balanced candidate shortlist; run no oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from materials_adv.oracle.candidate_selection import select_oracle_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--max-per-source", type=int, default=2)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.trajectory.read_text().splitlines() if line.strip()]
    frame = pd.DataFrame({
        "source_id": [r.get("source_id") for r in rows],
        "source_smiles": [r.get("source_smiles") for r in rows],
        "candidate_smiles": [r.get("proposed_candidate") for r in rows],
        "attack_family": [r.get("search_method") for r in rows],
        "target_model": [r.get("target_model") for r in rows],
        "model_drift_eV": [r.get("drift_from_original") for r in rows],
        "edit_count": [r.get("edit_count") for r in rows],
        "validity": [r.get("validity") for r in rows],
    })
    frame = frame[frame["validity"].eq("valid") & frame["edit_count"].gt(0)]
    selected = select_oracle_candidates(frame, n=args.n, max_per_source=args.max_per_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output, index=False)
    manifest = {
        "status": "prepared_not_executed",
        "oracle_calculations_performed": False,
        "input": str(args.trajectory),
        "n_selected": len(selected),
        "selection_uses_true_labels": False,
        "constraints": {"max_per_source": args.max_per_source, "canonical_candidate_dedup": True},
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
