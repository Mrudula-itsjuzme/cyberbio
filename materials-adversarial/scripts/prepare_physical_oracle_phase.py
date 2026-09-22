"""Prepare a fail-closed, execution-blocked physical-oracle phase.

This script performs only inexpensive local audits and RDKit structure generation.
It never submits or runs DFT and never reads the sealed ML test split.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem

from materials_adv.domain.chemistry.polymer_repeat import PolymerRepeatGraph
from materials_adv.domain.chemistry.surrogate_geometry import SurrogateGeometryBuilder


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/physical_oracle_phase/20260921"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_version(command: str, args: list[str]) -> dict:
    executable = shutil.which(command)
    if executable is None:
        return {"available": False, "path": None, "version": None}
    try:
        result = subprocess.run(
            [executable, *args], capture_output=True, text=True, timeout=10, check=False
        )
        text = (result.stdout + result.stderr).strip().splitlines()
        return {
            "available": True,
            "path": executable,
            "version": text[0] if text else "version output unavailable",
            "returncode": result.returncode,
        }
    except Exception as exc:
        return {"available": True, "path": executable, "version": None, "error": str(exc)}


def environment_report() -> dict:
    meminfo = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        meminfo[key] = value.strip()
    scheduler = {name: shutil.which(name) for name in ("sbatch", "srun", "qsub", "bsub")}
    pseudo_files = sorted(
        str(path) for base in (ROOT, Path("/tmp"))
        for path in base.rglob("*.UPF")
    )
    return {
        "status": "BLOCKED_LOCAL_DFT_EXECUTION",
        "software": {
            "pw.x": command_version("pw.x", ["-version"]),
            "bands.x": command_version("bands.x", ["-version"]),
            "projwfc.x": command_version("projwfc.x", ["-version"]),
            "mpirun": command_version("mpirun", ["--version"]),
            "orca_quantum_chemistry": {
                "available": False,
                "note": "/usr/bin/orca is the GNOME accessibility screen reader, not ORCA QC",
            },
            "rdkit": {"available": True, "version": rdBase.rdkitVersion},
        },
        "hardware": {
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "memory_total": meminfo.get("MemTotal"),
            "memory_available": meminfo.get("MemAvailable"),
            "gpu_usable": False,
            "gpu_note": "nvidia-smi cannot communicate with an NVIDIA driver",
        },
        "scheduler": scheduler,
        "pseudopotentials": {"count": len(pseudo_files), "files": pseudo_files},
        "network": {
            "status": "UNAVAILABLE_IN_SANDBOX_AUDIT",
            "evidence": "DNS resolution failed for quantum-espresso.org and its pseudopotential site",
        },
        "execution_allowed": False,
        "blocking_requirements": [
            "Quantum ESPRESSO executables",
            "one internally consistent pseudopotential family with verified hashes",
            "pseudopotential-recommended cutoff metadata",
            "at least one real timing/convergence smoke calculation",
        ],
    }


def fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol)


def diverse_calibration(frame: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    eligible = []
    for row in frame.itertuples(index=False):
        smiles = str(row.original_representation)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        attachments = sum(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms())
        formal_charge = int(Chem.GetFormalCharge(mol))
        if attachments != 2 or formal_charge != 0:
            continue
        elements = sorted({atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 0})
        eligible.append({
            "polymer_id": row.polymer_id,
            "original_psmiles": smiles,
            "canonical_psmiles": Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
            "dataset_bandgap_eV": float(row.property_value),
            "ml_split": row.split,
            "attachment_point_count": attachments,
            "formal_charge": formal_charge,
            "elements": ";".join(elements),
        })
    pool = pd.DataFrame(eligible).sort_values("polymer_id", kind="mergesort").reset_index(drop=True)
    pool["gap_stratum"] = pd.qcut(pool["dataset_bandgap_eV"], 3, labels=["low", "mid", "high"])
    selected = []
    per_stratum = n // 3
    for stratum in ("low", "mid", "high"):
        group = pool[pool["gap_stratum"].eq(stratum)].copy().reset_index(drop=True)
        fps = [fingerprint(value) for value in group["original_psmiles"]]
        chosen = [0]
        while len(chosen) < min(per_stratum, len(group)):
            remaining = [i for i in range(len(group)) if i not in chosen]
            next_index = max(
                remaining,
                key=lambda i: (
                    min(1.0 - DataStructs.TanimotoSimilarity(fps[i], fps[j]) for j in chosen),
                    -i,
                ),
            )
            chosen.append(next_index)
        selected.append(group.iloc[chosen])
    result = pd.concat(selected, ignore_index=True)
    result["selection_basis"] = "target-tertile plus deterministic fingerprint farthest-point sampling"
    result["oracle_role"] = "calibration"
    return result


def write_structures(calibration: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    config_path = ROOT / "hpc_oracle/configs/surrogate_structure_protocol.yaml"
    builder = SurrogateGeometryBuilder(str(config_path))
    structure_dir = RUN / "structures"
    structure_dir.mkdir(parents=True, exist_ok=True)
    records, failures = [], []
    for row in calibration.itertuples(index=False):
        for repeat_count in (2, 3, 4):
            mol, metadata = builder.build_surrogate(row.original_psmiles, repeat_count)
            structure_id = f"{row.polymer_id}_capped_n{repeat_count}"
            if mol is None:
                failures.append({
                    "structure_id": structure_id,
                    "polymer_id": row.polymer_id,
                    "original_psmiles": row.original_psmiles,
                    "stage": "structure_construction",
                    "failure_reason": metadata.get("error", "unknown construction failure"),
                })
                continue
            sdf = structure_dir / f"{structure_id}.sdf"
            writer = Chem.SDWriter(str(sdf))
            writer.write(mol)
            writer.close()
            conf = mol.GetConformer()
            xyz = np.asarray([
                [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                for i in range(mol.GetNumAtoms())
            ])
            span = xyz.max(axis=0) - xyz.min(axis=0)
            elements = sorted({atom.GetSymbol() for atom in mol.GetAtoms()})
            records.append({
                "structure_id": structure_id,
                "polymer_id": row.polymer_id,
                "original_psmiles": row.original_psmiles,
                "canonical_psmiles": row.canonical_psmiles,
                "source_or_candidate": "calibration_source",
                "construction_method": "finite hydrogen-terminated oligomer",
                "repeat_count": repeat_count,
                "capping_method": "implicit terminal hydrogen after wildcard removal",
                "cell_dimensions_angstrom": "UNASSIGNED_PENDING_VACUUM_CONVERGENCE",
                "periodic_directions": "none; molecular surrogate in future 3D supercell",
                "coordinate_source": "RDKit ETKDGv3 plus force-field preoptimization",
                "geometry_generation_seed": builder.config["embedding_seed"],
                "force_field": metadata.get("force_field_used"),
                "force_field_converged": metadata.get("force_field_converged"),
                "selected_conformer_energy": metadata.get("selected_conformer_energy"),
                "coordinate_span_x_angstrom": float(span[0]),
                "coordinate_span_y_angstrom": float(span[1]),
                "coordinate_span_z_angstrom": float(span[2]),
                "formal_charge": int(Chem.GetFormalCharge(mol)),
                "multiplicity": 1,
                "gap_stratum": row.gap_stratum,
                "elements": ";".join(elements),
                "sanitization_status": "passed",
                "structure_path": str(sdf.relative_to(ROOT)),
                "structure_sha256": sha256(sdf),
                "convergence_subset": False,
            })
    return pd.DataFrame(records), pd.DataFrame(failures)


def protocol_lock(required_elements: list[str]) -> dict:
    return {
        "status": "PREPARED_EXECUTION_BLOCKED",
        "oracle_class": "CALIBRATABLE_FINITE_OLIGOMER_SURROGATE",
        "historical_polyverse_match_claimed": False,
        "physical_quantity": "Kohn-Sham HOMO-LUMO gap of a hydrogen-terminated finite oligomer",
        "target_use": "empirically calibrated estimator of paired polymer-chain bandgap changes",
        "functional": "PBE (protocol choice; not claimed as historical polyVERSE setting)",
        "pseudopotential_policy": {
            "family": "one consistent SSSP release/family for every element",
            "required_elements": required_elements,
            "files": "BLOCKED_NOT_PRESENT",
            "hashes": "BLOCKED_NOT_PRESENT",
        },
        "cutoff_policy": "derive baseline from maximum recommended cutoff across selected UPFs; converge at 1.0x, 1.25x, 1.5x",
        "charge_density_cutoff_policy": "derive from the selected pseudopotential family and verify convergence; never assume a universal dual",
        "k_points": "Gamma only for isolated finite oligomer supercells; periodic-chain k-point study is not applicable to this surrogate",
        "occupations": "fixed occupations for finite closed-shell systems; smearing is disallowed unless a documented ambiguity study justifies it",
        "scf_threshold_grid_Ry": [1e-6, 1e-8, 1e-10],
        "vacuum_grid_angstrom": [12.0, 15.0, 18.0, 22.0],
        "geometry": {
            "initial": "ETKDGv3 plus MMFF94, UFF only when explicitly recorded",
            "dft_relaxation": "QE relax with fixed supercell after pseudopotentials are installed",
            "force_threshold": "to be locked before execution from QE units and verified in convergence pilot",
        },
        "spin": "closed-shell non-spin-polarized only after even-electron validation; otherwise fail pending explicit spin protocol",
        "bandgap_parser": "must parse occupied/unoccupied eigenvalues from verified QE output and reject metallic/smearing-ambiguous cases",
        "gate_to_paired_pilot": "blocked until calibration and oligomer-length behavior are observed and classified",
    }


def pair_is_topologically_eligible(source: str, candidate: str) -> tuple[bool, str, list[str]]:
    elements: set[str] = set()
    for label, smiles in (("source", source), ("candidate", candidate)):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, f"{label}_unparseable", []
        if sum(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()) != 2:
            return False, f"{label}_attachment_count_not_two", []
        if Chem.GetFormalCharge(mol) != 0:
            return False, f"{label}_nonneutral_requires_separate_charge_protocol", []
        if not PolymerRepeatGraph(smiles).is_unambiguous:
            return False, f"{label}_ambiguous_repeat_connectivity", []
        for atom in mol.GetAtoms():
            # RDKit can sanitize adversarial strings by silently assigning H to
            # hypervalent halogen/chalcogen atoms.  Those candidates are parser-
            # valid but not defensible inputs to an expensive physical oracle.
            if atom.GetSymbol() in {"F", "Cl", "Br", "I"} and atom.GetTotalNumHs() > 0:
                return False, f"{label}_hydrogenated_halogen_rejected", []
            if atom.GetSymbol() == "S" and atom.GetDegree() >= 3 and atom.GetTotalNumHs() > 0:
                return False, f"{label}_hypervalent_hydrogenated_sulfur_rejected", []
            if atom.GetNumRadicalElectrons() != 0:
                return False, f"{label}_radical_requires_separate_spin_protocol", []
        elements.update(atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 0)
    return True, "eligible_for_future_structure_construction", sorted(elements)


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    env = environment_report()
    (RUN / "environment_readiness.json").write_text(json.dumps(env, indent=2) + "\n")

    data = pd.read_csv(ROOT / "data/processed/processed.csv")
    # Explicitly exclude the sealed test split before selection.
    permitted = data[data["split"].isin(["train", "val"])].copy()
    calibration = diverse_calibration(permitted, n=12)
    if set(calibration["ml_split"]) - {"train", "val"}:
        raise AssertionError("sealed test sample entered calibration manifest")
    calibration.to_csv(RUN / "calibration_manifest.csv", index=False)

    structures, failures = write_structures(calibration)
    successful_n3 = structures[structures["repeat_count"].eq(3)]
    convergence_ids = set(
        successful_n3.groupby("gap_stratum", observed=True).head(1)["polymer_id"]
    )
    structures["convergence_subset"] = structures["polymer_id"].isin(convergence_ids)
    structures.to_csv(RUN / "structure_manifest.csv", index=False)
    failures.to_csv(RUN / "structure_failures.csv", index=False)
    convergence = structures[structures["convergence_subset"]].copy()
    convergence = convergence[convergence["repeat_count"].eq(3)]
    plan = []
    for row in convergence.itertuples(index=False):
        for parameter, values in (
            ("ecutwfc_multiplier", [1.0, 1.25, 1.5]),
            ("vacuum_angstrom", [12.0, 15.0, 18.0, 22.0]),
            ("scf_threshold_Ry", [1e-6, 1e-8, 1e-10]),
        ):
            for value in values:
                plan.append({
                    "structure_id": row.structure_id,
                    "gap_stratum": calibration.loc[calibration.polymer_id.eq(row.polymer_id), "gap_stratum"].iloc[0],
                    "parameter": parameter,
                    "value": value,
                    "status": "BLOCKED_MISSING_QE_AND_PSEUDOPOTENTIALS",
                    "acceptance_observation": "report absolute and paired gap changes; no post-hoc threshold",
                })
    pd.DataFrame(plan).to_csv(RUN / "convergence_plan.csv", index=False)

    shortlist = pd.read_csv(ROOT / "results/physical_oracle_preparation/20260921/candidate_shortlist.csv")
    eligibility = [
        pair_is_topologically_eligible(row.source_smiles, row.candidate_smiles)
        for row in shortlist.itertuples(index=False)
    ]
    shortlist["structure_eligibility"] = [item[0] for item in eligibility]
    shortlist["structure_eligibility_reason"] = [item[1] for item in eligibility]
    shortlist["pair_elements"] = [";".join(item[2]) for item in eligibility]
    eligible_pairs = shortlist[shortlist["structure_eligibility"]].copy()
    # Round-robin across attack/model strata and enforce distinct source molecules.
    pairs = (
        eligible_pairs.sort_values(["attack_family", "target_model", "model_drift_eV"], ascending=[True, True, False])
        .drop_duplicates(["attack_family", "target_model"])
        .drop_duplicates("source_canonical")
        .head(4)
        .copy()
    )
    pairs["oracle_status"] = "BLOCKED_PENDING_CALIBRATION"
    pairs["delta_G_signed_eV"] = np.nan
    pairs["delta_T_eV"] = np.nan
    pairs["E_adv_eV"] = np.nan
    pairs["signed_discrepancy_R_eV"] = np.nan
    pairs.to_csv(RUN / "paired_pilot_manifest.csv", index=False)

    elements = sorted(
        {element for value in structures["elements"] for element in value.split(";")}
        | {element for value in pairs["pair_elements"] for element in value.split(";") if element}
    )
    (RUN / "required_elements.json").write_text(json.dumps(elements, indent=2) + "\n")
    pd.DataFrame({
        "element": elements,
        "pseudopotential_family": "SSSP_TO_BE_INSTALLED_CONSISTENT_RELEASE",
        "filename": "BLOCKED_NOT_PRESENT",
        "sha256": "BLOCKED_NOT_PRESENT",
        "recommended_ecutwfc_Ry": "BLOCKED_NOT_PRESENT",
        "recommended_ecutrho_Ry": "BLOCKED_NOT_PRESENT",
        "verified": False,
    }).to_csv(RUN / "required_pseudopotentials.csv", index=False)

    protocol = protocol_lock(elements)
    (RUN / "protocol_lock.json").write_text(json.dumps(protocol, indent=2) + "\n")

    estimate = {
        "status": "PLANNING_RANGE_NOT_MEASURED",
        "warning": "No real QE timing exists; estimates must be replaced after one representative smoke job",
        "convergence_jobs": len(plan),
        "calibration_structures": len(structures),
        "paired_pilot_structures_if_gate_passes": 4 * 2 * 3,
        "planning_core_hours_per_structure": [8, 64],
        "planning_storage_per_structure_GB": [0.5, 5.0],
        "full_20_pair_structures_at_three_lengths": 20 * 2 * 3,
        "full_shortlist_planning_core_hours": [20 * 2 * 3 * 8, 20 * 2 * 3 * 64],
        "full_shortlist_planning_storage_GB": [20 * 2 * 3 * 0.5, 20 * 2 * 3 * 5.0],
    }
    (RUN / "compute_estimate.json").write_text(json.dumps(estimate, indent=2) + "\n")
    print(RUN)


if __name__ == "__main__":
    main()
