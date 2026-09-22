"""Fail-closed Quantum ESPRESSO backend for finite molecular surrogates."""
from __future__ import annotations

import hashlib
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from hpc_oracle.backend.base import EnvironmentStatus, OracleBackend, ParsedOutput

RY_TO_EV = 13.605693122994
ATOMIC_MASSES = {"H": 1.00794, "C": 12.0107, "N": 14.0067, "O": 15.9994,
                 "F": 18.998403, "P": 30.973762, "S": 32.065,
                 "Cl": 35.453, "Br": 79.904}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class QuantumEspressoBackend(OracleBackend):
    """Prepare and validate isolated, closed-shell QE calculations."""

    REQUIRED_EXECUTABLES = ("pw.x", "bands.x", "projwfc.x")

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def _executable(self, name: str) -> Optional[Path]:
        configured = self.config.get("executables", {}).get(name)
        if configured:
            return Path(configured).expanduser().resolve()
        found = shutil.which(name)
        return Path(found).resolve() if found else None

    def validate_environment(self) -> EnvironmentStatus:
        missing: List[str] = []
        resolved: Dict[str, str] = {}
        expected_hashes = self.config.get("executable_sha256", {})
        for name in self.REQUIRED_EXECUTABLES:
            path = self._executable(name)
            if path is None or not path.is_file() or not path.stat().st_mode & 0o111:
                missing.append(f"{name} executable not found or not executable")
                continue
            resolved[name] = str(path)
            expected = expected_hashes.get(name)
            if not expected:
                missing.append(f"Missing pinned SHA256 for {name}")
            elif _sha256(path) != expected:
                missing.append(f"SHA256 mismatch for {name}")

        pseudo_raw = self.config.get("pseudo_dir")
        pseudo_dir = Path(pseudo_raw).expanduser().resolve() if pseudo_raw else None
        manifest = self.config.get("pseudopotentials", {})
        if pseudo_dir is None or not pseudo_dir.is_dir():
            missing.append(f"Pseudopotential directory not found: {pseudo_raw}")
        if not manifest:
            missing.append("Missing pinned pseudopotential manifest")
        elif pseudo_dir:
            releases = set()
            for element, entry in manifest.items():
                pp = pseudo_dir / entry.get("filename", "")
                releases.add(entry.get("release"))
                if not pp.is_file():
                    missing.append(f"Missing pseudopotential for {element}: {pp.name}")
                elif not entry.get("sha256") or _sha256(pp) != entry["sha256"]:
                    missing.append(f"SHA256 mismatch for {element}: {pp.name}")
                if entry.get("xc", "").upper() != "PBE":
                    missing.append(f"Non-PBE pseudopotential for {element}")
                if entry.get("z_valence") is None:
                    missing.append(f"Missing valence electron count for {element}")
            if len(releases) != 1 or None in releases:
                missing.append("Pseudopotentials do not share one pinned SSSP release")

        for key in ("functional", "cutoff_wfc", "cutoff_rho"):
            if key not in self.config:
                missing.append(f"Missing required parameter: {key}")
        if self.config.get("functional", "").upper() != "PBE":
            missing.append("Backend is pinned to the PBE SSSP release")
        if self.config.get("cutoff_wfc", 0) <= 0 or self.config.get("cutoff_rho", 0) <= 0:
            missing.append("Cutoffs must be positive")

        version = None
        pw = self._executable("pw.x")
        if pw and not missing:
            try:
                result = subprocess.run([str(pw), "-v"], input="", text=True,
                                        capture_output=True, timeout=30, check=False)
                match = re.search(r"PWSCF\s+v\.([\w.\-]+)", result.stdout + result.stderr)
                version = match.group(1) if match else None
                expected = str(self.config.get("qe_version", ""))
                if not version or version != expected:
                    missing.append(f"QE version mismatch: expected {expected}, observed {version}")
            except (OSError, subprocess.SubprocessError) as exc:
                missing.append(f"Could not execute pw.x: {exc}")
        return EnvironmentStatus(not missing, resolved.get("pw.x"), version,
                                 str(pseudo_dir) if pseudo_dir else None, missing)

    def prepare_structure(self, external_structure_path: str,
                          config: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from rdkit import Chem
        except ImportError as exc:
            raise RuntimeError("RDKit is required to read SDF structures") from exc
        path = Path(external_structure_path).resolve()
        if not path.is_file() or path.suffix.lower() not in {".sdf", ".mol"}:
            raise ValueError(f"Expected an existing SDF/MOL structure: {path}")
        mol = Chem.MolFromMolFile(str(path), removeHs=False, sanitize=True)
        if mol is None or mol.GetNumConformers() != 1 or not mol.GetNumAtoms():
            raise ValueError(f"Could not read one valid 3D molecule from {path}")
        conformer = mol.GetConformer()
        if not conformer.Is3D():
            raise ValueError("Structure does not contain 3D coordinates")
        charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        requested_charge = int(config.get("charge", self.config.get("charge", 0)))
        if charge != requested_charge:
            raise ValueError(f"SDF formal charge {charge} != requested charge {requested_charge}")

        atoms, elements = [], set()
        for index, atom in enumerate(mol.GetAtoms()):
            symbol = atom.GetSymbol()
            if symbol not in self.config.get("pseudopotentials", {}):
                raise ValueError(f"No pinned pseudopotential for element {symbol}")
            point = conformer.GetAtomPosition(index)
            xyz = [float(point.x), float(point.y), float(point.z)]
            if not all(math.isfinite(value) for value in xyz):
                raise ValueError("Non-finite atomic coordinate")
            atoms.append({"element": symbol, "coordinates_angstrom": xyz})
            elements.add(symbol)
        valence = sum(float(self.config["pseudopotentials"][a["element"]]["z_valence"])
                      for a in atoms) - requested_charge
        multiplicity = int(config.get("multiplicity", self.config.get("multiplicity", 1)))
        if not valence.is_integer() or valence <= 0:
            raise ValueError(f"Invalid valence electron count: {valence}")
        if multiplicity != 1 or int(valence) % 2:
            raise ValueError("Only even-electron closed-shell singlets are supported")
        vacuum = float(config.get("vacuum_angstrom", self.config.get("vacuum_angstrom", 12.0)))
        if vacuum < 8.0:
            raise ValueError("Vacuum must be at least 8 angstrom")
        spans = []
        for axis in range(3):
            values = [a["coordinates_angstrom"][axis] for a in atoms]
            spans.append(max(values) - min(values))
        cell_length = max(spans) + 2.0 * vacuum
        for axis in range(3):
            values = [a["coordinates_angstrom"][axis] for a in atoms]
            shift = cell_length / 2.0 - (min(values) + max(values)) / 2.0
            for atom in atoms:
                atom["coordinates_angstrom"][axis] += shift
        return {"source_path": str(path), "source_sha256": _sha256(path),
                "atoms": atoms, "elements": sorted(elements), "nat": len(atoms),
                "ntyp": len(elements), "formal_charge": requested_charge,
                "multiplicity": multiplicity, "electron_count": int(valence),
                "occupied_bands": int(valence) // 2,
                "cell_angstrom": [cell_length] * 3, "vacuum_angstrom": vacuum}

    def write_input(self, structure_data: Dict[str, Any], config: Dict[str, Any],
                    output_dir: str) -> None:
        out = Path(output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        calculation = config.get("calculation", "scf")
        if calculation not in {"scf", "relax"}:
            raise ValueError("Only scf and relax calculations are supported")
        prefix = config.get("prefix", "cyberbio_smoke")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", prefix):
            raise ValueError("Unsafe QE prefix")
        scratch = out / "scratch"
        scratch.mkdir(exist_ok=True)
        occupied = structure_data["occupied_bands"]
        nbnd = int(config.get("nbnd", occupied + max(4, math.ceil(0.2 * occupied))))
        if nbnd <= occupied:
            raise ValueError("nbnd must include unoccupied states")
        species = [f"{e} {ATOMIC_MASSES[e]:.8f} {self.config['pseudopotentials'][e]['filename']}"
                   for e in structure_data["elements"]]
        required_wfc = max(float(self.config["pseudopotentials"][e]["cutoff_wfc"])
                           for e in structure_data["elements"])
        required_rho = max(float(self.config["pseudopotentials"][e]["cutoff_rho"])
                           for e in structure_data["elements"])
        ecutwfc = float(config.get("cutoff_wfc", required_wfc))
        ecutrho = float(config.get("cutoff_rho", required_rho))
        if ecutwfc < required_wfc or ecutrho < required_rho:
            raise ValueError(
                f"Cutoffs {ecutwfc}/{ecutrho} Ry are below the present-species "
                f"SSSP recommendations {required_wfc}/{required_rho} Ry"
            )
        positions = [f"{a['element']} " + " ".join(f"{v:.10f}" for v in a["coordinates_angstrom"])
                     for a in structure_data["atoms"]]
        extra = "  tprnfor = .true.,\n  tstress = .true.,\n" if calculation == "relax" else ""
        text = f"""&CONTROL
  calculation = '{calculation}',
  prefix = '{prefix}',
  pseudo_dir = '{Path(self.config['pseudo_dir']).resolve()}',
  outdir = '{scratch}',
  disk_io = 'low',
  verbosity = 'high',
{extra}/
&SYSTEM
  ibrav = 0,
  nat = {structure_data['nat']},
  ntyp = {structure_data['ntyp']},
  input_dft = 'PBE',
  ecutwfc = {ecutwfc:.8f},
  ecutrho = {ecutrho:.8f},
  occupations = 'fixed',
  tot_charge = {structure_data['formal_charge']},
  nspin = 1,
  nbnd = {nbnd},
  assume_isolated = 'mt',
  nosym = .true.,
  noinv = .true.,
/
&ELECTRONS
  conv_thr = {float(config.get('conv_thr', self.config.get('conv_thr', 1e-6))):.10e},
  electron_maxstep = {int(config.get('electron_maxstep', 120))},
  mixing_beta = {float(config.get('mixing_beta', 0.3)):.8f},
  diagonalization = 'david',
/
"""
        if calculation == "relax":
            text += "&IONS\n  ion_dynamics = 'bfgs',\n/\n"
        cell = structure_data["cell_angstrom"]
        text += "ATOMIC_SPECIES\n" + "\n".join(species) + "\n"
        text += "ATOMIC_POSITIONS angstrom\n" + "\n".join(positions) + "\n"
        text += ("K_POINTS gamma\nCELL_PARAMETERS angstrom\n"
                 f"{cell[0]:.10f} 0.0 0.0\n0.0 {cell[1]:.10f} 0.0\n0.0 0.0 {cell[2]:.10f}\n")
        (out / f"pw.{calculation}.in").write_text(text, encoding="utf-8")

    def parse_output(self, output_dir: str) -> ParsedOutput:
        directory = Path(output_dir)
        candidates = (directory / "pw.scf.out", directory / "pw.relax.out")
        output = next((p for p in candidates if p.is_file()), candidates[0])
        if not output.is_file():
            return ParsedOutput(False, None, None, None, [], "Output file not found", None,
                                failure_class="missing_output")
        content = output.read_text(encoding="utf-8", errors="replace")
        warnings = re.findall(r"^\s*Warning:\s*(.+)$", content, re.MULTILINE | re.IGNORECASE)
        failures = [
            (r"convergence NOT achieved", "SCF convergence failed", "scf_not_converged"),
            (r"%{3,}|Error in routine|stopping \.{3}", "QE reported a fatal error", "qe_fatal_error"),
        ]
        for pattern, reason, classification in failures:
            if re.search(pattern, content, re.IGNORECASE):
                return ParsedOutput(False, None, None, None, warnings, reason, None,
                                    failure_class=classification)
        if "JOB DONE." not in content:
            return ParsedOutput(False, None, None, None, warnings, "Missing QE completion marker", None,
                                failure_class="incomplete_output")
        if "convergence has been achieved" not in content:
            return ParsedOutput(False, None, None, None, warnings, "No SCF convergence marker", None,
                                failure_class="missing_convergence_marker")
        edges = re.findall(r"highest occupied, lowest unoccupied level \(ev\):\s*([-+\d.Ee]+)\s+([-+\d.Ee]+)",
                           content, re.IGNORECASE)
        if not edges:
            return ParsedOutput(False, None, None, None, warnings, "HOMO/LUMO edges absent", None,
                                failure_class="missing_band_edges")
        homo, lumo = map(float, edges[-1])
        gap = lumo - homo
        if not all(math.isfinite(v) for v in (homo, lumo, gap)) or gap <= 0:
            return ParsedOutput(False, None, None, None, warnings, "Invalid finite orbital gap", None,
                                homo_eV=homo, lumo_eV=lumo, failure_class="invalid_band_gap")
        energies = re.findall(r"!\s+total energy\s+=\s+([-+\d.Ee]+)\s+Ry", content)
        electrons = re.findall(r"number of electrons\s*=\s*([-+\d.Ee]+)", content)
        fermi = re.findall(r"the Fermi energy is\s+([-+\d.Ee]+)\s+ev", content, re.IGNORECASE)
        times = re.findall(r"^\s*PWSCF\s*:.*?WALL\s*$", content, re.MULTILINE)
        seconds = self._parse_wall_seconds(times[-1]) if times else None
        energy = float(energies[-1]) * RY_TO_EV if energies else None
        electron_count = float(electrons[-1]) if electrons else None
        if energy is None or electron_count is None:
            return ParsedOutput(False, gap, energy, None, warnings,
                                "Required energy/electron metadata absent", seconds,
                                homo, lumo, "missing_required_metadata", electron_count)
        return ParsedOutput(True, gap, energy, float(fermi[-1]) if fermi else None,
                            warnings, None, seconds, homo, lumo, None, electron_count)

    @staticmethod
    def _parse_wall_seconds(line: str) -> Optional[float]:
        match = re.search(r"(?:(\d+)h)?(?:(\d+)m)?([\d.]+)s\s+WALL", line)
        if not match:
            return None
        hours, minutes, seconds = match.groups()
        return float(hours or 0) * 3600 + float(minutes or 0) * 60 + float(seconds)

    def validate_result(self, parsed: ParsedOutput) -> bool:
        return bool(parsed.converged and parsed.failure_reason is None and
                    parsed.failure_class is None and parsed.bandgap_eV is not None and
                    math.isfinite(parsed.bandgap_eV) and parsed.bandgap_eV > 0 and
                    parsed.total_energy_eV is not None and parsed.electron_count is not None)
