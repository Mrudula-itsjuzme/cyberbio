"""Verified, deterministic equivalent-PSMILES transformations.

The validator treats RDKit dummy atoms as polymer attachment points and requires
both canonical graph identity and identical attachment-point invariants.  It
therefore rejects plausible-looking strings which change repeat-unit chemistry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import Counter
from typing import Any

import numpy as np
from rdkit import Chem


@dataclass(frozen=True)
class EquivalenceResult:
    equivalent: bool
    reason: str
    validation_metadata: dict[str, Any]


@dataclass(frozen=True)
class EquivalentVariant:
    family: str
    source_representation: str
    variant_representation: str
    transformation_parameters: dict[str, Any]
    generation_seed: int
    subset: str

    def metadata(self) -> dict[str, Any]:
        return asdict(self)


def _mol(text: str) -> Chem.Mol | None:
    try:
        return Chem.MolFromSmiles(text, sanitize=True)
    except Exception:
        return None


def _attachment_signature(mol: Chem.Mol) -> list[tuple[int, tuple[int, ...], tuple[float, ...]]]:
    signature = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        neighbors = tuple(sorted(n.GetAtomicNum() for n in atom.GetNeighbors()))
        bonds = tuple(sorted(float(b.GetBondTypeAsDouble()) for b in atom.GetBonds()))
        signature.append((atom.GetDegree(), neighbors, bonds))
    return sorted(signature)

def normalize_attachment_tokens(text: str) -> str:
    """Normalize RDKit's bare dummy atom to the frozen vocabulary spelling."""
    out=[]; i=0
    while i < len(text):
        if text.startswith("[*]",i): out.append("[*]"); i+=3
        elif text[i]=="*": out.append("[*]"); i+=1
        else: out.append(text[i]); i+=1
    return "".join(out)

def graph_evidence(text: str) -> dict[str, Any]:
    mol=_mol(text)
    if mol is None: return {"parse_ok":False}
    from rdkit.Chem import rdMolDescriptors
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return {"parse_ok":True,"atom_count":mol.GetNumAtoms(),"bond_count":mol.GetNumBonds(),
        "element_counts":dict(sorted(Counter(a.GetSymbol() for a in mol.GetAtoms()).items())),
        "formula":rdMolDescriptors.CalcMolFormula(mol),"formal_charge":sum(a.GetFormalCharge() for a in mol.GetAtoms()),
        "stereocenters":sorted(label for _,label in Chem.FindMolChiralCenters(mol,includeUnassigned=True,useLegacyImplementation=False)),
        "bond_stereo":sorted(str(b.GetStereo()) for b in mol.GetBonds() if b.GetStereo()!=Chem.BondStereo.STEREONONE),
        "attachment_signature":_attachment_signature(mol),"connected_components":len(Chem.GetMolFrags(mol)),
        "canonical_nonisomeric":Chem.MolToSmiles(mol,canonical=True,isomericSmiles=False),
        "canonical_isomeric":Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True)}


def is_equivalent(source: str, variant: str) -> EquivalenceResult:
    """Return a chemistry-aware equivalence decision and auditable evidence."""
    left, right = _mol(source), _mol(variant)
    base = {"source_attachment_count": None, "variant_attachment_count": None}
    if left is None or right is None:
        return EquivalenceResult(False, "rdkit_parse_failure", base)
    left_sig, right_sig = _attachment_signature(left), _attachment_signature(right)
    base.update({
        "source_attachment_count": len(left_sig), "variant_attachment_count": len(right_sig),
        "source_attachment_signature": left_sig, "variant_attachment_signature": right_sig,
    })
    if left_sig != right_sig:
        return EquivalenceResult(False, "attachment_point_semantics_changed", base)
    left_canon = Chem.MolToSmiles(left, canonical=True, isomericSmiles=True)
    right_canon = Chem.MolToSmiles(right, canonical=True, isomericSmiles=True)
    base.update({"source_canonical": left_canon, "variant_canonical": right_canon})
    if left_canon != right_canon:
        return EquivalenceResult(False, "canonical_chemical_graph_changed", base)
    le,re=graph_evidence(source),graph_evidence(variant)
    for key in ("atom_count","bond_count","element_counts","formula","formal_charge","stereocenters","bond_stereo","connected_components"):
        if le[key]!=re[key]:
            base["mismatched_field"]=key
            return EquivalenceResult(False,f"{key}_changed",base)
    return EquivalenceResult(True, "canonical_graph_and_attachment_semantics_match", base)


def generate_equivalents(source: str, *, seed: int, target: int = 30,
                         max_attempts: int = 400) -> tuple[list[EquivalentVariant], list[dict[str, Any]]]:
    """Generate unique verified serializations; never return an uncertain candidate."""
    mol = _mol(source)
    if mol is None:
        return [], [{"attempt": 0, "reason": "source_parse_failure"}]
    rng = np.random.default_rng(seed)
    seen = {source}
    accepted: list[EquivalentVariant] = []
    failed: list[dict[str, Any]] = []
    n_atoms = mol.GetNumAtoms()
    for attempt in range(max_attempts):
        if len(accepted) >= target:
            break
        if attempt < n_atoms:
            root = attempt % n_atoms
            candidate = Chem.MolToSmiles(mol, canonical=False, rootedAtAtom=root,
                                         doRandom=False, isomericSmiles=True)
            family = "start_point_shift"
            params = {"root_atom": root, "attempt": attempt}
            subset = "MIPS_REPEAT_SHIFT_COMPATIBLE"
        elif attempt % 3 == 0:
            order = rng.permutation(n_atoms).tolist()
            candidate = Chem.MolToSmiles(Chem.RenumberAtoms(mol, order), canonical=False,
                                         isomericSmiles=True)
            family = "randomized_atom_ordering"
            params = {"atom_order": order, "attempt": attempt}
            subset = "BROADER_EQUIVALENCE"
        elif attempt % 3 == 1:
            root = int(rng.integers(0, n_atoms))
            order = rng.permutation(n_atoms).tolist()
            renumbered = Chem.RenumberAtoms(mol, order)
            candidate = Chem.MolToSmiles(renumbered, canonical=False,
                                         rootedAtAtom=order.index(root), isomericSmiles=True)
            family = "branch_order_permutation"
            params = {"root_atom": root, "atom_order": order, "attempt": attempt}
            subset = "BROADER_EQUIVALENCE"
        else:
            order = list(reversed(range(n_atoms)))
            candidate = Chem.MolToSmiles(Chem.RenumberAtoms(mol, order), canonical=False,
                                         isomericSmiles=True)
            family = "orientation_reversal"
            params = {"atom_order": order, "attempt": attempt}
            subset = "BROADER_EQUIVALENCE"
        if candidate in seen:
            failed.append({"attempt": attempt, "family": family, "reason": "duplicate"})
            continue
        decision = is_equivalent(source, candidate)
        if not decision.equivalent:
            failed.append({"attempt": attempt, "family": family, "reason": decision.reason,
                           "candidate": candidate})
            continue
        seen.add(candidate)
        accepted.append(EquivalentVariant(family, source, candidate, params, seed, subset))
    return accepted, failed
