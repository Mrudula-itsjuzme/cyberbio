"""Polymer-semantic repeat and translation transforms for linear PSMILES.

The transformations operate on the periodic polymer implied by two degree-one
``[*]`` attachment atoms. They do not conflate a finite molecular graph with
the infinite periodic polymer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from rdkit import Chem

from ..data.tokenizer import detokenize, tokenize
from .equivalence import normalize_attachment_tokens


@dataclass(frozen=True)
class RepeatShiftVariant:
    source: str
    variant: str
    transform_family: str
    transform_parameters: dict[str, Any]
    equivalence_status: str
    equivalence_reason: str

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _parts(text: str):
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    stars = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(stars) != 2 or any(mol.GetAtomWithIdx(i).GetDegree() != 1 for i in stars):
        return None
    neighbors = [mol.GetAtomWithIdx(i).GetNeighbors()[0].GetIdx() for i in stars]
    bonds = [mol.GetBondBetweenAtoms(i, n).GetBondType() for i, n in zip(stars, neighbors)]
    if bonds[0] != bonds[1]:
        return None
    return mol, stars, neighbors, bonds[0]


def _remove_stars(mol, stars):
    editable = Chem.RWMol(mol)
    mapping = {
        i: i - sum(j < i for j in stars)
        for i in range(mol.GetNumAtoms())
        if i not in stars
    }
    for i in sorted(stars, reverse=True):
        editable.RemoveAtom(i)
    return editable, mapping


def _stereo_inventory(mol) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return representation-independent atom and double-bond stereo labels."""
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    atom_stereo = tuple(sorted(
        atom.GetProp("_CIPCode")
        for atom in mol.GetAtoms()
        if atom.HasProp("_CIPCode")
    ))
    bond_stereo = tuple(sorted(
        str(bond.GetStereo())
        for bond in mol.GetBonds()
        if bond.GetStereo() != Chem.BondStereo.STEREONONE
    ))
    return atom_stereo, bond_stereo


def periodic_closure(text: str):
    """Return the one-cell periodic closure and original backbone path."""
    parts = _parts(text)
    if not parts:
        return None
    mol, stars, neighbors, bond_type = parts
    editable, mapping = _remove_stars(mol, stars)
    left, right = mapping[neighbors[0]], mapping[neighbors[1]]
    if left == right:
        # Both attachment markers on one atom do not define a translatable
        # linear repeat-unit backbone.
        return None
    # Compute before adding the closure edge: branch bonds are not shift cuts.
    path = list(Chem.GetShortestPath(editable, left, right))
    if editable.GetBondBetweenAtoms(left, right) is None:
        editable.AddBond(left, right, bond_type)
    try:
        Chem.SanitizeMol(editable)
    except Exception:
        return None
    return editable.GetMol(), left, right, bond_type, path


def periodic_signature(text: str) -> str | None:
    closure = periodic_closure(text)
    if closure is None:
        return None
    return Chem.MolToSmiles(closure[0], canonical=True, isomericSmiles=True)


def _cut_with_stars(closed, i: int, j: int, bond_type):
    editable = Chem.RWMol(closed)
    editable.RemoveBond(i, j)
    for atom in (i, j):
        star = editable.AddAtom(Chem.Atom(0))
        editable.AddBond(atom, star, bond_type)
    try:
        Chem.SanitizeMol(editable)
        return normalize_attachment_tokens(
            Chem.MolToSmiles(
                editable, canonical=False, rootedAtAtom=i, isomericSmiles=True
            )
        )
    except Exception:
        return None


def _build_repeat(source: str, factor: int) -> str | None:
    parts = _parts(source)
    if not parts or factor < 2:
        return None
    mol, stars, neighbors, bond_type = parts
    core_editable, mapping = _remove_stars(mol, stars)
    core = core_editable.GetMol()
    left, right = mapping[neighbors[0]], mapping[neighbors[1]]
    repeated = Chem.RWMol(core)
    for k in range(1, factor):
        offset = repeated.GetNumAtoms()
        repeated = Chem.RWMol(Chem.CombineMols(repeated.GetMol(), core))
        repeated.AddBond((k - 1) * core.GetNumAtoms() + right, offset + left, bond_type)
    left_star = repeated.AddAtom(Chem.Atom(0))
    right_star = repeated.AddAtom(Chem.Atom(0))
    repeated.AddBond(left_star, left, bond_type)
    repeated.AddBond((factor - 1) * core.GetNumAtoms() + right, right_star, bond_type)
    try:
        Chem.SanitizeMol(repeated)
        return normalize_attachment_tokens(
            Chem.MolToSmiles(
                repeated, canonical=False, rootedAtAtom=left_star, isomericSmiles=True
            )
        )
    except Exception:
        return None


def is_repeat_shift_equivalent(
    source: str, variant: str, family: str, parameters: dict[str, Any]
):
    """Validate periodic semantics, not merely canonical finite-graph identity."""
    source_parts, variant_parts = _parts(source), _parts(variant)
    if not source_parts or not variant_parts:
        return False, "invalid_or_non_linear_attachment_semantics", {}

    factor = int(parameters.get("repeat_factor", 1))
    source_charge = sum(a.GetFormalCharge() for a in source_parts[0].GetAtoms())
    variant_charge = sum(a.GetFormalCharge() for a in variant_parts[0].GetAtoms())
    if source_charge * factor != variant_charge:
        return False, "formal_charge_mismatch", {}
    source_atom_stereo, source_bond_stereo = _stereo_inventory(source_parts[0])
    variant_atom_stereo, variant_bond_stereo = _stereo_inventory(variant_parts[0])
    if tuple(sorted(source_atom_stereo * factor)) != variant_atom_stereo or tuple(
        sorted(source_bond_stereo * factor)
    ) != variant_bond_stereo:
        return False, "stereochemistry_mismatch", {
            "source_atom_stereo": source_atom_stereo,
            "variant_atom_stereo": variant_atom_stereo,
            "source_bond_stereo": source_bond_stereo,
            "variant_bond_stereo": variant_bond_stereo,
        }

    if family == "SHIFT":
        source_sig = periodic_signature(source)
        variant_sig = periodic_signature(variant)
        ok = source_sig is not None and source_sig == variant_sig
        return (
            ok,
            "periodic_star_linked_graph_match" if ok else "periodic_graph_mismatch",
            {"source_periodic": source_sig, "variant_periodic": variant_sig},
        )

    if family == "REPEAT":
        expected = _build_repeat(source, factor)
        expected_mol = Chem.MolFromSmiles(expected) if expected else None
        ok = expected_mol is not None and Chem.MolToSmiles(
            expected_mol, canonical=True, isomericSmiles=True
        ) == Chem.MolToSmiles(
            variant_parts[0], canonical=True, isomericSmiles=True
        )
        return (
            ok,
            "exact_tiled_repeat_graph_match" if ok else "repeat_graph_mismatch",
            {"repeat_factor": factor, "expected_repeat": expected},
        )

    if family == "SEGMENTATION":
        repeated = _build_repeat(source, factor)
        repeated_sig = periodic_signature(repeated) if repeated else None
        variant_sig = periodic_signature(variant)
        ok = repeated_sig is not None and repeated_sig == variant_sig
        return (
            ok,
            "repeated_cell_periodic_graph_match" if ok else "segmentation_graph_mismatch",
            {"repeat_factor": factor, "repeated_periodic": repeated_sig,
             "variant_periodic": variant_sig},
        )

    return False, "unsupported_family", {}


def shift_variants(source: str, cap: int = 20) -> list[RepeatShiftVariant]:
    closure = periodic_closure(source)
    if not closure:
        return []
    closed, _left, _right, _closure_bond, path = closure
    output: list[RepeatShiftVariant] = []
    seen = {source}
    for offset, (i, j) in enumerate(zip(path, path[1:]), 1):
        bond = closed.GetBondBetweenAtoms(i, j)
        variant = _cut_with_stars(closed, i, j, bond.GetBondType())
        params = {"shift_offset": offset, "repeat_factor": 1,
                  "orientation": "forward"}
        if variant and variant not in seen:
            ok, reason, _ = is_repeat_shift_equivalent(source, variant, "SHIFT", params)
            if ok:
                output.append(RepeatShiftVariant(
                    source, variant, "SHIFT", params, "VERIFIED_EQUIVALENT", reason
                ))
                seen.add(variant)
        if len(output) >= cap:
            break
    return output


def repeat_variant(source: str, factor: int = 2) -> RepeatShiftVariant | None:
    variant = _build_repeat(source, factor)
    if variant is None:
        return None
    params = {"repeat_factor": factor, "shift_offset": 0,
              "orientation": "forward"}
    ok, reason, _ = is_repeat_shift_equivalent(source, variant, "REPEAT", params)
    return RepeatShiftVariant(
        source, variant, "REPEAT", params,
        "VERIFIED_EQUIVALENT" if ok else "NOT_EQUIVALENT", reason
    )


def generate_repeat_shift(source: str, cap: int = 20) -> list[RepeatShiftVariant]:
    """Generate translation, repetition, and repeat-then-translation variants."""
    output = shift_variants(source, cap=max(0, cap - 1))
    repeated = repeat_variant(source, 2)
    if repeated and repeated.equivalence_status == "VERIFIED_EQUIVALENT":
        if repeated.variant != source:
            output.append(repeated)
        for shifted in shift_variants(
            repeated.variant, cap=max(0, cap - len(output))
        ):
            params = {**shifted.transform_parameters, "repeat_factor": 2}
            ok, reason, _ = is_repeat_shift_equivalent(
                source, shifted.variant, "SEGMENTATION", params
            )
            if ok:
                output.append(RepeatShiftVariant(
                    source, shifted.variant, "SEGMENTATION", params,
                    "VERIFIED_EQUIVALENT", reason
                ))
            if len(output) >= cap:
                break
    return output


def tokenizer_roundtrip(text: str, vocab) -> dict[str, Any]:
    tokens = tokenize(text)
    vocabulary = set(vocab)
    return {
        "roundtrip": detokenize(tokens) == text,
        "unknown_count": sum(token not in vocabulary for token in tokens),
        "token_count": len(tokens),
        "within_max_length": len(tokens) <= 256,
    }
