import pytest

from hpc_oracle.backend.qe import QuantumEspressoBackend


def _backend():
    return QuantumEspressoBackend({
        "pseudo_dir": "/tmp/pseudo", "functional": "PBE",
        "cutoff_wfc": 100.0, "cutoff_rho": 800.0,
        "pseudopotentials": {
            "C": {"filename": "C.UPF", "z_valence": 4.0,
                  "cutoff_wfc": 45.0, "cutoff_rho": 360.0},
            "H": {"filename": "H.UPF", "z_valence": 1.0,
                  "cutoff_wfc": 80.0, "cutoff_rho": 320.0},
        },
    })


def test_parser_accepts_complete_converged_molecular_output(tmp_path):
    (tmp_path / "pw.scf.out").write_text("""
 number of electrons = 14.00
 convergence has been achieved in 11 iterations
 !    total energy = -12.500000 Ry
 highest occupied, lowest unoccupied level (ev): -5.2000 1.3000
 PWSCF        :      1m2.50s WALL
 JOB DONE.
""")
    backend = _backend()
    parsed = backend.parse_output(str(tmp_path))
    assert backend.validate_result(parsed)
    assert parsed.bandgap_eV == pytest.approx(6.5)
    assert parsed.runtime_seconds == pytest.approx(62.5)
    assert parsed.electron_count == pytest.approx(14.0)


@pytest.mark.parametrize("body,failure_class", [
    ("convergence NOT achieved\nJOB DONE.\n", "scf_not_converged"),
    ("convergence has been achieved\n", "incomplete_output"),
    ("convergence has been achieved\nJOB DONE.\n", "missing_band_edges"),
    ("convergence has been achieved\nhighest occupied, lowest unoccupied level (ev): 1 1\nJOB DONE.\n", "invalid_band_gap"),
])
def test_parser_fails_closed(tmp_path, body, failure_class):
    (tmp_path / "pw.scf.out").write_text(body)
    parsed = _backend().parse_output(str(tmp_path))
    assert not parsed.converged
    assert parsed.failure_class == failure_class
    assert not _backend().validate_result(parsed)


def test_write_input_contains_atoms_cell_species_and_isolation(tmp_path):
    pseudo_dir = tmp_path / "pseudo"
    pseudo_dir.mkdir()
    backend = _backend()
    backend.config["pseudo_dir"] = str(pseudo_dir)
    structure = {
        "atoms": [{"element": "C", "coordinates_angstrom": [10.0, 10.0, 10.0]},
                  {"element": "H", "coordinates_angstrom": [11.0, 10.0, 10.0]}],
        "elements": ["C", "H"], "nat": 2, "ntyp": 2,
        "formal_charge": 0, "electron_count": 6, "occupied_bands": 3,
        "cell_angstrom": [22.0, 22.0, 22.0],
    }
    backend.write_input(structure, {"calculation": "scf"}, str(tmp_path / "job"))
    text = (tmp_path / "job" / "pw.scf.in").read_text()
    for expected in ("ATOMIC_SPECIES", "ATOMIC_POSITIONS angstrom",
                     "CELL_PARAMETERS angstrom", "occupations = 'fixed'",
                     "assume_isolated = 'mt'", "ecutwfc = 80", "ecutrho = 360"):
        assert expected in text
    assert "MISSING" not in text


def test_write_input_rejects_cutoffs_below_present_species_recommendations(tmp_path):
    backend = _backend()
    backend.config["pseudo_dir"] = str(tmp_path)
    backend.config["pseudopotentials"]["C"].update(cutoff_wfc=45, cutoff_rho=360)
    structure = {"atoms": [{"element": "C", "coordinates_angstrom": [1, 1, 1]}],
                 "elements": ["C"], "nat": 1, "ntyp": 1, "formal_charge": 0,
                 "electron_count": 4, "occupied_bands": 2, "cell_angstrom": [20] * 3}
    with pytest.raises(ValueError, match="below the present-species"):
        backend.write_input(structure, {"cutoff_wfc": 40, "cutoff_rho": 320},
                            str(tmp_path / "job"))
