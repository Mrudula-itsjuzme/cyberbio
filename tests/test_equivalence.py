import os
import json
import pytest

def test_cohort_manifest_exists():
    assert os.path.exists('results/representation_equivalence/cohort_manifest.json')

def test_equivalence_validation_exists():
    assert os.path.exists('results/representation_equivalence/equivalence_validation.csv')
    
def test_figures_exist():
    figures = [
        'equivalent_drift_distribution.png',
        'worst_case_equivalence_drift.png',
        'clean_error_vs_equivalence_drift.png',
        'disagreement_vs_prediction_error.png',
        'drift_by_transformation_family.png',
        'repeat_shift_vs_broader_equivalence.png',
        'drift_vs_length_change.png',
        'source_level_failure_examples.png'
    ]
    for fig in figures:
        assert os.path.exists(os.path.join('results/representation_equivalence', fig))
