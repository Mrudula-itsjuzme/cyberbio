"""Reusable, artifact-safe experiment orchestration."""

from .pipeline import (
    AttackCandidate,
    ExperimentPipeline,
    ExperimentResult,
    compare_paired_records,
    generate_candidate_bank,
    score_candidate_bank,
)
from .closed_loop import (
    ATTACK_TAXONOMY,
    AttackCategory,
    ClosedLoopExperiment,
    ClosedLoopSettings,
    DefenseSpec,
)

__all__ = [
    "AttackCandidate",
    "ExperimentPipeline",
    "ExperimentResult",
    "compare_paired_records",
    "generate_candidate_bank",
    "score_candidate_bank",
    "ATTACK_TAXONOMY",
    "AttackCategory",
    "ClosedLoopExperiment",
    "ClosedLoopSettings",
    "DefenseSpec",
]
from .representation_attribution import AttributionSettings, RepresentationAttributionExperiment
from .attack_efficiency import AttackEfficiencyBenchmark, AttackEfficiencySettings

__all__ = ["AttributionSettings", "RepresentationAttributionExperiment",
           "AttackEfficiencyBenchmark", "AttackEfficiencySettings"]
