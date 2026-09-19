from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class AttackResult:
    source_id: str
    source_sequence: str
    candidate_sequence: str
    attack_name: str
    attack_family: str
    seed: int
    query_count: int
    generation_count: int
    runtime_seconds: float
    valid_rdkit: bool
    constraint_pass: bool
    tanimoto_similarity: float
    edit_distance: int
    source_prediction: float
    candidate_prediction: float
    prediction_drift: float
    objective_value: float
    duplicate_proposals: int = 0
    failure_reason: str = ""
    llm_calls: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    prompt_mode: str = ""
    total_proposals: int = 0
    parsed_proposals: int = 0
    unique_proposals: int = 0
    rdkit_valid_proposals: int = 0
    constraint_pass_proposals: int = 0
    evaluated_proposals: int = 0
    successful_proposals: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def signed_shift(self) -> float:
        return self.candidate_prediction - self.source_prediction

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_sequence": self.source_sequence,
            "candidate_sequence": self.candidate_sequence,
            "attack_name": self.attack_name,
            "attack_family": self.attack_family,
            "seed": self.seed,
            "query_count": self.query_count,
            "generation_count": self.generation_count,
            "runtime_seconds": self.runtime_seconds,
            "valid_rdkit": self.valid_rdkit,
            "constraint_pass": self.constraint_pass,
            "tanimoto_similarity": self.tanimoto_similarity,
            "edit_distance": self.edit_distance,
            "source_prediction": self.source_prediction,
            "candidate_prediction": self.candidate_prediction,
            "prediction_drift": self.prediction_drift,
            "signed_shift": self.signed_shift,
            "objective_value": self.objective_value,
            "duplicate_proposals": self.duplicate_proposals,
            "failure_reason": self.failure_reason,
            "llm_calls": self.llm_calls,
            "llm_input_tokens": self.llm_input_tokens,
            "llm_output_tokens": self.llm_output_tokens,
            "prompt_mode": self.prompt_mode,
            "total_proposals": self.total_proposals,
            "parsed_proposals": self.parsed_proposals,
            "unique_proposals": self.unique_proposals,
            "rdkit_valid_proposals": self.rdkit_valid_proposals,
            "constraint_pass_proposals": self.constraint_pass_proposals,
            "evaluated_proposals": self.evaluated_proposals,
            "successful_proposals": self.successful_proposals,
            "metadata": self.metadata,
        }
