from __future__ import annotations

from dataclasses import dataclass, field

from .contracts import CandidateRecord


@dataclass
class HardExampleBuffer:
    capacity: int = 1000
    records: list[CandidateRecord] = field(default_factory=list)

    def add_top_k(self, candidates: list[CandidateRecord], k: int, round_id: int) -> list[CandidateRecord]:
        eligible = [c for c in candidates if c.validity_status == "valid" and c.drift is not None]
        chosen = sorted(eligible, key=lambda c: (-float(c.drift), c.source_id, c.candidate_input))[:k]
        for item in chosen:
            item.metadata["selected_round"] = round_id
        self.records = (self.records + chosen)[-self.capacity :]
        return chosen

    def state(self) -> list[dict]:
        return [r.to_dict() for r in self.records]

