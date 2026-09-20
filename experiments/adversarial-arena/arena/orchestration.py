from __future__ import annotations

import csv, hashlib, json, time
from pathlib import Path
from typing import Sequence

from .adaptive import ThompsonController
from .buffers import HardExampleBuffer
from .contracts import Attacker, Defender, SourceSample
from .metrics import attack_metrics, gain
from .storage import RunStorage


class ArenaRunner:
    def __init__(self, *, defender: Defender, attackers: dict[str, Attacker], validator,
                 storage: RunStorage, seed: int, hard_example_k: int = 4,
                 controller: ThompsonController | None = None):
        self.defender, self.attackers, self.validator = defender, attackers, validator
        self.storage, self.seed, self.hard_example_k = storage, seed, hard_example_k
        self.controller = controller
        self.buffer = HardExampleBuffer()

    def _score(self, candidates):
        for record in candidates:
            self.validator.validate(record)
            if record.validity_status != "valid":
                continue
            clean, changed = self.defender.predict([record.original_input, record.candidate_input])
            record.clean_prediction, record.challenging_prediction = clean, changed
            record.drift = record.score = abs(changed-clean)
            record.target_query_count += 2

    def _evaluate_attackers(self, samples: Sequence[SourceSample]) -> tuple[list[dict], dict[str, float | None]]:
        matrix, means = [], {}
        for name, attacker in self.attackers.items():
            records = attacker.propose(samples, self.defender); self._score(records)
            summary = attack_metrics(records); means[name] = summary["mean_drift"]
            matrix.append({"defender_checkpoint": self.defender.checkpoint_id, "attacker": name,
                           "mean_robustness_effect": summary["mean_drift"], "n": summary["valid"]})
        return matrix, means

    def run(self, *, clean_train: Sequence[SourceSample], clean_validation: Sequence[SourceSample], rounds: int, config: dict) -> dict:
        self.storage.write_json(self.storage.path/"config.json", config)
        self.storage.write_json(self.storage.path/"environment.json", self.storage.metadata())
        dataset_text = "\n".join(f"{s.source_id}\t{s.input}\t{s.target}" for s in [*clean_train,*clean_validation])
        self.storage.write_json(self.storage.path/"dataset.json", {"sha256": hashlib.sha256(dataset_text.encode()).hexdigest(),
            "train": len(clean_train), "validation": len(clean_validation)})
        summaries, evolution = [], []
        self.storage.transition("IDLE")
        # D0 is genuinely fit on the real clean split before the first attack.
        self.storage.transition("UPDATING_DEFENDER", -1, "clean baseline")
        self.defender.train(clean_train, [], round_id=0)
        previous_means: dict[str, float | None] = {}
        for round_id in range(rounds):
            started = time.monotonic(); rdir = self.storage.round_dir(round_id)
            try:
                arm = self.controller.select() if self.controller else list(self.attackers)[round_id % len(self.attackers)]
                self.storage.transition("RUNNING_ATTACK", round_id, arm)
                records = self.attackers[arm].propose(clean_validation, self.defender)
                self.storage.transition("VALIDATING", round_id)
                self._score(records)
                before = attack_metrics(records)
                selected = self.buffer.add_top_k(records, self.hard_example_k, round_id)
                self.storage.write_jsonl(rdir/"candidates.jsonl", [r.to_dict() for r in records])
                self.storage.write_jsonl(rdir/"hard_examples.jsonl", [r.to_dict() for r in selected])
                self.storage.write_json(rdir/"attacker_metrics.json", before)
                self.storage.transition("UPDATING_DEFENDER", round_id)
                training = self.defender.train(clean_train, self.buffer.records, round_id=round_id+1)
                checkpoint = self.defender.checkpoint(str(rdir/"defender_checkpoint.json"))
                clean_metrics = self.defender.evaluate(clean_validation)
                self.storage.transition("EVALUATING", round_id)
                matrix, means = self._evaluate_attackers(clean_validation)
                with (rdir/"cross_attack_matrix.csv").open("w", newline="") as handle:
                    writer=csv.DictWriter(handle, fieldnames=matrix[0].keys()); writer.writeheader(); writer.writerows(matrix)
                active_after = means.get(arm)
                defender_gain = gain(before.get("mean_drift"), active_after)
                reward = 0.0 if before.get("mean_drift") in (None, 0) else min(1.0, float(before["mean_drift"])/(before["mean_drift"]+1.0))
                if self.controller: self.controller.update(arm, reward)
                defender_metrics={**training, "clean_metrics": clean_metrics, "robust_drift": active_after,
                    "robustness_gain_ratio": defender_gain, "checkpoint": checkpoint}
                self.storage.write_json(rdir/"defender_metrics.json", defender_metrics)
                summary={"round":round_id,"strongest_attacker":max(means,key=lambda x: means[x] if means[x] is not None else -1),
                    "active_attacker":arm, **before, "defender_clean_rmse":clean_metrics["rmse"],
                    "defender_robustness":active_after,"defender_gain":defender_gain,
                    "hard_example_buffer_size":len(self.buffer.records),"wall_clock_seconds":time.monotonic()-started}
                summaries.append(summary)
                evolution.extend({"round":round_id,**row} for row in matrix)
                self.storage.write_json(rdir/"round_summary.json", summary)
                self.storage.write_json(rdir/"attacker_state.json", self.attackers[arm].get_state())
                if self.controller: self.storage.write_json(rdir/"controller_state.json", self.controller.state())
                self.storage.write_json(rdir/"COMPLETE.json", {"round":round_id})
                self.storage.transition("ROUND_COMPLETE", round_id)
                previous_means = means
            except Exception as exc:
                self.storage.write_json(rdir/"FAILED.json", {"round":round_id,"error":repr(exc),
                    "previous_checkpoint_preserved":self.defender.checkpoint_id})
                self.storage.transition("FAILED", round_id, repr(exc)); raise
        final=self.storage.path/"final"; final.mkdir(exist_ok=True)
        self._write_csv(final/"arena_round_summary.csv", summaries)
        self._write_csv(final/"arena_cross_attack_matrix.csv", evolution)
        self._write_csv(final/"arena_attacker_dynamics.csv", [{"round":s["round"],"attacker":s["active_attacker"],"mean_drift":s["mean_drift"],"validity_rate":s["validity_rate"]} for s in summaries])
        self._write_csv(final/"arena_defender_dynamics.csv", [{"round":s["round"],"clean_rmse":s["defender_clean_rmse"],"robust_drift":s["defender_robustness"],"gain":s["defender_gain"]} for s in summaries])
        self._write_csv(final/"arena_hard_example_stats.csv", [{"round":s["round"],"buffer_size":s["hard_example_buffer_size"]} for s in summaries])
        result={"run_id":self.storage.path.name,"status":"RUN_COMPLETE","rounds":summaries,
                "controller":self.controller.state() if self.controller else None,
                "scientific_status":"SMOKE_ONLY_NON_CANONICAL"}
        self.storage.write_json(self.storage.path/"run_summary.json", result)
        self.storage.transition("RUN_COMPLETE", rounds-1)
        return result

    @staticmethod
    def _write_csv(path: Path, rows: list[dict]) -> None:
        if not rows: path.write_text(""); return
        with path.open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

