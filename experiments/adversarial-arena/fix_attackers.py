import re
with open("arena/attackers.py", "r") as f:
    lines = f.read().split("\n")

start_idx = -1
end_idx = -1
in_adapter = False
for i, line in enumerate(lines):
    if line.startswith("class ExistingAttackAdapter"):
        in_adapter = True
    if in_adapter and line.startswith("    def propose("):
        start_idx = i
    if start_idx != -1 and line.startswith("    def get_state"):
        end_idx = i - 1
        break

replacement = """    def propose(self, samples: Sequence[SourceSample], defender: Defender) -> list[CandidateRecord]:
        self.source_accounting = []
        from materials_adv.data.tokenizer import tokenize

        attack = self.factory(self.rng, defender)
        records: list[CandidateRecord] = []
        for sample in samples:
            outcomes = attack.generate(tokenize(sample.input), n_variants=self.proposal_budget)
                        
            source_queries = sum(int(dict(out.params).get("search_candidate_model_queries", 0)) for out in outcomes)
            self.source_accounting.append({
                "source_id": sample.source_id,
                "clean_reference_queries": 0,
                "attack_target_queries": source_queries,
                "attack_query_budget": self.proposal_budget,
                "validator_calls": 0,
                "proposals_generated": len(outcomes)
            })

            for step, outcome in enumerate(outcomes):
                params = dict(outcome.params)
                queries = int(params.get("search_candidate_model_queries", 0))
                records.append(CandidateRecord(
                    source_id=sample.source_id, original_input=sample.input,
                    candidate_input=outcome.adversarial_representation,
                    attack_family=self.name, attack_variant=outcome.attack_type,
                    proposal_step=step, target_query_count=queries,
                    candidate_attack_query_count=queries,
                    true_target=sample.target, edit_distance=outcome.number_of_changes,
                    metadata=params,
                ))
        self.proposed += len(records)
        return records
"""

lines = lines[:start_idx] + replacement.split("\n") + lines[end_idx:]
with open("arena/attackers.py", "w") as f:
    f.write("\n".join(lines))
