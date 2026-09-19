import json
from pathlib import Path

def main():
    base_dir = Path(__file__).resolve().parent.parent / "results/raw/llm_full"
    out_dir = Path(__file__).resolve().parent.parent / "results/frozen_candidate_banks"
    
    if not base_dir.exists():
        print(f"Directory {base_dir} does not exist yet. No candidates to export.")
        return
        
    out_dir.mkdir(parents=True, exist_ok=True)
    
    modes = ["llm_operator_constrained", "llm_iterative"]
    
    for mode in modes:
        out_file = out_dir / f"{mode}.jsonl"
        exported = 0
        
        with open(out_file, "w") as fout:
            mode_stripped = mode.replace("llm_", "")
            
            for fpath in base_dir.rglob("raw_*.jsonl"):
                if mode_stripped not in str(fpath):
                    continue
                    
                parts = fpath.parts
                budget = 0
                for p in parts:
                    if p.startswith("budget_"):
                        budget = int(p.split("_")[1])
                
                with open(fpath, "r") as fin:
                    for line in fin:
                        data = json.loads(line)
                        if data.get("prediction_drift", 0.0) > 0.0 and data.get("valid_rdkit") and data.get("constraint_pass"):
                            meta = data.get("metadata", {})
                            export_data = {
                                "source_id": data.get("source_id", "UNAVAILABLE"),
                                "source_sequence": data.get("source_sequence", "UNAVAILABLE"),
                                "candidate_sequence": data.get("candidate_sequence", "UNAVAILABLE"),
                                "mode": data.get("prompt_mode", "UNAVAILABLE"),
                                "seed": data.get("seed", "UNAVAILABLE"),
                                "query_budget": budget,
                                "prediction_drift": data.get("prediction_drift", "UNAVAILABLE"),
                                "signed_shift": data.get("candidate_prediction", 0) - data.get("source_prediction", 0),
                                "tanimoto_similarity": data.get("tanimoto_similarity", "UNAVAILABLE"),
                                "edit_distance": data.get("edit_distance", "UNAVAILABLE"),
                                "constraint_metadata": meta.get("rationale", "UNAVAILABLE"),
                                "provider": meta.get("provider", "UNAVAILABLE"),
                                "model": meta.get("model", "UNAVAILABLE"),
                                "prompt_hash": meta.get("prompt_hash", "UNAVAILABLE"),
                                "run_id": meta.get("run_id", "UNAVAILABLE"),
                                "git_commit": meta.get("git_commit", "UNAVAILABLE"),
                                "source_bank_hash": meta.get("source_bank_hash", "UNAVAILABLE"),
                                "checkpoint_hash": meta.get("checkpoint_hash", "UNAVAILABLE")
                            }
                            fout.write(json.dumps(export_data) + "\n")
                            exported += 1
                            
        print(f"Exported {exported} successful candidates to {out_file}")

if __name__ == "__main__":
    main()
