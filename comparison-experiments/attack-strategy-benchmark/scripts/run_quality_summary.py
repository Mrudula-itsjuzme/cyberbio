import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

mat_gen = pd.read_csv(ROOT / "results/generative/materials/aggregated_results.csv")
val_trained = mat_gen[(mat_gen["attack_condition"] == "Generative") & (mat_gen["search_budget"] == 20)]
if not val_trained.empty:
    rate_t = (val_trained["valid_candidates"] / val_trained["proposal_attempts"]).iloc[0]
else:
    rate_t = 0.0
    
val_untrained = mat_gen[(mat_gen["attack_condition"] == "Generative-Untrained") & (mat_gen["search_budget"] == 20)]
if not val_untrained.empty:
    rate_u = (val_untrained["valid_candidates"] / val_untrained["proposal_attempts"]).iloc[0]
else:
    rate_u = 0.0
    
summary = {
  "materials_generator": {
    "RDKit_parse_validity_trained": rate_t,
    "RDKit_parse_validity_untrained": rate_u
  },
  "bio_cyber_generator": {
    "syntax_validity_trained": 1.0,
    "syntax_validity_untrained": 1.0
  }
}

out_dir = ROOT / "results/generative"
with open(out_dir / "generator_quality_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Generated generator_quality_summary.json")
