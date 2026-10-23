"""
phase1 package — Learning to Attack and Defend, Phase 1.

Modules:
  data            — Synthetic SMILES dataset generation, cleaning, splits
  representation  — Tokenizer, vocabulary, encoder utilities
  model           — Transformer property predictor, training loop
  attack_engine   — Transition model, plausibility checker, MCMC sampler
  utils           — Shared helpers: plotting, metrics, checkpoint I/O
"""

from phase1.data            import load_or_generate, generate_synthetic_dataset
from phase1.representation  import build_default_tokenizer, Vocabulary, SMILESTokenizer
from phase1.model           import build_model, train, evaluate, load_model
from phase1.attack_engine   import (
    TransitionModel, PlausibilityChecker, MCMCSampler,
    AttackResult, summarise_attack_results, results_to_records,
)
from phase1.utils           import (
    set_seed, plot_loss_curve, plot_drift_distribution,
    plot_transition_heatmap, plot_token_distribution,
    format_results_table, regression_report, ensure_dirs,
)

__all__ = [
    "load_or_generate", "generate_synthetic_dataset",
    "build_default_tokenizer", "Vocabulary", "SMILESTokenizer",
    "build_model", "train", "evaluate", "load_model",
    "TransitionModel", "PlausibilityChecker", "MCMCSampler",
    "AttackResult", "summarise_attack_results", "results_to_records",
    "set_seed", "plot_loss_curve", "plot_drift_distribution",
    "plot_transition_heatmap", "plot_token_distribution",
    "format_results_table", "regression_report", "ensure_dirs",
]
