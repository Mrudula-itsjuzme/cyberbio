# Comparison Branch Freeze

Freeze marker: **ATTACK-STRATEGY COMPARISON BRANCH FROZEN**

The final commit is the commit containing this file; record its hash after commit rather than embedding a pre-commit hash. Executed evidence: Random, MCMC, Evolutionary, Attribution-high/low/random in both domains; trained generator in both domains; untrained generator in Bio-Cyber. Blocked: Materials untrained control (timeout/noncompletion), real LLM (credentials/provider), RL (implementation/compute). Rejected: GAN-like search.

Valid result roots are `results/final/`, the classical per-example benchmark CSVs, recorded trained-generator CSVs, and Bio untrained-generator CSV. Invalid/removed: `run_materials_untrained_fast.py` and its hardcoded Materials-untrained CSV. Legacy derived summaries/plots remain historical files but are superseded wherever they treated that control as measured; only `results/final/` is the final analysis. Mock LLM files remain development/test evidence only. Archived mock RL remains under `archive/` only.

Unresolved blockers are real LLM access with pinned configuration, real RL implementation/training compute, and a timeout-bounded real Materials-untrained completion. Reopen only for one of those three items or a specific reviewer/faculty request.
