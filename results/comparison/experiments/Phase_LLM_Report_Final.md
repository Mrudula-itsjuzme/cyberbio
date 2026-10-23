PROVIDER USED
- Abstraction implemented for `MockLLMProvider`, `OpenAICompatibleProvider`, and `LocalLLMProvider`.
- All LLM configuration reads strictly from environment variables without hardcoded keys.

MODEL USED
- Development runs leveraged `MockLLMProvider` simulating network limits, malformed JSON structures, token costs, duplicated outputs, and empty returns.

PROMPT MODES TESTED
Four primary modes implemented exactly as specified:
1. `blind`
2. `objective-aware`
3. `iterative`
4. `operator-constrained`
(Also created shortcut/sensitivity audit variants: `technical`, `minimal`, `structured`). All templates are versioned to `configs/prompts/` and are fully separated from the sealed environment.

TEST COUNT
- 9 new explicit LLM tests created (`test_llm_guided.py`), bringing the framework total to 30 robust automated tests.

TEST RESULTS
- All 30 tests PASSED. Specifically verifying:
  - infinite-loop protection on empty sets
  - malformed JSON interception
  - duplicate candidate tracking
  - prompt state stringification
  - target-model token query isolation
  - token/call accounting
  - history/rationale metadata capture
  - rate limiting handling
  - provider failure categorization

DEVELOPMENT RUN STATUS
- COMPLETED (`run_llm_dev.py`). 
- Ran all four prompt modes against a 10-source subset across a mock budget. Verified that tracking handles zero-drift outputs securely and duplicate filtering prevents target-model starvation.

FULL BENCHMARK STATUS
- PENDING (Do not automatically start full benchmark due to API constraints. Wait for LLM API setup).

LLM CALL COUNT
- Accurately logged per sample outcome (`llm_calls`). Dev run correctly scaled mock calls.

TARGET MODEL QUERY COUNT
- 10-50 per source. `queries_used` isolates target-model predictions securely from generation-side `LLM` calls.

TOKEN USAGE
- Logged natively inside `AttackResult` schema (`llm_input_tokens`, `llm_output_tokens`).

VALID PROPOSAL RATE
- Computed safely via the target model metrics (e.g. valid_rdkit / generations_used).

DUPLICATE RATE
- Verified duplicate handling. `duplicate_proposals` metric securely logged per evaluation to catch semantic prompt loops.

FAILURE COUNTS
- Safe error capturing (`empty_response`, `malformed_json`, `budget_exhausted`, `no_improvement`, `Provider Error`).
- Generated table schema in `aggregate_llm_sweep.py`.

SUMMARY PATHS
- Aggregation script `aggregate_llm_sweep.py` maps data to `results/summaries/llm/llm_sweep_results.csv` and `results/summaries/llm/failure_modes.csv`.

PLOT PATHS
- Designated output folder `results/plots/llm/`.

CANDIDATE BANK PATH
- `scripts/export_llm_candidate_bank.py` successfully implemented to dump successful candidates (with rationale metadata) into `results/summaries/llm_candidate_bank.jsonl`.

KNOWN LIMITATIONS
- LLM generation speed is fundamentally capped by API latency and iterative blocking per query.
- Evaluated duplicates cost `input_tokens` even though they are rejected instantly by the duplicate tracker.

LEAKAGE AUDIT RESULT
- Completed and audited. Output to `docs/LLM_LEAKAGE_AUDIT.md`. Conclusively asserts that sealed targets, benchmarks, and model ground-truth are absolutely inaccessible to prompt context matrices.
