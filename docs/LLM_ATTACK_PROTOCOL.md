# LLM Attack Protocol

Status: **BLOCKED_EXTERNAL_CREDENTIALS** in both domains. Existing recorded LLM sweeps used `MockLLMProvider` and are development tests only.

Modes are unguided/blind, constrained/operator-constrained, and feedback/iterative. The sanitized template supplies the source, permitted operators, validator constraints, remaining proposal and target-query budgets, and optionally prior validator feedback; it never supplies sealed labels or test targets. Expected JSON contains candidate sequence, edit description, and rationale.

The configured OpenAI-compatible adapter defaults to model `gpt-4o-mini`, temperature 0.7, JSON response format, a 30-second call timeout, and provider-reported token accounting. A production run must pin provider/model/version, add explicit `max_tokens` (currently absent), candidate count, retry policy, and immutable prompt hashes before execution. Each candidate passes the same domain validator; the LLM cannot declare validity. Log prompts, raw/sanitized outputs, parser outcome, validator outcome, API calls, input/output tokens, latency, proposals, target queries, and best drift. No `llm_attack_results.csv` is emitted because no real run exists.
