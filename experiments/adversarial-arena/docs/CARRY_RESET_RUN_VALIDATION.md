# Run Validation

All raw artifacts successfully parsed and verified against strict invariants:
- **Round-0 Control**: Passed. All methods yielded ~0 G_carry on Round 0.
- **Carry State Integrity**: Passed. `after R(t)` hashes exactly matched `before R(t+1)` hashes.
- **Reset State Integrity**: Passed. `before R(t)` adaptive state uniformly matched canonical initialization.
- **RNG Integrity**: Passed. Evaluator arrays `carry_rng_schedule` and `reset_rng_schedule` matched exactly.
- **Reference SHA Integrity**: Passed. Real `defender_checkpoint.json` files were reloaded via `TransPolymerDefender` and securely hashed. All 27 expected paired events referenced identical exact byte hashes.
- **D0 Identity**: Passed.
