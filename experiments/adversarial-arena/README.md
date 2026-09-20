# CyberBio Adversarial Arena

An offline, non-canonical attacker/defender co-evolution sandbox. It wraps the
existing `materials-adversarial` attacks and validator, preserves every round,
and exposes saved artifacts through a local app.

Run the verified small loop from the repository root:

```bash
materials-adversarial/.venv/bin/python experiments/adversarial-arena/scripts/run_smoke.py --run-id my_smoke
```

Run the API and frontend:

```bash
cd experiments/adversarial-arena
uvicorn app.backend.main:app --reload --port 8000
cd app/frontend && npm install && npm run dev -- --host 127.0.0.1
```

The smoke adapter is deliberately labelled non-canonical. Scientific claims
require the pending Transformer adapter and a pre-registered multi-seed pilot.

