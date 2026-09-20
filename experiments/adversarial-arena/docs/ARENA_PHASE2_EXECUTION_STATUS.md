# Phase 2 Execution Status

* Branch: `experiments/adversarial-arena-probabilistic`
* Git Status: Clean (but files untracked inside the detached worktree).
* Running processes: The previous experiment task (PID from the Sandbox) was running but was manually killed.
* `results/experiment.pid`: empty / dead.
* `results/experiment.log`: empty (log captured in the sandbox task log instead).

| Curriculum      | Seed     | Rounds Completed | Status               | Last Checkpoint | Artifacts Present |
|-----------------|----------|------------------|----------------------|-----------------|-------------------|
| baseline        | 20260920 | 3                | COMPLETED            | D_3             | yes               |
| baseline        | 20260921 | 3                | COMPLETED            | D_3             | yes               |
| baseline        | 20260922 | 3                | COMPLETED            | D_3             | yes               |
| probabilistic   | 20260920 | 3                | COMPLETED            | D_3             | yes               |
| probabilistic   | 20260921 | 3                | COMPLETED            | D_3             | yes               |
| probabilistic   | 20260922 | ?                | KILLED / INTERRUPTED | ?               | partial           |
| mixed           | 20260920 | 0                | NOT STARTED          | -               | no                |
| mixed           | 20260921 | 0                | NOT STARTED          | -               | no                |
| mixed           | 20260922 | 0                | NOT STARTED          | -               | no                |

(Note: All completed runs above are marked INVALIDATED_BY_PROTOCOL_FIX because they ran with incorrect state persistence and query accounting logic. They must be rerun or resumed after fixes.)
