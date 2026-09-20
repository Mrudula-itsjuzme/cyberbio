# RL Attack Protocol

Status: **BLOCKED_COMPUTE / NOT_RUN**. No real environment, policy checkpoint, coefficients, or result exists; archived mock RL is excluded.

Proposed MDP: state `s_t` contains source/current tokens, current/source prediction, remaining budgets, edit history, and validator history. Action `a_t` selects edit type, position, and replacement. Transition `s_(t+1)=T(s_t,a_t)` applies the edit, validator, and when valid a target query. A candidate reward is `r_t=lambda_d*Delta|f(x_t)-f(x)|-lambda_i I_invalid-lambda_e edit_cost-lambda_q query_cost`; coefficients must be preregistered and ablated before it becomes the real reward. Policy options include masked categorical actor-critic or PPO; invalid actions should be masked where deterministic.

Training needs many target queries, multiple seeds, held-out sources, reward-ablation checks, and checkpoint provenance. Until those resources exist, compute cost and wall time are unknown rather than estimated. Promotion requires held-out benefit over random/MCMC at matched full cost.
