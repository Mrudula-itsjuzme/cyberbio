# Local app guide

The FastAPI service reads `results/arena` and offers run, round, metric, candidate,
and matrix endpoints. The React/Vite client renders the loop, live/saved panels,
round timeline, candidate inspector, cross-attack heatmap, and RQ status. A new
smoke run requires an explicit `confirmed: true` request after the resolved
configuration is shown. The service binds locally and exposes no network attack.
Stopping is explicitly unimplemented rather than falsely acknowledged.

