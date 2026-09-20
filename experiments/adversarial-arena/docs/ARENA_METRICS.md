# Metrics

Attacker artifacts contain validity rate; mean, median, p90 and maximum drift;
target queries; drift/query; proposal efficiency; and unique-candidate diversity.
Defender artifacts contain clean MAE/RMSE/R², training loss, robust drift, buffer
size, and the gain ratio. System artifacts contain round state and wall time.

Candidate diversity is currently unique-string fraction, not chemical diversity.
MCMC query accounting comes from its canonical terminal metadata plus the Arena's
two scoring queries. RandomSMILES counts the two Arena scoring queries.

