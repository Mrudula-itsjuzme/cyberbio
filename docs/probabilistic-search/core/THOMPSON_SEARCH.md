# Thompson Search

Status: core algorithm implemented and unit-tested; canonical adapters blocked.

The controller treats each existing perturbation family as an arm. In binary
mode it starts from a Beta(1,1) prior, samples one value per arm, selects the
largest, and updates alpha after a valid above-threshold effect or beta
otherwise. Continuous mode uses an online empirical Gaussian approximation.
Validity-aware mode requires both domain eligibility and the success rule.

The clean prediction consumes one target-model query. Invalid proposals consume
validation work but no target-model query. Posterior state is logged after every
selected proposal. A canonical benchmark result will not be emitted until the
shared source/model/validator/edit-family adapters pass accounting tests.

This is an established method applied to perturbation-family selection; no
novelty claim is made.
