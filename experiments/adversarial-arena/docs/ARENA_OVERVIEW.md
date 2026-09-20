# Arena overview

The Arena is a round-based local research instrument. A defender is fit, an
attacker proposes candidates, the canonical validator decides validity, and
valid failures are ranked into a replay buffer. The defender is then updated
and all configured attackers re-evaluate the new checkpoint. Artifacts, state
transitions, adaptive policy state, and checkpoints are written atomically per
round. Existing `materials_adv` attack and validation code is used through
adapters; canonical results are never written.

The included ridge defender proves orchestration cheaply on real polyVERSE
rows. It is not TransPolymer and cannot support scientific conclusions.

