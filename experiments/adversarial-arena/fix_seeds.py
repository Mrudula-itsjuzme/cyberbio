import re
with open("arena/probabilistic_attackers.py", "r") as f:
    text = f.read()

text = re.sub(
    r'round_seed = self.base_seed \+ self\._round_id \* 1000 \+ source_index',
    r'round_seed = self._rng_schedule.pop(0) if hasattr(self, "_rng_schedule") and self._rng_schedule else self.base_seed + self._round_id * 1000 + source_index',
    text
)
with open("arena/probabilistic_attackers.py", "w") as f:
    f.write(text)
