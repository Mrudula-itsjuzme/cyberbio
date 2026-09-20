#!/bin/bash
set -e

# 1. Fix Validator Fail-Open
sed -i 's/return Validation(True, True, ())/return Validation(False, False, (), str(e))/g' arena/prob_lab/canonical.py || true
sed -i 's/return Validation(True, True, ())/return Validation(False, False, (), str(e))/g' arena/probabilistic_attackers.py || true
# Wait, let's verify where _ArenaValidator is. I will grep for _ArenaValidator and edit it.
