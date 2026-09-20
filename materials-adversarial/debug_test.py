import pytest
from tests.test_thompson_search import *

def test_debug():
    good_arm = MockProposal("good", returns_valid=True, mock_drift=100.0)
    predictor = MockPredictor({"good": good_arm})
    rng = np.random.default_rng(42)
    
    search = ThompsonSamplingSearch(
        predictor=predictor,
        proposals=[good_arm],
        rng=rng,
        query_budget=5,
        proposal_budget=10,
        require_plausible=False,
        reward_mode="binary",
        threshold=50.0
    )
    result = search.search("C")
    for i, entry in enumerate(result.trace):
        print(f"Iter {i}: rejection={entry.get('rejection_reason')}, dup={entry.get('duplicate')}, queried={entry.get('queried')}")
test_debug()
