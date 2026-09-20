import re

file_path = "tests/test_thompson_search.py"
with open(file_path, "r") as f:
    content = f.read()

# Add test for duplicates
dup_test = '''
def test_duplicate_arm_penalty():
    # A mock proposal that always generates the exact same sequence
    # meaning all proposals after the first will be duplicates.
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
    
    # Trace length should be 10 (since proposal budget is 10 and it loops proposing same thing)
    assert len(result.trace) == 10
    
    # First proposal is fresh, posterior should be updated
    assert result.trace[0]["posterior_updated"] is True
    assert result.trace[0]["duplicate"] is False
    assert result.trace[0]["reward"] == 1.0
    assert result.trace[0]["alpha_after"] == 2.0
    assert result.trace[0]["beta_after"] == 1.0
    
    # 2nd through 10th proposals are duplicates
    for i in range(1, 10):
        assert result.trace[i]["posterior_updated"] is False
        assert result.trace[i]["duplicate"] is True
        assert result.trace[i]["reward"] == 0.0
        assert result.trace[i]["alpha_after"] == 2.0
        assert result.trace[i]["beta_after"] == 1.0
'''

# Update test_budget_enforcement
budget_test = '''
def test_budget_enforcement():
    # To test exactly hitting the budget, we need it to propose unique candidates
    class UniqueMockProposal(MockProposal):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.count = 0
        def propose(self, tokens):
            self.count += 1
            new_tokens = tuple(list(tokens) + [f"[{self._name}{self.count}]"])
            outcome = AttackOutcome(original_tokens=tuple(tokens), adversarial_tokens=new_tokens, attack_type=self._name)
            return Proposal(tokens=new_tokens, operator=self._name, outcome=outcome)
            
    unique_arm = UniqueMockProposal("good", returns_valid=True, mock_drift=100.0)
    predictor = MockPredictor({"good": unique_arm})
    rng = np.random.default_rng(42)
    budget = 7
    search = ThompsonSamplingSearch(
        predictor=predictor,
        proposals=[unique_arm],
        rng=rng,
        query_budget=budget,
        proposal_budget=50,
        require_plausible=False
    )
    result = search.search("C")
    assert result.candidate_model_queries == budget
    assert result.candidate_proposals >= budget
    assert result.candidate_proposals <= 50
'''

# Remove old test_budget_enforcement and append the new tests
content = re.sub(r'def test_budget_enforcement\(\):.*?(?=\ndef test_reproducibility)', budget_test + "\n", content, flags=re.DOTALL)
content += dup_test

with open(file_path, "w") as f:
    f.write(content)
print("Updated tests.")
