import re

file_path = "src/materials_adv/attacks/search/strategies.py"
with open(file_path, "r") as f:
    content = f.read()

# 1. Update BlackBoxSearch.__init__ to explicitly accept proposal_budget
init_pattern = r'def __init__\(self, predictor: PredictorProtocol, proposal: ProposalProtocol,\s*rng: np\.random\.Generator, \*, query_budget: int,\s*require_plausible: bool = True, max_changes: int \| None = None,\s*max_proposal_multiplier: int = 50\):'
new_init = '''def __init__(self, predictor: PredictorProtocol, proposal: ProposalProtocol,
                 rng: np.random.Generator, *, query_budget: int,
                 require_plausible: bool = True, max_changes: int | None = None,
                 max_proposal_multiplier: int = 50, proposal_budget: int | None = None):'''
content = re.sub(init_pattern, new_init, content)

# Replace self.max_proposals assignment
max_prop_pattern = r'self\.max_proposals = query_budget \* max_proposal_multiplier'
new_max_prop = '''self.proposal_budget = proposal_budget if proposal_budget is not None else query_budget * max_proposal_multiplier'''
content = re.sub(max_prop_pattern, new_max_prop, content)

# Replace self.max_proposals usage throughout
content = content.replace("self.max_proposals", "self.proposal_budget")

# 2. Update ThompsonSamplingSearch.__init__
t_init_pattern = r'def __init__\(\s*self,\s*predictor: PredictorProtocol,\s*proposals: Sequence\[ProposalProtocol\],\s*rng: np\.random\.Generator,\s*\*,[\s\S]*?update_posterior: bool = True\s*\):'
t_new_init = '''def __init__(
        self,
        predictor: PredictorProtocol,
        proposals: Sequence[ProposalProtocol],
        rng: np.random.Generator,
        *,
        query_budget: int,
        require_plausible: bool = True,
        max_changes: int | None = None,
        max_proposal_multiplier: int = 50,
        proposal_budget: int | None = None,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        reward_mode: str = "binary",
        threshold: float = 0.0,
        update_posterior: bool = True
    ):'''
content = re.sub(t_init_pattern, t_new_init, content)

t_super_pattern = r'max_proposal_multiplier=max_proposal_multiplier\s*\)'
t_new_super = '''max_proposal_multiplier=max_proposal_multiplier,
            proposal_budget=proposal_budget
        )'''
content = re.sub(t_super_pattern, t_new_super, content, count=1)

# 3. Update ThompsonSamplingSearch._run for duplicate checking
dup_pattern = r'''            if self\.update_posterior:
                arms\[chosen_arm_idx\]\[0\] \+= reward
                arms\[chosen_arm_idx\]\[1\] \+= \(1\.0 - reward\)
            
            # Record strictly requested diagnostic fields in the trace entry'''
new_dup = '''            # 4. Check Duplicate and Update Posterior
            is_duplicate = entry.get("rejection_reason") == "duplicate_cached_candidate"
            entry["duplicate"] = is_duplicate
            
            if self.update_posterior and not is_duplicate:
                arms[chosen_arm_idx][0] += reward
                arms[chosen_arm_idx][1] += (1.0 - reward)
                entry["posterior_updated"] = True
            else:
                entry["posterior_updated"] = False
                if is_duplicate:
                    reward = 0.0 # Force reward to 0 for logging on duplicate
            
            # Record strictly requested diagnostic fields in the trace entry'''
content = re.sub(dup_pattern, new_dup, content)

with open(file_path, "w") as f:
    f.write(content)
print("Updated strategies.py")
