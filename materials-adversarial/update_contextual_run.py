import re

file_path = "src/materials_adv/attacks/search/strategies.py"
with open(file_path, "r") as f:
    content = f.read()

old_run = """    def _run(self, original: str) -> SearchResult:
        current = Candidate(tokens=tuple(original), drift=0.0)
        best = current
        trace = []
        
        # Initialize arms
        arms_state = []
        for p in self.proposals:
            name = p.names[0]
            # Tracking precision matrix Lambda and precision-mean vector xi
            if name in self.mu_priors and name in self.cov_priors:
                mu = self.mu_priors[name].copy()
                cov = self.cov_priors[name].copy()
                Lambda = np.linalg.inv(cov)
                xi = Lambda @ mu
            else:
                dim = len(self.context_features)
                Lambda = np.eye(dim)
                xi = np.zeros(dim)
            arms_state.append({"name": name, "Lambda": Lambda, "xi": xi, "proposal": p})
            
        z = self.context_features
        z_outer = np.outer(z, z) / self.noise_variance
        
        proposals_generated = 0
        model_queries = 0
        
        for iteration in range(self.proposal_budget):
            if model_queries >= self.query_budget:
                break
                
            # Sample from posterior and calculate scores
            scores = []
            for arm in arms_state:
                Lambda = arm["Lambda"]
                xi = arm["xi"]
                # cov = inv(Lambda)
                cov = np.linalg.inv(Lambda)
                mu = cov @ xi
                # Sample theta
                # To ensure numerical stability, use multivariate_normal
                try:
                    theta_hat = self.rng.multivariate_normal(mu, cov)
                except np.linalg.LinAlgError:
                    # Fallback to mean if covariance is singular due to precision issues
                    theta_hat = mu
                
                score = np.dot(z, theta_hat)
                scores.append(score)
                
            # Choose arm
            chosen_arm_idx = np.argmax(scores)
            chosen_arm = arms_state[chosen_arm_idx]
            self.proposal = chosen_arm["proposal"]
            
            candidate, entry = self._evaluate(current)
            proposals_generated += 1
            if entry.get("queried", False) and entry.get("rejection_reason") != "duplicate_cached_candidate":
                model_queries += 1
                
            # Reward
            reward = 0.0
            accepted = False
            validity = entry.get("representation_valid", False)
            if self.require_plausible:
                validity = validity and entry.get("plausible", False)
                
            drift = candidate.drift if candidate is not None else 0.0
            
            if self.reward_mode == "binary":
                if validity and drift >= self.threshold:
                    reward = 1.0
                    
            if candidate is not None and candidate.drift > current.drift:
                current = candidate
                best = self._better(best, candidate)
                accepted = True
                
            is_duplicate = entry.get("rejection_reason") == "duplicate_cached_candidate"
            entry["duplicate"] = is_duplicate
            
            if self.update_posterior and not is_duplicate:
                # Bayesian Linear Update
                chosen_arm["Lambda"] += z_outer
                chosen_arm["xi"] += (z * reward) / self.noise_variance
                entry["posterior_updated"] = True
            else:
                entry["posterior_updated"] = False
                if is_duplicate:
                    reward = 0.0
                    
            entry["iteration"] = iteration
            entry["chosen_arm"] = chosen_arm_idx
            entry["chosen_arm_name"] = chosen_arm["name"]
            entry["reward"] = reward
            entry["score"] = float(scores[chosen_arm_idx])
            trace.append(entry)
            
        return SearchResult(
            original=original,
            best_adversarial=best.text,
            best_drift=best.drift,
            trace=trace,
            candidate_proposals=proposals_generated,
            candidate_model_queries=model_queries
        )"""

new_run = """    def _run(self, original: _Candidate, evaluate, record) -> _Candidate:
        best, current = original, original
        iteration = 0
        
        # Initialize arms
        arms_state = []
        for p in self.proposals:
            name = p.names[0]
            # Tracking precision matrix Lambda and precision-mean vector xi
            if name in self.mu_priors and name in self.cov_priors:
                mu = self.mu_priors[name].copy()
                cov = self.cov_priors[name].copy()
                Lambda = np.linalg.inv(cov)
                xi = Lambda @ mu
            else:
                dim = len(self.context_features)
                Lambda = np.eye(dim)
                xi = np.zeros(dim)
            arms_state.append({"name": name, "Lambda": Lambda, "xi": xi, "proposal": p})
            
        z = self.context_features
        z_outer = np.outer(z, z) / self.noise_variance
        
        for _ in range(self.proposal_budget):
            iteration += 1
            
            # Sample from posterior and calculate scores
            scores = []
            for arm in arms_state:
                Lambda = arm["Lambda"]
                xi = arm["xi"]
                # cov = inv(Lambda)
                cov = np.linalg.inv(Lambda)
                mu = cov @ xi
                # Sample theta
                try:
                    theta_hat = self.rng.multivariate_normal(mu, cov)
                except np.linalg.LinAlgError:
                    theta_hat = mu
                
                score = np.dot(z, theta_hat)
                scores.append(score)
                
            # Choose arm
            chosen_arm_idx = int(np.argmax(scores))
            chosen_arm = arms_state[chosen_arm_idx]
            
            # Generate a proposal from the chosen arm
            precomputed = chosen_arm["proposal"].propose(current.tokens)
            
            candidate, proposed, entry = evaluate(current, precomputed_proposal=precomputed)
            
            # Reward
            reward = 0.0
            accepted = False
            validity = entry.get("representation_valid", False)
            if self.require_plausible:
                validity = validity and entry.get("plausible", False)
                
            drift = candidate.drift if candidate is not None else 0.0
            
            if self.reward_mode == "binary":
                if validity and drift >= self.threshold:
                    reward = 1.0
                    
            if candidate is not None and candidate.drift > current.drift:
                current = candidate
                best = self._better(best, candidate)
                accepted = True
                
            is_duplicate = entry.get("rejection_reason") == "duplicate_cached_candidate"
            entry["duplicate"] = is_duplicate
            
            if self.update_posterior and not is_duplicate:
                # Bayesian Linear Update
                chosen_arm["Lambda"] += z_outer
                chosen_arm["xi"] += (z * reward) / self.noise_variance
                entry["posterior_updated"] = True
            else:
                entry["posterior_updated"] = False
                if is_duplicate:
                    reward = 0.0
                    
            entry["iteration"] = iteration
            entry["chosen_arm"] = chosen_arm_idx
            entry["chosen_arm_name"] = chosen_arm["name"]
            entry["reward"] = reward
            entry["score"] = float(scores[chosen_arm_idx])
            
            record(candidate, proposed, entry, accepted)
            
        return best"""

content = content.replace(old_run, new_run)
with open(file_path, "w") as f:
    f.write(content)
print("Updated ContextualThompson._run")
