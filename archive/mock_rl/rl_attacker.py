import random

class RLAttacker:
    def __init__(self, action_space=["substitution", "insertion", "deletion", "rearrangement"], max_budget=50):
        self.action_space = action_space
        self.max_budget = max_budget
    
    def attack(self, initial_sequence, black_box_model, constraints):
        """
        Simplest defensible RL (e.g. Q-learning or REINFORCE mock)
        using observable quantities: prediction drift, validity.
        """
        best_candidate = initial_sequence
        best_reward = 0.0
        
        for _ in range(self.max_budget):
            # Sample action (mock)
            action = random.choice(self.action_space)
            
            # Form candidate (mock mutation)
            candidate = initial_sequence + "C" # mock mutation
            
            if constraints(candidate):
                drift = black_box_model(candidate) - black_box_model(initial_sequence)
                reward = drift
                
                if reward > best_reward:
                    best_reward = reward
                    best_candidate = candidate
                    
        return best_candidate
