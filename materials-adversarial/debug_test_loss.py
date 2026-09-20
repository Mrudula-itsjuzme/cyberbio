import torch
from src.materials_adv.training.invariance import anchored_consistency_loss, worst_consistency_loss, objective

def test_consistency_loss():
    # Deterministic dummy data
    source = torch.tensor([[1.0], [2.0]])
    variants = torch.tensor([[[1.5], [1.2]], [[2.0], [2.4]]]) # 2 sources, 2 variants each
    
    # anchored_consistency_loss = mean_j (f(x_j') - f(x))^2
    # For source 0: (1.5 - 1.0)^2 = 0.25, (1.2 - 1.0)^2 = 0.04 -> mean = 0.145
    # For source 1: (2.0 - 2.0)^2 = 0.0, (2.4 - 2.0)^2 = 0.16 -> mean = 0.08
    # Batch mean = (0.145 + 0.08) / 2 = 0.1125
    expected_anchored = 0.1125
    anchored = anchored_consistency_loss(source, variants).item()
    print(f"Anchored: expected {expected_anchored:.4f}, got {anchored:.4f}")
    assert abs(anchored - expected_anchored) < 1e-6
    
    # worst_consistency_loss
    # For source 0: max drift is 1.5 - 1.0 -> 0.25
    # For source 1: max drift is 2.4 - 2.0 -> 0.16
    # Batch mean = (0.25 + 0.16) / 2 = 0.205
    expected_worst = 0.205
    worst = worst_consistency_loss(source, variants).item()
    print(f"Worst: expected {expected_worst:.4f}, got {worst:.4f}")
    assert abs(worst - expected_worst) < 1e-6
    
    prop_loss = torch.tensor(1.0)
    cons_loss = torch.tensor(2.0)
    
    # Check lambda = 0
    obj_zero = objective(prop_loss, cons_loss, 0.0).item()
    assert obj_zero == 1.0
    print("Lambda 0 objective is purely property loss")

    # Check lambda = 0.5
    obj_half = objective(prop_loss, cons_loss, 0.5).item()
    assert obj_half == 2.0
    print("Lambda 0.5 objective is property + 0.5 * consistency")

if __name__ == "__main__":
    test_consistency_loss()
