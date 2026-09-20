import os
for fname in ["scripts/build_reference_trajectory.py", "scripts/run_carry_reset_pair.py"]:
    with open(fname, "r") as f:
        content = f.read()
    
    bad_init = """TransPolymerDefender(
        project_root="../../materials-adversarial",
        seed=args.seed,
        checkpoint_dir="../../materials-adversarial/results/models/transformer_regressor",
        lambda_rep=0.5,
        learning_rate=0.001
    )"""
    good_init = """TransPolymerDefender(
        project_root="../../materials-adversarial",
        seed=args.seed,
        checkpoint_dir="results/models/transformer_regressor",
        lambda_rep=0.5,
        learning_rate=0.001
    )"""
    
    content = content.replace(bad_init, good_init)
    
    with open(fname, "w") as f:
        f.write(content)
