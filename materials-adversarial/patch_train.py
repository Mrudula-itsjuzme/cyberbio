import re

with open("src/materials_adv/training/train.py", "r") as f:
    content = f.read()

# Add imports
content = content.replace("from ..data.scaler import TargetScaler",
                          "from ..data.scaler import TargetScaler\nfrom ..evaluation.metrics import length_linear_regression, length_stratified_metrics")

# Add exploratory flag
content = content.replace("write_back_config: bool = True, seed: int = None):",
                          "write_back_config: bool = True, seed: int = None, exploratory_residualize: bool = False):")

# Compute lengths
length_code = """
    # Compute lengths for baseline gate and stratifications
    from ..data.tokenizer import tokenize
    train_lengths = np.array([len(tokenize(x)) for x in train_df["psmiles"]])
    val_lengths = np.array([len(tokenize(x)) for x in val_df["psmiles"]])
    test_lengths = np.array([len(tokenize(x)) for x in test_df["psmiles"]])
    
    # Baseline gate
    val_baseline = length_linear_regression(train_lengths, train_df["target"].values, val_lengths, val_df["target"].values)
    test_baseline = length_linear_regression(train_lengths, train_df["target"].values, test_lengths, test_df["target"].values)
    logger.info(f"Baseline Gate (Length-only LR) - Val MAE: {val_baseline['mae']:.4f} K, Test MAE: {test_baseline['mae']:.4f} K")
    
    if exploratory_residualize:
        logger.warning("EXPLORATORY: Residualizing length from training targets.")
        train_baseline = length_linear_regression(train_lengths, train_df["target"].values, train_lengths, train_df["target"].values)
        train_preds = np.polyval([train_baseline["slope"], train_baseline["intercept"]], train_lengths)
        train_df["target"] = train_df["target"].values - train_preds
"""

content = content.replace('train_ds = PolymerDataset(train_df, vocab, max_len, scaler)',
                          length_code + '\n    train_ds = PolymerDataset(train_df, vocab, max_len, scaler)')

# Adjust evaluation logging
eval_log = """
    test_mae = np.mean(np.abs(test_preds_inv - test_targets_inv))
    test_rmse = np.sqrt(np.mean((test_preds_inv - test_targets_inv)**2))
"""

eval_stratified = """
    if exploratory_residualize:
        # Re-add length prediction
        test_length_preds = np.polyval([test_baseline["slope"], test_baseline["intercept"]], test_lengths)
        test_preds_inv += test_length_preds
        val_length_preds = np.polyval([val_baseline["slope"], val_baseline["intercept"]], val_lengths)
        val_preds_inv += val_length_preds

    test_mae = np.mean(np.abs(test_preds_inv - test_targets_inv))
    test_rmse = np.sqrt(np.mean((test_preds_inv - test_targets_inv)**2))
    
    # Stratified eval
    stratified = length_stratified_metrics(test_targets_inv, test_preds_inv, test_lengths, n_bins=4)
    logger.info("Test MAE by length bins: " + ", ".join(f"{k}: {v['mae']:.2f}" for k, v in stratified.items()))
    
    if test_mae >= test_baseline['mae']:
        logger.warning("CONFOUND ALERT: Transformer failed to beat the length-only baseline!")
"""

content = content.replace(eval_log, eval_stratified)

# also need to update val_preds_inv in loop if residualizing
val_eval = """
        val_preds_inv = scaler.inverse_transform(val_preds)
        val_targets_inv = scaler.inverse_transform(val_targets)
"""
val_eval_stratified = """
        val_preds_inv = scaler.inverse_transform(val_preds)
        val_targets_inv = scaler.inverse_transform(val_targets)
        if exploratory_residualize:
            val_preds_inv += np.polyval([val_baseline["slope"], val_baseline["intercept"]], val_lengths)
            # targets were NOT residualized in val_df
"""
content = content.replace(val_eval, val_eval_stratified)

with open("src/materials_adv/training/train.py", "w") as f:
    f.write(content)

