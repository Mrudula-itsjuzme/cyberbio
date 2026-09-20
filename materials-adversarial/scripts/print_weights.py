import pandas as pd
import numpy as np
from materials_adv.attacks.search.context import ContextFeatureExtractor, ContextScaler

def train_contextual_prior(train_data: pd.DataFrame, feature_names: list[str]):
    features = train_data[[f"feat_{f}" for f in feature_names]].values
    scaler = ContextScaler()
    scaler.fit(features)
    scaled_features = scaler.transform(features)
    mu_priors = {}
    alpha = 1.0
    lambda_ = 1.0
    for arm in train_data["arm"].unique():
        arm_df = train_data[train_data["arm"] == arm]
        idx = arm_df.index
        X = scaled_features[idx]
        y = arm_df["mean_reward"].values
        prec = lambda_ * np.eye(X.shape[1]) + alpha * (X.T @ X)
        cov = np.linalg.inv(prec)
        mu = cov @ (alpha * X.T @ y)
        mu_priors[arm] = mu
    return mu_priors

train_data = pd.read_csv("results/contextual_thompson/context_arm_training_data.csv")
extractor = ContextFeatureExtractor(["C", "O", "N", "c", "1", "2", "(", ")", "="])
mu_priors = train_contextual_prior(train_data, extractor.feature_names)

print("Learned Feature Coefficients:")
for arm, mu in mu_priors.items():
    print(f"\nArm: {arm}")
    for i, name in enumerate(extractor.feature_names):
        print(f"  {name}: {mu[i]:.4f}")
