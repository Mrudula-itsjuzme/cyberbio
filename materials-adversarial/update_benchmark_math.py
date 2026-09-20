import re

file_path = "scripts/run_contextual_benchmark.py"
with open(file_path, "r") as f:
    content = f.read()

# Remove sklearn import
content = content.replace("from sklearn.linear_model import BayesianRidge\n", "")

# Update train_contextual_prior
new_func = """def train_contextual_prior(train_data: pd.DataFrame, feature_names: list[str]):
    # Train scaler
    features = train_data[[f"feat_{f}" for f in feature_names]].values
    scaler = ContextScaler()
    scaler.fit(features)
    scaled_features = scaler.transform(features)
    
    mu_priors = {}
    cov_priors = {}
    
    # Simple Bayesian Ridge Regression with numpy
    # alpha is precision of the noise, lambda_ is precision of the weights
    alpha = 1.0
    lambda_ = 1.0
    
    for arm in train_data["arm"].unique():
        arm_df = train_data[train_data["arm"] == arm]
        idx = arm_df.index
        X = scaled_features[idx]
        y = arm_df["mean_reward"].values
        
        # prec = lambda * I + alpha * X^T X
        prec = lambda_ * np.eye(X.shape[1]) + alpha * (X.T @ X)
        cov = np.linalg.inv(prec)
        
        # mu = cov @ (alpha * X^T y)
        mu = cov @ (alpha * X.T @ y)
        
        mu_priors[arm] = mu
        cov_priors[arm] = cov
        
    return scaler, mu_priors, cov_priors"""

content = re.sub(r'def train_contextual_prior\(.*?\):.*?return scaler, mu_priors, cov_priors', new_func, content, flags=re.DOTALL)

with open(file_path, "w") as f:
    f.write(content)
print("Replaced sklearn with numpy Bayesian logic.")
