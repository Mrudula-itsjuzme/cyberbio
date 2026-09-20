import pandas as pd
import scipy.stats
import numpy as np

# Load dataset
df = pd.read_csv('results/validity_boundary/candidate_boundary_dataset.csv')
df_valid = df[df['validity_status'] == 'valid'].copy()

# 2. Source-level statistic
source_stats = df_valid.groupby('source_id').agg(
    mean_u_valid=('u_valid', 'mean'),
    max_drift=('drift', 'max'),
    mean_drift=('drift', 'mean'),
    candidate_count=('candidate_id', 'count')
).reset_index()
source_rho, source_pval = scipy.stats.spearmanr(source_stats['mean_u_valid'], source_stats['max_drift'])
source_stats.to_csv('results/validity_boundary/canonical_source_level_stats.csv', index=False)
print(f"Source-level result: rho={source_rho:.3f}, p={source_pval:.3e}")

# 3. Method-stratified associations
method_rows = []
for m in df_valid['method'].unique():
    sub = df_valid[df_valid['method'] == m]
    if len(sub) > 5:
        m_rho, m_p = scipy.stats.spearmanr(sub['u_valid'], sub['drift'])
        
        top10_thresh = sub['drift'].quantile(0.90)
        top5_thresh = sub['drift'].quantile(0.95)
        
        # safely handle empty slices if quantiles fall weirdly
        top10_slice = sub[sub['drift'] >= top10_thresh]
        top5_slice = sub[sub['drift'] >= top5_thresh]
        
        method_rows.append({
            'method': m,
            'n_candidates': len(sub),
            'n_sources': sub['source_id'].nunique(),
            'rho': m_rho,
            'p_value': m_p,
            'mean_u_valid': sub['u_valid'].mean(),
            'top10_mean_u_valid': top10_slice['u_valid'].mean() if len(top10_slice)>0 else np.nan,
            'top5_mean_u_valid': top5_slice['u_valid'].mean() if len(top5_slice)>0 else np.nan
        })

method_df = pd.DataFrame(method_rows)
method_df.to_csv('results/validity_boundary/canonical_boundary_by_method.csv', index=False)
print("Method-stratified saved.")

# 4. Probe stability stats
print("Probe stability (u_valid_std):")
print(f"Mean:   {df_valid['u_valid_std'].mean():.4f}")
print(f"Median: {df_valid['u_valid_std'].median():.4f}")
print(f"P90:    {df_valid['u_valid_std'].quantile(0.90):.4f}")
print(f"Max:    {df_valid['u_valid_std'].max():.4f}")

# Method counts verification
counts = pd.read_csv('results/validity_boundary/canonical_cohort_counts.csv')
print("\nMethod Counts:")
print(counts[['method', 'valid_candidates', 'unique_sources']])
