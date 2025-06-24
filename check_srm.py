import pandas as pd
from scipy.stats import chi2_contingency
import glob
import os

# Load all acquisition files
acquisition_files = glob.glob("DailyAcquisitions/*.csv")

dfs = []
for file in acquisition_files:
    date_str = os.path.basename(file).replace(".csv", "")
    try:
        df = pd.read_csv(file)
        df['acquisition_date'] = pd.to_datetime(date_str).date()
        dfs.append(df)
    except Exception as e:
        print(f"[WARNING] Could not process file {file}: {e}")

df_all = pd.concat(dfs, ignore_index=True)

# Group by experiment_number and experiment_group
group_counts = df_all.groupby(['experiment_number', 'experiment_group'])['user_pseudo_id'].nunique().reset_index()
group_counts.rename(columns={'user_pseudo_id': 'user_count'}, inplace=True)

# SRM test with first/last seen
srm_results = []
experiment_ids = group_counts['experiment_number'].unique()

for exp_id in experiment_ids:
    subset = group_counts[group_counts['experiment_number'] == exp_id]
    counts = subset['user_count'].values
    labels = subset['experiment_group'].astype(str).values

    if len(counts) > 1:
        chi2, p, dof, expected = chi2_contingency([counts])

        # Determine acquisition dates for this experiment
        exp_data = df_all[df_all['experiment_number'] == exp_id]
        dates = pd.to_datetime(exp_data['acquisition_date'])

        first_seen = dates.min().date()
        last_seen = dates.max().date()
        p99_seen = dates.quantile(0.99).date()
        delay_days = (last_seen - p99_seen).days
        delayed_flag = "YES" if delay_days > 5 else "NO"

        if delayed_flag == "YES":
            print(f"[WARNING] Experiment {exp_id} has delayed users ({delay_days} days after p99)")

        srm_results.append({
            'experiment_number': exp_id,
            'groups': ', '.join(labels),
            'counts': ', '.join(map(str, counts)),
            'p_value': round(p, 5),
            'SRM_flag': 'FAIL' if p < 0.05 else 'OK',
            'first_seen': first_seen,
            'p99_seen': p99_seen,
            'last_seen': last_seen,
            'delayed_users_found': delayed_flag,
            'delay_days': delay_days
        })

# Export results
srm_df = pd.DataFrame(srm_results)
print(srm_df)
srm_df.to_csv("SRM_Report.csv", index=False)
