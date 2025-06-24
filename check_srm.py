# This script performs SRM (Sample Ratio Mismatch) checks for A/B tests
# based on user acquisition data stored in the `DailyAcquisitions/` folder.
#
# It loads all CSV files in the folder, each representing a day's worth
# of user acquisition data. It then:
# - Groups users by `experiment_number` and `experiment_group`
# - Counts unique users in each group
# - Performs a chi-squared test to detect any mismatch in sample ratios
#
# If the distribution of users across groups is statistically unlikely
# (p-value < 0.05), the test is flagged as "FAIL", indicating a potential
# issue with randomization or traffic allocation.
#
# The final results are saved to `SRM_Report.csv` with the following columns:
# - experiment_number
# - groups (variant labels)
# - counts (number of users per group)
# - p_value (from chi-squared test)
# - SRM_flag (OK or FAIL)

import pandas as pd
from scipy.stats import chi2_contingency
import glob

acquisition_files = glob.glob("DailyAcquisitions/*.csv")
df_all = pd.concat([pd.read_csv(file) for file in acquisition_files], ignore_index=True)

df = pd.DataFrame(df_all)

# Group by experiment_number and experiment_group
group_counts = df.groupby(['experiment_number', 'experiment_group'])['user_pseudo_id'].nunique().reset_index()
group_counts.rename(columns={'user_pseudo_id': 'user_count'}, inplace=True)

# SRM test
srm_results = []
experiment_ids = group_counts['experiment_number'].unique()

for exp_id in experiment_ids:
    subset = group_counts[group_counts['experiment_number'] == exp_id]
    counts = subset['user_count'].values
    labels = subset['experiment_group'].astype(str).values

    if len(counts) > 1:
        chi2, p, dof, expected = chi2_contingency([counts])
        srm_results.append({
            'experiment_number': exp_id,
            'groups': ', '.join(labels),
            'counts': ', '.join(map(str, counts)),
            'p_value': round(p, 5),
            'SRM_flag': 'FAIL' if p < 0.05 else 'OK'
        })

srm_df = pd.DataFrame(srm_results)
print(srm_df)
srm_df.to_csv("SRM_Report.csv", index=False)

