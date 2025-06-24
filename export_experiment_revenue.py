# This script extracts users from a specific Firebase A/B test experiment,
# merges their corresponding Ad and IAP revenue across all available days,
# and outputs a clean dataset with the following columns:
# - user_pseudo_id
# - experiment_group
# - ad_revenue (sum of all ad revenues for that user)
# - iap_revenue (sum of all in-app purchase revenues for that user)
# - total_revenue (ad_revenue + iap_revenue)
#
# The output CSV is saved into the "ExperimentResults/" folder.

import pandas as pd
import glob
import os
from tqdm import tqdm
from datetime import datetime

# Experiment number to filter
experiment_number = 46

# Create output folder
output_dir = "ExperimentResults"
os.makedirs(output_dir, exist_ok=True)

# Load all players assigned to the given experiment
acq_files = glob.glob("DailyAcquisitions/*.csv")
acq_df = pd.concat([pd.read_csv(file) for file in acq_files], ignore_index=True)
exp_users = acq_df[acq_df['experiment_number'] == experiment_number][['user_pseudo_id', 'experiment_group']].drop_duplicates()

# Load and reshape Ad Revenue data
ad_files = glob.glob("DailyUserAdRevenue/*.csv")
ad_dfs = []
for file in tqdm(ad_files, desc="Loading Ad Revenue"):
    df = pd.read_csv(file)
    date = os.path.splitext(os.path.basename(file))[0]
    df['date'] = date
    ad_dfs.append(df)
ad_df = pd.concat(ad_dfs, ignore_index=True)
ad_pivot = ad_df.pivot_table(index='user_pseudo_id', columns='date', values='revenue_sum', aggfunc='sum').fillna(0)
ad_pivot.columns = [f"ad_revenue_D{idx}" for idx in range(len(ad_pivot.columns))]
ad_pivot['total_ad_revenue'] = ad_pivot.sum(axis=1)
ad_pivot.reset_index(inplace=True)

# Load and reshape IAP Revenue data
iap_files = glob.glob("DailyUserIAPRevenue/*.csv")
iap_dfs = []
for file in tqdm(iap_files, desc="Loading IAP Revenue"):
    df = pd.read_csv(file)
    date = os.path.splitext(os.path.basename(file))[0]
    df['date'] = date
    iap_dfs.append(df)
iap_df = pd.concat(iap_dfs, ignore_index=True)
iap_pivot = iap_df.pivot_table(index='user_pseudo_id', columns='date', values='total_usd_revenue', aggfunc='sum').fillna(0)
iap_pivot.columns = [f"iap_revenue_D{idx}" for idx in range(len(iap_pivot.columns))]
iap_pivot['total_iap_revenue'] = iap_pivot.sum(axis=1)
iap_pivot.reset_index(inplace=True)

# Merge all data
merged = exp_users.merge(ad_pivot, on='user_pseudo_id', how='left')
merged = merged.merge(iap_pivot, on='user_pseudo_id', how='left')
merged = merged.fillna(0)
merged['total_revenue'] = merged['total_ad_revenue'] + merged['total_iap_revenue']

# Save result
date_str = datetime.today().strftime("%Y-%m-%d")
output_file = os.path.join(output_dir, f"experiment_{experiment_number}_revenue_{date_str}.csv")
merged.to_csv(output_file, index=False)

