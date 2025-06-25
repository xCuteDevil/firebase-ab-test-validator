import os
import argparse
import pandas as pd

REPORTS_DIR = "Reports"

def format_percent_diff(df, value_col, baseline_group):
    baseline_value = df.loc[baseline_group, value_col]
    def calc_diff(row):
        if row.name == baseline_group:
            return "0.00%"
        diff = (row[value_col] - baseline_value) / baseline_value * 100
        return f"{diff:+.2f}%"
    return df.apply(calc_diff, axis=1)

def summarize_revenue(experiment_number):
    exp_path = os.path.join(REPORTS_DIR, str(experiment_number))
    if not os.path.isdir(exp_path):
        print(f"[ERROR] Folder '{exp_path}' not found.")
        return

    combined = []
    files_found = 0

    for filename in os.listdir(exp_path):
        if filename.endswith(".csv"):
            filepath = os.path.join(exp_path, filename)
            try:
                df = pd.read_csv(filepath, usecols=['user_pseudo_id', 'experiment_group', 'total_ad_revenue', 'total_iap_revenue'])
                combined.append(df)
                files_found += 1
            except Exception as e:
                print(f"[WARNING] Skipping '{filename}': {e}")

    if not combined:
        print("[INFO] No usable data found.")
        return

    data = pd.concat(combined, ignore_index=True).fillna(0)
    data['total_revenue'] = data['total_ad_revenue'] + data['total_iap_revenue']

    print(f"\n[SUMMARY] Experiment {experiment_number} — Processed files: {files_found}\n")

    # --- TOTAL REVENUE ---
    total_stats = data.groupby('experiment_group').agg(
        Users=('user_pseudo_id', 'nunique'),
        Total_Revenue=('total_revenue', 'sum'),
        Revenue_per_User=('total_revenue', 'mean'),
        Std_Dev=('total_revenue', 'std')
    )
    baseline_group = sorted(total_stats.index)[0]
    total_stats['% Difference from Baseline'] = format_percent_diff(total_stats, 'Revenue_per_User', baseline_group)

    print("=== TOTAL REVENUE ===")
    print(total_stats.rename(columns={
        'Total_Revenue': 'Total Revenue',
        'Revenue_per_User': 'Revenue per User',
        'Std_Dev': 'Standard Deviation'
    }))
    print()

    # --- AD REVENUE ---
    ad_stats = data.groupby('experiment_group').agg(
        Users=('user_pseudo_id', 'nunique'),
        Total_Ad_Revenue=('total_ad_revenue', 'sum'),
        Revenue_per_User=('total_ad_revenue', 'mean'),
        Std_Dev=('total_ad_revenue', 'std')
    )
    baseline_group_ad = sorted(ad_stats.index)[0]
    ad_stats['% Difference from Baseline'] = format_percent_diff(ad_stats, 'Revenue_per_User', baseline_group_ad)

    print("=== AD REVENUE ===")
    print(ad_stats.rename(columns={
        'Total_Ad_Revenue': 'Total Ad Revenue',
        'Revenue_per_User': 'Revenue per User',
        'Std_Dev': 'Standard Deviation'
    }))
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize experiment revenue metrics by group.")
    parser.add_argument("experiment_number", type=int, help="Experiment number (e.g. 44)")
    args = parser.parse_args()

    summarize_revenue(args.experiment_number)
