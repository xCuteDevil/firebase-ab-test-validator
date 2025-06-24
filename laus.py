import os
import argparse
import pandas as pd

REPORTS_DIR = "Reports"

def sum_revenue_by_group(experiment_number):
    exp_path = os.path.join(REPORTS_DIR, str(experiment_number))

    if not os.path.exists(exp_path):
        print(f"[ERROR] Experiment folder '{exp_path}' does not exist.")
        return

    combined = []
    files_found = 0

    for filename in os.listdir(exp_path):
        if not filename.endswith(".csv"):
            continue

        filepath = os.path.join(exp_path, filename)
        try:
            df = pd.read_csv(filepath, usecols=['experiment_group', 'total_ad_revenue', 'total_iap_revenue'])
            combined.append(df)
            files_found += 1
        except Exception as e:
            print(f"[WARNING] Skipping file {filename}: {e}")

    if not combined:
        print("[INFO] No valid data files found.")
        return

    all_data = pd.concat(combined, ignore_index=True)
    all_data = all_data.fillna(0)

    grouped = all_data.groupby('experiment_group')[['total_ad_revenue', 'total_iap_revenue']].sum()
    grouped['total_revenue'] = grouped['total_ad_revenue'] + grouped['total_iap_revenue']
    grouped = grouped.sort_index()

    print(f"\n[SUMMARY] Experiment {experiment_number} — Revenue by Group")
    print(f"  Processed files: {files_found}\n")
    print(grouped.round(2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sum revenue for a given experiment, grouped by experiment_group.")
    parser.add_argument("experiment_number", type=int, help="Experiment number (e.g. 44)")
    args = parser.parse_args()

    sum_revenue_by_group(args.experiment_number)
