import os
import pandas as pd
from datetime import datetime, timedelta
import argparse
import sys

REPORTS_DIR = "Reports"
AD_REVENUE_DIR = "DailyUserAdRevenue"
IAP_REVENUE_DIR = "DailyUserIAPRevenue"

def parse_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        print(f"Invalid date format: '{date_str}'. Expected format: YYYY-MM-DD.")
        sys.exit(1)

def get_dates_to_process(start_date_str, end_date_str=None):
    start_date = parse_date(start_date_str)
    
    if end_date_str:
        end_date = parse_date(end_date_str)
        if end_date < start_date:
            print("End date must be greater than or equal to start date.")
            sys.exit(1)
        return [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    else:
        return [start_date]

def daterange(start_date, days):
    for n in range(days + 1):
        yield start_date + timedelta(n)

def process_report(report_path, experiment_number):
    df = pd.read_csv(report_path)

    if df.empty:
        print(f"[INFO] Skipping empty report: {report_path}")
        return

    if 'acquisition_date' in df.columns:
        print(f"[INFO] Skipping already enriched report: {report_path}")
        return

    if 'user_pseudo_id' not in df.columns or 'experiment_group' not in df.columns:
        print(f"[WARNING] Required columns missing in {report_path}, skipping.")
        return

    # Extract acquisition date from filename
    acq_date_str = os.path.basename(report_path).split(".csv")[0]
    acq_date = datetime.strptime(acq_date_str, "%Y-%m-%d").date()

    user_ids = set(df['user_pseudo_id'].astype(str))
    max_days = 90

    for idx, current_date in enumerate(daterange(acq_date, max_days)):
        ad_file = os.path.join(AD_REVENUE_DIR, f"{current_date}.csv")
        iap_file = os.path.join(IAP_REVENUE_DIR, f"{current_date}.csv")

        ad_day = pd.read_csv(ad_file) if os.path.exists(ad_file) else pd.DataFrame()
        iap_day = pd.read_csv(iap_file) if os.path.exists(iap_file) else pd.DataFrame()

        if not ad_day.empty and 'user_pseudo_id' in ad_day.columns and 'revenue_sum' in ad_day.columns:
            ad_day = ad_day[ad_day['user_pseudo_id'].astype(str).isin(user_ids)]
            ad_map = ad_day.groupby('user_pseudo_id')['revenue_sum'].sum()
        else:
            ad_map = {}

        if not iap_day.empty and 'user_pseudo_id' in iap_day.columns and 'total_usd_revenue' in iap_day.columns:
            iap_day = iap_day[iap_day['user_pseudo_id'].astype(str).isin(user_ids)]
            iap_map = iap_day.groupby('user_pseudo_id')['total_usd_revenue'].sum()
        else:
            iap_map = {}

        df[f"AdRev_D{idx}"] = df['user_pseudo_id'].map(ad_map).fillna(0)
        df[f"IAPRev_D{idx}"] = df['user_pseudo_id'].map(iap_map).fillna(0)

        if ad_day.empty and iap_day.empty:
            break

    # Compute totals
    ad_cols = [col for col in df.columns if col.startswith("AdRev_D")]
    iap_cols = [col for col in df.columns if col.startswith("IAPRev_D")]

    df['total_ad_revenue'] = df[ad_cols].sum(axis=1)
    df['total_iap_revenue'] = df[iap_cols].sum(axis=1)
    df['total_revenue'] = df['total_ad_revenue'] + df['total_iap_revenue']

    df.to_csv(report_path, index=False)
    print(f"[OK] Updated {report_path} with revenue data.")

def main():
    parser = argparse.ArgumentParser(description="Update experiment reports with revenue data.")
    parser.add_argument("start_date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("end_date", type=str, nargs="?", help="Optional end date (YYYY-MM-DD)")
    args = parser.parse_args()

    dates = get_dates_to_process(args.start_date, args.end_date)

    for experiment in os.listdir(REPORTS_DIR):
        exp_path = os.path.join(REPORTS_DIR, experiment)
        if not os.path.isdir(exp_path):
            continue

        for report in os.listdir(exp_path):
            if not report.endswith(".csv"):
                continue

            # Extract acquisition date from filename
            report_date_str = report.replace(".csv", "")
            try:
                report_date = datetime.strptime(report_date_str, "%Y-%m-%d").date()
            except ValueError:
                print(f"[WARNING] Unexpected file format: {report}, skipping.")
                continue

            if report_date in dates:
                report_path = os.path.join(exp_path, report)
                process_report(report_path, experiment)

if __name__ == "__main__":
    main()
