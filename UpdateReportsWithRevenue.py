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

    # Extract acquisition date
    acq_date_str = os.path.basename(report_path).split(".csv")[0]
    acq_date = datetime.strptime(acq_date_str, "%Y-%m-%d").date()

    SRM_CSV = "SRM_Report.csv"
    srm_df = pd.read_csv(SRM_CSV)
    srm_df['last_seen'] = pd.to_datetime(srm_df['last_seen']).dt.date

    # Get last_seen from SRM_Report
    last_seen_row = srm_df[srm_df['experiment_number'] == int(experiment_number)]
    if last_seen_row.empty:
        print(f"[WARNING] No SRM info found for experiment {experiment_number}, skipping.")
        return
    last_seen = last_seen_row.iloc[0]['last_seen']
    max_days = (last_seen - acq_date).days
    if max_days < 0:
        print(f"[INFO] Acquisition date after experiment end for {report_path}, skipping.")
        return

    user_ids = set(df['user_pseudo_id'].astype(str))

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

        new_columns = pd.DataFrame({
            f"AdRev_D{idx}": df['user_pseudo_id'].map(ad_map).fillna(0),
            f"IAPRev_D{idx}": df['user_pseudo_id'].map(iap_map).fillna(0)
        })
        df = pd.concat([df.reset_index(drop=True), new_columns.reset_index(drop=True)], axis=1)

        if ad_day.empty and iap_day.empty:
            break

    # Compute totals
    ad_cols = [col for col in df.columns if col.startswith("AdRev_D")]
    iap_cols = [col for col in df.columns if col.startswith("IAPRev_D")]

    df['total_ad_revenue'] = df[ad_cols].sum(axis=1)
    df['total_iap_revenue'] = df[iap_cols].sum(axis=1)
    df['total_revenue'] = df['total_ad_revenue'] + df['total_iap_revenue']

    import re

    day_numbers = []
    for col in df.columns:
        match = re.match(r"AdRev_D(\d+)$", col)
        if match:
            day_numbers.append(int(match.group(1)))

    last_day = max(day_numbers, default=0)


    # Build new filename with _Dx suffix
    base_name = os.path.basename(report_path).replace(".csv", "")
    new_name = f"{base_name}_D{last_day}.csv"
    new_path = os.path.join(os.path.dirname(report_path), new_name)

    # Remove old file to avoid duplicates
    os.remove(report_path)

    # Save with new name
    df.to_csv(new_path, index=False)
    print(f"[OK] Updated {new_name} with revenue data.")


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
