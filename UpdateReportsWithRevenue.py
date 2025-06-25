import os
import pandas as pd
from datetime import datetime, timedelta
import argparse
import sys
import re

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

def process_report(report_path, experiment_number):
    df = pd.read_csv(report_path)

    if df.empty:
        print(f"[INFO] Skipping empty report: {report_path}")
        return

    if 'user_pseudo_id' not in df.columns or 'experiment_group' not in df.columns:
        print(f"[WARNING] Required columns missing in {report_path}, skipping.")
        return

    match = re.match(r"(\d{4}-\d{2}-\d{2})(?:_D(\d+))?\.csv", os.path.basename(report_path))
    if not match:
        print(f"[WARNING] Unexpected file format: {report_path}, skipping.")
        return

    acq_date_str, current_d_str = match.groups()
    acq_date = datetime.strptime(acq_date_str, "%Y-%m-%d").date()
    current_d = int(current_d_str) if current_d_str else -1

    SRM_CSV = "SRM_Report.csv"
    srm_df = pd.read_csv(SRM_CSV)
    srm_df['last_seen'] = pd.to_datetime(srm_df['last_seen']).dt.date

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
    updated = False

    for d in range(current_d + 1, max_days + 1):
        day = acq_date + timedelta(days=d)
        ad_path = os.path.join(AD_REVENUE_DIR, f"{day}.csv")
        iap_path = os.path.join(IAP_REVENUE_DIR, f"{day}.csv")

        if not (os.path.exists(ad_path) and os.path.exists(iap_path)):
            break

        ad_day = pd.read_csv(ad_path)
        iap_day = pd.read_csv(iap_path)

        ad_day = ad_day[ad_day['user_pseudo_id'].astype(str).isin(user_ids)] if 'revenue_sum' in ad_day.columns else pd.DataFrame()
        iap_day = iap_day[iap_day['user_pseudo_id'].astype(str).isin(user_ids)] if 'total_usd_revenue' in iap_day.columns else pd.DataFrame()

        ad_map = ad_day.groupby('user_pseudo_id')['revenue_sum'].sum() if not ad_day.empty else {}
        iap_map = iap_day.groupby('user_pseudo_id')['total_usd_revenue'].sum() if not iap_day.empty else {}

        new_cols = pd.DataFrame({
            f"AdRev_D{d}": df['user_pseudo_id'].map(ad_map).fillna(0),
            f"IAPRev_D{d}": df['user_pseudo_id'].map(iap_map).fillna(0)
        })
        df = pd.concat([df, new_cols], axis=1)

        updated = True

    if not updated:
        return
    ad_cols = sorted(
        [col for col in df.columns if col.startswith("AdRev_D") and re.search(r"\d+", col)],
        key=lambda x: int(re.search(r"\d+", x).group())
    )
    iap_cols = sorted(
        [col for col in df.columns if col.startswith("IAPRev_D") and re.search(r"\d+", col)],
        key=lambda x: int(re.search(r"\d+", x).group())
    )

    df = df.copy()
    df['total_ad_revenue'] = df[ad_cols].sum(axis=1)
    df['total_iap_revenue'] = df[iap_cols].sum(axis=1)
    df['total_revenue'] = df['total_ad_revenue'] + df['total_iap_revenue']

    last_day = max([int(re.search(r"\d+", col).group()) for col in ad_cols if re.search(r"\d+", col)], default=0)

    new_name = f"{acq_date}_D{last_day}.csv"
    new_path = os.path.join(os.path.dirname(report_path), new_name)

    if report_path != new_path:
        os.remove(report_path)

    df.to_csv(new_path, index=False)
    print(f"[OK] Updated {new_name} with D{last_day} revenue data.")

def main():
    parser = argparse.ArgumentParser(description="Update experiment reports with daily revenue data.")
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

            match = re.match(r"(\d{4}-\d{2}-\d{2})(?:_D\d+)?\.csv", report)
            if not match:
                continue

            report_date = parse_date(match.group(1))
            if report_date in dates:
                report_path = os.path.join(exp_path, report)
                process_report(report_path, experiment)

if __name__ == "__main__":
    main()
