import argparse
import pandas as pd
import os
from datetime import datetime, timedelta
import sys

ACQ_DIR = "DailyAcquisitions"
REPORTS_DIR = "Reports"

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
        num_days = (end_date - start_date).days + 1
        return [start_date + timedelta(days=i) for i in range(num_days)]
    else:
        return [start_date]

def load_acquisitions(dates):
    all_acqs = []

    for date in dates:
        file_path = os.path.join(ACQ_DIR, f"{date}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, usecols=['user_pseudo_id', 'experiment_number', 'experiment_group'])
            df['acquisition_date'] = str(date)
            all_acqs.append(df)
        else:
            print(f"[WARNING] Acquisition file not found for {date}, skipping...")

    if not all_acqs:
        print("No acquisition data found for the selected dates.")
        sys.exit(1)

    return pd.concat(all_acqs, ignore_index=True)

def save_rows_to_reports(acq_df):
    # Remove rows where experiment info is missing
    acq_df = acq_df.dropna(subset=['experiment_number', 'experiment_group'])

    for _, row in acq_df.iterrows():
        experiment = str(int(row['experiment_number']))
        acq_date = row['acquisition_date']
        user_row = row.drop(labels='acquisition_date')

        # Create directory for experiment if needed
        experiment_dir = os.path.join(REPORTS_DIR, experiment)
        os.makedirs(experiment_dir, exist_ok=True)

        # File path to write to
        report_file = os.path.join(experiment_dir, f"{acq_date}.csv")

        # Append user to the file
        if os.path.exists(report_file):
            user_row.to_frame().T.to_csv(report_file, mode='a', header=False, index=False)
        else:
            user_row.to_frame().T.to_csv(report_file, mode='w', header=True, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate daily experiment reports based on acquisitions.")
    parser.add_argument("start_date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("end_date", type=str, nargs="?", help="Optional end date (YYYY-MM-DD)")

    args = parser.parse_args()
    dates = get_dates_to_process(args.start_date, args.end_date)

    acq_df = load_acquisitions(dates)
    save_rows_to_reports(acq_df)
    print("User rows saved to corresponding experiment reports.")
