import argparse
from datetime import datetime, timedelta
import sys

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate daily experiment reports based on acquisitions.")
    parser.add_argument("start_date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("end_date", type=str, nargs="?", help="Optional end date (YYYY-MM-DD)")

    args = parser.parse_args()
    dates = get_dates_to_process(args.start_date, args.end_date)

    print("Dates to process:")
    for d in dates:
        print(d)
