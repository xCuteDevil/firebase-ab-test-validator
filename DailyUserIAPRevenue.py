# This script is used to download daily user IAP revenue data from BigQuery
# It estimates the total size of the data to be processed and asks for confirmation
# before running the queries. The script saves the results in CSV files in a specified directory.
# The script uses the google-cloud-bigquery library to interact with BigQuery:
# pip install google-cloud-bigquery
# outputs the data into /BQ_downloads/DailyUserIAPRevenue in format YYYY-MM-DD.csv
# arguments:
# start_date: str: Start date in YYYY-MM-DD format (or single date for one day query)
# end_date: str: End date in YYYY-MM-DD format (optional)
# json_key is not part of the submitted solution

import os
import argparse
from google.cloud import bigquery as bq
from datetime import datetime, timedelta

# Setting credentials
json_key = r"hexapolis-bcb77-bdb7e18f6ae5.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = json_key

# Define the directory for storing CSV files
output_directory = "DailyUserIAPRevenue"
os.makedirs(output_directory, exist_ok=True)

def bq_estimate_total_size(start_date, end_date):
    client = bq.Client()
    total_bytes = 0

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # Each day in the interval
    current_dt = start_dt
    while current_dt <= end_dt:
        date_str = current_dt.strftime("%Y-%m-%d")
        filename = os.path.join(output_directory, f"{date_str}.csv")
        
        if not os.path.exists(filename):
            # Query for user IAP revenue on the current date
            query = f"""
                SELECT
                    user_pseudo_id,
                    SUM(price_usd) AS total_usd_revenue
                FROM (
                    SELECT DISTINCT
                    user_pseudo_id,
                    product_id,
                    FLOOR(event_timestamp / 60000000) AS rounded_timestamp, -- 1 minutes
                    price_usd
                    FROM
                    `hexapolis-bcb77.Events.Currency_events`
                    WHERE
                    event_date = "{date_str}"
                    AND price_usd IS NOT NULL
                ) AS unique_transactions
                GROUP BY
                    user_pseudo_id
                ORDER BY
                    total_usd_revenue DESC;
            """

            # Run dry run to estimate size
            dry_run_job = client.query(query, job_config=bq.QueryJobConfig(dry_run=True))
            total_bytes += dry_run_job.total_bytes_processed

        current_dt += timedelta(days=1)
    
    # Convert total bytes to gigabytes
    total_gb = total_bytes / (1024 ** 3)
    print(f"Total estimated bytes processed: {total_gb:.2f} GB")
    return total_gb

def bq_run_queries(start_date, end_date):
    client = bq.Client()

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # Each day in the interval
    current_dt = start_dt
    while current_dt <= end_dt:
        date_str = current_dt.strftime("%Y-%m-%d")
        filename = os.path.join(output_directory, f"{date_str}.csv")
        
        # Only run the query if the file doesn't already exist
        if not os.path.exists(filename):
            # Query for user IAP revenue on the current date
            query = f"""
                SELECT
                    user_pseudo_id,
                    SUM(price_usd) AS total_usd_revenue
                FROM (
                    SELECT DISTINCT
                    user_pseudo_id,
                    product_id,
                    FLOOR(event_timestamp / 60000000) AS rounded_timestamp, -- 1 minutes
                    price_usd
                    FROM
                    `hexapolis-bcb77.Events.Currency_events`
                    WHERE
                    event_date = "{date_str}"
                    AND price_usd IS NOT NULL
                ) AS unique_transactions
                GROUP BY
                    user_pseudo_id
                ORDER BY
                    total_usd_revenue DESC;
            """

            # Run the actual query
            query_job = client.query(query)
            query_result = query_job.result(timeout=600)

            # Save each date's result directly in the output directory
            df_iter = query_result.to_dataframe_iterable()
            for i, df in enumerate(df_iter):
                df.to_csv(filename, header=(i == 0), index=False)  # Overwrite mode (no 'a' for append)

            print(f"Data for {date_str} saved to: {filename}")
        else:
            print(f"File for {date_str} already exists, skipping.")

        current_dt += timedelta(days=1)


# Argument parsing
parser = argparse.ArgumentParser(description="Export daily IAP revenue data for a date range.")
parser.add_argument("start_date", type=str, help="Start date in YYYY-MM-DD format (or single date for one day query)")
parser.add_argument("end_date", type=str, nargs='?', help="End date in YYYY-MM-DD format (optional)")
args = parser.parse_args()

start_date = args.start_date
end_date = args.end_date if args.end_date else start_date

# Estimate the total size
total_gb = bq_estimate_total_size(start_date, end_date)

# Ask for confirmation before running the queries
proceed = input(f"Estimated total data size is {total_gb:.2f} GB. Proceed with query execution? (y/n): ").strip().lower()
if proceed == 'y':
    # Execute the queries for each day in the interval
    bq_run_queries(start_date, end_date)
else:
    print("Query execution canceled.")