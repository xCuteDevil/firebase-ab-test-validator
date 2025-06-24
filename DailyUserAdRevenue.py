# This script quiries the BigQuery database for daily user ad revenue data and saves the results in CSV files in a specified directory.
# The script estimates the total size of the data to be processed and asks for confirmation before running the queries.
# The script uses the google-cloud-bigquery library to interact with BigQuery:
# pip install google-cloud-bigquery
# outputs the data into /BQ_downloads/DailyUserAdRevenue in format YYYY-MM-DD.csv
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
output_directory = "DailyUserAdRevenue"
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
            # Query for users ad revenues on the current date
            query = f"""
                SELECT
                    user_pseudo_id,
                    SUM(ad_revenue) AS revenue_sum,
                    COUNTIF(new_event_name = "ad_impression") AS ad_impression_count,
                    MAX(time_in_game) AS max_time_in_game,
                    MAX(session_number) AS max_session_number,
                    ARRAY_AGG(STRUCT(max_world, max_level) ORDER BY max_world DESC, max_level DESC LIMIT 1)[OFFSET(0)].max_world AS max_world,
                    ARRAY_AGG(STRUCT(max_world, max_level) ORDER BY max_world DESC, max_level DESC LIMIT 1)[OFFSET(0)].max_level AS max_level
                FROM 
                    `hexapolis-bcb77.Events.Ads_events`
                WHERE 
                    event_date = "{date_str}"
                GROUP BY 
                    user_pseudo_id
            """

            # Run dry run to estimate size
            dry_run_job = client.query(query, job_config=bq.QueryJobConfig(dry_run=True))
            total_bytes += dry_run_job.total_bytes_processed

        # Move to the next day
        current_dt += timedelta(days=1)
    
    # Convert total bytes to gigabytes
    total_gb = total_bytes / (1024 ** 3)
    print(f"Total estimated bytes processed: {total_gb:.2f} GB")
    return total_gb


def bq_run_queries(start_date, end_date):
    client = bq.Client()

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    current_dt = start_dt
    while current_dt <= end_dt:
        date_str = current_dt.strftime("%Y-%m-%d")
        filename = os.path.join(output_directory, f"{date_str}.csv")
        
        # Only run the query if the file doesn't already exist
        if not os.path.exists(filename):
            # Query for users ad revenues on the current date
            query = f"""
                SELECT
                    user_pseudo_id,
                    SUM(ad_revenue) AS revenue_sum,
                    COUNTIF(new_event_name = "ad_impression") AS ad_impression_count,
                    MAX(time_in_game) AS max_time_in_game,
                    MAX(session_number) AS max_session_number,
                    ARRAY_AGG(STRUCT(max_world, max_level) ORDER BY max_world DESC, max_level DESC LIMIT 1)[OFFSET(0)].max_world AS max_world,
                    ARRAY_AGG(STRUCT(max_world, max_level) ORDER BY max_world DESC, max_level DESC LIMIT 1)[OFFSET(0)].max_level AS max_level
                FROM 
                    `hexapolis-bcb77.Events.Ads_events`
                WHERE 
                    event_date = "{date_str}"
                GROUP BY 
                    user_pseudo_id
            """

            # Run the actual query
            query_job = client.query(query)
            query_result = query_job.result(timeout=600)  # Set timeout to 600 seconds

            # Save each date's result directly in the output directory
            df_iter = query_result.to_dataframe_iterable()
            for i, df in enumerate(df_iter):
                df.to_csv(filename, header=(i == 0), index=False)

            print(f"Data for {date_str} saved to: {filename}")
        else:
            print(f"File for {date_str} already exists, skipping.")

        # Move to the next day
        current_dt += timedelta(days=1)


# Argument parsing
parser = argparse.ArgumentParser(description="Export daily ad revenue data for a date range.")
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