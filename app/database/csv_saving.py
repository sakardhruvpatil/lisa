# csv_saving.py
# Converting the database to CSV file for understanding of a normal user
import sys
sys.path.insert(0, '/home/dp/lisa/app/utils')
import pandas as pd
import os
from logger import log_bug, log_print

def save_collection_to_csv(collection, filename):
    try:
        data = list(collection.find())
        if data:
            df = pd.DataFrame(data)
            # Drop MongoDB-specific '_id' column if it exists
            if '_id' in df.columns:
                df.drop(columns=['_id'], inplace=True)
            
            # Handle unhashable types by excluding them from duplicates check
            unhashable_columns = [col for col in df.columns if isinstance(df[col].iloc[0], list)]
            if os.path.exists(filename):
                # If file exists, update it by appending new data
                df_existing = pd.read_csv(filename)
                # Avoid duplicating rows by checking for existing data, ignoring unhashable columns
                subset_columns = [col for col in df.columns if col not in unhashable_columns]
                combined_df = pd.concat([df_existing, df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=subset_columns)
            else:
                combined_df = df
            
            # Save the combined DataFrame back to CSV
            combined_df.to_csv(filename, index=False)
            log_print(f"Collection saved to {filename}.")
        else:
            log_print(f"No data found in collection to save for {filename}.")
    except Exception as e:
        log_bug(f"Failed to save collection to CSV. Filename: {filename}. Exception: {e}")

def save_history_to_csv(history_collection, filename):
    try:
        data = list(history_collection.find())
        rows = []

        for entry in data:
            date = entry.get("date")
            total_bedsheets = entry.get("total_bedsheets", 0)
            total_accepted = entry.get("total_accepted", 0)
            total_rejected = entry.get("total_rejected", 0)
            
            # Main row with the date and total values
            rows.append({
                "Date": date,
                "Total Bedsheets": total_bedsheets,
                "Total Accepted": total_accepted,
                "Total Rejected": total_rejected,
                "Threshold": "",  # Empty for the main row
                "Accepted": "",
                "Rejected": "",
            })
            
            # Subsequent rows for each threshold entry within the date
            for threshold_entry in entry.get("thresholds", []):
                rows.append({
                    "Date": "",  # Empty to avoid repeating the date
                    "Total Bedsheets": "",  # Empty for threshold sub-row
                    "Total Accepted": "",  # Empty for threshold sub-row
                    "Total Rejected": "",  # Empty for threshold sub-row
                    "Threshold": threshold_entry.get("set_threshold"),
                    "Accepted": threshold_entry.get("accepted", 0),
                    "Rejected": threshold_entry.get("rejected", 0),
                })
        
        # Create DataFrame from formatted data
        new_df = pd.DataFrame(rows)

        if os.path.exists(filename):
            # Load existing CSV data
            existing_df = pd.read_csv(filename)
            
            # Remove any rows in existing_df that have the same date as in new_df
            unique_dates = new_df['Date'].dropna().unique()
            existing_df = existing_df[~existing_df['Date'].isin(unique_dates)]
            
            # Concatenate and remove duplicates based on all columns to keep only latest
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df

        # Save the combined DataFrame to CSV
        combined_df.to_csv(filename, index=False)
        log_print(f"History collection saved to {filename}.")
    except Exception as e:
        log_bug(f"Failed to save history to CSV. Filename: {filename}. Exception: {e}")
