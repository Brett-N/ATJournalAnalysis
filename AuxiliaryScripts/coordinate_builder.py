import os
import pandas as pd

def process_csv_files(directory_path):
    # Ensure the path ends with a '/'
    if not directory_path.endswith('/'):
        directory_path += '/'

    # Print the content of the directory
    content = os.listdir(directory_path)
    if not content:
        print("Directory is empty. Exiting.")
        return
    else:
        print(f"Directory content: {content}")

    # Set the processed directory to the specified path
    processed_directory = r'C:\Users\19802\Downloads\Coordinates\processed'
    if not os.path.exists(processed_directory):
        os.makedirs(processed_directory)
        print("Created 'processed' directory at specified location.")

    # Iterate over each file in the directory
    for filename in content:
        if filename.endswith('.csv'):
            print(f"Processing {filename}...")
            # Extract the year from the filename
            year = filename.split('.')[0]
            # Read the CSV file into a DataFrame
            filepath = os.path.join(directory_path, filename)
            df = pd.read_csv(filepath)

            # Remove rows with empty "Latitude" and "Longitude"
            df = df.dropna(subset=['Latitude', 'Longitude'])

            # Check the column name for "Date" or "date"
            date_column = 'Date' if 'Date' in df.columns else 'date'
            
            # Append the year to the date column
            df[date_column] = df[date_column].astype(str) + ', ' + year

            # Save the modified DataFrame to the specified 'processed' directory
            new_filepath = os.path.join(processed_directory, filename)
            df.to_csv(new_filepath, index=False)
            print(f"Processed and saved {filename} to the specified 'processed' directory.")
        else:
            print(f"Skipped {filename} as it's not a .csv file.")

if __name__ == "__main__":
    directory = input("Enter the path to the directory containing the .csv files: ")
    process_csv_files(directory)