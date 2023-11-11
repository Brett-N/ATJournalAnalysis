import os
import pandas as pd

def process_folder(folder_path):
    all_dfs = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            all_dfs.append(df)

    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Save the combined dataframe to an xlsx file
    output_name = "Combined.xlsx"
    combined_df.to_excel(os.path.join(folder_path, output_name), index=False)
    print(f"Saved to {output_name}")

if __name__ == '__main__':
    directory = input("Enter the folder path containing the .csv files: ")
    process_folder(directory)