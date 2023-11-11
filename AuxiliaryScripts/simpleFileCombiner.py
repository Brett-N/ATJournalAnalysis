import os
import pandas as pd

def combine_xlsx_files(directory):
    all_dfs = []

    for filename in os.listdir(directory):
        if filename.endswith(".xlsx"):
            filepath = os.path.join(directory, filename)
            
            # Read the Excel file into a DataFrame
            df = pd.read_excel(filepath)
            all_dfs.append(df)

    # Combine all DataFrames into a single DataFrame
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Save the combined DataFrame to a new Excel file
    output_filepath = os.path.join(directory, "combined_output.xlsx")
    combined_df.to_excel(output_filepath, index=False)

    print(f"All files combined and saved to {output_filepath}")

if __name__ == "__main__":
    dir_path = input("Enter the path to the directory containing the .xlsx files: ")
    combine_xlsx_files(dir_path)