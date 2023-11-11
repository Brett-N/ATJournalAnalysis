import os
import pandas as pd

# Prompt the user for the input Excel file name
input_file_name = input("Enter the input Excel file name (e.g., Emotions.xlsx): ")

try:
    # Load the Excel file into a pandas DataFrame
    df = pd.read_excel(input_file_name)

    # Get the base name of the input file (without extension)
    base_name = os.path.splitext(os.path.basename(input_file_name))[0]

    # Group the DataFrame by the "label" column
    grouped = df.groupby("label")

    # Get the directory of the input file
    output_directory = os.path.dirname(input_file_name)

    # Iterate through each group and save it to a separate file
    for label, group in grouped:
        output_file_name = os.path.join(output_directory, f"{base_name}_{label}.xlsx")
        group.to_excel(output_file_name, index=False, engine="openpyxl")
        print(f"Saved {len(group)} rows with label '{label}' to {output_file_name}")

    print("Separation complete!")
except FileNotFoundError:
    print(f"Error: The file '{input_file_name}' was not found.")
except Exception as e:
    print(f"An error occurred: {str(e)}")