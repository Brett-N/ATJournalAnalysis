import pandas as pd

def extract_first_500_rows(input_filepath, output_filepath):
    # Read the xlsx file
    df = pd.read_csv(input_filepath)

    # Take the first 500 rows
    df_500 = df.head(500)

    # Write to a new xlsx file
    df_500.to_csv(output_filepath, index=False)

if __name__ == "__main__":
    input_file = input("Enter the path to the .xlsx file: ")
    output_file = input("Enter the path for the output file: ")

    extract_first_500_rows(input_file, output_file)
    print(f"First 500 rows extracted and saved to {output_file}")