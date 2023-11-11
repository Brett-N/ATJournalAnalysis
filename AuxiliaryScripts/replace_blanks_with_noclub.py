import pandas as pd

def replace_blank_cells(filename):
    # Read the Excel file
    df = pd.read_excel(filename, engine='openpyxl')
    
    # Check if "Trail Club" column exists
    if "Trail club" not in df.columns:
        print("The 'Trail Club' column does not exist in the provided file.")
        return
    
    # Replace blank cells in the "Trail Club" column with "No Club"
    df['Trail club'].fillna("No Club", inplace=True)
    
    # Save the changes back to the Excel file
    df.to_excel(filename, index=False, engine='openpyxl')
    print("Blank cells in 'Trail club' column have been replaced.")

if __name__ == "__main__":
    filename = input("Enter the path to your .xlsx file: ")
    try:
        replace_blank_cells(filename)
    except Exception as e:
        print(f"An error occurred: {e}")