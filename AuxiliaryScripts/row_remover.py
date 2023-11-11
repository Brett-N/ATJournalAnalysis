import pandas as pd

def remove_blank_state_rows(input_file, output_file):
    # Read the xlsx file into a pandas DataFrame
    df = pd.read_excel(input_file, engine='openpyxl')
    
    # Filter out rows where 'State' is blank or NaN
    df = df[df['State'].notna()]
    
    # Write the filtered DataFrame back to the xlsx file
    df.to_excel(output_file, index=False, engine='openpyxl')

# Example usage:
file_path = 'C:\\Users\\19802\\VSCodeStuff\\FinalProgram\\Emotions.xlsx'
output_path = 'C:\\Users\\19802\\VSCodeStuff\\FinalProgram\\EmotionsTrimmed.xlsx'  # This can be the same as the input file if you want to overwrite it
remove_blank_state_rows(file_path, output_path)