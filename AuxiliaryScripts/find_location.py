import pandas as pd

# Ask the user for the paths to the input files
blog_posts_file = input("Enter the path to the blog posts Excel file: ")
locations_file = input("Enter the path to the locations Excel file: ")

# Read the Excel files into DataFrames
blog_posts_df = pd.read_excel(blog_posts_file)
locations_df = pd.read_excel(locations_file)

# Function to clean and convert latitude values
def clean_latitude(value):
    try:
        # Remove any white spaces and convert to float
        return float(str(value).strip())
    except ValueError:
        # If conversion fails, return None
        return None

# Clean 'Latitude' columns for both DataFrames
blog_posts_df['Latitude'] = blog_posts_df['Latitude'].apply(clean_latitude)
locations_df['Latitude'] = locations_df['Latitude'].apply(clean_latitude)

# Remove rows where 'Latitude' values could not be converted to a float
blog_posts_df = blog_posts_df.dropna(subset=['Latitude'])
locations_df = locations_df.dropna(subset=['Latitude'])

# Round 'Latitude' columns for both DataFrames to two decimal places
blog_posts_df['Rounded Latitude'] = blog_posts_df['Latitude'].round(2)
locations_df['Rounded Latitude'] = locations_df['Latitude'].round(2)

# Merge the blog_posts_df with locations_df on 'Rounded Latitude'
merged_df = pd.merge(blog_posts_df, locations_df[['Rounded Latitude', 'Shelter Name']], 
                     left_on='Rounded Latitude', right_on='Rounded Latitude', how='left')

# Rename the 'Shelter Name' column to 'Location Name'
merged_df = merged_df.rename(columns={'Shelter Name': 'Location Name'})

# Check for unmatched rows and print them out
unmatched_rows = merged_df[merged_df['Location Name'].isnull()]
if not unmatched_rows.empty:
    print("The following rows in the blog posts file did not find a match in the locations file:")
    print(unmatched_rows[['Hiker trail name', 'Latitude']])

# Filter out unmatched rows
matched_df = merged_df.dropna(subset=['Location Name'])

# Ask the user for the path to save the updated file
output_file = input("Enter the path to save the updated blog posts Excel file: ")

# Save the updated DataFrame back to the specified Excel file
matched_df.to_excel(output_file, index=False)

print(f"Updated blog posts saved to '{output_file}'")